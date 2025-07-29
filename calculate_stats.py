import os
import json
import logging
import argparse
from collections import defaultdict, Counter
from itertools import combinations

import numpy as np
from scipy.spatial import KDTree
from sklearn.linear_model import RANSACRegressor

# --- Configuration ---
# K for K-Nearest Neighbors in inter-line spacing calculation
INTER_LINE_K = 10
# K for Heuristic Graph neighbor search
HEURISTIC_GRAPH_K = 6
# Cosine similarity threshold for opposite neighbors
OPPOSITE_NEIGHBOR_COS_SIM_THRESHOLD = -0.8

# --- Utility Functions ---

def setup_logging():
    """Configures the logging for the script."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

class NumpyEncoder(json.JSONEncoder):
    """ Custom encoder for numpy data types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def describe_distribution(data: np.ndarray, name: str) -> dict:
    """Computes descriptive statistics for a 1D numpy array."""
    if data.size == 0:
        logging.warning(f"Distribution '{name}' is empty. Returning NaN statistics.")
        return {
            "count": 0, "mean": float('nan'), "std": float('nan'),
            "min": float('nan'), "25%": float('nan'), "50%": float('nan'),
            "75%": float('nan'), "max": float('nan')
        }
    
    stats = {
        "count": len(data),
        "mean": np.mean(data),
        "std": np.std(data),
        "min": np.min(data),
        "25%": np.percentile(data, 25),
        "50%": np.percentile(data, 50),
        "75%": np.percentile(data, 75),
        "max": np.max(data)
    }
    return stats

def load_page_data(page_id: int, base_path: str) -> (dict, np.ndarray, np.ndarray):
    """Loads all data files for a single page."""
    logging.info(f"Loading data for page {page_id}...")
    try:
        dims_path = os.path.join(base_path, f"{page_id}_dims.txt")
        inputs_path = os.path.join(base_path, f"{page_id}_inputs_unnormalized.txt")
        labels_path = os.path.join(base_path, f"{page_id}_labels_textline.txt")

        dims_arr = np.loadtxt(dims_path)
        dims = {"width": dims_arr[0], "height": dims_arr[1]}
        
        points = np.loadtxt(inputs_path) # [x, y, s]
        labels = np.loadtxt(labels_path, dtype=int)

        assert points.shape[0] == labels.shape[0], \
            f"Page {page_id}: Mismatch between number of points ({points.shape[0]}) and labels ({labels.shape[0]})"
        assert points.shape[1] == 3, \
            f"Page {page_id}: Points data should have 3 columns (x, y, s)"

        return dims, points, labels
    except FileNotFoundError as e:
        logging.error(f"File not found for page {page_id}: {e}")
        return None, None, None
    except Exception as e:
        logging.error(f"Error loading data for page {page_id}: {e}")
        return None, None, None


# --- Core Statistical Calculation Functions ---

def calculate_intra_line_stats(lines_data: dict) -> dict:
    """
    Calculates statistics that exist *within* each text line.
    - Relative Horizontal Spacing (RHS)
    - Vertical Baseline Jitter
    - Intra-Line Font Size Variation (CV)
    - Local Writing Angle
    """
    logging.info("Calculating intra-line statistics...")
    all_rhs, all_jitter, all_font_size_cv, all_angles = [], [], [], []

    for line_label, line_data in lines_data.items():
        points = line_data['points']
        n_points = len(points)
        
        if n_points < 2:
            logging.warning(f"Line {line_label} has < 2 points. Skipping spacing/angle/jitter calculations.")
            continue

        # 1. Intra-Line Font Size Variation
        font_sizes = points[:, 2]
        if np.mean(font_sizes) > 0:
            cv = np.std(font_sizes) / np.mean(font_sizes)
            all_font_size_cv.append(cv)

        # Build KD-Tree for efficient intra-line neighbor search
        line_kdtree = KDTree(points[:, :2]) # Use only x, y for spatial search
        
        # Query for the nearest neighbor for each point in the line
        # k=2 because the point itself is the 0-th neighbor
        distances, indices = line_kdtree.query(points[:, :2], k=2)

        # 2. Relative Horizontal Spacing & 3. Local Writing Angle
        for i in range(n_points):
            neighbor_idx = indices[i, 1]
            p1 = points[i]
            p2 = points[neighbor_idx]

            dist = distances[i, 1]
            s_avg = (p1[2] + p2[2]) / 2.0
            if s_avg > 0:
                all_rhs.append(dist / s_avg)

            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            all_angles.append(np.arctan2(dy, dx))
            
        # 4. Vertical Baseline Jitter (using RANSAC for robustness)
        X = points[:, 0].reshape(-1, 1)
        y = points[:, 1]
        try:
            ransac = RANSACRegressor(random_state=42)
            ransac.fit(X, y)
            y_pred = ransac.predict(X)
            residuals = y - y_pred
            # Normalize jitter by font size
            jitter = residuals / points[:, 2]
            all_jitter.extend(jitter.tolist())
        except ValueError as e:
            logging.warning(f"Could not fit RANSAC for line {line_label} (n_points={n_points}): {e}")


    return {
        "relative_horizontal_spacing": np.array(all_rhs),
        "vertical_baseline_jitter": np.array(all_jitter),
        "intra_line_font_size_cv": np.array(all_font_size_cv),
        "local_writing_angle_rad": np.array(all_angles)
    }


def calculate_inter_line_stats(all_points: np.ndarray, lines_data: dict, page_dims: dict) -> dict:
    """
    Calculates statistics that describe relationships *between* text lines.
    - Relative Vertical Spacing (Line Spacing)
    - Line Alignment (Left, Right, Center)
    """
    logging.info("Calculating inter-line statistics...")
    all_rvs = []
    
    # 1. Relative Vertical Spacing (RVS)
    if len(all_points) > INTER_LINE_K:
        all_labels = np.array([ld['label'] for ld in lines_data.values() for _ in range(len(ld['points']))])
        # Build a KD-Tree of all points on the page
        page_kdtree = KDTree(all_points[:, :2])
        
        # For each point, find neighbors and filter for those on other lines
        distances, indices = page_kdtree.query(all_points[:, :2], k=INTER_LINE_K)
        
        for i in range(len(all_points)):
            current_label = all_labels[i]
            neighbor_indices = indices[i, 1:] # Exclude self
            neighbor_labels = all_labels[neighbor_indices]
            
            # Find neighbors on different lines
            other_line_mask = neighbor_labels != current_label
            
            if np.any(other_line_mask):
                other_line_neighbors = all_points[neighbor_indices[other_line_mask]]
                vertical_distances = np.abs(all_points[i, 1] - other_line_neighbors[:, 1])
                min_vd = np.min(vertical_distances)
                
                # Normalize by current point's font size
                if all_points[i, 2] > 0:
                    all_rvs.append(min_vd / all_points[i, 2])
    else:
        logging.warning("Not enough points on page to calculate inter-line spacing.")

    # 2. Line Alignment
    line_x_mins, line_x_maxs, line_centers = [], [], []
    page_width = page_dims['width']
    
    for line_label, line_data in lines_data.items():
        points = line_data['points']
        if len(points) > 0:
            x_coords = points[:, 0]
            x_min = np.min(x_coords)
            x_max = np.max(x_coords)
            
            # Normalize by page width for comparability
            line_x_mins.append(x_min / page_width)
            line_x_maxs.append(x_max / page_width)
            line_centers.append(((x_min + x_max) / 2.0) / page_width)

    return {
        "relative_vertical_spacing": np.array(all_rvs),
        "line_alignment_left_normalized": np.array(line_x_mins),
        "line_alignment_right_normalized": np.array(line_x_maxs),
        "line_alignment_center_normalized": np.array(line_centers)
    }

def calculate_page_level_stats(all_points: np.ndarray, page_dims: dict) -> dict:
    """
    Calculates global statistics for the entire page.
    - Page Aspect Ratio (Width / Height)
    - Page Density (Character and Ink)
    """
    logging.info("Calculating page-level statistics...")
    
    # --- NEW: Calculate Aspect Ratio ---
    width = page_dims.get('width', 0)
    height = page_dims.get('height', 0)
    
    if height > 0:
        aspect_ratio = width / height
    else:
        logging.warning("Page height is 0, cannot calculate aspect ratio. Setting to NaN.")
        aspect_ratio = float('nan')
        
    # --- Existing Density Calculations ---
    n_chars = len(all_points)
    if n_chars == 0:
        return {
            "aspect_ratio": aspect_ratio,
            "character_density": 0, 
            "ink_density": 0
        }
        
    x_coords, y_coords, sizes = all_points[:, 0], all_points[:, 1], all_points[:, 2]
    
    # Use text block bounding box for density, not full page dimensions
    text_block_width = np.max(x_coords) - np.min(x_coords)
    text_block_height = np.max(y_coords) - np.min(y_coords)
    text_block_area = text_block_width * text_block_height

    if text_block_area == 0:
        logging.warning("Text block area is zero, cannot calculate density.")
        char_density = float('inf')
        ink_density = float('inf')
    else:
        # Total area of "ink", approximating each char as a circle
        ink_area = np.sum(np.pi * (sizes / 2.0)**2)
        
        char_density = n_chars / text_block_area
        ink_density = ink_area / text_block_area

    return {
        "aspect_ratio": aspect_ratio,
        "character_density": char_density,
        "ink_density": ink_density
    }

def calculate_graph_based_stats(all_points: np.ndarray, page_dims: dict) -> dict:
    """
    Calculates statistics based on the heuristic graph construction method.
    - Heuristic Degree Distribution
    - Overlap Distribution
    """
    logging.info("Calculating graph-based statistics...")
    n_points = len(all_points)
    if n_points < HEURISTIC_GRAPH_K:
        logging.warning(f"Not enough points ({n_points}) for graph-based stats. Need at least {HEURISTIC_GRAPH_K}.")
        return {"heuristic_degree": np.array([]), "overlap": np.array([])}

    # Step 1: Normalization
    max_dim = max(page_dims['width'], page_dims['height'])
    max_s = np.max(all_points[:, 2])
    
    normalized_points = np.copy(all_points)
    normalized_points[:, :2] /= max_dim
    if max_s > 0:
        normalized_points[:, 2] /= max_s

    # Step 2: Heuristic Graph Construction
    kdtree = KDTree(normalized_points[:, :2])
    heuristic_directed_edges = []
    
    for i in range(n_points):
        # Find neighbors
        _, neighbor_indices = kdtree.query(normalized_points[i, :2], k=HEURISTIC_GRAPH_K)
        neighbor_indices = neighbor_indices[1:] # Exclude self
        
        best_pair = None
        min_dist_sum = float('inf')
        
        # Identify opposite pairs
        for n1_idx, n2_idx in combinations(neighbor_indices, 2):
            vec1 = normalized_points[n1_idx, :2] - normalized_points[i, :2]
            vec2 = normalized_points[n2_idx, :2] - normalized_points[i, :2]
            
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0: continue
            
            cosine_sim = np.dot(vec1, vec2) / (norm1 * norm2)
            
            if cosine_sim < OPPOSITE_NEIGHBOR_COS_SIM_THRESHOLD:
                dist_sum = norm1 + norm2
                if dist_sum < min_dist_sum:
                    min_dist_sum = dist_sum
                    best_pair = (n1_idx, n2_idx)

        # Create edges from the optimal pair
        if best_pair:
            heuristic_directed_edges.append((i, best_pair[0]))
            heuristic_directed_edges.append((i, best_pair[1]))
            
    # Step 3.1: Heuristic Degree
    degrees = np.zeros(n_points, dtype=int)
    for u, v in heuristic_directed_edges:
        degrees[u] += 1
        degrees[v] += 1
        
    # Step 3.2: Overlap
    edge_counts = Counter(heuristic_directed_edges)
    overlaps = []
    processed_edges = set()
    for u, v in heuristic_directed_edges:
        # To avoid double counting, create a canonical representation (min, max)
        edge_key = tuple(sorted((u, v)))
        if edge_key not in processed_edges:
            overlap_val = edge_counts.get((u, v), 0) + edge_counts.get((v, u), 0)
            overlaps.append(overlap_val)
            processed_edges.add(edge_key)

    return {
        "heuristic_degree": degrees,
        "overlap": np.array(overlaps)
    }

def aggregate_statistics(all_page_stats: list) -> dict:
    """Aggregates statistics from all pages into a single summary."""
    logging.info("Aggregating statistics from all pages...")
    if not all_page_stats:
        logging.warning("No page statistics to aggregate.")
        return {}

    # Initialize lists to collect all raw values from all pages
    collated = {
        'intra_line': defaultdict(list),
        'inter_line': defaultdict(list),
        'page_level': defaultdict(list),
        'graph_based': defaultdict(list)
    }

    num_pages_processed = len(all_page_stats)

    for page_stat in all_page_stats:
        # The `page_stat` object now only contains the stat categories
        for category, stats in page_stat.items():
            # This check is now more robust. It ensures `stats` is a dictionary.
            if not isinstance(stats, dict):
                logging.warning(f"Skipping non-dictionary item '{category}' in aggregation.")
                continue

            for stat_name, values in stats.items():
                if isinstance(values, np.ndarray):
                    collated[category][stat_name].append(values)
                else: # For single-value page-level stats
                    collated[category][stat_name].append(np.array([values]))

    # Concatenate and describe the distributions
    aggregated_summary = {}
    for category, stats in collated.items():
        aggregated_summary[category] = {}
        for stat_name, list_of_arrays in stats.items():
            # Check if there's anything to concatenate
            if not list_of_arrays:
                logging.warning(f"No data found for statistic '{stat_name}' in category '{category}'. Skipping.")
                continue
            
            full_distribution = np.concatenate(list_of_arrays)
            aggregated_summary[category][stat_name] = describe_distribution(full_distribution, f"{category}.{stat_name}")
    
    # Add a count of the number of pages processed
    aggregated_summary['metadata'] = {'num_pages_processed': num_pages_processed}

    return aggregated_summary


def main():
    """Main execution function."""
    setup_logging()
    parser = argparse.ArgumentParser(description="Calculate typographical and layout statistics from manuscript data.")
    parser.add_argument("dataset_path", type=str, help="Path to the base dataset directory.")
    parser.add_argument("--output_per_page", type=str, default="stats_per_page.json", help="Output JSON file for per-page statistics.")
    parser.add_argument("--output_aggregated", type=str, default="stats_aggregated.json", help="Output JSON file for aggregated statistics.")
    
    args = parser.parse_args()

    # Discover pages in the dataset directory
    try:
        files = os.listdir(args.dataset_path)
        page_ids = sorted(list(set([int(f.split('_')[0]) for f in files if f.split('_')[0].isdigit()])))
        logging.info(f"Found {len(page_ids)} pages in '{args.dataset_path}': {page_ids}")
    except Exception as e:
        logging.critical(f"Could not read dataset directory '{args.dataset_path}': {e}")
        return

    all_page_raw_stats_for_json = []
    stats_for_aggregation = []
    
    for page_id in page_ids:
        dims, points, labels = load_page_data(page_id, args.dataset_path)
        if dims is None:
            continue
        
        # Pre-process data into a more usable structure
        lines_data = defaultdict(lambda: {'points': [], 'indices': []})
        for i, (point, label) in enumerate(zip(points, labels)):
            lines_data[label]['points'].append(point)
            lines_data[label]['indices'].append(i)
            lines_data[label]['label'] = label
        
        for label in lines_data:
            lines_data[label]['points'] = np.array(lines_data[label]['points'])

        # Calculate all statistics for the current page
        page_stats_data = {
            'intra_line': calculate_intra_line_stats(lines_data),
            'inter_line': calculate_inter_line_stats(points, lines_data, dims),
            'page_level': calculate_page_level_stats(points, dims),
            'graph_based': calculate_graph_based_stats(points, dims),
        }
        stats_for_aggregation.append(page_stats_data)
        
        # Create a separate dictionary for JSON export that includes the page_id
        page_stats_with_id = {'page_id': page_id, **page_stats_data}
        all_page_raw_stats_for_json.append(page_stats_with_id)
    
    # Save per-page raw statistics
    logging.info(f"Saving per-page statistics to '{args.output_per_page}'...")
    with open(args.output_per_page, 'w') as f:
        json.dump(all_page_raw_stats_for_json, f, cls=NumpyEncoder, indent=4)

    # Aggregate and save summary statistics
    # We now pass only the list of statistical dictionaries, without the page_id
    aggregated_stats = aggregate_statistics(stats_for_aggregation)
    logging.info(f"Saving aggregated statistics to '{args.output_aggregated}'...")
    with open(args.output_aggregated, 'w') as f:
        json.dump(aggregated_stats, f, cls=NumpyEncoder, indent=4)

    logging.info("Processing complete.")


if __name__ == '__main__':
    main()