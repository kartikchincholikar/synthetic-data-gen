# augment.py

import os
import sys
import yaml
import time
import logging
import argparse
import numpy as np
from pathlib import Path
from functools import partial
from multiprocessing import Pool, cpu_count
from types import SimpleNamespace # We will use this to create the adapter

from tqdm import tqdm

# Ensure the manuscript_generator package is in the Python path
sys.path.append(str(Path(__file__).parent.resolve()))

# --- Imports from the existing codebase ---
from manuscript_generator.augmentations import phase1_content, phase2_distortion, phase3_page
from manuscript_generator.augmentations import phase4_page
from manuscript_generator.core.registry import AUGMENTATIONS
from manuscript_generator.core.textbox import TextBox, TextBoxType
from manuscript_generator.utils.plotter import visualize_page, Page

# --- Imports for the new augmentation pipeline ---
from manuscript_generator.configs.augmentation_config import AugmentationConfig


def setup_logging():
    """Configures logging for the script."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(processName)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def load_real_page_data(page_id: str, base_path: Path) -> (dict, np.ndarray, np.ndarray):
    """Loads all data files for a single real-data page."""
    try:
        dims_path = base_path / f"{page_id}_dims.txt"
        inputs_path = base_path / f"{page_id}_inputs_unnormalized.txt"
        labels_path = base_path / f"{page_id}_labels_textline.txt"

        assert dims_path.exists(), f"Dimension file not found: {dims_path}"
        assert inputs_path.exists(), f"Inputs file not found: {inputs_path}"
        assert labels_path.exists(), f"Labels file not found: {labels_path}"

        dims_arr = np.loadtxt(dims_path)
        dims = {"width": dims_arr[0], "height": dims_arr[1]}
        
        points = np.loadtxt(inputs_path)
        labels = np.loadtxt(labels_path, dtype=int)

        assert points.shape[0] == labels.shape[0], \
            f"Page {page_id}: Mismatch between points ({points.shape[0]}) and labels ({labels.shape[0]})"
        assert points.shape[1] == 3, f"Page {page_id}: Points data must have 3 columns (x, y, s)"

        return dims, points, labels
    except Exception as e:
        logging.error(f"Failed to load data for page {page_id}: {e}")
        return None, None, None

def save_augmented_data(page_id: str, output_dir: Path, dims: dict, points: np.ndarray, labels: np.ndarray):
    """Saves the augmented data for a single page to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)

    longest_dim = max(dims['width'], dims['height'])
    assert longest_dim > 0, "Page dimensions must be positive."

    points_normalized = points.copy()
    points_normalized[:, :2] /= longest_dim
    points_normalized[:, 2] /= longest_dim

    np.savetxt(output_dir / f"{page_id}_inputs_unnormalized.txt", points, fmt="%.2f %.2f %d")
    np.savetxt(output_dir / f"{page_id}_inputs_normalized.txt", points_normalized, fmt="%.6f %.6f %.6f")
    np.savetxt(output_dir / f"{page_id}_labels_textline.txt", labels, fmt="%d")
    with open(output_dir / f"{page_id}_dims.txt", 'w') as f:
        f.write(f"{dims['width']} {dims['height']}")


def _augment_single_instance(task_info: tuple):
    """
    Worker function to perform augmentation on a single real data sample.
    """
    page_id, aug_idx, seed, config, input_dir, output_dir = task_info
    rng = np.random.default_rng(seed)
    
    dims, points, labels = load_real_page_data(page_id, input_dir)
    if dims is None:
        return f"Skipped page {page_id} due to loading error."

    aug_points, aug_labels = points.copy(), labels.copy()

    # --- START OF CORRECTION: THE ADAPTER PATTERN ---
    # Create an adapter object that mimics the structure of the original Config
    # so that we can pass it to the reused augmentation functions.
    config_adapter = SimpleNamespace()

    # Map Phase 1 config: The old functions expect a `textbox_content` attribute.
    config_adapter.textbox_content = SimpleNamespace(
        font_size_variation=config.phase1_content.font_size_variation.model_dump(),
        point_level_jitter=config.phase1_content.point_level_jitter.model_dump(),
        # Add a dummy entry for congestion_jitter as the original code might check for it
        congestion_jitter={'enabled': False}
    )

    # Map Phase 2 config: The old functions expect `textbox_distortion`.
    config_adapter.textbox_distortion = config.phase2_distortion

    # Map Phase 3 config: The old function expects `page_augmentations`.
    p3_dropout_config = config.phase3_page.point_dropout
    config_adapter.page_augmentations = SimpleNamespace(
        point_dropout={
            'enabled': p3_dropout_config.enabled,
            # The original function expects the distribution under the key 'probability'
            'probability': p3_dropout_config.dropout_probability_dist.model_dump()
        }
    )
    # --- END OF CORRECTION ---

    # --- AUGMENTATION PIPELINE ---
    
    # Phase 1: Content Augmentations
    p1_config = config.phase1_content
    # CORRECTED: Pass the config_adapter instead of the original config.
    if p1_config.font_size_variation.enabled and rng.random() < p1_config.font_size_variation.probability:
        aug_points = AUGMENTATIONS['font_size_variation'](aug_points, aug_labels, config_adapter, rng)
    if p1_config.point_level_jitter.enabled and rng.random() < p1_config.point_level_jitter.probability:
        aug_points = AUGMENTATIONS['point_level_jitter'](aug_points, aug_labels, config_adapter, rng)

    # Phase 2: Geometric Distortions (adapted for page-level)
    p2_config = config.phase2_distortion
    distortions = list(p2_config.model_dump().keys())
    rng.shuffle(distortions)

    center_transform = np.array([dims['width'] / 2.0, dims['height'] / 2.0, 0])
    centered_points = aug_points - center_transform

    dummy_textbox = TextBox(box_type=TextBoxType.MAIN_TEXT)
    dummy_textbox.points_local = centered_points
    dummy_textbox.width = dims['width']
    dummy_textbox.height = dims['height']
    
    for aug_name in distortions:
        aug_conf = getattr(p2_config, aug_name)
        if aug_conf.enabled and rng.random() < aug_conf.probability:
            # CORRECTED: Pass the config_adapter here as well.
            dummy_textbox.points_local = AUGMENTATIONS[aug_name](dummy_textbox.points_local, dummy_textbox, config_adapter, rng)

    aug_points = dummy_textbox.points_local + center_transform
    
    # Phase 4: Rigid Page Transforms
    # These are new and expect the new `AugmentationConfig` format, so no adapter needed.
    p4_config = config.phase4_page
    page_dims_dict = {'width': dims['width'], 'height': dims['height']}
    for aug_name, aug_conf in [
        ("page_rotation", p4_config.page_rotation),
        ("page_translation", p4_config.page_translation),
        ("page_mirror", p4_config.page_mirror)
    ]:
         if aug_conf.enabled and rng.random() < aug_conf.probability:
            aug_points = AUGMENTATIONS[aug_name](aug_points, page_dims_dict, config, rng)

    # Phase 3: Point Dropout (Applied last)
    p3_config = config.phase3_page.point_dropout
    if p3_config.enabled and rng.random() < p3_config.probability:
        # CORRECTED: Pass the config_adapter. The logic is now much cleaner.
        aug_points, kept_indices = AUGMENTATIONS['point_dropout'](aug_points, None, config_adapter, rng)
        aug_labels = aug_labels[kept_indices]

    # Save Augmented Data
    aug_page_id = f"{page_id}{aug_idx}"
    save_augmented_data(aug_page_id, output_dir, dims, aug_points, aug_labels)
    
    if config.general.visualize and (int(page_id) * config.general.num_augmentations_per_sample + aug_idx) % config.general.visualize_every_n == 0:
        viz_page = Page(width=int(dims['width']), height=int(dims['height']))
        viz_page.points = aug_points
        viz_page.textline_labels = aug_labels
        
        class DummyVizConfig:
            coloring = "textline"
            point_size_multiplier = 1.0
        
        viz_path = output_dir / f"{aug_page_id}.png"
        visualize_page(viz_page, DummyVizConfig(), viz_path)

    return None

def main():
    """Main execution function to run the augmentation pipeline."""
    setup_logging()
    parser = argparse.ArgumentParser(description="Augment real manuscript data using a defined pipeline.")
    parser.add_argument(
        '--config',
        type=str,
        default='configs/augmentation_config.yaml',
        help='Path to the YAML configuration file for augmentations.'
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    assert config_path.exists(), f"Configuration file not found at {config_path}"

    logging.info(f"Loading augmentation configuration from: {config_path}")
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    try:
        config = AugmentationConfig(**config_dict)
    except Exception as e:
        logging.critical(f"Error validating configuration: {e}")
        return

    input_dir = Path(config.general.input_dir)
    output_dir = Path(config.general.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    assert input_dir.is_dir(), f"Input directory not found: {input_dir}"
    
    files = os.listdir(input_dir)
    page_ids = sorted(list(set([f.split('_')[0] for f in files if f.endswith('_dims.txt')])))
    logging.info(f"Found {len(page_ids)} pages to augment in '{input_dir}'.")
    if not page_ids:
        logging.warning("No pages found. Exiting.")
        return

    tasks = []
    for page_id in page_ids:
        for i in range(config.general.num_augmentations_per_sample):
            seed = config.general.base_seed + int(page_id) * config.general.num_augmentations_per_sample + i
            tasks.append((page_id, i, seed, config, input_dir, output_dir))
    
    logging.info(f"Total augmentations to generate: {len(tasks)}")
    
    start_time = time.time()
    num_workers = config.general.num_workers
    if num_workers == -1:
        num_workers = cpu_count()

    if num_workers > 1 and len(tasks) > 1:
        logging.info(f"Using {num_workers} parallel workers.")
        with Pool(processes=num_workers) as pool:
            results = list(tqdm(pool.imap_unordered(_augment_single_instance, tasks), total=len(tasks), desc="Augmenting Data"))
    else:
        logging.info("Using a single process (sequential).")
        results = [_augment_single_instance(task) for task in tqdm(tasks, desc="Augmenting Data")]

    duration = time.time() - start_time
    errors = [r for r in results if r is not None]
    
    logging.info(f"\n--- Augmentation Complete ---")
    logging.info(f"Total time: {duration:.2f} seconds.")
    logging.info(f"Successfully generated: {len(tasks) - len(errors)} samples.")
    if errors:
        logging.warning(f"Encountered {len(errors)} errors. See logs for details.")

if __name__ == "__main__":
    main()