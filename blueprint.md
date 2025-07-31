
You are an expert in domain randomization, synthetic data generation, and data augmentation. I need your help in adapting a codebase which generates synthetic data to perform augmentation of real data. Please file

Below you will find python code which generates synthetic data for a layout analysis task - namely text lines detection. The synthetic dataset will have 2D points as inputs representing the location of the characters on the page. It will also have font size of each character (this can be different as we generate synthetic data for handwritten pages). The inputs are in normalized and unnormalized format. The labels are in a simple txt file, which marks all points in belonging to a particular text line with the same label.

## Directory sturcture
The generated synthetic data (using generate.py) of all pages is stored in the folder base-data. It looks something like this for page 0

-   **`0_dims.txt`**: To store the Page Dimensions and other stats. This will look like:
```
2729 943
```

-   **`0_inputs_unnormalized.txt`**: `x y font_size` in raw generation units.This will look like:
```
955.99 835.16 16
1009.06 829.70 12
1062.73 829.10 14
1115.60 828.71 16
so on...

```

-   **`0_inputs_normalized.txt`**: `x_norm y_norm font_size_norm`.
    -   `x_norm`, `y_norm`: Coords scaled such that the longest page dimension maps to `[0, 1]`.
    -   `font_size_norm`: Font size divided by the longest page dimension

This will look like:
```
0.458019 0.329295 0.008226
0.468476 0.331876 0.009123
0.479381 0.333814 0.007307
0.489580 0.334551 0.010435
so on...
```

-   **`0_labels_textline.txt`**: `textline_id` (globally unique integer).This will look like:
```
0
0
0
0
so on..
```

Now I want your help in adapting this code to reuse it to AUGMENT real-life data which will be stored in same format in the directory 'real-data'. For example data for real page0 will look like:
-   **`0_dims.txt`**
-   **`0_inputs_unnormalized.txt`**
-   **`0_inputs_normalized.txt`**
-   **`0_labels_textline.txt`**

Hence I want you to create the following files:
augment.py - which will take in data from 'real-data' directory, augment it, and save the augmented data in the repository 'augmented-data' in a similar format.
augmentation_config.yaml
augmentation_config.py
phase4_page.py - augmentations specific to real data.

Do not change any existing code. You make only create new files. Try to reuse code as much as possible (especially the augmentation code)

Details about augmentations:
We want to do all three augmentation phases, but with some nuances.
Phase 1 augmentations (similar to synthetic data)
Phase 2 augmentations (but at the page level instead of textbox level, because we don't have text box labels)
Phase 3 augmentation (similar to synthetic data)

You may also create real data specific augmentations in a new Phase 4
Phase 4 augmentations (these would be specific to real data only):
- translations (page level)
- rotations (page level)
- mirror (page level horizontal and vertical)
- any other augmentation you feel would be good

Please set the augmentation levels to be a bit milder than synthetic data in augmentation_config.yaml
Write entire files please. Also, write robust code, with lots of logging and 'assert' statement. The new files should not break any existing functionality.

Before coding, please study the relevant files from the codebase below:


