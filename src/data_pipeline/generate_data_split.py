"""
generate_data_split.py

Description:
    Generates a .csv file specifying the train/validation/test split for dataset files.

    Input files are expected in:
        - DATA_DIR/event/<composer>
    Output, a CSV file mapping filenames to splits is saved as:
        - DATA_DIR/dataset_split.csv (default location unless --output is specified)

Usage:
    python3 generate_data_split.py [--train_ratio TRAIN_RATIO] [--output OUTPUT_PATH]

Arguments:
    --train_ratio   Proportion of data to allocate to the training set. 
                    The remaining data is split equally between validation and test.
                    Defaults to 0.8. (optional)

    --output        Path to save the output CSV file. Defaults to DATA_DIR/dataset_split.csv. (optional)

Example:
    # Generate a split with 80% training data
    python3 generate_data_split.py --train_ratio 0.8
"""
import os
import random
import pandas as pd

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.common import get_file_list
from utils.constants import DATA_DIR


def generate_split(fname_list, train_ratio=0.8):
    """Create a train/validation/test split from a list of file names.

    The split is: 
        - train: `train_ratio`
        - validation: `(1 - train_ratio) / 2`
        - test: `(1 - train_ratio) / 2`

    Args:
        fname_list (list): List of file paths or identifiers to split.
        train_ratio (float, optional):  Proportion of data to allocate to the training set. Defaults to 0.8.

    Returns:
        pandas.DataFrame: A DataFrame with two columns: 'filename' and 'split', 
                          where 'split' is one of {'train', 'val', 'test'}.
    """
    val_ratio = (1 - train_ratio) / 2

    n_file = len(fname_list)
    idx = list(range(n_file))
    random.shuffle(idx)

    n_train = round(n_file * train_ratio)
    n_val = round(n_file * val_ratio)
    train_idx = idx[: n_train]
    test_idx = idx[n_train: n_train + n_val]
    val_idx = idx[n_train + n_val:]

    fname_list = ['/'.join(i.split("/")[-2:]) for i in fname_list]
    df = pd.DataFrame({'fname': fname_list})
    df.loc[train_idx, 'split'] = 'train'
    df.loc[val_idx, 'split'] = 'val'
    df.loc[test_idx, 'split'] = 'test'

    return df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_ratio", dest="train_ratio", type=float, default=0.8,
                        help="Train/validation/test split ratio. Default to 0.8.")
    parser.add_argument("--output", dest="output", default=f"{DATA_DIR}/dataset_split.csv",
                        type=str, help="Output file name. Default to DATA_DIR/dataset_split.csv.")
    args = parser.parse_args()

    if os.path.exists(args.output):
        res = ""
        while res not in ["y", "n"]:
            res = input(f"Overwrite {args.output}? y/n")
        if res == "n":
            sys.exit(0)

    fname_list = get_file_list(os.path.join(DATA_DIR, "event"))
    df = generate_split(fname_list, args.train_ratio)
    df.to_csv(args.output, index=False)
