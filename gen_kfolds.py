import argparse
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from src.utils import make_dir


parser = argparse.ArgumentParser()
parser.add_argument("--input_csv", help="Path to all csv file", type=str)
parser.add_argument("--k", help="Number of folds", type=int)
parser.add_argument("--save_dir", help="Folder to save split files to", type=str)
parser.add_argument("--seed", help="Random seed", type=int)
args = parser.parse_args()


def main():
    df = pd.read_csv(args.input_csv)
    np.random.seed(args.seed)
    make_dir(save_dir = args.save_dir)

    kf = KFold(n_splits=args.k, random_state=args.seed, shuffle=True)
    for fold, (train_index, test_index) in enumerate(kf.split(df)):
        print("FOLD:", fold+1, "TRAIN:", train_index, "TEST:", test_index)
        df_train = df.iloc[train_index]
        df_test = df.iloc[test_index]
        df_train.to_csv(os.path.join(args.save_dir,f'train_fold{fold+1}.csv'), index=False)
        df_test.to_csv(os.path.join(args.save_dir,f'test_fold{fold+1}.csv'), index=False)


if __name__ == "__main__":
    main()