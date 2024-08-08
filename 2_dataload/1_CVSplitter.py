import numpy as np
import pandas as pd
import os

class DatasetSplitter:
    """Divide the dataset into k-folds for cross-validation and save CSV files to a specific path."""

    def __init__(self, dataset, out_path, num_folds=5, seed=42):
        self.files = pd.read_csv(dataset)
        self.num_folds = num_folds
        self.seed = seed
        self.base_path = out_path

    def split_and_save(self):
        total_samples = len(self.files)
        indices = np.arange(total_samples)
        np.random.seed(self.seed)
        np.random.shuffle(indices)

        fold_size = total_samples // self.num_folds
        folds = []

        for i in range(self.num_folds):
            start = i * fold_size
            end = start + fold_size if i != self.num_folds - 1 else total_samples
            fold_indices = indices[start:end]
            folds.append(fold_indices)

        for i in range(self.num_folds):
            val_indices = folds[i]
            train_indices = np.hstack([folds[j] for j in range(self.num_folds) if j != i])

            train_path = self.files.iloc[train_indices].sort_index()
            train_size = len(train_path)
            train_pos_count = sum(train_path['label'].tolist())
            train_pos_ration = round(train_pos_count / train_size *100, 2)

            val_path = self.files.iloc[val_indices].sort_index()
            val_size = len(val_path)
            val_pos_count = sum(val_path['label'].tolist())
            val_pos_ration = round(val_pos_count / val_size * 100, 2)

            train_path.to_csv(os.path.join(self.base_path, f'train_fold{i}.csv'), index=False)
            val_path.to_csv(os.path.join(self.base_path, f'val_fold{i}.csv'), index=False)

            print(f"Fold {i}: Train({train_size}) including positive img{train_pos_count}({train_pos_ration}%), "
                  f"and validation({val_size}) including positive img{val_pos_count}({val_pos_ration}%) datasets are saved.")

if __name__ == '__main__':
    dataset_path = '/1_dataset/labels.csv'  # Update this to your dataset path
    out_path = "/1_dataset/"
    splitter = DatasetSplitter(dataset_path, out_path, num_folds=5, seed=42)
    splitter.split_and_save()
