import monai
import pandas as pd
import torch
from monai.transforms import Randomizable, AddChannel
from torch.utils.data import Dataset
import nibabel as nib



"""Construct the project's DataSet"""
class DTDataset(Dataset):

    def __init__(self, file, transform=None, seed=1):
        if type(file) is str:
            data = pd.read_csv(file)
        else:
            data = file
        self.dataA = data['image_pathA'].tolist()
        self.dataB = data['image_pathB'].tolist()
        self.labels = data['label'].tolist()
        self.wbc = data['wbc'].tolist()
        self.ne = data['ne'].tolist()
        self.d_d = data['d_d'].tolist()
        self.lactic = data['lactic'].tolist()
        self.transform = transform
        self.seed = seed

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        imgA = nib.load(self.dataA[index]).get_fdata()
        imgB = nib.load(self.dataB[index]).get_fdata()
        name = self.dataA[index][-11: -7]
        label = self.labels[index]
        wbc = self.wbc[index]
        ne = self.ne[index]
        d_d = self.d_d[index]
        lactic = self.lactic[index]
        if self.transform is not None:
            # Consistent pre-processing is guaranteed for the returned images(imgA and imgB)
            for t in self.transform.transforms:
                if isinstance(t, Randomizable):
                    t.set_random_state(seed=self.seed)
            imgA = self.transform(imgA)
            for t in self.transform.transforms:
                if isinstance(t, Randomizable):
                    t.set_random_state(seed=self.seed)
            imgB = self.transform(imgB)
            add_channel_transform = AddChannel()
            imgA = add_channel_transform(imgA)
            imgB = add_channel_transform(imgB)
        else:
            add_channel_transform = AddChannel()
            imgA = add_channel_transform(imgA)
            imgB = add_channel_transform(imgB)
        return {'imgA': imgA, 'imgB': imgB, 'label': label, 'index': name, 'wbc': wbc, 'ne': ne, 'd_d': d_d, 'lactic': lactic}

    def get_labels(self):
        return self.labels

    def update_seed(self, new_seed):
        self.seed = new_seed

    @staticmethod
    def collate_fn(batch):
        imgA_batch = torch.stack([sample['imgA'] for sample in batch], dim=0)
        imgB_batch = torch.stack([sample['imgB'] for sample in batch], dim=0)
        label_batch = torch.tensor([sample['label'] for sample in batch])
        index_batch = torch.tensor([sample['index'] for sample in batch])
        wbc_batch = torch.tensor([sample['wbc'] for sample in batch])
        ne_batch = torch.tensor([sample['ne'] for sample in batch])
        d_d_batch = torch.tensor([sample['d_d'] for sample in batch])
        lactic_batch = torch.tensor([sample['lactic'] for sample in batch])

        return {
            'imgA': imgA_batch,
            'imgB': imgB_batch,
            'label': label_batch,
            'index': index_batch,
            'wbc': wbc_batch,
            'ne': ne_batch,
            'd_d': d_d_batch,
            'lactic': lactic_batch
        }



