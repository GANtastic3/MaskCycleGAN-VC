"""
Dataset class for voice conversion. Returns pairs of Mel-spectrograms from
two datasets as well as corresponding masks.
"""

from torch.utils.data.dataset import Dataset
import torch
import numpy as np


class VCDataset(Dataset):
    def __init__(self, datasetA, datasetB=None, n_frames=64, max_mask_len=25, valid=False):
        self.datasetA = datasetA
        self.datasetB = datasetB
        self.n_frames = n_frames
        self.valid = valid
        self.max_mask_len = max_mask_len

    def __getitem__(self, index):
        dataset_A = self.datasetA
        dataset_B = self.datasetB
        n_frames = self.n_frames
        
        if self.valid:
            if dataset_B is None:  # only return datasetA utterance
                return dataset_A[index]
            else:
                return dataset_A[index], dataset_B[index]

        self.length = min(len(dataset_A), len(dataset_B))
        num_samples = min(len(dataset_A), len(dataset_B))

        train_data_A_idx = np.arange(len(dataset_A))
        train_data_B_idx = np.arange(len(dataset_B))
        np.random.shuffle(train_data_A_idx)  # Why do we shuffle?
        np.random.shuffle(train_data_B_idx)
        train_data_A_idx_subset = train_data_A_idx[:num_samples]
        train_data_B_idx_subset = train_data_B_idx[:num_samples]

        train_data_A = list()
        train_mask_A = list()
        train_data_B = list()
        train_mask_B = list()

        for idx_A, idx_B in zip(train_data_A_idx_subset, train_data_B_idx_subset):
            data_A = dataset_A[idx_A]
            frames_A_total = data_A.shape[1]
            assert frames_A_total >= n_frames
            start_A = np.random.randint(frames_A_total - n_frames + 1)
            end_A = start_A + n_frames
            mask_size_A = np.random.randint(0, self.max_mask_len)
            assert n_frames > mask_size_A
            mask_start_A = np.random.randint(0, n_frames - mask_size_A)
            mask_A = np.ones_like(data_A[:, start_A:end_A])
            mask_A[:, mask_start_A:mask_start_A + mask_size_A] = 0.
            train_data_A.append(data_A[:, start_A:end_A])
            train_mask_A.append(mask_A)

            data_B = dataset_B[idx_B]
            frames_B_total = data_B.shape[1]
            assert frames_B_total >= n_frames
            start_B = np.random.randint(frames_B_total - n_frames + 1)
            end_B = start_B + n_frames
            mask_size_B = np.random.randint(0, self.max_mask_len)
            assert n_frames > mask_size_B
            mask_start_B = np.random.randint(0, n_frames - mask_size_B)
            mask_B = np.ones_like(data_A[:, start_A:end_A])
            mask_B[:, mask_start_B:mask_start_B + mask_size_B] = 0.
            train_data_B.append(data_B[:, start_B:end_B])
            train_mask_B.append(mask_B)

        train_data_A = np.array(train_data_A)
        train_data_B = np.array(train_data_B)
        train_mask_A = np.array(train_mask_A)
        train_mask_B = np.array(train_mask_B)

        return train_data_A[index], train_mask_A[index],  train_data_B[index], train_mask_B[index]

    def __len__(self):
        if self.datasetB is None:
            return len(self.datasetA)
        else:
            return min(len(self.datasetA), len(self.datasetB))


if __name__ == '__main__':
    # Trivial test for dataset class
    trainA = np.random.randn(162, 24, 554)
    trainB = np.random.randn(158, 24, 554)
    dataset = VCDataset(trainA, trainB)
    trainLoader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=2,
                                              shuffle=True)
    for i, (A, mask_A, B, mask_B) in enumerate(trainLoader):
        print(A.shape, mask_B.shape, B.shape, mask_B.shape)
        assert A.shape == mask_B.shape == B.shape == mask_B.shape
        break
