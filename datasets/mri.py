from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import h5py
import sigpy as sp


class SKM_TEA_MRI(Dataset):
    def __init__(self, target_files, center_slice=256, num_slices=100, echoes=int):
        self.target_files        = target_files
        self.num_slices       = num_slices
        self.center_slice     = center_slice
        self.ACS_size         = 32
        self.echoes         = echoes
        self._labels = torch.zeros(1)

    def __len__(self):
        # half are used for T1, and the other for T2
        return (len(self.target_files) * self.num_slices)//2

    def __getitem__(self, idx_):

        if self.echoes == 0:
            # use even slices for T1
            idx = idx_ * 2
        elif self.echoes == 1:
            # use odd slices for T2
            idx = idx_ * 2 + 1
        else:
            raise NotImplementedError("joint reconstruction not supported")
        # Convert to numerical
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Separate slice and sample
        sample_idx = idx // self.num_slices
        slice_idx  = self.center_slice + np.mod(idx, self.num_slices) - self.num_slices // 2


        if self.echoes==0:
            # Load MRI samples and maps
            with h5py.File(self.target_files[sample_idx], 'r') as contents:
                # Get k-space for specific slice
                target = np.asarray(contents['target'][slice_idx,:,:,0,0])# shape = [H,W]

            ksp = sp.resize(sp.fft(target,axes = (0,)), (160,160))
            target = sp.ifft(ksp, axes=(0,))

        elif self.echoes==1:
            # Load MRI samples and maps
            with h5py.File(self.target_files[sample_idx], 'r') as contents:
                # Get k-space for specific slice
                target = np.asarray(contents['target'][slice_idx,:,:,1,0])# shape = [H,W]
            ksp = sp.resize(sp.fft(target,axes = (0,)), (160,160))
            target = sp.ifft(ksp, axes=(0,))

        elif self.echoes==2:
            # Load MRI samples and maps
            with h5py.File(self.target_files[sample_idx], 'r') as contents:
                # Get k-space for specific slice
                target = np.asarray(contents['target'][slice_idx,:,:,:,0])# shape = [H,W,2]

            ksp = sp.resize(sp.fft(target,axes = (0,)), (160,160,2))
            target = sp.ifft(ksp, axes=(0,))



        norm_const = np.percentile(np.abs(target), 99)
        # print(norm_const)
        gt_img_cplx_norm = target/norm_const
        if self.echoes ==2:
            gt_img_2ch_norm  = torch.view_as_real(torch.tensor(gt_img_cplx_norm))
            gt_img_2ch_norm=gt_img_2ch_norm.permute(-1,-2,0,1).reshape(4,160,160)
        else:
            gt_img_2ch_norm  = torch.view_as_real(torch.tensor(gt_img_cplx_norm)).permute(-1,0,1)



        op = gt_img_2ch_norm
        return op, 0
