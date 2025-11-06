'''
Demystifying MMD GANs. Authors: Mikołaj Bińkowski, Danica J. Sutherland, Michael Arbel, Arthur Gretton.

Computation method:
1 Render the gt and pred results in exactly the same way.
2 Put the two rendered image sets into /images_path1 and /images_path2.
'''
from tqdm import tqdm
import os
import numpy as np
import torch
import torchvision.transforms as TF
from PIL import Image
from scipy import linalg
from pytorch_fid.inception import InceptionV3
from sklearn.metrics.pairwise import polynomial_kernel, rbf_kernel
import sys


class ImagePathDataset(torch.utils.data.Dataset):
    def __init__(self, files, transforms=None):
        self.files = files
        self.transforms = transforms

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path = self.files[i]
        img = Image.open(path).convert('RGB')
        if self.transforms is not None:
            img = self.transforms(img)
        return img

def polynomial_mmd(features_generated, features_real, degree=3, gamma=None, coef0=1):
    kernel_xx = np.mean(polynomial_kernel(features_generated, features_generated, degree=degree, gamma=gamma, coef0=coef0))
    kernel_yy = np.mean(polynomial_kernel(features_real, features_real, degree=degree, gamma=gamma, coef0=coef0))
    kernel_xy = np.mean(polynomial_kernel(features_generated, features_real, degree=degree, gamma=gamma, coef0=coef0))
    mmd = kernel_xx + kernel_yy - 2 * kernel_xy
    return mmd


# path1 = '/home/myubt/Projects/house_diffusion-main/scripts/metrics/3-channel-semantics-128'
# path2 = '/home/myubt/Projects/house_diffusion-main/scripts/metrics/testimgs1'
# batch_size = 64
# device = 'cuda:0'
# dims = 2048
# block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
# model = InceptionV3([block_idx]).to(device).eval()
#
#
# files1 = [os.path.join(path1, fn) for fn in os.listdir(path1)]
# dataset1 = ImagePathDataset(files1, transforms=TF.ToTensor())
# dataloader1 = torch.utils.data.DataLoader(dataset1,
#                                          batch_size=batch_size,
#                                          shuffle=False,
#                                          drop_last=False,
#                                          num_workers=1)
# pred_arr1 = np.empty((len(files1), dims)) # np.ndarray([len(test set), 2048])
# start_idx1 = 0
# for batch1 in tqdm(dataloader1):
#     batch1 = batch1.to(device) # torch.Size([64, 3, 256, 256])
#     '''In the FID calculator we also use the Inception network.
#     Inception is essentially a feature extraction network; the last layer outputs class probabilities.
#     We remove the final fully connected or pooling layer so we obtain a 2048-dimensional feature vector.'''
#     with torch.no_grad():
#         pred1 = model(batch1)[0] # torch.Size([64, 2048, 1, 1])
#     pred1 = pred1.squeeze(3).squeeze(2).cpu().numpy() # np.ndarray([64, 2048])
#     pred_arr1[start_idx1:start_idx1 + batch_size] = pred1
#     start_idx1 = start_idx1 + batch_size
#
#
# ''' Do the same operation for the other image set. '''
# files2 = [os.path.join(path2, fn) for fn in os.listdir(path2)]
# dataset2 = ImagePathDataset(files2, transforms=TF.ToTensor())
# dataloader2 = torch.utils.data.DataLoader(dataset2,
#                                          batch_size=batch_size,
#                                          shuffle=False,
#                                          drop_last=False,
#                                          num_workers=1)
# pred_arr2 = np.empty((len(files2), dims))
# start_idx2 = 0
# for batch2 in tqdm(dataloader2):
#     batch2 = batch2.to(device)
#     with torch.no_grad():
#         pred2 = model(batch2)[0]
#     pred2 = pred2.squeeze(3).squeeze(2).cpu().numpy()
#     pred_arr2[start_idx2:start_idx2 + batch_size] = pred2
#     start_idx2 = start_idx2 + batch_size
#
# # Use the function above to compute the kernel MMD; this mmd is the KID score.
# mmd = polynomial_mmd(pred_arr1, pred_arr2)
# print("KID score:", mmd * 1000)

def kid2(path1, path2, kid_batch_size, kid_device):
    dims = 2048
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx]).to(kid_device).eval()

    files1 = [os.path.join(path1, fn) for fn in os.listdir(path1)]
    dataset1 = ImagePathDataset(files1, transforms=TF.ToTensor())
    dataloader1 = torch.utils.data.DataLoader(dataset1,
                                              batch_size=kid_batch_size,
                                              shuffle=False,
                                              drop_last=False,
                                              num_workers=1)
    pred_arr1 = np.empty((len(files1), dims))  # np.ndarray([len(test set), 2048])
    start_idx1 = 0
    for batch1 in tqdm(dataloader1):
        batch1 = batch1.to(kid_device)  # torch.Size([64, 3, 256, 256])
        '''In the FID calculator we also use the Inception network.
        Inception is essentially a feature extraction network; the last layer outputs the image class.
        We remove the final fully connected or pooling layer so we obtain a 2048-dimensional feature vector.'''
        with torch.no_grad():
            pred1 = model(batch1)[0]  # torch.Size([64, 2048, 1, 1])
        pred1 = pred1.squeeze(3).squeeze(2).cpu().numpy()  # np.ndarray([64, 2048])
        pred_arr1[start_idx1:start_idx1 + kid_batch_size] = pred1
        start_idx1 = start_idx1 + kid_batch_size

    ''' Do the same operation for the other image set. '''
    files2 = [os.path.join(path2, fn) for fn in os.listdir(path2)]
    dataset2 = ImagePathDataset(files2, transforms=TF.ToTensor())
    dataloader2 = torch.utils.data.DataLoader(dataset2,
                                              batch_size=kid_batch_size,
                                              shuffle=False,
                                              drop_last=False,
                                              num_workers=1)
    pred_arr2 = np.empty((len(files2), dims))
    start_idx2 = 0
    for batch2 in tqdm(dataloader2):
        batch2 = batch2.to(kid_device)
        with torch.no_grad():
            pred2 = model(batch2)[0]
        pred2 = pred2.squeeze(3).squeeze(2).cpu().numpy()
        pred_arr2[start_idx2:start_idx2 + kid_batch_size] = pred2
        start_idx2 = start_idx2 + kid_batch_size

    # Use the function above to compute the kernel MMD; this mmd is the KID score.
    mmd = polynomial_mmd(pred_arr1, pred_arr2)
    return mmd * 1000