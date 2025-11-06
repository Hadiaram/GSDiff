'''
Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp
Hochreiter. 2017. GANs trained by a two time-scale update rule converge to a local
Nash equilibrium. Advances in Neural Information Processing Systems 30 (2017).

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

def fid(path1, path2, fid_batch_size, fid_device):
    dims = 2048
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx]).to(fid_device).eval()

    files1 = [os.path.join(path1, fn) for fn in os.listdir(path1)]
    dataset1 = ImagePathDataset(files1, transforms=TF.ToTensor())
    dataloader1 = torch.utils.data.DataLoader(dataset1,
                                             batch_size=fid_batch_size,
                                             shuffle=False,
                                             drop_last=False,
                                             num_workers=0)  # Set to 0 for Windows compatibility
    pred_arr1 = np.empty((len(files1), dims)) # np.ndarray([len(test set), 2048])
    start_idx1 = 0
    for batch1 in tqdm(dataloader1):
        batch1 = batch1.to(fid_device) # torch.Size([64, 3, 256, 256])
        '''In FID we also employ the Inception network.
        It serves purely as a feature extractor whose final layer would normally output class logits.
        We remove the final fully-connected / pooling layer so we obtain a 2048‑D feature vector per image.'''
        with torch.no_grad():
            pred1 = model(batch1)[0] # torch.Size([64, 2048, 1, 1])
        pred1 = pred1.squeeze(3).squeeze(2).cpu().numpy() # np.ndarray([64, 2048])
        pred_arr1[start_idx1:start_idx1 + fid_batch_size] = pred1
        start_idx1 = start_idx1 + fid_batch_size
    # Feature space is 2048-D; mu is the 2048-D mean feature vector for an image set.
    # The larger the mean difference, the more dissimilar the two distributions.
    mu1 = np.mean(pred_arr1, axis=0) # np.ndarray([2048])
    # Covariance matrix: diagonal = variance of each dimension; off-diagonals = cross-dimension correlations.
    # Roughly captures the shape of the distribution.
    sigma1 = np.cov(pred_arr1, rowvar=False) # np.ndarray([2048, 2048])

    ''' Do the same processing for the second image set. '''
    files2 = [os.path.join(path2, fn) for fn in os.listdir(path2)]
    dataset2 = ImagePathDataset(files2, transforms=TF.ToTensor())
    dataloader2 = torch.utils.data.DataLoader(dataset2,
                                             batch_size=fid_batch_size,
                                             shuffle=False,
                                             drop_last=False,
                                             num_workers=0)  # Set to 0 for Windows compatibility
    pred_arr2 = np.empty((len(files2), dims))
    start_idx2 = 0
    for batch2 in tqdm(dataloader2):
        batch2 = batch2.to(fid_device)
        with torch.no_grad():
            pred2 = model(batch2)[0]
        pred2 = pred2.squeeze(3).squeeze(2).cpu().numpy()
        pred_arr2[start_idx2:start_idx2 + fid_batch_size] = pred2
        start_idx2 = start_idx2 + fid_batch_size
    mu2 = np.mean(pred_arr2, axis=0)
    sigma2 = np.cov(pred_arr2, rowvar=False)

    eps = 1e-6 # Numerical stability
    diff = mu1 - mu2 # Mean difference
    covmean = linalg.sqrtm(sigma1.dot(sigma2)) # Matrix square root of sigma1 * sigma2 (not element-wise)
    tr_covmean = np.trace(covmean) # Trace
    fid_value = (diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)

    return fid_value