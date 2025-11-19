import os
from torch.utils.data import Dataset
import torch
import numpy as np
from . import tiny_graph
from .path_utils import get_data_path

torch.set_printoptions(threshold=np.inf, linewidth=999999)
np.set_printoptions(threshold=np.inf, linewidth=999999)


class RPlanGEdgeSemanSimplified(Dataset):
    '''
    hould have storaged data without padding, augmentation and normalization in advance.
   data is graphs, (V, E), V has attributes(coords, ...), E has adjacency matrices.
   (although we use ordered data structure like ndarray, we only use the order in adjacency matrices(instead of adjacency lists)
   to facilitate the data loading. we don't use the order in nn, to meet permutation invariability of graph nodes.)
    '''
    def __init__(self, mode):
        '''(1)data reading. np.load()
           (2)data filtering(generate_corner_number in eval and not in train).
           (3)normalization(you could also do this in __getitem__() even after batch loading, but this will lead to longer time in training and sampling).
           purpose: make nn do not need to learn distributions scale shifting, more easier to converge.
           (4)padding and attn mask generating.

           but if dataset very big, memory can't stand, you should storage each one as a file and read it'''
        super().__init__()
        self.mode = mode
        '''train(65763) & val(3000) & test(3000)'''
        if self.mode == 'train':
            self.files = [f for f in os.listdir(get_data_path('rplang-v3-withsemantics-withboundary', 'train')) if f.endswith('.npy')]
        elif self.mode == 'val':
            self.files = [f for f in os.listdir(get_data_path('rplang-v3-withsemantics-withboundary', 'val')) if f.endswith('.npy')]
        elif self.mode == 'test':
            self.files = [f for f in os.listdir(get_data_path('rplang-v3-withsemantics-withboundary', 'test')) if f.endswith('.npy')]
        else:
            assert 0, 'mode error'
        self.files = sorted(self.files, key=lambda x: int(x[:-4]), reverse=False)

    def __len__(self):
        '''return len(dataset)'''
        return len(self.files)

    def __getitem__(self, index):
        '''(1)get ndarray item by index.
          (2)random augmentation.
          return all unbatched things in ndarray in a dict'''

        if self.mode == 'train':
            graph = np.load(get_data_path('rplang-v3-withsemantics', 'train', self.files[index]), allow_pickle=True).item()
        elif self.mode == 'val':
            graph = np.load(get_data_path('rplang-v3-withsemantics', 'val', self.files[index]), allow_pickle=True).item()
        elif self.mode == 'test':
            graph = np.load(get_data_path('rplang-v3-withsemantics', 'test', self.files[index]), allow_pickle=True).item()
        else:
            assert 0, 'mode error'

        '''coords_withsemantics, (53, 16)'''
        corners_withsemantics = graph['corner_list_np_normalized_padding_withsemantics']
        # 初始化一个n*9的新数组(53, 9)
        corners_withsemantics_simplified = np.zeros((corners_withsemantics.shape[0], 9))
        # 复制第0、1列
        corners_withsemantics_simplified[:, 0:2] = corners_withsemantics[:, 0:2]
        # 计算新的第2列
        corners_withsemantics_simplified[:, 2] = (corners_withsemantics[:, [2, 6, 12]]).sum(axis=1)
        # 计算新的第3列
        corners_withsemantics_simplified[:, 3] = (corners_withsemantics[:, [3, 7, 8, 9, 10]]).sum(axis=1)
        # 计算新的第4列
        corners_withsemantics_simplified[:, 4] = (corners_withsemantics[:, [13, 14]]).sum(axis=1)
        # 复制第4、5、11、15列
        corners_withsemantics_simplified[:, 5] = corners_withsemantics[:, 4]
        corners_withsemantics_simplified[:, 6] = corners_withsemantics[:, 5]
        corners_withsemantics_simplified[:, 7] = corners_withsemantics[:, 11]
        corners_withsemantics_simplified[:, 8] = corners_withsemantics[:, 15]

        '''corners padding mask, (53, 1)'''
        corners_padding_mask = graph['padding_mask']

        '''attn 1 matrix, (53, 53)'''
        # Generate global attention matrix if not present (for converted data)
        if 'global_matrix_np_padding' in graph:
            global_attn_matrix = graph['global_matrix_np_padding'].astype(bool)
        else:
            # Create attention matrix based on padding mask
            # Only positions where both corners are valid (not padded) should be True
            n = corners_withsemantics.shape[0]
            valid_mask = corners_padding_mask.squeeze().astype(bool)  # (n,)
            # Create outer product to get matrix showing where both corners are valid
            global_attn_matrix = np.outer(valid_mask, valid_mask)  # (n, n)

        '''edges, (2809, 1)'''
        # Generate edges if not present (for converted data)
        if 'edges' in graph:
            edges = graph['edges']
        else:
            # Create edges based on the attention matrix structure
            # edges should be a flattened version of the adjacency matrix
            # Only include edges where attention matrix is True
            n = corners_withsemantics.shape[0]
            # Create a binary edge matrix (0 or 1) for all positions
            # In the depadding process, only edges where global_attn_matrix is True are kept
            # So we need edges to be size (n*n, 1) but values should be 1 for valid edges
            edges = np.ones((n * n, 1), dtype=np.float32)

        return corners_withsemantics_simplified, global_attn_matrix, corners_padding_mask, edges