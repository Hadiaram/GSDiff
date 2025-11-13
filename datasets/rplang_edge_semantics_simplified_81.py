import os
from torch.utils.data import Dataset
import torch
import numpy as np
import random
from PIL import Image, ImageDraw
from tqdm import tqdm
from .path_utils import get_data_path

torch.set_printoptions(threshold=np.inf, linewidth=999999)
np.set_printoptions(threshold=np.inf, linewidth=999999)


class RPlanGEdgeSemanSimplified_81(Dataset):
    def __init__(self, mode): # 不随机数据
        super().__init__()
        self.mode = mode
        '''train(65763) & val(3000) & test(3000)'''
        if self.mode == 'train':
            self.files = os.listdir(get_data_path('rplang-v3-withsemantics-withboundary', 'train'))
        elif self.mode == 'val':
            self.files = os.listdir(get_data_path('rplang-v3-withsemantics-withboundary', 'val'))
        elif self.mode == 'test':
            self.files = os.listdir(get_data_path('rplang-v3-withsemantics-withboundary', 'test'))
        else:
            assert 0, 'mode error'
        self.files = sorted(self.files, key=lambda x: int(x[:-4]), reverse=False)
        self.ftmps = []
        for fn in tqdm(self.files):
            feat_data = np.load(get_data_path('prerunning_cnn_featuremaps', fn), allow_pickle=True)
            # Handle different numpy array formats
            if isinstance(feat_data, np.ndarray):
                if feat_data.ndim == 0:
                    feat_dict = feat_data.item()
                elif feat_data.size == 1:
                    feat_dict = feat_data.flatten()[0]
                else:
                    feat_dict = feat_data
            else:
                feat_dict = feat_data
            self.ftmps.append(feat_dict[16][0])


    def __len__(self):
        '''return len(dataset)'''
        return len(self.files)

    def __getitem__(self, index):

        # if self.mode == 'train':
        #     feat = np.load('../datasets/prerunning_cnn_featuremaps/' + self.files[index], allow_pickle=True).item()
        #     # featmap_64 = feat[64][0] # ndarray(256, 64, 64)
        #     # featmap_32 = feat[32][0] # ndarray(512, 32, 32)
        #     featmap_16 = feat[16][0] # ndarray(1024, 16, 16)
        # elif self.mode == 'val':
        #     feat = np.load('../datasets/prerunning_cnn_featuremaps/' + self.files[index], allow_pickle=True).item()
        #     # featmap_64 = feat[64][0] # ndarray(256, 64, 64)
        #     # featmap_32 = feat[32][0] # ndarray(512, 32, 32)
        #     featmap_16 = feat[16][0] # ndarray(1024, 16, 16)
        # elif self.mode == 'test':
        #     feat = np.load('../datasets/prerunning_cnn_featuremaps/' + self.files[index], allow_pickle=True).item()
        #     # featmap_64 = feat[64][0] # ndarray(256, 64, 64)
        #     # featmap_32 = feat[32][0] # ndarray(512, 32, 32)
        #     featmap_16 = feat[16][0] # ndarray(1024, 16, 16)
        # else:
        #     assert 0

        if self.mode == 'train':
            data = np.load(get_data_path('rplang-v3-withsemantics', 'train', self.files[index]), allow_pickle=True)
        elif self.mode == 'val':
            data = np.load(get_data_path('rplang-v3-withsemantics', 'val', self.files[index]), allow_pickle=True)
        elif self.mode == 'test':
            data = np.load(get_data_path('rplang-v3-withsemantics', 'test', self.files[index]), allow_pickle=True)
        else:
            assert 0, 'mode error'

        # Handle different numpy array formats
        if isinstance(data, np.ndarray):
            if data.ndim == 0:
                # 0-dimensional array containing an object (expected format)
                graph = data.item()
            elif data.size == 1:
                # Array with single element
                graph = data.flatten()[0]
            else:
                # Large array detected - likely wrong format
                raise ValueError(
                    f"ERROR: Loaded numpy file contains a raw array instead of a dictionary.\n"
                    f"  File: {self.files[index]}\n"
                    f"  Array shape: {data.shape}, dtype: {data.dtype}, size: {data.size}\n\n"
                    f"Expected format: A dictionary containing these keys:\n"
                    f"  - 'corner_list_np_normalized_padding_withsemantics': array of shape (53, 16)\n"
                    f"  - 'padding_mask': array of shape (53, 1)\n"
                    f"  - 'global_matrix_np_padding': array of shape (53, 53)\n"
                    f"  - 'edges': edge adjacency data\n\n"
                    f"How to fix:\n"
                    f"  1. Your data conversion script should save a DICTIONARY, not a raw array\n"
                    f"  2. Use: np.save(filepath, your_dict) where your_dict contains all required keys\n"
                    f"  3. See datasets/rplan-process4.py lines 1072-1079 for an example\n"
                    f"  4. Or see DATA_FORMAT_GUIDE.md for detailed instructions"
                )
        elif isinstance(data, dict):
            # Already a dict (shouldn't happen with np.load but handle it)
            graph = data
        else:
            raise ValueError(f"Unexpected data type from np.load: {type(data)}")

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



        
        '''attn 1 matrix, (53, 53)'''
        # global_attn_matrix = structure_graph['global_matrix_np_padding'].astype(np.uint8)
        global_attn_matrix = np.ones((53, 53), dtype=np.uint8)
        '''corners padding mask, (53, 1)'''
        corners_padding_mask = graph['padding_mask'] # uint8

        featmap_16 = self.ftmps[index]
 



        

        # return featmap_64, featmap_32, featmap_16, corners_withsemantics_simplified, global_attn_matrix, corners_padding_mask
        # return featmap_32, featmap_16, corners_withsemantics_simplified, global_attn_matrix, corners_padding_mask
        return featmap_16, corners_withsemantics_simplified, global_attn_matrix, corners_padding_mask



