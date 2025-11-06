import os
from torch.utils.data import Dataset
import torch
import numpy as np
import random
from PIL import Image, ImageDraw

torch.set_printoptions(threshold=np.inf, linewidth=999999)
np.set_printoptions(threshold=np.inf, linewidth=999999)


class RPlanGEdgeSemanSimplified_56_32(Dataset):
    def __init__(self, mode):  # deterministic data (no random augmentation)
        super().__init__()
        self.mode = mode
        '''train(65763) & val(3000) & test(3000)'''
        if self.mode == 'train':
            self.files = os.listdir('../datasets/rplang-v3-withsemantics-withboundary/train')
        elif self.mode == 'val':
            self.files = os.listdir('../datasets/rplang-v3-withsemantics-withboundary/val')
        elif self.mode == 'test':
            self.files = os.listdir('../datasets/rplang-v3-withsemantics-withboundary/test')
        else:
            assert 0, 'mode error'
        self.files = sorted(self.files, key=lambda x: int(x[:-4]), reverse=False)


    def __len__(self):
        '''return len(dataset)'''
        return len(self.files)

    def __getitem__(self, index):

        if self.mode == 'train':
            feat = np.load('../datasets/prerunning_cnn_featuremaps/' + self.files[index], allow_pickle=True).item()
            # featmap_64 = feat[64][0] # ndarray(256, 64, 64)
            # featmap_32 = feat[32][0] # ndarray(512, 32, 32)
            featmap_16 = feat[16][0] # ndarray(1024, 16, 16)
        elif self.mode == 'val':
            feat = np.load('../datasets/prerunning_cnn_featuremaps/' + self.files[index], allow_pickle=True).item()
            # featmap_64 = feat[64][0] # ndarray(256, 64, 64)
            # featmap_32 = feat[32][0] # ndarray(512, 32, 32)
            featmap_16 = feat[16][0] # ndarray(1024, 16, 16)
        elif self.mode == 'test':
            feat = np.load('../datasets/prerunning_cnn_featuremaps/' + self.files[index], allow_pickle=True).item()
            # featmap_64 = feat[64][0] # ndarray(256, 64, 64)
            # featmap_32 = feat[32][0] # ndarray(512, 32, 32)
            featmap_16 = feat[16][0] # ndarray(1024, 16, 16)
        else:
            assert 0





















    

        if self.mode == 'train':
            graph = np.load('../datasets/rplang-v3-withsemantics/train/' + self.files[index], allow_pickle=True).item()
        elif self.mode == 'val':
            graph = np.load('../datasets/rplang-v3-withsemantics/val/' + self.files[index], allow_pickle=True).item()
        elif self.mode == 'test':
            graph = np.load('../datasets/rplang-v3-withsemantics/test/' + self.files[index], allow_pickle=True).item()
        else:
            assert 0, 'mode error'

        '''coords_withsemantics, (53, 16)'''
        corners_withsemantics = graph['corner_list_np_normalized_padding_withsemantics']
    # Initialize a new n*9 array (53, 9)
        corners_withsemantics_simplified = np.zeros((corners_withsemantics.shape[0], 9))
    # Copy columns 0 and 1 (x, y coordinates)
        corners_withsemantics_simplified[:, 0:2] = corners_withsemantics[:, 0:2]
    # Compute new column 2: sum of original semantic columns 2, 6, 12
        corners_withsemantics_simplified[:, 2] = (corners_withsemantics[:, [2, 6, 12]]).sum(axis=1)
    # Compute new column 3: sum of original semantic columns 3, 7, 8, 9, 10
        corners_withsemantics_simplified[:, 3] = (corners_withsemantics[:, [3, 7, 8, 9, 10]]).sum(axis=1)
    # Compute new column 4: sum of original semantic columns 13 and 14
        corners_withsemantics_simplified[:, 4] = (corners_withsemantics[:, [13, 14]]).sum(axis=1)
    # Copy original columns 4, 5, 11, 15 into simplified columns 5, 6, 7, 8
        corners_withsemantics_simplified[:, 5] = corners_withsemantics[:, 4]
        corners_withsemantics_simplified[:, 6] = corners_withsemantics[:, 5]
        corners_withsemantics_simplified[:, 7] = corners_withsemantics[:, 11]
        corners_withsemantics_simplified[:, 8] = corners_withsemantics[:, 15]



        
        '''attn 1 matrix, (53, 53)'''
        global_attn_matrix = graph['global_matrix_np_padding'].astype(bool)
        '''corners padding mask, (53, 1)'''
        corners_padding_mask = graph['padding_mask']

        '''edges, (2809, 1)'''
        edges = graph['edges']
 



        

        # return featmap_64, featmap_32, featmap_16, corners_withsemantics_simplified, global_attn_matrix, corners_padding_mask
        # return featmap_32, featmap_16, corners_withsemantics_simplified, global_attn_matrix, corners_padding_mask
        return featmap_16, corners_withsemantics_simplified, global_attn_matrix, corners_padding_mask, edges



