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


class RPlanGEdgeSemanSimplified_81_WithEdges(Dataset):
    """
    Dataset class for Stage 2 edge training with 150 corners (boundary-constrained).
    Returns: feat_16, corners_withsemantics, global_attn_matrix, corners_padding_mask, edges
    """
    def __init__(self, mode):
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
        # Sort by filename (supports both numeric and descriptive names)
        try:
            # Try numeric sort first (for backward compatibility)
            self.files = sorted(self.files, key=lambda x: int(x[:-4]), reverse=False)
        except ValueError:
            # Fall back to string sort for descriptive filenames
            self.files = sorted(self.files, reverse=False)

        # Preload CNN feature maps
        print(f"Loading CNN feature maps for {self.mode} set...")
        self.ftmps = []
        for fn in tqdm(self.files):
            self.ftmps.append(np.load(get_data_path('prerunning_cnn_featuremaps', fn), allow_pickle=True).item()[16][0])


    def __len__(self):
        '''return len(dataset)'''
        return len(self.files)

    def __getitem__(self, index):
        # Load graph data
        if self.mode == 'train':
            graph = np.load(get_data_path('rplang-v3-withsemantics', 'train', self.files[index]), allow_pickle=True).item()
        elif self.mode == 'val':
            graph = np.load(get_data_path('rplang-v3-withsemantics', 'val', self.files[index]), allow_pickle=True).item()
        elif self.mode == 'test':
            graph = np.load(get_data_path('rplang-v3-withsemantics', 'test', self.files[index]), allow_pickle=True).item()
        else:
            assert 0, 'mode error'

        '''coords_withsemantics, (150, 16) -> (150, 9)'''
        corners_withsemantics = graph['corner_list_np_normalized_padding_withsemantics']
        # Initialize new (150, 9) array
        corners_withsemantics_simplified = np.zeros((corners_withsemantics.shape[0], 9))
        # Copy columns 0, 1 (coordinates)
        corners_withsemantics_simplified[:, 0:2] = corners_withsemantics[:, 0:2]
        # Calculate new column 2
        corners_withsemantics_simplified[:, 2] = (corners_withsemantics[:, [2, 6, 12]]).sum(axis=1)
        # Calculate new column 3
        corners_withsemantics_simplified[:, 3] = (corners_withsemantics[:, [3, 7, 8, 9, 10]]).sum(axis=1)
        # Calculate new column 4
        corners_withsemantics_simplified[:, 4] = (corners_withsemantics[:, [13, 14]]).sum(axis=1)
        # Copy columns 4, 5, 11, 15
        corners_withsemantics_simplified[:, 5] = corners_withsemantics[:, 4]
        corners_withsemantics_simplified[:, 6] = corners_withsemantics[:, 5]
        corners_withsemantics_simplified[:, 7] = corners_withsemantics[:, 11]
        corners_withsemantics_simplified[:, 8] = corners_withsemantics[:, 15]

        '''attn 1 matrix, (150, 150)'''
        max_corners = corners_withsemantics.shape[0]
        global_attn_matrix = graph.get('global_matrix_np_padding', np.ones((max_corners, max_corners), dtype=np.uint8))
        if global_attn_matrix.dtype != bool:
            global_attn_matrix = global_attn_matrix.astype(bool)

        '''corners padding mask, (150, 1)'''
        corners_padding_mask = graph['padding_mask']  # uint8

        '''edges, (22500, 1) for 150x150=22500'''
        # Load edges from graph data (flattened adjacency matrix)
        edges = graph.get('edges', np.zeros((max_corners * max_corners, 1), dtype=np.float64))

        # Get preloaded CNN feature map
        featmap_16 = self.ftmps[index]

        return featmap_16, corners_withsemantics_simplified, global_attn_matrix, corners_padding_mask, edges
