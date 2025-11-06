import os
from torch.utils.data import Dataset
import torch
import numpy as np
import random
from PIL import Image, ImageDraw

torch.set_printoptions(threshold=np.inf, linewidth=999999)
np.set_printoptions(threshold=np.inf, linewidth=999999)


class RPlanGEdgeSemanSimplified_78_10_prerunCNN(Dataset):
    def __init__(self, mode, random_training_data):
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

        self.random_training_data = random_training_data

    def __len__(self):
        '''return len(dataset)'''
        return len(self.files)

    def __getitem__(self, index):

        if self.mode == 'train':
            if not self.random_training_data:
                graph = np.load('../datasets/rplang-v3-withsemantics-withboundary/train/' + self.files[index],
                                allow_pickle=True).item()
                fileid = graph['file_id']  # integer id
                '''coord, (53, 2)'''
                corners = graph['corner_list_np_normalized_padding_withsemantics'][:, 0:2]

                '''corners padding mask, (53, 1)'''
                corners_padding_mask = graph['padding_mask']  # uint8
                '''boundary vertex indices, (53, 2)'''
                boundary_vertex_indices = graph['boundary_vertex_indices']  # [1 1] means boundary corner; [0 0] means non-boundary or padding
                boundary_infos = np.load('../datasets/rplang-v3-withsemantics-withboundary-v2/train/' + self.files[index],
                                         allow_pickle=True).item()
                boundary_adjacency_matrix = boundary_infos['boundary_adjacency_matrix'].astype(bool)  # Boundary adjacency matrix (bool)
            
                
        elif self.mode == 'val':
            graph = np.load('../datasets/rplang-v3-withsemantics-withboundary/val/' + self.files[index],
                            allow_pickle=True).item()
            fileid = graph['file_id']  # integer id
            corners = graph['corner_list_np_normalized_padding_withsemantics'][:, 0:2]
            '''corners padding mask, (53, 1)'''
            corners_padding_mask = graph['padding_mask']  # uint8
            '''boundary vertex indices, (53, 2)'''
            boundary_vertex_indices = graph['boundary_vertex_indices']  # [1 1] means boundary corner; [0 0] means non-boundary or padding
            boundary_infos = np.load('../datasets/rplang-v3-withsemantics-withboundary-v2/val/' + self.files[index],
                                     allow_pickle=True).item()
            boundary_adjacency_matrix = boundary_infos['boundary_adjacency_matrix'].astype(bool)  # Boundary adjacency matrix (bool)

        elif self.mode == 'test':
            graph = np.load('../datasets/rplang-v3-withsemantics-withboundary/test/' + self.files[index],
                            allow_pickle=True).item()
            fileid = graph['file_id']  # integer id
            corners = graph['corner_list_np_normalized_padding_withsemantics'][:, 0:2]
            '''corners padding mask, (53, 1)'''
            corners_padding_mask = graph['padding_mask']  # uint8
            '''boundary vertex indices, (53, 2)'''
            boundary_vertex_indices = graph['boundary_vertex_indices']  # [1 1] means boundary corner; [0 0] means non-boundary or padding
            boundary_infos = np.load('../datasets/rplang-v3-withsemantics-withboundary-v2/test/' + self.files[index],
                                     allow_pickle=True).item()
            boundary_adjacency_matrix = boundary_infos['boundary_adjacency_matrix'].astype(bool)  # Boundary adjacency matrix (bool)
        else:
            assert 0
        # Draw image for CNN; coordinates already normalized so later we normalize pixel values to [-1, 1]
        corners_unnorm = ((corners * boundary_vertex_indices) * 128 + 128).astype(np.int32)
        polygon_edges = np.triu(boundary_adjacency_matrix, 1)  # Take upper-triangular part (each undirected edge once)
        # Create a 256x256 white image canvas
        image = Image.new("RGB", (256, 256), "white")
        draw = ImageDraw.Draw(image)
        
        # Define line colors and widths (multi-scale boundary rendering)
        lines_info = [
            {'width': 7, 'color': "black"},
            {'width': 5, 'color': "black"},
            {'width': 3, 'color': "black"},
            {'width': 1, 'color': "black"}
        ]
        
        # Render polygon edges in multiple passes
        for line in lines_info:
            for i in range(len(corners_unnorm)):
                for j in range(i + 1, len(corners_unnorm)):
                    if polygon_edges[i, j]:
                        start_point = tuple(corners_unnorm[i])
                        end_point = tuple(corners_unnorm[j])
                        draw.line([start_point, end_point], fill=line['color'], width=line['width'])
        for i in range(len(corners_unnorm)):
            point = tuple(corners_unnorm[i])
            if boundary_vertex_indices[i, 0] == 1:
                draw.rectangle([point[0] - 3, point[1] - 3, point[0] + 3, point[1] + 3,], fill="black")

                
        
        # Normalize image pixel values: (pixel - 128) / 128 -> range roughly [-1, 1]
        normalized_image_array = (np.array(image).astype(np.int32) - 128) / 128
        normalized_image_array = normalized_image_array.transpose((2, 0, 1))
        return normalized_image_array, fileid

