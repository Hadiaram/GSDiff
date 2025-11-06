import os
from torch.utils.data import Dataset
import torch
import numpy as np
import random
from PIL import Image, ImageDraw

torch.set_printoptions(threshold=np.inf, linewidth=999999)
np.set_printoptions(threshold=np.inf, linewidth=999999)


class RPlanGEdgeSemanSimplified_78_11(Dataset):
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
                '''coord, (53, 2)'''
                corners = graph['corner_list_np_normalized_padding_withsemantics'][:, 0:2]

                '''corners padding mask, (53, 1)'''
                corners_padding_mask = graph['padding_mask']  # uint8
                '''boundary vertex indices, (53, 2)'''
                boundary_vertex_indices = graph['boundary_vertex_indices']  # [1 1] means a boundary corner; [0 0] means a non-boundary corner or padding
                boundary_infos = np.load('../datasets/rplang-v3-withsemantics-withboundary-v2/train/' + self.files[index],
                                         allow_pickle=True).item()
                boundary_adjacency_matrix = boundary_infos['boundary_adjacency_matrix'].astype(bool)  # boundary adjacency matrix
            else:
                # Randomly generate a polygon
                graph = np.load('../datasets/rplang-v3-withsemantics-withboundary/train/' + self.files[index],
                                allow_pickle=True).item()
                corners = np.zeros((53, 2))
                boundary_corner_number = int((graph['boundary_vertex_indices'].sum() / 2).item())
                # Set grid size
                grid_size = 256
                # Initialize vertex list
                vertices = []
                # Add vertices
                for i in range(boundary_corner_number):
                    x, y = np.random.randint(0, grid_size, size=2)
                    vertices.append((x, y))
                # Vertex list
                vertices = np.stack(vertices, axis=0)
                # Normalize coordinates to [-1, 1]
                vertices = (vertices - 128) / 128
                corners[0:len(vertices), :] = vertices
                '''corners padding mask, (53, 1)'''
                corners_padding_mask = graph['padding_mask'] * 0
                corners_padding_mask[0:len(vertices), 0:1] = 1
                # boundary_vertex_indices (duplicate for both columns)
                boundary_vertex_indices = np.concatenate((corners_padding_mask, corners_padding_mask), axis=1)
                # boundary_adjacency_matrix (simple ring polygon)
                boundary_adjacency_matrix = np.zeros((53, 53)).astype(bool)
                for i in range(boundary_corner_number):
                    boundary_adjacency_matrix[i, (i + 1) % boundary_corner_number] = True
                boundary_adjacency_matrix = np.logical_or(boundary_adjacency_matrix, boundary_adjacency_matrix.T)
                
        elif self.mode == 'val':
            graph = np.load('../datasets/rplang-v3-withsemantics-withboundary/val/' + self.files[index],
                            allow_pickle=True).item()
            corners = graph['corner_list_np_normalized_padding_withsemantics'][:, 0:2]
            '''corners padding mask, (53, 1)'''
            corners_padding_mask = graph['padding_mask']  # uint8
            '''boundary vertex indices, (53, 2)'''
            boundary_vertex_indices = graph['boundary_vertex_indices']  # [1 1] means a boundary corner; [0 0] means a non-boundary corner or padding
            boundary_infos = np.load('../datasets/rplang-v3-withsemantics-withboundary-v2/val/' + self.files[index],
                                     allow_pickle=True).item()
            boundary_adjacency_matrix = boundary_infos['boundary_adjacency_matrix'].astype(bool)  # boundary adjacency matrix

        elif self.mode == 'test':
            graph = np.load('../datasets/rplang-v3-withsemantics-withboundary/test/' + self.files[index],
                            allow_pickle=True).item()
            corners = graph['corner_list_np_normalized_padding_withsemantics'][:, 0:2]
            '''corners padding mask, (53, 1)'''
            corners_padding_mask = graph['padding_mask']  # uint8
            '''boundary vertex indices, (53, 2)'''
            boundary_vertex_indices = graph['boundary_vertex_indices']  # [1 1] means a boundary corner; [0 0] means a non-boundary corner or padding
            boundary_infos = np.load('../datasets/rplang-v3-withsemantics-withboundary-v2/test/' + self.files[index],
                                     allow_pickle=True).item()
            boundary_adjacency_matrix = boundary_infos['boundary_adjacency_matrix'].astype(bool)  # boundary adjacency matrix
        else:
            assert 0
        # Render the image for CNN and normalize 0–255 pixel values to [-1, 1], same normalization as coordinates
        corners_unnorm = ((corners * boundary_vertex_indices) * 128 + 128).astype(np.int32)
        polygon_edges = np.triu(boundary_adjacency_matrix, 1)  # take the upper triangular portion
    # Create a 256x256 white image
        image = Image.new("RGB", (256, 256), "white")
        draw = ImageDraw.Draw(image)
        
    # Define line colors and widths
        lines_info = [
            {'width': 7, 'color': "black"},
            {'width': 5, 'color': "black"},
            {'width': 3, 'color': "black"},
            {'width': 1, 'color': "black"}
        ]
        
    # Draw lines
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

                
        
        # Normalize image pixel values: (pixel_value - 128) / 128
        normalized_image_array = (np.array(image).astype(np.int32) - 128) / 128
        normalized_image_array = normalized_image_array.transpose((2, 0, 1))
        return normalized_image_array

