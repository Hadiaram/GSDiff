import sys

import numpy as np

sys.path.append('/home/user00/HSZ/gsdiff-main')
sys.path.append('/home/user00/HSZ/gsdiff-main/datasets')
sys.path.append('/home/user00/HSZ/gsdiff-main/gsdiff')
sys.path.append('/home/user00/HSZ/gsdiff-main/scripts/metrics')


import math
import torch
import shutil
from torch.optim import AdamW, SGD
from torch.utils.data import DataLoader
from itertools import cycle
from .rplang_edge_semantics_simplified import RPlanGEdgeSemanSimplified
from gsdiff.utils import *
import torch.nn.functional as F
from scripts.metrics.fid import fid
from scripts.metrics.kid import kid
import copy
import os
from tiny_graph import train as train_tiny_graph


train_fnids = [int(fnid[:-4]) for fnid in train_tiny_graph]


def deep_compare(a, b):
    '''Determine whether two Python data structures (dict/list/tuple/ndarray/scalar) are completely identical.'''
    # If the types differ, return False immediately
    if type(a) != type(b):
        return False

    # If a dict, compare keys then recursively compare corresponding values
    if isinstance(a, dict):
        if a.keys() != b.keys():
            return False
        return all(deep_compare(a[key], b[key]) for key in a)

    # If a list or tuple, compare element by element
    elif isinstance(a, (list, tuple)):
        if len(a) != len(b):
            return False
        return all(deep_compare(item1, item2) for item1, item2 in zip(a, b))

    # If a numpy array, use np.array_equal for exact match
    elif isinstance(a, np.ndarray):
        return np.array_equal(a, b)

    # Other types: direct comparison (int, float, str, etc.)
    else:
        return a == b

def check_subdir_file_counts(base_dir):
    subdirs = ['train']
    for subdir in subdirs:
        full_path = os.path.join(base_dir, subdir)
        file_list = [f for f in os.listdir(full_path)
                     if os.path.isfile(os.path.join(full_path, f))]
    print('File count: ' + str(len(file_list)))


bubblecategory = {}
bubblecategory[0] = 0
bubblecategory[1] = 0
bubblecategory[2] = 0
bubblecategory[3] = 0
bubblecategory[4] = 0
bubblecategory[5] = 0
edgecategory = {}
edgecategory[0] = 0
edgecategory[1] = 0

boundary_num = [] # max: 38 (76 dim)

'''Input data: rplang-v3-withsemantics'''
'''Output data: rplang-v3-withsemantics-withboundary and rplang-v3-withsemantics-withboundary-v2'''

if not os.path.exists('./rplandata/Data/rplang-v3-withsemantics-withboundary'):
    os.mkdir('./rplandata/Data/rplang-v3-withsemantics-withboundary')
os.mkdir('./rplandata/Data/rplang-v3-withsemantics-withboundary/train')

if not os.path.exists('./rplandata/Data/rplang-v3-withsemantics-withboundary-v2'):
    os.mkdir('./rplandata/Data/rplang-v3-withsemantics-withboundary-v2')
os.mkdir('./rplandata/Data/rplang-v3-withsemantics-withboundary-v2/train')

train_path = './rplandata/Data/rplang-v3-withsemantics/train'

train_files = os.listdir(train_path)
for train_file in tqdm(train_files):
    # print(train_file)
    train_graph = np.load(train_path + '/' + train_file, allow_pickle=True).item()

    '''file_id
        corners
        adjacency_matrix
        adjacency_list
        corners_np
        adjacency_matrix_np
        adjacency_list_np
        corner_list_np_normalized
        corner_list_np_normalized_padding
        padding_mask
        global_matrix_np_padding
        adjacency_matrix_np_padding
        edge_coords
        edges
        semantics
        corner_list_np_normalized_padding_withsemantics
        '''
    '''coords_withsemantics, (53, 16)'''
    corners_withsemantics = train_graph['corner_list_np_normalized_padding_withsemantics']
    # Initialize a new n*9 array (53, 9)
    corners_withsemantics_simplified = np.zeros((corners_withsemantics.shape[0], 9))
    # Copy columns 0 and 1 (coordinates)
    corners_withsemantics_simplified[:, 0:2] = corners_withsemantics[:, 0:2]
    # Compute new column 2 (label 0: living room / dining room / entrance)
    corners_withsemantics_simplified[:, 2] = (corners_withsemantics[:, [2, 6, 12]]).sum(axis=1)
    # Compute new column 3 (label 1: bedroom / study)
    corners_withsemantics_simplified[:, 3] = (corners_withsemantics[:, [3, 7, 8, 9, 10]]).sum(axis=1)
    # Compute new column 4 (label 2: cabinet)
    corners_withsemantics_simplified[:, 4] = (corners_withsemantics[:, [13, 14]]).sum(axis=1)
    # Others
    corners_withsemantics_simplified[:, 5] = corners_withsemantics[:, 4] # label 3: kitchen
    corners_withsemantics_simplified[:, 6] = corners_withsemantics[:, 5] # label 4: bathroom
    corners_withsemantics_simplified[:, 7] = corners_withsemantics[:, 11] # label 5: balcony
    corners_withsemantics_simplified[:, 8] = corners_withsemantics[:, 15] # label 6: exterior

    '''attn 1 matrix, (53, 53)'''
    global_attn_matrix = train_graph['global_matrix_np_padding'].astype(bool)
    '''corners padding mask, (53, 1)'''
    corners_padding_mask = train_graph['padding_mask']

    '''edges, (2809, 1)'''
    edges = train_graph['edges']
    corners_withsemantics_0_train = corners_withsemantics_simplified[None, :, :]
    global_attn_matrix_train = global_attn_matrix[None, :, :]
    corners_padding_mask_train = corners_padding_mask[None, :, :]
    edges_train = edges[None, :, :]
    corners_withsemantics_0_train = corners_withsemantics_0_train.clip(-1, 1)
    corners_0_train = (corners_withsemantics_0_train[0, :, :2] * 128 + 128).astype(int)
    semantics_0_train = corners_withsemantics_0_train[0, :, 2:].astype(int)
    global_attn_matrix_train = global_attn_matrix_train
    corners_padding_mask_train = corners_padding_mask_train
    edges_train = edges_train
    corners_0_train_depadded = corners_0_train[corners_padding_mask_train.squeeze() == 1][None, :, :]  # (n, 2)
    semantics_0_train_depadded = semantics_0_train[corners_padding_mask_train.squeeze() == 1][None, :, :]  # (n, 7)
    edges_train_depadded = edges_train[global_attn_matrix_train.reshape(1, -1, 1)][None, :, None]
    edges_train_depadded = np.concatenate((1 - edges_train_depadded, edges_train_depadded), axis=2)

    ''' get planar cycles'''
    # Shape (1,n,14) ndarray with 0/1; find indices of 1s in each subarray and replace 0-valued positions with 99999
    semantics_gt_i_transform_train = semantics_0_train_depadded
    semantics_gt_i_transform_indices_train = np.indices(semantics_gt_i_transform_train.shape)[-1]
    semantics_gt_i_transform_train = np.where(semantics_gt_i_transform_train == 1,
                                              semantics_gt_i_transform_indices_train, 99999)

    gt_i_points_train = [tuple(corner_with_seman_train) for corner_with_seman_train in
                         np.concatenate((corners_0_train_depadded, semantics_gt_i_transform_train), axis=-1).tolist()[
                             0]]
    # print(output_points)
    gt_i_edges_train = edges_to_coordinates(
        np.triu(edges_train_depadded[0, :, 1].reshape(len(gt_i_points_train), len(gt_i_points_train))).reshape(-1),
        gt_i_points_train)

    # print(gt_i_points_train)
    # print(gt_i_edges_train)

    d_rev_train, simple_cycles_train, simple_cycles_semantics_train = get_cycle_basis_and_semantic_2_semansimplified_4extractingboundary(
        gt_i_points_train,
        gt_i_edges_train)
    simple_cycles_train_ = []
    for sc in simple_cycles_train:
        sc_train = [(t[0], t[1]) for t in sc]
        simple_cycles_train_.append(sc_train)
        # print(sc_train)
    # for scs in simple_cycles_semantics_train:
    #     print(scs)
    polygons = simple_cycles_train_
    # for p in polygons:
    #     print(p)


    # Define a function to compute the angle between two vectors
    def angle(v1, v2):
        dot_product = np.dot(v1, v2)
        norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
        return np.degrees(np.arccos(dot_product / norm_product))


    non_flat_angles = []
    count = 0
    for polygon_i, polygon in enumerate(polygons):
        if simple_cycles_semantics_train[polygon_i] == 6:
            # Polygon vertices
            polygon.pop(-1)
            # Iterate each corner of the polygon
            for i in range(len(polygon)):
                p1, p2, p3 = np.array(polygon[i % len(polygon)]), np.array(polygon[(i + 1) % len(polygon)]), np.array(polygon[(i + 2) % len(polygon)])
                # print(p1, p2, p3)
                v1, v2 = p1 - p2, p3 - p2
                ang = angle(v1, v2)
                # print(ang)
                # If angle is within 10 degrees of 180 treat as flat (ignore); else record as non-flat
                if np.abs(ang - 180) < 10:
                    # print("Flat angle at point:", p2)
                    pass
                else:
                    non_flat_angles.append(tuple(p2.tolist()))
            non_flat_angles.insert(0, non_flat_angles[-1])
            non_flat_angles.pop(-1)
            # print(non_flat_angles)
            count += 1
        else:
            pass
    assert count == 1, 'count ==' + str(count) + '    ' + train_file

    # Assert there is only one type-6 polygon
    assert simple_cycles_semantics_train.count(6) == 1, train_file


    list1 = train_graph['corners']
    list2 = non_flat_angles
    if len(list1) >= 53:
        boundary_num.append(train_graph['file_id'])
    indices = [i for i, item in enumerate(list1) if item in list2]
    # print(list1)
    # print(list2)
    # print(indices)
    # print(train_graph['corners_np'])
    # print(train_graph['corner_list_np_normalized_padding_withsemantics'])


    boundary_vertex_indices_mask = np.zeros((53, 2), dtype=np.float32)
    for index in indices:
        boundary_vertex_indices_mask[index, :] = 1

    train_graph['boundary_vertex_indices'] = boundary_vertex_indices_mask
    # print(train_graph['boundary_vertex_indices'])

    # Boundary adjacency matrix
    boundary_adjacency_matrix = np.zeros_like(train_graph['adjacency_matrix_np_padding'])
    for i, coord in enumerate(list2):
        list1_i = list1.index(coord)
        list1_i_plus_1 = list1.index(list2[(i + 1) % len(list2)])
        boundary_adjacency_matrix[list1_i, list1_i_plus_1] = 1
        boundary_adjacency_matrix[list1_i_plus_1, list1_i] = 1

    # print(boundary_adjacency_matrix)
    # print(list2)
    # assert 0


    # Boundary coordinate sequence
    train_graph['boundary_vertex_coords_4cvae'] = list2
    train_graph['boundary_adjacency_matrix'] = boundary_adjacency_matrix



    v1 = copy.deepcopy(train_graph)
    del v1['boundary_vertex_coords_4cvae']
    del v1['boundary_adjacency_matrix']


    np.save(os.path.join('rplandata/Data/rplang-v3-withsemantics-withboundary-v2/train', f"{train_graph['file_id']}.npy"), train_graph)
    np.save(os.path.join('rplandata/Data/rplang-v3-withsemantics-withboundary/train',f"{v1['file_id']}.npy"), v1)





# Check subdirectory file counts
check_subdir_file_counts('./rplandata/Data/rplang-v3-withsemantics')
check_subdir_file_counts('./rplandata/Data/rplang-v3-withsemantics-withboundary')
check_subdir_file_counts('./rplandata/Data/rplang-v3-withsemantics-withboundary-v2')