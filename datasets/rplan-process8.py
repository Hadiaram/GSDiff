import sys
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

import os


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


if not os.path.exists('rplandata/Data/rplang-v3-bubble-diagram'):
    os.mkdir('./rplandata/Data/rplang-v3-bubble-diagram')
os.mkdir('./rplandata/Data/rplang-v3-bubble-diagram/train')


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
bubblenumber = {}
train_files = os.listdir('rplandata/Data/rplang-v3-withsemantics/train')
for train_file in tqdm(train_files):
    train_graph = np.load('rplandata/Data/rplang-v3-withsemantics/train/' + train_file, allow_pickle=True).item()

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
    # Copy original columns 4,5,11,15 into new columns 5,6,7,8 (labels 3 kitchen, 4 bathroom, 5 balcony, 6 exterior)
    corners_withsemantics_simplified[:, 5] = corners_withsemantics[:, 4]
    corners_withsemantics_simplified[:, 6] = corners_withsemantics[:, 5]
    corners_withsemantics_simplified[:, 7] = corners_withsemantics[:, 11]
    corners_withsemantics_simplified[:, 8] = corners_withsemantics[:, 15]

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

    '''Note on semantic extraction randomness:
    During our original experiments, ground-truth bubble (room) semantics were derived with a procedure containing inherent randomness.
    Under the RPLAN dataset terms we cannot release any portion of the dataset itself—only this extraction script.
    Running this same script yourself may yield slightly different bubble semantics compared to those we observed, but given the dataset scale
    the final aggregate metrics should not differ significantly. Alternatively, you can adopt get_cycle_basis_and_semantic_3_semansimplified
    which assigns room semantics deterministically and may offer marginal metric improvements relative to the paper's reported numbers.'''
    d_rev_train, simple_cycles_train, simple_cycles_semantics_train = get_cycle_basis_and_semantic_2_semansimplified(
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
    edges = [[(polygon[i], polygon[(i + 1) % len(polygon)]) for i in range(len(polygon))][:-1] for polygon in polygons]
    # print(edges)


    def get_adjacency_matrix(polygons):
        n = len(polygons)
        matrix = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                if any(set(edge) in [set(edge_j) for edge_j in polygons[j]] for edge in polygons[i]):
                    matrix[i][j] = 1
                    matrix[j][i] = 1
        return matrix


    adjacency_matrix = get_adjacency_matrix(edges)
    edgecategory[1] += np.sum(np.triu(np.array(adjacency_matrix)))
    edgecategory[0] += ((len(adjacency_matrix) * (len(adjacency_matrix) + 1)) / 2) - np.sum(np.triu(np.array(adjacency_matrix)))


    # Use the Shoelace formula to compute area, then the centroid formula to compute the centroid of a concave polygon.
    # When considering adjacency and drawing we can use the OpenCV library. The related Python code:
    # Compute the centroid of the polygon
    def get_polygon_centroid(polygon):
        area = 0
        x = 0
        y = 0
        for i in range(-1, len(polygon) - 1):
            step = (polygon[i][0] * polygon[i + 1][1]) - (polygon[i + 1][0] * polygon[i][1])
            area += step
            x += (polygon[i][0] + polygon[i + 1][0]) * step
            y += (polygon[i][1] + polygon[i + 1][1]) * step
        area /= 2
        x /= (6 * area)
        y /= (6 * area)
        return (int(x), int(y))




    # Compute the centroid for each polygon
    centroids = [get_polygon_centroid(polygon[:-1]) for polygon in polygons]


    # Save the bubble diagram room polygons (first and last point identical), centroids, types, and adjacency matrix.
    bbdiagram = {}
    bbdiagram['file_id'] = train_graph['file_id']
    bbdiagram['polygons'] = simple_cycles_train
    bbdiagram['centroids'] = centroids
    bbdiagram['semantics'] = simple_cycles_semantics_train
    
    for s in simple_cycles_semantics_train:
        bubblecategory[s] += 1

    bbdiagram['adjacency_matrix'] = adjacency_matrix
    bbdiagram['corner_number'] = len(train_graph['corners'])
    
    
    np.save(os.path.join('rplandata/Data/rplang-v3-bubble-diagram/train', f"{train_graph['file_id']}.npy"), bbdiagram)

