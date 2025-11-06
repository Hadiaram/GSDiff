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
    '''Determine whether two dictionary-like data structures are exactly identical.'''
    # If types differ, return False immediately
    if type(a) != type(b):
        return False

    # If dict, compare keys and their corresponding values
    if isinstance(a, dict):
        if a.keys() != b.keys():
            return False
        return all(deep_compare(a[key], b[key]) for key in a)

    # If list or tuple, compare each element sequentially
    elif isinstance(a, (list, tuple)):
        if len(a) != len(b):
            return False
        return all(deep_compare(item1, item2) for item1, item2 in zip(a, b))

    # If numpy array, use np.array_equal for comparison
    elif isinstance(a, np.ndarray):
        return np.array_equal(a, b)

    # Other types: compare directly (int, float, str, etc.)
    else:
        return a == b

def check_subdir_file_counts(base_dir):
    subdirs = ['val']
    for subdir in subdirs:
        full_path = os.path.join(base_dir, subdir)
        file_list = [f for f in os.listdir(full_path)
                     if os.path.isfile(os.path.join(full_path, f))]
    print('file count ' + str(len(file_list)))


if not os.path.exists('rplandata/Data/rplang-v3-bubble-diagram'):
    os.mkdir('./rplandata/Data/rplang-v3-bubble-diagram')
os.mkdir('./rplandata/Data/rplang-v3-bubble-diagram/val')


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
val_files = os.listdir('rplandata/Data/rplang-v3-withsemantics/val')
for val_file in tqdm(val_files):
    val_graph = np.load('rplandata/Data/rplang-v3-withsemantics/val/' + val_file, allow_pickle=True).item()

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
    corners_withsemantics = val_graph['corner_list_np_normalized_padding_withsemantics']
    # Initialize a new n*9 array (53, 9)
    corners_withsemantics_simplified = np.zeros((corners_withsemantics.shape[0], 9))
    # Copy columns 0 and 1
    corners_withsemantics_simplified[:, 0:2] = corners_withsemantics[:, 0:2]
    # Compute new column 2 by summing original columns 2, 6, 12
    corners_withsemantics_simplified[:, 2] = (corners_withsemantics[:, [2, 6, 12]]).sum(axis=1)
    # Compute new column 3 by summing original columns 3, 7, 8, 9, 10
    corners_withsemantics_simplified[:, 3] = (corners_withsemantics[:, [3, 7, 8, 9, 10]]).sum(axis=1)
    # Compute new column 4 by summing original columns 13, 14
    corners_withsemantics_simplified[:, 4] = (corners_withsemantics[:, [13, 14]]).sum(axis=1)
    # Copy original columns 4, 5, 11, 15 into simplified columns 5, 6, 7, 8
    corners_withsemantics_simplified[:, 5] = corners_withsemantics[:, 4]
    corners_withsemantics_simplified[:, 6] = corners_withsemantics[:, 5]
    corners_withsemantics_simplified[:, 7] = corners_withsemantics[:, 11]
    corners_withsemantics_simplified[:, 8] = corners_withsemantics[:, 15]

    '''attn 1 matrix, (53, 53)'''
    global_attn_matrix = val_graph['global_matrix_np_padding'].astype(bool)
    '''corners padding mask, (53, 1)'''
    corners_padding_mask = val_graph['padding_mask']

    '''edges, (2809, 1)'''
    edges = val_graph['edges']
    corners_withsemantics_0_val = corners_withsemantics_simplified[None, :, :]
    global_attn_matrix_val = global_attn_matrix[None, :, :]
    corners_padding_mask_val = corners_padding_mask[None, :, :]
    edges_val = edges[None, :, :]
    corners_withsemantics_0_val = corners_withsemantics_0_val.clip(-1, 1)
    corners_0_val = (corners_withsemantics_0_val[0, :, :2] * 128 + 128).astype(int)
    semantics_0_val = corners_withsemantics_0_val[0, :, 2:].astype(int)
    global_attn_matrix_val = global_attn_matrix_val
    corners_padding_mask_val = corners_padding_mask_val
    edges_val = edges_val
    corners_0_val_depadded = corners_0_val[corners_padding_mask_val.squeeze() == 1][None, :, :]  # (n, 2)
    semantics_0_val_depadded = semantics_0_val[corners_padding_mask_val.squeeze() == 1][None, :, :]  # (n, 7)
    edges_val_depadded = edges_val[global_attn_matrix_val.reshape(1, -1, 1)][None, :, None]
    edges_val_depadded = np.concatenate((1 - edges_val_depadded, edges_val_depadded), axis=2)

    ''' get planar cycles'''
    # Array shape (1, n, 14) containing 0/1; find index of each 1 and replace original 0 values with 99999
    semantics_gt_i_transform_val = semantics_0_val_depadded
    semantics_gt_i_transform_indices_val = np.indices(semantics_gt_i_transform_val.shape)[-1]
    semantics_gt_i_transform_val = np.where(semantics_gt_i_transform_val == 1,
                                              semantics_gt_i_transform_indices_val, 99999)

    gt_i_points_val = [tuple(corner_with_seman_val) for corner_with_seman_val in
                         np.concatenate((corners_0_val_depadded, semantics_gt_i_transform_val), axis=-1).tolist()[
                             0]]
    # print(output_points)
    gt_i_edges_val = edges_to_coordinates(
        np.triu(edges_val_depadded[0, :, 1].reshape(len(gt_i_points_val), len(gt_i_points_val))).reshape(-1),
        gt_i_points_val)

    # print(gt_i_points_val)
    # print(gt_i_edges_val)

    '''During our experiments, extracting bubble diagram ground-truth semantics involved randomness. Due to RPLAN dataset terms,
    we are not allowed to disclose any RPLAN dataset content. We can only provide this script for extracting bubble diagram data.
    Therefore, semantics you obtain with the same script may differ slightly from ours; however, given the dataset scale, final metrics
    should not diverge much statistically. You may alternatively use get_cycle_basis_and_semantic_3_semansimplified to extract room semantics
    and train topology-related models yourself; that method is deterministic and may yield improved metrics over those reported in the paper.'''
    d_rev_val, simple_cycles_val, simple_cycles_semantics_val = get_cycle_basis_and_semantic_2_semansimplified(
        gt_i_points_val,
        gt_i_edges_val)
    simple_cycles_val_ = []
    for sc in simple_cycles_val:
        sc_val = [(t[0], t[1]) for t in sc]
        simple_cycles_val_.append(sc_val)
        # print(sc_val)
    # for scs in simple_cycles_semantics_val:
    #     print(scs)
    polygons = simple_cycles_val_
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


    # Use the Shoelace formula to compute polygon area, then the centroid formula for (possibly concave) polygons.
    # When considering adjacency and rendering we could leverage OpenCV; reference Python implementation below:
    # Compute polygon centroid
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




    # Compute centroid for each polygon
    centroids = [get_polygon_centroid(polygon[:-1]) for polygon in polygons]


    # Save bubble diagram room polygons (closed), centroids, types, adjacency matrix.
    bbdiagram = {}
    bbdiagram['file_id'] = val_graph['file_id']
    bbdiagram['polygons'] = simple_cycles_val
    bbdiagram['centroids'] = centroids
    bbdiagram['semantics'] = simple_cycles_semantics_val
    
    for s in simple_cycles_semantics_val:
        bubblecategory[s] += 1

    bbdiagram['adjacency_matrix'] = adjacency_matrix
    bbdiagram['corner_number'] = len(val_graph['corners'])
    
    
    np.save(os.path.join('rplandata/Data/rplang-v3-bubble-diagram/val', f"{val_graph['file_id']}.npy"), bbdiagram)

