import math
from itertools import cycle

import networkx as nx
import numpy as np
from scipy.spatial import cKDTree
import time
import random

# # Record start time
start_time = time.time()



import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import *
import cv2
from gsdiff.utils import *


batch_size = 1
device = 'cpu'
import os
from torch.utils.data import Dataset
import torch
import numpy as np
from datasets import tiny_graph

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
            self.files = os.listdir('./datasets/rplang-v3-withsemantics/train')
        elif self.mode == 'val':
            self.files = os.listdir('./datasets/rplang-v3-withsemantics/val')
        elif self.mode == 'test':
            self.files = os.listdir('./datasets/rplang-v3-withsemantics/test')
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
            graph = np.load('./datasets/rplang-v3-withsemantics/train/' + self.files[index], allow_pickle=True).item()
        elif self.mode == 'val':
            graph = np.load('./datasets/rplang-v3-withsemantics/val/' + self.files[index], allow_pickle=True).item()
        elif self.mode == 'test':
            graph = np.load('./datasets/rplang-v3-withsemantics/test/' + self.files[index], allow_pickle=True).item()
        else:
            assert 0, 'mode error'

        '''coords_withsemantics, (53, 16)'''
        corners_withsemantics = graph['corner_list_np_normalized_padding_withsemantics']
    # Initialize a new n*9 array (53, 9)
        corners_withsemantics_simplified = np.zeros((corners_withsemantics.shape[0], 9))
    # Copy columns 0 and 1
        corners_withsemantics_simplified[:, 0:2] = corners_withsemantics[:, 0:2]
    # Compute new column 2 (sum of selected semantic channels)
        corners_withsemantics_simplified[:, 2] = (corners_withsemantics[:, [2, 6, 12]]).sum(axis=1)
    # Compute new column 3 (sum of selected semantic channels)
        corners_withsemantics_simplified[:, 3] = (corners_withsemantics[:, [3, 7, 8, 9, 10]]).sum(axis=1)
    # Compute new column 4 (sum of selected semantic channels)
        corners_withsemantics_simplified[:, 4] = (corners_withsemantics[:, [13, 14]]).sum(axis=1)
    # Copy original columns 4, 5, 11, 15 into simplified indices 5..8
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

        return corners_withsemantics_simplified, global_attn_matrix, corners_padding_mask, edges



dataset_test_for_gt_rendering = RPlanGEdgeSemanSimplified('test')
dataloader_test_for_gt_rendering = DataLoader(dataset_test_for_gt_rendering, batch_size=1, shuffle=False, num_workers=0,
                        drop_last=False, pin_memory=True)  # try different num_workers to be faster
dataloader_test_iter_for_gt_rendering = iter(cycle(dataloader_test_for_gt_rendering))


gts = []
for batch_count in tqdm(range(3000)):
    corners_withsemantics_0_test_batch, global_attn_matrix_test_batch, corners_padding_mask_test_batch, edges_test_batch = next(dataloader_test_iter_for_gt_rendering)
    for i in range(corners_withsemantics_0_test_batch.shape[0]):
        test_count = batch_count * 1 + i
        corners_withsemantics_0_test = corners_withsemantics_0_test_batch[i][None, :, :]
        global_attn_matrix_test = global_attn_matrix_test_batch[i][None, :, :]
        corners_padding_mask_test = corners_padding_mask_test_batch[i][None, :, :]
        edges_test = edges_test_batch[i][None, :, :]

        corners_withsemantics_0_test = corners_withsemantics_0_test.clamp(-1, 1).cpu().numpy()
        corners_0_test = (corners_withsemantics_0_test[0, :, :2] * (512 // 2) + (512 // 2)).astype(int)
        semantics_0_test = corners_withsemantics_0_test[0, :, 2:].astype(int)
        global_attn_matrix_test = global_attn_matrix_test.cpu().numpy()
        corners_padding_mask_test = corners_padding_mask_test.cpu().numpy()
        edges_test = edges_test.cpu().numpy()
        corners_0_test_depadded = corners_0_test[corners_padding_mask_test.squeeze() == 1][None, :, :]  # (n, 2)
        semantics_0_test_depadded = semantics_0_test[corners_padding_mask_test.squeeze() == 1][None, :, :]  # (n, 7)
        edges_test_depadded = edges_test[global_attn_matrix_test.reshape(1, -1, 1)][None, :, None]
        edges_test_depadded = np.concatenate((1 - edges_test_depadded, edges_test_depadded), axis=2)

        ''' get planar cycles'''
    # ndarray of shape (1, n, 14) containing 0/1; replace 1s by their index and 0s by 99999
        semantics_gt_i_transform_test = semantics_0_test_depadded
        semantics_gt_i_transform_indices_test = np.indices(semantics_gt_i_transform_test.shape)[-1]
        semantics_gt_i_transform_test = np.where(semantics_gt_i_transform_test == 1,
                                                 semantics_gt_i_transform_indices_test, 99999)

        gt_i_points_test = [tuple(corner_with_seman_test) for corner_with_seman_test in
                            np.concatenate((corners_0_test_depadded, semantics_gt_i_transform_test), axis=-1).tolist()[
                                0]]

        gt_i_edges_test = edges_to_coordinates(
            np.triu(edges_test_depadded[0, :, 1].reshape(len(gt_i_points_test), len(gt_i_points_test))).reshape(-1),
            gt_i_points_test)
    # List all edges and remove duplicates (treat as undirected)
        gt_i_edges_test_ = [list(p1)[:2] + list(p2)[:2] for (p1, p2) in gt_i_edges_test
                             if (p2, p1) not in gt_i_edges_test]
        # print(output_edges_test)
        d_rev_test, simple_cycles_test, simple_cycles_semantics_test = get_cycle_basis_and_semantic_3_semansimplified(
            gt_i_points_test,
            gt_i_edges_test)
        gts.append([gt_i_points_test, gt_i_edges_test_, d_rev_test, simple_cycles_test, simple_cycles_semantics_test])

def get_cycle_basis_and_semantic_3_semansimplified_area1(output_points, output_edges):
    # Dictionary mapping output point to its index
    d = {}
    for output_point_index, output_point in enumerate(output_points):
        d[output_point] = output_point_index  # Cannot handle duplicate points here; NMS removal required earlier
    d_rev = {}
    for output_point_index, output_point in enumerate(output_points):
        d_rev[output_point_index] = output_point  # Cannot handle duplicate points here; NMS removal required earlier
    es = []
    for output_edge in output_edges:
        es.append((d[output_edge[0]], d[output_edge[1]]))


    G = nx.Graph()
    for e in es:
        G.add_edge(e[0], e[1])
        G.add_edge(e[1], e[0])


    simple_cycles = []
    simple_cycles_number = []
    simple_cycles_semantics = []
    all_area = 0
    # print('Breakpoint 1', simple_cycles)
    bridges = list(nx.bridges(G))
    # Handling bridge (cut) edges: remove them so remaining components are isolated nodes or pure cycles
    for b in bridges:
        if (d_rev[b[0]], d_rev[b[1]]) in output_edges:
            output_edges.remove((d_rev[b[0]], d_rev[b[1]]))
            es.remove((b[0], b[1]))
            if G.has_edge(b[0], b[1]):
                G.remove_edge(b[0], b[1])
        if (d_rev[b[1]], d_rev[b[0]]) in output_edges:
            output_edges.remove((d_rev[b[1]], d_rev[b[0]]))
            es.remove((b[1], b[0]))
            if G.has_edge(b[1], b[0]):
                G.remove_edge(b[1], b[0])
    # After removing bridges, examine each remaining connected component (isolated node or cycle)
    connected_components = list(nx.connected_components(G))


    for c in connected_components:
        if len(c) == 1:
            pass
        else:
            simple_cycles_c = []
            simple_cycles_number_c = []
            simple_cycle_semantics_c = []
            # print(c) # {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23}
            # Collect point/edge subsets for this component
            output_points_c = [p for p in output_points if d[p] in c]
            output_edges_c = [e for e in output_edges if d[e[0]] in c or d[e[1]] in c]  # Fixed edge set (not deleted)
            output_edges_c_copy_for_traversing = copy.deepcopy(output_edges_c)  # Working copy for traversal; edges removed to cut complexity
            # print(output_points_c)
            # print(output_edges_c)

            # Enumerate all counter‑clockwise simple cycles in this component:
            # - Use indices in d as identifiers.
            # - Iterate each undirected edge: initial = smaller index endpoint; last = initial; current = larger.
            # - Gather all neighbor edges of current and compute their outgoing angles.
            # - Take reverse direction of (last -> current) as reference angle.
            # - Rotate CCW to select the last encountered neighbor edge.
            # - Its opposite endpoint becomes next.
            # - When next equals initial, we have a cycle [p0,...,p_{n-1},p0].
            # - Remove oriented edges (pi, pi+1) with pi < pi+1 from traversal copy to avoid duplicates.
            # - Set last = current; current = next; continue.

            for edge_c in output_edges_c:
                if edge_c not in output_edges_c_copy_for_traversing:
                    pass
                else:
                    try:
                        simple_cycle_semantics = []
                        simple_cycle = []
                        simple_cycle_number = []
                        point1 = edge_c[0]
                        point2 = edge_c[1]
                        point1_number = d[point1]
                        point2_number = d[point2]
                        # Initial point
                        initial_point = None
                        initial_point_number = None
                        if point1_number < point2_number:
                            initial_point = point1
                            initial_point_number = point1_number
                        else:
                            initial_point = point2
                            initial_point_number = point2_number
                        simple_cycle.append(initial_point)
                        simple_cycle_number.append(initial_point_number)
                        # Previous point
                        last_point = initial_point
                        last_point_number = initial_point_number
                        # Current point
                        current_point = None
                        current_point_number = None
                        if point1_number < point2_number:
                            current_point = point2
                            current_point_number = point2_number
                        else:
                            current_point = point1
                            current_point_number = point1_number
                        simple_cycle.append(current_point)
                        simple_cycle_number.append(current_point_number)
                        # Successor of initial point (used to decide loop termination)
                        next_initial_point = copy.deepcopy(current_point)
                        next_initial_point_number = copy.deepcopy(current_point_number)
                        # Next point placeholder
                        next_point = None
                        next_point_number = None
                        # Stop when next_point equals the successor of the initial point
                        while_count = 0
                        while next_point != next_initial_point and while_count < 100:
                            # Collect all neighbor edges of current_point
                            relevant_edges = []
                            for edge in output_edges_c:
                                if (edge[0] == current_point or edge[1] == current_point) and (not (edge[0] == current_point and edge[1] == current_point)):
                                    relevant_edges.append(edge)
                            # Compute outgoing angles of those neighbor edges
                            relevant_edges_degree = []
                            for relevant_edge in relevant_edges:
                                # Outgoing vector
                                vec = None
                                if relevant_edge[0] == current_point:
                                    vec = (
                                    relevant_edge[1][0] - relevant_edge[0][0], relevant_edge[1][1] - relevant_edge[0][1])
                                elif relevant_edge[1] == current_point:
                                    vec = (
                                    relevant_edge[0][0] - relevant_edge[1][0], relevant_edge[0][1] - relevant_edge[1][1])
                                else:
                                    assert 0
                                # Compute outgoing angle
                                vec_degree = x_axis_angle(vec)
                                relevant_edges_degree.append(vec_degree)
                            # Compute reverse (outgoing) direction from last_point to current_point and its angle
                            vec_from_current_point_to_last_point = None
                            vec_from_current_point_to_last_point_degree = None
                            for relevant_edge_ind, relevant_edge in enumerate(relevant_edges):
                                if relevant_edge == (current_point, last_point):
                                    vec_from_current_point_to_last_point = (
                                    relevant_edge[1][0] - relevant_edge[0][0], relevant_edge[1][1] - relevant_edge[0][1])
                                    vec_from_current_point_to_last_point_degree = relevant_edges_degree[relevant_edge_ind]
                                    relevant_edges.remove(relevant_edge)
                                    relevant_edges_degree.remove(vec_from_current_point_to_last_point_degree)
                                elif relevant_edge == (last_point, current_point):
                                    vec_from_current_point_to_last_point = (
                                    relevant_edge[0][0] - relevant_edge[1][0], relevant_edge[0][1] - relevant_edge[1][1])
                                    vec_from_current_point_to_last_point_degree = relevant_edges_degree[relevant_edge_ind]
                                    relevant_edges.remove(relevant_edge)
                                    relevant_edges_degree.remove(vec_from_current_point_to_last_point_degree)
                                else:
                                    continue
                            # From this angle rotate CCW to encounter the last neighbor edge of the current point
                            # Also record the unswept portion (the interior angle interval)
                            # Here we substitute interior-angle semantics with full semantics
                            rotate_deltas_counterclockwise = []
                            # Record interior angle region: CCW from previous angle to next angle
                            interior_angles = []
                            for relevant_edge_degree in relevant_edges_degree:
                                rotate_delta = rotate_degree_counterclockwise_from_counter_degree(
                                    vec_from_current_point_to_last_point_degree, relevant_edge_degree)
                                rotate_deltas_counterclockwise.append(rotate_delta)
                                interior_angles.append((relevant_edge_degree, vec_from_current_point_to_last_point_degree))
                            # print(rotate_deltas_counterclockwise)
                            # Index of the maximum rotation angle
                            max_rotate_index = rotate_deltas_counterclockwise.index(max(rotate_deltas_counterclockwise))
                            # Corresponding interior angle
                            interior_angle_counterclockwise = interior_angles[max_rotate_index]
                            # Derive the corresponding semantic region
                            # Gather all semantics of the current point ordered by the four quadrants
                            # current_point_semantic = [current_point[3], current_point[2], current_point[5],
                            #                           current_point[4], ]
                            current_point_semantic = [current_point[3], current_point[2], current_point[5],
                                                      current_point[4], current_point[6], current_point[7],
                                                      current_point[8]]
                            # Determine how much of the four quadrants this counter‑clockwise angle spans
                            # Method: rotate CCW from the smaller degree value to the larger to obtain quadrant coverage
                            # Then check: if the smaller degree corresponds to the interior angle's source direction, keep coverage;
                            # if the smaller degree is the interior angle's target direction, subtract coverage from 90 degrees.
                            interior_angle_counterclockwise_degree_smaller = min(interior_angle_counterclockwise)  # smaller degree
                            interior_angle_counterclockwise_degree_bigger = max(interior_angle_counterclockwise)  # larger degree
                            quadrant_smaller_to_bigger_counterclockwise = get_quadrant(
                                (interior_angle_counterclockwise_degree_smaller,
                                 interior_angle_counterclockwise_degree_bigger))
                            # print(quadrant_smaller_to_bigger_counterclockwise)
                            if interior_angle_counterclockwise.index(interior_angle_counterclockwise_degree_smaller) == 0:
                                pass
                            elif interior_angle_counterclockwise.index(interior_angle_counterclockwise_degree_smaller) == 1:
                                quadrant_smaller_to_bigger_counterclockwise = (
                                90 - quadrant_smaller_to_bigger_counterclockwise[0],
                                90 - quadrant_smaller_to_bigger_counterclockwise[1],
                                90 - quadrant_smaller_to_bigger_counterclockwise[2],
                                90 - quadrant_smaller_to_bigger_counterclockwise[3])
                            else:
                                assert 0
                            # Never assign semantic -1
                            current_point_semantic_valid = []
                            for qd, seman in enumerate(current_point_semantic):
                                if 1:
                                    current_point_semantic_valid.append(seman)
                                else:
                                    current_point_semantic_valid.append(-1)
                            # Accumulate all semantics for this vertex
                            simple_cycle_semantics.append(current_point_semantic_valid)

                            # Edge chosen by maximal rotation
                            max_rotate_edge = relevant_edges[max_rotate_index]
                            # Derive next point
                            if max_rotate_edge[0] == current_point:
                                next_point = max_rotate_edge[1]
                                next_point_number = d[next_point]
                            elif max_rotate_edge[1] == current_point:
                                next_point = max_rotate_edge[0]
                                next_point_number = d[next_point]
                            else:
                                assert 0
                            # Update traversal pointers and append current point into simple_cycle
                            last_point = current_point
                            last_point_number = current_point_number
                            current_point = next_point
                            current_point_number = next_point_number
                            simple_cycle.append(current_point)
                            simple_cycle_number.append(current_point_number)
                            while_count += 1
                        if len(simple_cycle) > 80:
                            continue
                        # (Optionally) append initial point for edge removal (omitted)
                        # simple_cycle.append(initial_point)
                        # simple_cycle_number.append(initial_point_number)
                        # Remove from remaining edges all with pi < p(i+1)
                        # print('------------------')
                        # print(simple_cycle)
                        # print(simple_cycle_number)
                        # print('------------------')
                        for point_number_ind, point_number in enumerate(simple_cycle_number):
                            if point_number_ind < len(simple_cycle_number) - 1:
                                edge_number = (point_number, simple_cycle_number[point_number_ind + 1])
                                # print(simple_cycle_number)
                                if edge_number[0] < edge_number[1]:
                                    if (d_rev[edge_number[0]], d_rev[edge_number[1]]) in output_edges_c_copy_for_traversing:
                                        output_edges_c_copy_for_traversing.remove(
                                            (d_rev[edge_number[0]], d_rev[edge_number[1]]))
                                    elif (
                                    d_rev[edge_number[1]], d_rev[edge_number[0]]) in output_edges_c_copy_for_traversing:
                                        output_edges_c_copy_for_traversing.remove(
                                            (d_rev[edge_number[1]], d_rev[edge_number[0]]))
                        # Remove closing vertex for area computation
                        simple_cycle.pop(-1)
                        simple_cycle_number.pop(-1)
                        # Store cycle if CCW area positive (negative would mean outermost, skip)
                        polygon_counterclockwise = [(int(p[0]), -int(p[1])) for p in simple_cycle]
                        polygon_counterclockwise.pop(-1)
                        # print('poly_area(polygon_counterclockwise)', poly_area(polygon_counterclockwise))
                        if poly_area(polygon_counterclockwise) > 0:
                            all_area += poly_area(polygon_counterclockwise)
                            simple_cycles_c.append(simple_cycle)
                            simple_cycles_number_c.append(simple_cycle_number)
                            # Majority semantic (skip largest outer cycle) tally semantics for this cycle
                            semantic_result = {}
                            for semantic_label in range(0, 7):
                                semantic_result[semantic_label] = 0
                            for everypoint_semantic in simple_cycle_semantics:
                                for _ in range(0, 7):
                                    if _ in everypoint_semantic:
                                        semantic_result[_] += 1
                            del semantic_result[6]

                            # print(semantic_result)
                            # If top counts tie choose uniformly at random (label 13 excluded in original setting)
                            this_cycle_semantic = sorted(semantic_result.items(), key=lambda d: d[1], reverse=True)
                            # print(this_cycle_semantic)
                            if this_cycle_semantic[0][1] > this_cycle_semantic[1][1]:
                                # Unique highest vote wins
                                this_cycle_result = this_cycle_semantic[0][0]
                            else:
                                # Tie-break priority order: cabinet(2) > bathroom(4) > kitchen(3) > bedroom(1) > balcony(5) > living room(0)
                                this_cycle_results = [i[0] for i in this_cycle_semantic if
                                                      i[1] == this_cycle_semantic[0][1]]
                                if 2 in this_cycle_results:
                                    this_cycle_result = 2
                                elif 4 in this_cycle_results:
                                    this_cycle_result = 4
                                elif 3 in this_cycle_results:
                                    this_cycle_result = 3
                                elif 1 in this_cycle_results:
                                    this_cycle_result = 1
                                elif 5 in this_cycle_results:
                                    this_cycle_result = 5
                                else:
                                    this_cycle_result = 0
                            # print(this_cycle_result)
                            simple_cycle_semantics_c.append(this_cycle_result)
                    except:
                        pass

            simple_cycles.extend(simple_cycles_c)
            simple_cycles_number.extend(simple_cycles_number_c)
            simple_cycles_semantics.extend(simple_cycle_semantics_c)

    # print([[(int(j[0]), int(j[1])) for j in i] for i in simple_cycles])

    # print(len(simple_cycles_number))
    # print(simple_cycles_semantics)

    return all_area


def get_cycle_basis_and_semantic_3_semansimplified_conn(output_points, output_edges):
    # Dictionary mapping output point to its index
    d = {}
    for output_point_index, output_point in enumerate(output_points):
        d[output_point] = output_point_index  # Cannot handle duplicate points; cannot remove NMS
    d_rev = {}
    for output_point_index, output_point in enumerate(output_points):
        d_rev[output_point_index] = output_point  # Cannot handle duplicate points; cannot remove NMS
    es = []
    for output_edge in output_edges:
        es.append((d[output_edge[0]], d[output_edge[1]]))


    G = nx.Graph()
    for e in es:
        G.add_edge(e[0], e[1])
        G.add_edge(e[1], e[0])


    simple_cycles = []
    simple_cycles_number = []
    simple_cycles_semantics = []
    all_area = 0
    # print('breakpoint1', simple_cycles)
    bridges = list(nx.bridges(G))
    # Handling virtual edges: remove them from the edge set and treat as multiple connected predictions
    for b in bridges:
        if (d_rev[b[0]], d_rev[b[1]]) in output_edges:
            output_edges.remove((d_rev[b[0]], d_rev[b[1]]))
            es.remove((b[0], b[1]))
            if G.has_edge(b[0], b[1]):
                G.remove_edge(b[0], b[1])
        if (d_rev[b[1]], d_rev[b[0]]) in output_edges:
            output_edges.remove((d_rev[b[1]], d_rev[b[0]]))
            es.remove((b[1], b[0]))
            if G.has_edge(b[1], b[0]):
                G.remove_edge(b[1], b[0])
    # After removing bridges inspect remaining components (only isolated nodes or cycles) and traverse cycles
    connected_components = list(nx.connected_components(G))

    # Number of connected components
    conn_nb = 0


    for c in connected_components:
        if len(c) == 1:
            pass
        else:
            conn_nb += 1


    return conn_nb


# Initialize dictionary storing all FID, KID values and alignment losses
metrics_dict = {}

# Iterate over all folders in current directory
for folder in os.listdir('./T'):
    if os.path.isdir('./T/' + folder) and '-' in folder:
        if len(folder.split('-')) == 2 and folder.split('-')[0].isupper() and folder.split('-')[1].isdigit():
            if int(folder.split('-')[1]) <= 48 and int(folder.split('-')[1]) % 2 == 0:
                # Initialize angle statistics
                angle_ranges = {
                    (5, 85): 0,
                    (95, 175): 0,
                    (185, 265): 0,
                    (275, 355): 0
                }
                edge_number = 0
                # Area (vector method)
                area1 = 0
                # Area (scalar / image method)
                area2 = 0
                # Room count accumulator
                rmnbs = 0
                # Connected component count accumulator
                conn_number = 0

                for i in tqdm(range(3000)):
                    file_name = f"vr4stat_{i}.npy"
                    file_path = os.path.join('./T/' + folder, file_name)

                    # Check file existence
                    if os.path.isfile(file_path):
                        # print(file_path)
                        # Load numpy data dict
                        data = np.load(file_path, allow_pickle=True).item()
                        # print(data)
                        # Collect all edges and remove duplicates (undirected)
                        output_edges_test = [list(p1)[:2] + list(p2)[:2] for (p1, p2) in data['output_edges_test']
                                             if (p2, p1) not in data['output_edges_test']]
                        # print(output_edges_test)
                        edge_number += len(output_edges_test)

                        # # Compute polar angle
                        # for edge in output_edges_test:
                        #     x1, y1, x2, y2 = edge
                        #     dx = x2 - x1
                        #     dy = y2 - y1
                        #
                        #     # Compute polar angle (degrees)
                        #     angle = math.degrees(math.atan2(dy, dx)) % 360
                        #
                        #     # Tally angle into bins
                        #     for angle_range in angle_ranges:
                        #         if angle_range[0] <= angle < angle_range[1]:
                        #             angle_ranges[angle_range] += 1
                        #             break
                        # # Vector-based area calculation
                        # area1 += get_cycle_basis_and_semantic_3_semansimplified_area1(data['output_points_test'], data['output_edges_test']) / (512**2)
                        # # Image-based area calculation
                        # img_area = cv2.imread(os.path.join('./T/' + folder + '/' + 'test_model1000000', f"test_pred_{i}.png"))
                        # # Create all-white array with same shape as image
                        # white_pixels = np.ones_like(img_area) * 255
                        # # Compare image to white array to locate non-white pixels
                        # non_white_pixels = np.any(img_area != white_pixels, axis=-1)
                        # # Count non-white pixels
                        # area2 += np.count_nonzero(non_white_pixels) / (512**2)

                        # Increment room count
                        rmnbs += len(data['simple_cycles_semantics_test'])

                        # # Connected component count (disabled)
                        # conn_number += get_cycle_basis_and_semantic_3_semansimplified_conn(data['output_points_test'], data['output_edges_test'])



                angle_notgood = sum(angle_ranges.values()) / edge_number
                area1 /= 3000
                area2 /= 3000
                rmnbs /= 3000
                conn_number /= 3000


                # Extract numeric id from folder name
                group, number = folder.split('-')
                # Accumulate metrics into dictionary
                if number not in metrics_dict:
                    metrics_dict[number] = {'angle_notgood': [], 'edge_number': [], 'area1': [], 'area2': [], 'rmnbs': [], 'conn_number': []}
                metrics_dict[number]['angle_notgood'].append(angle_notgood)
                metrics_dict[number]['edge_number'].append(edge_number)
                metrics_dict[number]['area1'].append(area1)
                metrics_dict[number]['area2'].append(area2)
                metrics_dict[number]['rmnbs'].append(rmnbs)
                metrics_dict[number]['conn_number'].append(conn_number)
# Output per-number averages
print(f"{'Number':<10}{'angle_notgood':<30}{'edge_number':<30}{'area1':<30}{'area2':<30}{'rmnbs':<30}{'conn_number':<30}")
for number, values in sorted(metrics_dict.items(), key=lambda x: int(x[0])):
    avg_angle_notgood = np.mean(np.array(values['angle_notgood']))
    avg_edge_number = np.mean(np.array(values['edge_number']))
    avg_area1 = np.mean(np.array(values['area1']))
    avg_area2 = np.mean(np.array(values['area2']))
    avg_rmnbs = np.mean(np.array(values['rmnbs']))
    avg_conn_number = np.mean(np.array(values['conn_number']))
    print(f"{number:<10}{avg_angle_notgood:<30}{avg_edge_number:<30}{avg_area1:<30}{avg_area2:<30}{avg_rmnbs:<30}{avg_conn_number:<30}")
# # Record end time
end_time = time.time()
print(end_time - start_time, 's')

