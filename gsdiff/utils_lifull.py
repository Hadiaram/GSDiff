import copy
import math
import os
import random
import networkx as nx
import torch
from tqdm import *
import cv2
import numpy as np


def euclidean_edge_match(input_array):
    # Target value array for discretizing soft edge predictions
    target_values = np.array([[0, 1], [1, 0]])
    # Compute Euclidean distance to both prototypes
    distances = np.linalg.norm(input_array[:, None, :] - target_values, axis=-1)
    # Argmin over the two prototypes
    min_indices = np.argmin(distances, axis=-1)
    # Map indices back to one-hot edge representation
    output_array = target_values[min_indices]
    return output_array

def visualize(result_corners_inverse_normalized, result_edges_unpaddinged, output_dir, timestep):
    '''timestep can be 'gt' or any int time'''
    for result_index, result in enumerate(result_corners_inverse_normalized):
        result = result[0]
        # print(result)
        img = np.ones((256, 256, 3), dtype=np.uint8) * 255
        # drew walls
        drew_walls = [] 
        for corner_index, corner in enumerate(result):
            corner_tuple = tuple(corner.tolist())
            # draw walls
            if result_edges_unpaddinged[result_index][0].shape[1] == 1:
                edges = result_edges_unpaddinged[result_index][0][:, 0]
            else:
                edges = result_edges_unpaddinged[result_index][0][:, 1]
            for edge_index, one_or_zero in enumerate(edges):
                if one_or_zero:
                    if corner_index in [edge_index % len(result), edge_index // len(result)] and (edge_index % len(result) != edge_index // len(result)):
                        # get another corner
                        another_corner = result[(edge_index % len(result)) if corner_index == (edge_index // len(result)) else (edge_index // len(result))]
                        another_corner_tuple = tuple(another_corner.tolist())
                        if (corner_tuple, another_corner_tuple) in drew_walls or (
                        another_corner_tuple, corner_tuple) in drew_walls:
                            pass
                        else:
                            # print(corner_tuple, another_corner_tuple)
                            cv2.line(img, corner_tuple, another_corner_tuple, color=(0, 0, 0), thickness=3)
                            drew_walls.append((corner_tuple, another_corner_tuple))
                            drew_walls.append((another_corner_tuple, corner_tuple))
        # draw corners
        for corner_index, corner in enumerate(result):
            corner_tuple = tuple(corner.tolist())
            cv2.circle(img, corner_tuple, radius=3, color=(0, 215, 255), thickness=-1)
        cv2.imwrite(output_dir + str(result_index) + '_' + str(timestep) + '.png', img)

def visualize_withsemantic(output_points, output_edges, simple_cycles, simple_cycles_semantics, output_dir, result_index, timestep):
    '''timestep can be 'gt' or any int time'''
    colors = {6: (0, 0, 0), 0: (222, 241, 244), 1: (159, 182, 234), 2: (92, 112, 107), 3: (95, 122, 224),
              4: (123, 121, 95), 5: (143, 204, 242)}
    img = np.ones((256, 256, 3), dtype=np.uint8) * 255
    # draw polygons
    for polygon_i, polygon in enumerate(simple_cycles):
        # Extract valid (x,y) coordinates for this polygon
        pts = np.array([(x, y) for x, y, *rest in polygon], np.int32)
        pts = pts.reshape((-1, 1, 2))
        # Fill and outline polygon
        cv2.fillPoly(img, [pts], color=colors[simple_cycles_semantics[polygon_i]])
        cv2.polylines(img, [pts], isClosed=True, color=(150, 150, 150), thickness=5)
    # draw corners
    # for corner in output_points:
    #     corner_tuple = tuple(list(corner)[:2])
    #     cv2.circle(img, corner_tuple, radius=3, color=(0, 215, 255), thickness=-1)
    cv2.imwrite(output_dir + str(result_index) + '_' + str(timestep) + '.png', img)

def visualize_35(result_corners_inverse_normalized, result_edges_unpaddinged, output_dir, timestep):
    '''timestep can be 'gt' or any int time'''
    for result_index, result in enumerate(result_corners_inverse_normalized):
        result = result[0]
        # print(result)
        img = np.ones((256, 256, 3), dtype=np.uint8) * 255
        # drew walls
        drew_walls = []
        for corner_index, corner in enumerate(result):
            corner_tuple = tuple(corner.tolist())
            # draw walls
            if result_edges_unpaddinged[result_index][0].shape[1] == 1:
                edges = result_edges_unpaddinged[result_index][0][:, 0]
            else:
                edges = result_edges_unpaddinged[result_index][0][:, 1]
            for edge_index, one_or_zero in enumerate(edges):
                if one_or_zero:
                    if corner_index in [edge_index % len(result), edge_index // len(result)] and (edge_index % len(result) != edge_index // len(result)):
                        # get another corner
                        another_corner = result[(edge_index % len(result)) if corner_index == (edge_index // len(result)) else (edge_index // len(result))]
                        another_corner_tuple = tuple(another_corner.tolist())
                        if (corner_tuple, another_corner_tuple) in drew_walls or (
                        another_corner_tuple, corner_tuple) in drew_walls:
                            pass
                        else:
                            # print(corner_tuple, another_corner_tuple)
                            cv2.line(img, corner_tuple, another_corner_tuple, color=(0, 0, 0), thickness=3)
                            drew_walls.append((corner_tuple, another_corner_tuple))
                            drew_walls.append((another_corner_tuple, corner_tuple))
        # draw corners
        for corner_index, corner in enumerate(result):
            corner_tuple = tuple(corner.tolist())
            cv2.circle(img, corner_tuple, radius=3, color=(0, 215, 255), thickness=-1)
        cv2.imwrite(output_dir + str(result_index) + '_' + str(timestep) + '.png', img)

def visualize_36(result_corners_inverse_normalized, result_edges_unpaddinged, output_dir, timestep):
    '''timestep can be 'gt' or any int time'''
    for result_index, result in enumerate(result_corners_inverse_normalized):
        result = result[0]
        # print(result)
        img = np.ones((256, 256, 3), dtype=np.uint8) * 255
        # draw corners
        for corner_index, corner in enumerate(result):
            corner_tuple = tuple(corner.tolist())
            cv2.circle(img, corner_tuple, radius=3, color=(0, 215, 255), thickness=-1)
        cv2.imwrite(output_dir + str(result_index) + '_' + str(timestep) + '.png', img)

def visualize_51(result_corners_inverse_normalized, result_semantics_inverse_normalized, output_dir, timestep):
    '''timestep can be 'gt' or any int time'''
    colors = {13:(0, 0, 0), 0:(255, 0, 0), 1:(0, 255, 0), 2:(0, 0, 255), 3:(255, 255, 0),
              4:(255, 0, 255), 5:(0, 255, 255), 6:(127, 0, 0), 7:(0, 127, 0),
              8:(0, 0, 127), 9:(127, 127, 0), 10:(127, 0, 127), 11:(0, 127, 127), 12:(127, 127, 127)}

    for result_index, result in enumerate(result_corners_inverse_normalized):
        result = result[0]
        # print(result)
        img = np.ones((256, 256, 3), dtype=np.uint8) * 255
        # draw corners
        for corner_index, corner in enumerate(result):
            corner_tuple = tuple(corner.tolist())
            cv2.circle(img, corner_tuple, radius=3, color=(0, 215, 255), thickness=-1)
            # draw semantics
            if timestep == 'gt' or timestep == 0:
                semantic = result_semantics_inverse_normalized[result_index][0][corner_index].tolist()
                # print(semantic)
                semans = []
                for seman_i in range(14):
                    if semantic[seman_i] > 0:
                        for _ in range(semantic[seman_i]):
                            semans.append(colors[seman_i])
                # print(semans)
                # print(corner_tuple)
                for ii, seman in enumerate(semans):
                    img[corner_tuple[1] + 4, corner_tuple[0] + ii, 0] = seman[0]
                    img[corner_tuple[1] + 4, corner_tuple[0] + ii, 1] = seman[1]
                    img[corner_tuple[1] + 4, corner_tuple[0] + ii, 2] = seman[2]

        cv2.imwrite(output_dir + str(result_index) + '_' + str(timestep) + '.png', img)

def visualize_33(result_corners_inverse_normalized, result_edges_unpaddinged, output_dir, timestep):
    # print(len(result_edges_unpaddinged)) # bs
    # print(len(result_edges_unpaddinged[0]), len(result_edges_unpaddinged[1])) # 1, 1
    # print(len(result_edges_unpaddinged[0][0]), len(result_edges_unpaddinged[1][0])) # 18 ** 2, 22 ** 2
    # print(result_edges_unpaddinged[0][0][-1], result_edges_unpaddinged[1][0][-1])  # 18 ** 2, 22 ** 2
    # Discretize edges by mapping to the closer prototype between [0 1] and [1 0]
    result_edges_unpaddinged_discrete = []
    for bs_i in range(len(result_edges_unpaddinged)):
        aaa = []
        for _ in range(1):
            e = euclidean_edge_match(result_edges_unpaddinged[bs_i][_])
            aaa.append(e)
        result_edges_unpaddinged_discrete.append(aaa)
    result_edges_unpaddinged = result_edges_unpaddinged_discrete

    '''timestep can be 'gt' or any int time'''
    for result_index, result in enumerate(result_corners_inverse_normalized):
        result = result[0]
        # print(result)
        img = np.ones((256, 256, 3), dtype=np.uint8) * 255
        # drew walls
        drew_walls = []
        for corner_index, corner in enumerate(result):
            corner_tuple = tuple(corner.tolist())
            # draw walls
            if result_edges_unpaddinged[result_index][0].shape[1] == 1:
                edges = result_edges_unpaddinged[result_index][0][:, 0]
            else:
                edges = result_edges_unpaddinged[result_index][0][:, 1]
            for edge_index, one_or_zero in enumerate(edges):
                if one_or_zero:
                    if corner_index in [edge_index % len(result), edge_index // len(result)] and (edge_index % len(result) != edge_index // len(result)):
                        # get another corner
                        another_corner = result[(edge_index % len(result)) if corner_index == (edge_index // len(result)) else (edge_index // len(result))]
                        another_corner_tuple = tuple(another_corner.tolist())
                        if (corner_tuple, another_corner_tuple) in drew_walls or (
                        another_corner_tuple, corner_tuple) in drew_walls:
                            pass
                        else:
                            # print(corner_tuple, another_corner_tuple)
                            cv2.line(img, corner_tuple, another_corner_tuple, color=(0, 0, 0), thickness=3)
                            drew_walls.append((corner_tuple, another_corner_tuple))
                            drew_walls.append((another_corner_tuple, corner_tuple))
        # draw corners
        for corner_index, corner in enumerate(result):
            corner_tuple = tuple(corner.tolist())
            cv2.circle(img, corner_tuple, radius=3, color=(0, 215, 255), thickness=-1)
        cv2.imwrite(output_dir + str(result_index) + '_' + str(timestep) + '.png', img)


def chemistry_visualize(result_atoms_unpaddinged, result_bonds_unpaddinged, output_dir, timestep):
    print(timestep)
    import numpy as np
    from rdkit import Chem
    from rdkit.Chem import AllChem

    sample_numbers = len(result_atoms_unpaddinged)
    for i in range(sample_numbers):
        print(i)
        print(result_atoms_unpaddinged[i])
        print(result_bonds_unpaddinged[i].reshape(1, len(result_atoms_unpaddinged[i][0]), len(result_atoms_unpaddinged[i][0]), 5))

    # Atom and bond type one-hot ndarrays
        atom_array = result_atoms_unpaddinged[i][0]  # Replace with your atom array
        bond_array = result_bonds_unpaddinged[i][0]  # Replace with your bond array

    # Define atom symbol list
        atom_symbols = ["C", "N", "O", "F"]

    # Convert one-hot arrays to an RDKit molecule
        def ndarray_to_molecule(atom_array, bond_array):
            # Build atom list
            atoms = [atom_symbols[np.argmax(atom_one_hot)] for atom_one_hot in atom_array]

            # Create an editable empty molecule
            mol = Chem.EditableMol(Chem.Mol())

            # Add atoms
            for atom in atoms:
                mol.AddAtom(Chem.Atom(atom))

            # Add bonds (upper triangle only)
            num_atoms = len(atom_array)
            for i in range(num_atoms):
                for j in range(i + 1, num_atoms):
                    bond_type = np.argmax(bond_array[i * num_atoms + j])  # bond type (0:none,1:single,2:double,3:triple)
                    if bond_type > 0:
                        mol.AddBond(i, j, Chem.BondType.values[bond_type])


            # Get immutable molecule and list atoms/bonds
            mol = mol.GetMol()
            for a in mol.GetAtoms():
                print(a.GetIdx(), a.GetSymbol())
            for b in mol.GetBonds():
                print(b.GetBeginAtomIdx(), b.GetEndAtomIdx(), b.GetBondType())

            return mol

        mol = ndarray_to_molecule(atom_array, bond_array)
        from rdkit.Chem import Draw

    # Generate 2D coordinates
        AllChem.Compute2DCoords(mol)

    # Render molecule to file
        Draw.MolToFile(mol, output_dir + str(i) + '_' + str(timestep) + '.png', size=(256, 256))

def inverse_normalize_remove_padding(result_corners, results_edges, results_corners_numbers):
    result_corners_inverse_normalized = []
    for i, result in enumerate(result_corners):
        result_corner_inverse_normalized = np.array((result * 128 + 128).cpu()).astype(np.uint8)[:,
                                    :results_corners_numbers[i], :]
        result_corners_inverse_normalized.append(result_corner_inverse_normalized)

    result_edges_unpaddinged = []
    for j, l in enumerate(results_edges):
        result_edge_unpaddinged = []
        for _ in range(53):
            if _ < results_corners_numbers[j]:
                result_edge_unpaddinged.append(np.array(l.cpu())[:, _ * 53:_ * 53 + results_corners_numbers[j], :])
        result_edge_unpaddinged = np.concatenate(result_edge_unpaddinged, axis=1)
        result_edges_unpaddinged.append(result_edge_unpaddinged)
    return result_corners_inverse_normalized, result_edges_unpaddinged

def edges_remove_padding(results_edges, results_corners_numbers):
    result_edges_unpaddinged = []
    for j, l in enumerate(results_edges):
        result_edge_unpaddinged = []
        for _ in range(53):
            if _ < results_corners_numbers[j]:
                result_edge_unpaddinged.append(np.array(l.cpu())[:, _ * 53:_ * 53 + results_corners_numbers[j], :])
        result_edge_unpaddinged = np.concatenate(result_edge_unpaddinged, axis=1)
        result_edges_unpaddinged.append(result_edge_unpaddinged)
    return result_edges_unpaddinged

def inverse_normalize_remove_padding_51(result_corners, result_semantics, results_corners_numbers):
    result_corners_inverse_normalized = []
    for i, result in enumerate(result_corners):
        result_corner_inverse_normalized = np.array((result * 128 + 128).cpu()).astype(np.uint8)[:,
                                    :results_corners_numbers[i], :]
        result_corners_inverse_normalized.append(result_corner_inverse_normalized)

    result_semantics_inverse_normalized = []
    for i, result in enumerate(result_semantics):
        result_semantic_inverse_normalized = np.round(np.array(result.cpu())).astype(np.int8)[:,
                                           :results_corners_numbers[i], :]
        result_semantics_inverse_normalized.append(result_semantic_inverse_normalized)
    return result_corners_inverse_normalized, result_semantics_inverse_normalized

def inverse_normalize_and_remove_padding(result_corners, result_semantics, results_corners_numbers):
    result_corners_inverse_normalized = []
    for i, result in enumerate(result_corners):
        result_corner_inverse_normalized = np.array((result * 128 + 128).cpu()).astype(np.uint8)[:,
                                    :results_corners_numbers[i], :]
        result_corners_inverse_normalized.append(result_corner_inverse_normalized)

    result_semantics_inverse_normalized = []
    for i, result in enumerate(result_semantics):
        result_semantic_inverse_normalized = np.round(np.array(result.cpu())).astype(np.int8)[:,
                                           :results_corners_numbers[i], :]
        result_semantics_inverse_normalized.append(result_semantic_inverse_normalized)
    return result_corners_inverse_normalized, result_semantics_inverse_normalized

def inverse_normalize_and_remove_padding_100(result_corners, result_semantics, results_corners_numbers):
    # print(result_corners) # length 3000, each element (1,53,2) tensor
    # print(result_semantics) # length 3000, each element (1,53,8) tensor
    # print(results_corners_numbers) # length 3000, each (53,) tensor; 0=corner, 1=non-corner


    result_corners_inverse_normalized = []
    for i, result in enumerate(result_corners):
        result_corner_inverse_normalized = np.array((result * 128 + 128).cpu()).astype(np.uint8)[:, np.where((results_corners_numbers[i]).cpu().numpy() == 0)[0], :]
        # print(result_corner_inverse_normalized)
        result_corners_inverse_normalized.append(result_corner_inverse_normalized)

    result_semantics_inverse_normalized = []
    for i, result in enumerate(result_semantics):
        result_semantic_inverse_normalized = np.round(np.array(result.cpu())).astype(np.int8)[:, np.where((results_corners_numbers[i]).cpu().numpy() == 0)[0], :]
        result_semantics_inverse_normalized.append(result_semantic_inverse_normalized)
    return result_corners_inverse_normalized, result_semantics_inverse_normalized

def inverse_normalize_and_remove_padding_4testing(result_corners, result_semantics, results_corners_numbers, resolution=512):
    result_corners_inverse_normalized = []
    for i, result in enumerate(result_corners):
        result_corner_inverse_normalized = np.array((result * (resolution // 2) + (resolution // 2)).cpu()).astype(np.int32)[:,
                                    :results_corners_numbers[i], :]
        result_corners_inverse_normalized.append(result_corner_inverse_normalized)

    result_semantics_inverse_normalized = []
    for i, result in enumerate(result_semantics):
        result_semantic_inverse_normalized = np.round(np.array(result.cpu())).astype(np.int8)[:,
                                           :results_corners_numbers[i], :]
        result_semantics_inverse_normalized.append(result_semantic_inverse_normalized)
    return result_corners_inverse_normalized, result_semantics_inverse_normalized

def inverse_normalize_and_remove_padding_100_4testing(result_corners, result_semantics, results_corners_numbers, resolution=512):
    # print(result_corners) # length 3000, each element (1,53,2) tensor
    # print(result_semantics) # length 3000, each element (1,53,8) tensor
    # print(results_corners_numbers) # length 3000, each (53,) tensor; 0=corner,1=non-corner

    result_corners_inverse_normalized = []
    for i, result in enumerate(result_corners):
        result_corner_inverse_normalized = np.array((result * (resolution // 2) + (resolution // 2)).cpu()).astype(np.int32)[:, np.where((results_corners_numbers[i]).cpu().numpy() == 0)[0], :]
        # print(result_corner_inverse_normalized)
        result_corners_inverse_normalized.append(result_corner_inverse_normalized)

    result_semantics_inverse_normalized = []
    for i, result in enumerate(result_semantics):
        result_semantic_inverse_normalized = np.round(np.array(result.cpu())).astype(np.int8)[:, np.where((results_corners_numbers[i]).cpu().numpy() == 0)[0], :]
        result_semantics_inverse_normalized.append(result_semantic_inverse_normalized)
    return result_corners_inverse_normalized, result_semantics_inverse_normalized


def chemistry_remove_padding(result_atoms, result_bonds, results_atoms_numbers):
    result_atoms_unpaddinged = []
    for i, k in enumerate(result_atoms):
        result_atoms_unpaddinged.append(np.array(k.cpu()).astype(np.uint8)[:,
                                    :results_atoms_numbers[i], :])

    result_bonds_unpaddinged = []
    for j, l in enumerate(result_bonds):
        result_bond_unpaddinged = []
        for _ in range(9):
            if _ < results_atoms_numbers[j]:
                result_bond_unpaddinged.append(np.array(l.cpu())[:, _ * 9:_ * 9 + results_atoms_numbers[j], :])
        result_bond_unpaddinged = np.concatenate(result_bond_unpaddinged, axis=1)
        result_bonds_unpaddinged.append(result_bond_unpaddinged)

    return result_atoms_unpaddinged, result_bonds_unpaddinged

def edges_prior_distribution():
    # edges_category1_count = 0
    # edges_category2_count = 0
    # for f in tqdm(os.listdir('../datasets/rplang/train')):
    #     g = np.load('../datasets/rplang/train/' + f, allow_pickle=True).item()
    #     e1_and_2 = g['global_matrix_np_padding']
    #     e2 = g['adjacency_matrix_np_padding']
    #     e2_number = np.sum(e2)
    #     e1_and_2_number = np.sum(e1_and_2)
    #     e1_number = e1_and_2_number - e2_number
    #     edges_category1_count += e1_number
    #     edges_category2_count += e2_number
    # for f in tqdm(os.listdir('../datasets/rplang/test')):
    #     g = np.load('../datasets/rplang/test/' + f, allow_pickle=True).item()
    #     e1_and_2 = g['global_matrix_np_padding']
    #     e2 = g['adjacency_matrix_np_padding']
    #     e2_number = np.sum(e2)
    #     e1_and_2_number = np.sum(e1_and_2)
    #     e1_number = e1_and_2_number - e2_number
    #     edges_category1_count += e1_number
    #     edges_category2_count += e2_number
    # print(edges_category1_count, edges_category2_count)
    # return [edges_category1_count / (edges_category1_count + edges_category2_count),
    #         edges_category2_count / (edges_category1_count + edges_category2_count)]
    return np.array([0.8894, 0.1106], dtype=np.float64)

def atoms_prior_distribution():
    return np.array([0.7230, 0.1151, 0.1593, 0.0026], dtype=np.float64)

def bonds_prior_distribution():
    return np.array([0.7261, 0.2384, 0.0274, 0.0081, 0.0], dtype=np.float64)

def masked_softmax(x, mask, **kwargs):
    x_masked = x.clone()
    x_masked[mask == False] = -float("inf")
    return torch.softmax(x_masked, **kwargs)

def get_quadrant(angle):
    # Angle span across quadrants (two vectors assumed non-overlapping)
    if angle[0] < angle[1]: # case 1
        if 0 <= angle[0] < 90 and 0 <= angle[1] < 90:
            quadrant = (angle[1] - angle[0], 0, 0, 0)
        elif 0 <= angle[0] < 90 and 90 <= angle[1] < 180:
            quadrant = (90 - angle[0], angle[1] - 90, 0, 0)
        elif 0 <= angle[0] < 90 and 180 <= angle[1] < 270:
            quadrant = (90 - angle[0], 90, angle[1] - 180, 0)
        elif 0 <= angle[0] < 90 and 270 <= angle[1] < 360:
            quadrant = (90 - angle[0], 90, 90, angle[1] - 270)
        elif 90 <= angle[0] < 180 and 90 <= angle[1] < 180:
            quadrant = (0, angle[1] - angle[0], 0, 0)
        elif 90 <= angle[0] < 180 and 180 <= angle[1] < 270:
            quadrant = (0, 180 - angle[0], angle[1] - 180, 0)
        elif 90 <= angle[0] < 180 and 270 <= angle[1] < 360:
            quadrant = (0, 180 - angle[0], 90, angle[1] - 270)
        elif 180 <= angle[0] < 270 and 180 <= angle[1] < 270:
            quadrant = (0, 0, angle[1] - angle[0], 0)
        elif 180 <= angle[0] < 270 and 270 <= angle[1] < 360:
            quadrant = (0, 0, 270 - angle[0], angle[1] - 270)
        elif 270 <= angle[0] < 360 and 270 <= angle[1] < 360:
            quadrant = (0, 0, 0, angle[1] - angle[0])
    else: # case 2
        if 0 <= angle[1] < 90 and 0 <= angle[0] < 90:
            quadrant_ = (angle[0] - angle[1], 0, 0, 0)
        elif 0 <= angle[1] < 90 and 90 <= angle[0] < 180:
            quadrant_ = (90 - angle[1], angle[0] - 90, 0, 0)
        elif 0 <= angle[1] < 90 and 180 <= angle[0] < 270:
            quadrant_ = (90 - angle[1], 90, angle[0] - 180, 0)
        elif 0 <= angle[1] < 90 and 270 <= angle[0] < 360:
            quadrant_ = (90 - angle[1], 90, 90, angle[0] - 270)
        elif 90 <= angle[1] < 180 and 90 <= angle[0] < 180:
            quadrant_ = (0, angle[0] - angle[1], 0, 0)
        elif 90 <= angle[1] < 180 and 180 <= angle[0] < 270:
            quadrant_ = (0, 180 - angle[1], angle[0] - 180, 0)
        elif 90 <= angle[1] < 180 and 270 <= angle[0] < 360:
            quadrant_ = (0, 180 - angle[1], 90, angle[0] - 270)
        elif 180 <= angle[1] < 270 and 180 <= angle[0] < 270:
            quadrant_ = (0, 0, angle[0] - angle[1], 0)
        elif 180 <= angle[1] < 270 and 270 <= angle[0] < 360:
            quadrant_ = (0, 0, 270 - angle[1], angle[0] - 270)
        elif 270 <= angle[1] < 360 and 270 <= angle[0] < 360:
            quadrant_ = (0, 0, 0, angle[0] - angle[1])
        quadrant = (90 - quadrant_[0], 90 - quadrant_[1], 90 - quadrant_[2], 90 - quadrant_[3])
    return quadrant

def poly_area(points): # Oriented polygon area (counter-clockwise positive, clockwise negative)
    s = 0
    points_count = len(points)
    for i in range(points_count):
        point = points[i]
        point2 = points[(i + 1) % points_count]
        s += (point[0] - point2[0]) * (point[1] + point2[1])
    return s / 2

def rotate_degree_clockwise_from_counter_degree(src_degree, dest_degree):
    delta = src_degree - dest_degree
    return delta if delta >= 0 else 360 + delta

def rotate_degree_counterclockwise_from_counter_degree(src_degree, dest_degree):
    delta = dest_degree - src_degree
    return delta if delta >= 0 else 360 + delta


def x_axis_angle(y):
    # Image coordinate frame: (1,0) mapped to 0 degrees; measure counter-clockwise to 360 degrees
    # print('-------------')
    # print(y)
    y_right_hand = (y[0], -y[1])
    # print(y_right_hand)

    x = (1, 0)
    inner = x[0] * y_right_hand[0] + x[1] * y_right_hand[1]
    # print(inner)
    y_norm2 = (y_right_hand[0] ** 2 + y_right_hand[1] ** 2) ** 0.5
    # print(y_norm2)
    cosxy = inner / y_norm2
    # print(cosxy)
    angle = math.acos(cosxy)
    # print(angle, math.degrees(angle))
    # print('-------------')
    return math.degrees(angle) if y_right_hand[1] >= 0 else 360 - math.degrees(angle)

def get_results_float_with_semantic(best_result):
    if 1:
        preds = best_result[2]
        # All points and edges
        output_points = []
        output_edges = []
        for triplet in preds:
            this_preds = triplet[0]
            last_edges = triplet[1]
            this_edges = triplet[2]
            for this_pred in this_preds:
                point = (this_pred['points'].tolist()[0], this_pred['points'].tolist()[1],
                         this_pred['semantic_left_up'].item(), this_pred['semantic_right_up'].item(),
                         this_pred['semantic_right_down'].item(), this_pred['semantic_left_down'].item())
                output_points.append(point)
            for last_edge in last_edges:
                point1 = (last_edge[0]['points'].tolist()[0], last_edge[0]['points'].tolist()[1],
                         last_edge[0]['semantic_left_up'].item(), last_edge[0]['semantic_right_up'].item(),
                         last_edge[0]['semantic_right_down'].item(), last_edge[0]['semantic_left_down'].item())
                point2 = (last_edge[1]['points'].tolist()[0], last_edge[1]['points'].tolist()[1],
                          last_edge[1]['semantic_left_up'].item(), last_edge[1]['semantic_right_up'].item(),
                          last_edge[1]['semantic_right_down'].item(), last_edge[1]['semantic_left_down'].item())
                edge = (point1, point2)
                output_edges.append(edge)
            for this_edge in this_edges:
                point1 = (this_edge[0]['points'].tolist()[0], this_edge[0]['points'].tolist()[1],
                          this_edge[0]['semantic_left_up'].item(), this_edge[0]['semantic_right_up'].item(),
                          this_edge[0]['semantic_right_down'].item(), this_edge[0]['semantic_left_down'].item())
                point2 = (this_edge[1]['points'].tolist()[0], this_edge[1]['points'].tolist()[1],
                          this_edge[1]['semantic_left_up'].item(), this_edge[1]['semantic_right_up'].item(),
                          this_edge[1]['semantic_right_down'].item(), this_edge[1]['semantic_left_down'].item())
                edge = (point1, point2)
                output_edges.append(edge)
        return output_points, output_edges

def get_cycle_basis_and_semantic_2(output_points, output_edges):
    # Build a mapping from point tuple to index
    d = {}
    for output_point_index, output_point in enumerate(output_points):
        d[output_point] = output_point_index  # Cannot handle duplicate points; upstream NMS required
    d_rev = {}
    for output_point_index, output_point in enumerate(output_points):
        d_rev[output_point_index] = output_point  # Cannot handle duplicate points; upstream NMS required
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
    # Debug: initial simple_cycles container
    bridges = list(nx.bridges(G))
    # Remove bridge edges so remaining components are either isolated points or cycles
    for b in bridges:
        if (d_rev[b[0]], d_rev[b[1]]) in output_edges:
            output_edges.remove((d_rev[b[0]], d_rev[b[1]]))
            es.remove((b[0], b[1]))
            G.remove_edge(b[0], b[1])
        if (d_rev[b[1]], d_rev[b[0]]) in output_edges:
            output_edges.remove((d_rev[b[1]], d_rev[b[0]]))
            es.remove((b[1], b[0]))
            G.remove_edge(b[1], b[0])
    # After removing bridges, traverse each remaining connected component (isolated node or cycle)
    connected_components = list(nx.connected_components(G))


    for c in connected_components:
        if len(c) == 1:
            pass
        else:
            simple_cycles_c = []
            simple_cycles_number_c = []
            simple_cycle_semantics_c = []
            # print(c) # {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23}
            # Get corresponding point and edge sets
            output_points_c = [p for p in output_points if d[p] in c]
            output_edges_c = [e for e in output_edges if d[e[0]] in c or d[e[1]] in c]  # Fixed edge set (not deleted)
            output_edges_c_copy_for_traversing = copy.deepcopy(output_edges_c)  # Working copy for traversal; edges removed to reduce complexity
            # print(output_points_c)
            # print(output_edges_c)

            # Method to enumerate every counter-clockwise simple cycle in this connected component
            # Use d's indices as identifiers for points in this connected component
            # Iterate through the undirected edge set output_edges_c:
            # For each undirected edge: set the initial point to the endpoint with the smaller index; previous = initial, current = larger index.
            # Collect all neighbor edges of the current point and compute their outgoing (polar) angles.
            # Compute the reverse (outgoing) direction of the vector from last_point to current_point; get its angle in [0, 2pi).
            # Rotate counter‑clockwise starting from that angle to find the last (rightmost before wrap) neighbor edge of current_point.
            # Take the opposite endpoint of that chosen edge as the next point.
            # When next_point equals the initial point, we obtain a cycle of the form [p0, p1, ..., p_{n-1}, p0].
            # After recording a cycle, remove from the remaining edge set every oriented edge (pi, pi+1) with pi < pi+1 (including the starting edge) to avoid duplicates.
            # Update: last_point = current_point; current_point = next_point; continue traversal.

            for edge_c in output_edges_c:
                if edge_c not in output_edges_c_copy_for_traversing:
                    pass
                else:
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
                    # The successor of the initial point (used to decide when the while loop terminates)
                    next_initial_point = copy.deepcopy(current_point)
                    next_initial_point_number = copy.deepcopy(current_point_number)
                    # Next point placeholder
                    next_point = None
                    next_point_number = None
                    # Loop until next_point equals next_initial_point (i.e., we closed the cycle)
                    while next_point != next_initial_point:
                        # Collect all neighbor edges of the current point
                        relevant_edges = []
                        for edge in output_edges_c:
                            if (edge[0] == current_point or edge[1] == current_point) and (not (edge[0] == current_point and edge[1] == current_point)):
                                relevant_edges.append(edge)
                        # Compute outgoing angles for each neighbor edge
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
                        # Rotate counter‑clockwise from that angle to the last encountered neighbor edge (defines interior corner)
                        # Record the untouched swept region (the interior angle interval)
                        # (Here we substitute full semantics for the interior angle semantics)
                        rotate_deltas_counterclockwise = []
                        # Store interior angle interval (counter‑clockwise) from previous direction to candidate direction
                        interior_angles = []
                        for relevant_edge_degree in relevant_edges_degree:
                            rotate_delta = rotate_degree_counterclockwise_from_counter_degree(
                                vec_from_current_point_to_last_point_degree, relevant_edge_degree)
                            rotate_deltas_counterclockwise.append(rotate_delta)
                            interior_angles.append((relevant_edge_degree, vec_from_current_point_to_last_point_degree))
                        # print(rotate_deltas_counterclockwise)
                        # Index of maximum CCW rotation
                        max_rotate_index = rotate_deltas_counterclockwise.index(max(rotate_deltas_counterclockwise))
                        # Interior angle (as a pair of boundary directions)
                        interior_angle_counterclockwise = interior_angles[max_rotate_index]
                        # Determine semantic region for this angle
                        # First gather all semantic channels of current point ordered by quadrants
                        # current_point_semantic = [current_point[3], current_point[2], current_point[5],
                        #                           current_point[4], ]
                        current_point_semantic = [current_point[3], current_point[2], current_point[5],
                                                  current_point[4], current_point[6], current_point[7],
                                                  current_point[8], current_point[9], current_point[10],
                                                  current_point[11], current_point[12], current_point[13],
                                                  current_point[14], current_point[15]]
                        # Compute how much of each quadrant this CCW angle spans:
                        # Take smaller->larger angle extent; if smaller is the source direction use it directly;
                        # if smaller is the target direction subtract from 90 degrees.
                        interior_angle_counterclockwise_degree_smaller = min(interior_angle_counterclockwise)  # smaller angle
                        interior_angle_counterclockwise_degree_bigger = max(interior_angle_counterclockwise)  # larger angle
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
                        # Never set semantics to -1 here
                        current_point_semantic_valid = []
                        for qd, seman in enumerate(current_point_semantic):
                            if 1:
                                current_point_semantic_valid.append(seman)
                            else:
                                current_point_semantic_valid.append(-1)
                        # Record all semantics for this step
                        simple_cycle_semantics.append(current_point_semantic_valid)

                        # Chosen edge (max rotation)
                        max_rotate_edge = relevant_edges[max_rotate_index]
                        # Determine next point
                        if max_rotate_edge[0] == current_point:
                            next_point = max_rotate_edge[1]
                            next_point_number = d[next_point]
                        elif max_rotate_edge[1] == current_point:
                            next_point = max_rotate_edge[0]
                            next_point_number = d[next_point]
                        else:
                            assert 0
                        # Update last/current and append current point to the simple cycle list
                        last_point = current_point
                        last_point_number = current_point_number
                        current_point = next_point
                        current_point_number = next_point_number
                        simple_cycle.append(current_point)
                        simple_cycle_number.append(current_point_number)
                    # Finally append the initial point (close cycle for subsequent edge removal)
                    # simple_cycle.append(initial_point)
                    # simple_cycle_number.append(initial_point_number)
                    # Traverse simple_cycle_number and remove every edge (pi, pi+1) with pi < pi+1 (including the starting edge) from the remaining edge set
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
                    # No need to duplicate the first point when computing polygon area
                    simple_cycle.pop(-1)
                    simple_cycle_number.pop(-1)
                    # Store polygon (area computed CCW; if area < 0 skip since it indicates the largest outer reversed loop)
                    polygon_counterclockwise = [(int(p[0]), -int(p[1])) for p in simple_cycle]
                    polygon_counterclockwise.pop(-1)
                    # print('poly_area(polygon_counterclockwise)', poly_area(polygon_counterclockwise))
                    if poly_area(polygon_counterclockwise) > 0:
                        simple_cycles_c.append(simple_cycle)
                        simple_cycles_number_c.append(simple_cycle_number)
                        # Derive the semantic label for this simple cycle by majority vote (skip outermost reversed cycle)
                        semantic_result = {}
                        for semantic_label in range(0, 14):
                            semantic_result[semantic_label] = 0
                        for everypoint_semantic in simple_cycle_semantics:
                            for _ in range(0, 14):
                                if _ in everypoint_semantic:
                                    semantic_result[_] += 1
                        del semantic_result[13]

                        # print(semantic_result)
                        # If top vote counts tie choose uniformly at random (label 13 excluded)
                        this_cycle_semantic = sorted(semantic_result.items(), key=lambda d: d[1], reverse=True)
                        # print(this_cycle_semantic)
                        this_cycle_result = None
                        if this_cycle_semantic[0][1] > this_cycle_semantic[1][1]:
                            # Unique highest vote wins directly
                            this_cycle_result = this_cycle_semantic[0][0]
                        else:
                            # Collect all labels with max count and pick one uniformly at random
                            this_cycle_results = [i[0] for i in this_cycle_semantic if
                                                  i[1] == this_cycle_semantic[0][1]]
                            this_cycle_result = this_cycle_results[random.randint(0, len(this_cycle_results) - 1)]
                        # print(this_cycle_result)
                        simple_cycle_semantics_c.append(this_cycle_result)

            simple_cycles.extend(simple_cycles_c)
            simple_cycles_number.extend(simple_cycles_number_c)
            simple_cycles_semantics.extend(simple_cycle_semantics_c)

    # print([[(int(j[0]), int(j[1])) for j in i] for i in simple_cycles])

    # print(len(simple_cycles_number))
    # print(simple_cycles_semantics)

    return d_rev, simple_cycles, simple_cycles_semantics


def get_cycle_basis_and_semantic_2_semansimplified(output_points, output_edges):
    # Dictionary mapping each output point to its index
    d = {}
    for output_point_index, output_point in enumerate(output_points):
        d[output_point] = output_point_index  # Cannot handle duplicate points here; NMS must have removed them earlier
    d_rev = {}
    for output_point_index, output_point in enumerate(output_points):
        d_rev[output_point_index] = output_point  # Cannot handle duplicate points here; NMS must have removed them earlier
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
    # print('Breakpoint 1', simple_cycles)
    bridges = list(nx.bridges(G))
    # Handle bridge (virtual) edges by removing them; remaining components are processed independently
    for b in bridges:
        if (d_rev[b[0]], d_rev[b[1]]) in output_edges:
            output_edges.remove((d_rev[b[0]], d_rev[b[1]]))
            es.remove((b[0], b[1]))
            G.remove_edge(b[0], b[1])
        if (d_rev[b[1]], d_rev[b[0]]) in output_edges:
            output_edges.remove((d_rev[b[1]], d_rev[b[0]]))
            es.remove((b[1], b[0]))
            G.remove_edge(b[1], b[0])
    # After removing bridge edges, only isolated nodes or pure cycles remain; iterate through each cycle component
    connected_components = list(nx.connected_components(G))


    for c in connected_components:
        if len(c) == 1:
            pass
        else:
            simple_cycles_c = []
            simple_cycles_number_c = []
            simple_cycle_semantics_c = []
            # print(c) # {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23}
            # Collect corresponding point and edge sets for this component
            output_points_c = [p for p in output_points if d[p] in c]
            output_edges_c = [e for e in output_edges if d[e[0]] in c or d[e[1]] in c]  # Fixed edge set (not removed)
            output_edges_c_copy_for_traversing = copy.deepcopy(output_edges_c)  # Working copy for traversal; edges removed to reduce time complexity
            # print(output_points_c)
            # print(output_edges_c)

            # Method to enumerate all counter‑clockwise simple cycles in this component
            # Use indices in mapping d as the local identifiers for points in this component
            # Iterate through the undirected edge set output_edges_c:
            # For each undirected edge: initial = smaller index endpoint; last = initial; current = larger index
            # Collect all neighbor edges of current point and compute their outgoing angles
            # Compute reverse (outgoing) direction from last_point to current_point; get its angle in [0, 2pi)
            # From that reference angle rotate CCW to select the last encountered neighbor edge
            # Take the opposite endpoint of that neighbor edge as next point
            # When next_point equals the initial point we obtain a cycle of form [p0, p1, ..., p_{n-1}, p0]
            # After recording the cycle remove every edge (pi, pi+1) with pi < pi+1 (including the starting edge) and continue
            # Update last_point = current_point; current_point = next_point

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
                        # Successor of the initial point (used to determine while-loop termination)
                        next_initial_point = copy.deepcopy(current_point)
                        next_initial_point_number = copy.deepcopy(current_point_number)
                        # Next point placeholder
                        next_point = None
                        next_point_number = None
                        # End when next_point equals the successor of the initial point
                        while_count = 0
                        while next_point != next_initial_point and while_count < 1000:
                            # Collect all neighbor edges of the current point
                            relevant_edges = []
                            for edge in output_edges_c:
                                if (edge[0] == current_point or edge[1] == current_point) and (not (edge[0] == current_point and edge[1] == current_point)):
                                    relevant_edges.append(edge)
                            # Compute outgoing angles for all neighbor edges of the current point
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
                            # Compute reverse (outgoing) direction of (last_point -> current_point) and its angle
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
                            # From this angle rotate counter‑clockwise to the last neighbor edge of the current point
                            # Also record the unswept portion (the interior angle interval)
                            # Replace interior‑angle semantics with full semantics
                            rotate_deltas_counterclockwise = []
                            # Record interior angle region: counter‑clockwise from previous angle to next angle
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
                            # Then check: if the smaller degree corresponds to the interior angle's source direction, the quadrant coverage stands as is;
                            # If the smaller degree is the interior angle's target direction, subtract the span from 90°
                            interior_angle_counterclockwise_degree_smaller = min(interior_angle_counterclockwise)  # smaller degree value
                            interior_angle_counterclockwise_degree_bigger = max(interior_angle_counterclockwise)  # larger degree value
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
                            # Never set the semantic value to -1
                            current_point_semantic_valid = []
                            for qd, seman in enumerate(current_point_semantic):
                                if 1:
                                    current_point_semantic_valid.append(seman)
                                else:
                                    current_point_semantic_valid.append(-1)
                            # Tally all semantic values
                            simple_cycle_semantics.append(current_point_semantic_valid)

                            # Corresponding edge
                            max_rotate_edge = relevant_edges[max_rotate_index]
                            # Corresponding next point
                            if max_rotate_edge[0] == current_point:
                                next_point = max_rotate_edge[1]
                                next_point_number = d[next_point]
                            elif max_rotate_edge[1] == current_point:
                                next_point = max_rotate_edge[0]
                                next_point_number = d[next_point]
                            else:
                                assert 0
                            # Reassign last/current/next points and append current point to simple_cycle
                            last_point = current_point
                            last_point_number = current_point_number
                            current_point = next_point
                            current_point_number = next_point_number
                            simple_cycle.append(current_point)
                            simple_cycle_number.append(current_point_number)
                            while_count += 1
                        if len(simple_cycle) > 800:
                            continue
                        # Finally append the initial point (for subsequent edge removal)
                        # simple_cycle.append(initial_point)
                        # simple_cycle_number.append(initial_point_number)
                        # Traverse simple_cycle_number and delete from the remaining set every edge (pi, pi+1) with pi < pi+1 (including the starting edge)
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
                        # No need to explicitly close the loop when computing area
                        simple_cycle.pop(-1)
                        simple_cycle_number.pop(-1)
                        # Store cycle: compute area assuming CCW; if area is negative skip (indicates the outer largest face)
                        polygon_counterclockwise = [(int(p[0]), -int(p[1])) for p in simple_cycle]
                        polygon_counterclockwise.pop(-1)
                        # print('poly_area(polygon_counterclockwise)', poly_area(polygon_counterclockwise))
                        if poly_area(polygon_counterclockwise) > 0:
                            simple_cycles_c.append(simple_cycle)
                            simple_cycles_number_c.append(simple_cycle_number)
                            # Compute majority (common) semantics excluding the largest outer cycle; assign and record semantics for this simple_cycle
                            semantic_result = {}
                            for semantic_label in range(0, 7):
                                semantic_result[semantic_label] = 0
                            for everypoint_semantic in simple_cycle_semantics:
                                for _ in range(0, 7):
                                    if _ in everypoint_semantic:
                                        semantic_result[_] += 1
                            del semantic_result[6]

                            # print(semantic_result)
                            # If the highest vote count ties, pick one uniformly at random (ignoring label 13)
                            this_cycle_semantic = sorted(semantic_result.items(), key=lambda d: d[1], reverse=True)
                            # print(this_cycle_semantic)
                            this_cycle_result = None
                            if this_cycle_semantic[0][1] > this_cycle_semantic[1][1]:
                                # Adopt the uniquely highest vote
                                this_cycle_result = this_cycle_semantic[0][0]
                            else:
                                # Find all labels with the max vote and select one uniformly at random
                                this_cycle_results = [i[0] for i in this_cycle_semantic if
                                                      i[1] == this_cycle_semantic[0][1]]
                                this_cycle_result = this_cycle_results[random.randint(0, len(this_cycle_results) - 1)]
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

    return d_rev, simple_cycles, simple_cycles_semantics

def get_cycle_basis_and_semantic_3_semansimplified(output_points, output_edges):
    # Dictionary relating indices to output points
    d = {}
    for output_point_index, output_point in enumerate(output_points):
        d[output_point] = output_point_index  # Cannot handle duplicate points here; do not remove prior NMS
    d_rev = {}
    for output_point_index, output_point in enumerate(output_points):
        d_rev[output_point_index] = output_point  # Cannot handle duplicate points here; do not remove prior NMS
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
    # print('Breakpoint 1', simple_cycles)
    bridges = list(nx.bridges(G))
    # Handling virtual edges: simply remove them from the edge set and treat the result as multiple connected components
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
    # After removing the bridge edge set, only isolated nodes or pure cycles remain; iterate all remaining cycles
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
            output_edges_c = [e for e in output_edges if d[e[0]] in c or d[e[1]] in c]  # Fixed edge set (not removed)
            output_edges_c_copy_for_traversing = copy.deepcopy(output_edges_c)  # Working copy for traversal; edges removed to reduce time complexity
            # print(output_points_c)
            # print(output_edges_c)

            # Method to enumerate all counter‑clockwise simple cycles in this component
            # Use indices in mapping d as the local identifiers for points in this component
            # Iterate through the undirected edge set output_edges_c:
            # For each undirected edge: initial = smaller index endpoint; last = initial; current = larger index
            # Compute all neighbor edges of the current point and their outgoing angles
            # Compute the reverse direction of (last -> current) (standard polar angle in [0, 2π))
            # From that reference angle rotate CCW to the last encountered neighbor edge
            # Use that edge's other endpoint as the next point
            # When next == initial we obtain a cycle like [p0,p1,...,pn-1,p0]
            # Scan the cycle and remove from remaining edges every edge with pi < p(i+1)
            # Update last/current pointers (last <- current, current <- next)

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
                        # Point right after the initial point (loop termination sentinel)
                        next_initial_point = copy.deepcopy(current_point)
                        next_initial_point_number = copy.deepcopy(current_point_number)
                        # Placeholder for next point
                        next_point = None
                        next_point_number = None
                        # Stop when next point equals the post‑initial sentinel (or safety count exceeded)
                        while_count = 0
                        while next_point != next_initial_point and while_count < 100:
                            # Collect all neighbor edges of current point
                            relevant_edges = []
                            for edge in output_edges_c:
                                if (edge[0] == current_point or edge[1] == current_point) and (not (edge[0] == current_point and edge[1] == current_point)):
                                    relevant_edges.append(edge)
                            # Compute outgoing angle for each neighbor edge
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
                                # Outgoing angle
                                vec_degree = x_axis_angle(vec)
                                relevant_edges_degree.append(vec_degree)
                            # Reverse (outgoing) direction of incoming edge last_point -> current_point
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
                            # Rotate CCW from this angle to the last encountered neighbor edge
                            # Record the remaining (unscanned) angular interval = interior angle
                            # We treat interior semantics as the full semantic signal here
                            rotate_deltas_counterclockwise = []
                            # Record interior angle region (CCW: previous direction -> neighbor direction)
                            interior_angles = []
                            for relevant_edge_degree in relevant_edges_degree:
                                rotate_delta = rotate_degree_counterclockwise_from_counter_degree(
                                    vec_from_current_point_to_last_point_degree, relevant_edge_degree)
                                rotate_deltas_counterclockwise.append(rotate_delta)
                                interior_angles.append((relevant_edge_degree, vec_from_current_point_to_last_point_degree))
                            # print(rotate_deltas_counterclockwise)
                            # Index of maximal rotation
                            max_rotate_index = rotate_deltas_counterclockwise.index(max(rotate_deltas_counterclockwise))
                            # Retrieve the corresponding interior angle pair
                            interior_angle_counterclockwise = interior_angles[max_rotate_index]
                            # Derive semantic region
                            # Gather current point semantics ordered by quadrants
                            # current_point_semantic = [current_point[3], current_point[2], current_point[5],
                            #                           current_point[4], ]
                            current_point_semantic = [current_point[3], current_point[2], current_point[5],
                                                      current_point[4], current_point[6], current_point[7],
                                                      current_point[8]]
                            # Compute quadrant coverage of the CCW interior angle:
                            # rotate from smaller degree to larger; if smaller is the source keep as is;
                            # if smaller is the target subtract coverage from 90°.
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
                            # Never set semantic to -1
                            current_point_semantic_valid = []
                            for qd, seman in enumerate(current_point_semantic):
                                if 1:
                                    current_point_semantic_valid.append(seman)
                                else:
                                    current_point_semantic_valid.append(-1)
                            # Accumulate semantics for this vertex
                            simple_cycle_semantics.append(current_point_semantic_valid)

                            # Edge chosen by maximal rotation
                            max_rotate_edge = relevant_edges[max_rotate_index]
                            # Determine next point
                            if max_rotate_edge[0] == current_point:
                                next_point = max_rotate_edge[1]
                                next_point_number = d[next_point]
                            elif max_rotate_edge[1] == current_point:
                                next_point = max_rotate_edge[0]
                                next_point_number = d[next_point]
                            else:
                                assert 0
                            # Reassign (last,current,next) and append current to cycle
                            last_point = current_point
                            last_point_number = current_point_number
                            current_point = next_point
                            current_point_number = next_point_number
                            simple_cycle.append(current_point)
                            simple_cycle_number.append(current_point_number)
                            while_count += 1
                        if len(simple_cycle) > 80:
                            continue
                        # (Optional) append initial point again (edge removal bookkeeping)
                        # simple_cycle.append(initial_point)
                        # simple_cycle_number.append(initial_point_number)
                        # Remove every edge with index ordering pi < p(i+1) from traversal set
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
                        # No need to explicitly close cycle for area calculation
                        simple_cycle.pop(-1)
                        simple_cycle_number.pop(-1)
                        # Keep if CCW area positive (negative implies outermost boundary)
                        polygon_counterclockwise = [(int(p[0]), -int(p[1])) for p in simple_cycle]
                        polygon_counterclockwise.pop(-1)
                        # print('poly_area(polygon_counterclockwise)', poly_area(polygon_counterclockwise))
                        if poly_area(polygon_counterclockwise) > 0:
                            simple_cycles_c.append(simple_cycle)
                            simple_cycles_number_c.append(simple_cycle_number)
                            # Majority semantic (outermost skipped) – tally and record
                            semantic_result = {}
                            for semantic_label in range(0, 7):
                                semantic_result[semantic_label] = 0
                            for everypoint_semantic in simple_cycle_semantics:
                                for _ in range(0, 7):
                                    if _ in everypoint_semantic:
                                        semantic_result[_] += 1
                            del semantic_result[6]

                            # print(semantic_result)
                            # If top vote count ties choose randomly (label 13 excluded)
                            this_cycle_semantic = sorted(semantic_result.items(), key=lambda d: d[1], reverse=True)
                            # print(this_cycle_semantic)
                            if this_cycle_semantic[0][1] > this_cycle_semantic[1][1]:
                                # Unique top vote wins
                                this_cycle_result = this_cycle_semantic[0][0]
                            else:
                                # Tie resolution priority: cabinet(2) > bathroom(4) > kitchen(3) > bedroom(1) > balcony(5) > livingroom(0)
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

    return d_rev, simple_cycles, simple_cycles_semantics

def get_cycle_basis_and_semantic_3_semansimplified_lifull(output_points, output_edges):
    # Dictionary mapping output points to indices
    d = {}
    for output_point_index, output_point in enumerate(output_points):
        d[output_point] = output_point_index  # Cannot handle duplicate points here; NMS cannot be removed
    d_rev = {}
    for output_point_index, output_point in enumerate(output_points):
        d_rev[output_point_index] = output_point  # Cannot handle duplicate points here; NMS cannot be removed
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
    # print('breakpoint1', simple_cycles)
    bridges = list(nx.bridges(G))
    # Handling virtual edges: remove them from the edge set and treat each resulting connected prediction independently
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
    # After removing bridges inspect remaining connected components (only isolated nodes or pure cycles) and traverse all cycles
    connected_components = list(nx.connected_components(G))


    for c in connected_components:
        if len(c) == 1:
            pass
        else:
            simple_cycles_c = []
            simple_cycles_number_c = []
            simple_cycle_semantics_c = []
            # Collect corresponding point/edge subset for this component
            output_points_c = [p for p in output_points if d[p] in c]
            output_edges_c = [e for e in output_edges if d[e[0]] in c or d[e[1]] in c]  # Fixed edge set (not removed)
            output_edges_c_copy_for_traversing = copy.deepcopy(output_edges_c)  # Working traversal edge set; edges removed to reduce complexity


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
                        # Point immediately after the initial point (termination sentinel)
                        next_initial_point = copy.deepcopy(current_point)
                        next_initial_point_number = copy.deepcopy(current_point_number)
                        # Next point placeholder
                        next_point = None
                        next_point_number = None
                        # Stop when next point equals the post‑initial sentinel or safety limit reached
                        while_count = 0
                        while next_point != next_initial_point and while_count < 100:
                            # Collect all neighbor edges of current point
                            relevant_edges = []
                            for edge in output_edges_c:
                                if (edge[0] == current_point or edge[1] == current_point) and (not (edge[0] == current_point and edge[1] == current_point)):
                                    relevant_edges.append(edge)
                            # Compute outgoing angles of neighbor edges
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
                                # Outgoing angle
                                vec_degree = x_axis_angle(vec)
                                relevant_edges_degree.append(vec_degree)
                            # Reverse (outgoing) direction of incoming edge last_point -> current_point and its angle
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
                            # Rotate CCW from this angle until the last (max rotation) neighbor edge is reached
                            # Also record the unscanned portion (the interior angle interval)
                            # Substitute interior-angle semantics with the full semantics
                            rotate_deltas_counterclockwise = []
                            # Record interior angle interval: CCW from previous angle to neighbor angle
                            interior_angles = []
                            for relevant_edge_degree in relevant_edges_degree:
                                rotate_delta = rotate_degree_counterclockwise_from_counter_degree(
                                    vec_from_current_point_to_last_point_degree, relevant_edge_degree)
                                rotate_deltas_counterclockwise.append(rotate_delta)
                                interior_angles.append((relevant_edge_degree, vec_from_current_point_to_last_point_degree))
                            # print(rotate_deltas_counterclockwise)
                            # Index of maximal rotation
                            max_rotate_index = rotate_deltas_counterclockwise.index(max(rotate_deltas_counterclockwise))
                            # Corresponding interior angle
                            interior_angle_counterclockwise = interior_angles[max_rotate_index]
                            # Derive corresponding semantic region
                            # Gather all semantics of the current point ordered by quadrants
                            # current_point_semantic = [current_point[3], current_point[2], current_point[5],
                            #                           current_point[4], ]
                            current_point_semantic = [current_point[3], current_point[2], current_point[5],
                                                      current_point[4], current_point[6], current_point[7],
                                                      current_point[8], current_point[9], current_point[10], current_point[11], current_point[12], current_point[13], current_point[14]]
                            # Compute how much of each quadrant this CCW interior angle covers
                            # Method: rotate CCW from the smaller degree to the larger to determine coverage
                            # Then: if the smaller degree is the 'source' direction use coverage directly;
                            # if it is the 'target' direction subtract coverage from 90°.
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
                            # Never set semantic to -1
                            current_point_semantic_valid = []
                            for qd, seman in enumerate(current_point_semantic):
                                if 1:
                                    current_point_semantic_valid.append(seman)
                                else:
                                    current_point_semantic_valid.append(-1)
                            # Accumulate all semantics
                            simple_cycle_semantics.append(current_point_semantic_valid)

                            # Chosen edge
                            max_rotate_edge = relevant_edges[max_rotate_index]
                            # Next point
                            if max_rotate_edge[0] == current_point:
                                next_point = max_rotate_edge[1]
                                next_point_number = d[next_point]
                            elif max_rotate_edge[1] == current_point:
                                next_point = max_rotate_edge[0]
                                next_point_number = d[next_point]
                            else:
                                assert 0
                            # Reassign last/current/next and append current point to simple_cycle
                            last_point = current_point
                            last_point_number = current_point_number
                            current_point = next_point
                            current_point_number = next_point_number
                            simple_cycle.append(current_point)
                            simple_cycle_number.append(current_point_number)
                            while_count += 1
                        if len(simple_cycle) > 80:
                            continue
                        # (Optional) append initial point again (for edge removal bookkeeping)
                        # simple_cycle.append(initial_point)
                        # simple_cycle_number.append(initial_point_number)
                        # Remove from remaining edge set all edges with pi < p(i+1) (including this one)
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
                        # No explicit closure needed for area computation
                        simple_cycle.pop(-1)
                        simple_cycle_number.pop(-1)
                        # Keep if CCW area > 0 (negative => outer boundary)
                        polygon_counterclockwise = [(int(p[0]), -int(p[1])) for p in simple_cycle]
                        polygon_counterclockwise.pop(-1)
                        # print('poly_area(polygon_counterclockwise)', poly_area(polygon_counterclockwise))
                        if poly_area(polygon_counterclockwise) > 0:
                            simple_cycles_c.append(simple_cycle)
                            simple_cycles_number_c.append(simple_cycle_number)
                            # Majority semantic (largest outer cycle skipped) – compute and record label
                            semantic_result = {}
                            for semantic_label in range(0, 13):
                                semantic_result[semantic_label] = 0
                            for everypoint_semantic in simple_cycle_semantics:
                                for _ in range(0, 13):
                                    if _ in everypoint_semantic:
                                        semantic_result[_] += 1
                            del semantic_result[11]
                            del semantic_result[12]

                            # print(semantic_result)
                            # If top vote counts tie pick uniformly at random (label 13 ignored)
                            this_cycle_semantic = sorted(semantic_result.items(), key=lambda d: d[1], reverse=True)
                            # print(this_cycle_semantic)
                            if this_cycle_semantic[0][1] > this_cycle_semantic[1][1]:
                                # Unique top vote wins
                                this_cycle_result = this_cycle_semantic[0][0]
                            else:
                                # For tied max votes apply priority: 0>10>1>8>9>5>6>2>4>3>7 (fallback = raw frequency)
                                this_cycle_results = [i[0] for i in this_cycle_semantic if
                                                      i[1] == this_cycle_semantic[0][1]]
                                if 0 in this_cycle_results:
                                    this_cycle_result = 0
                                elif 10 in this_cycle_results:
                                    this_cycle_result = 10
                                elif 1 in this_cycle_results:
                                    this_cycle_result = 1
                                elif 8 in this_cycle_results:
                                    this_cycle_result = 8
                                elif 9 in this_cycle_results:
                                    this_cycle_result = 9
                                elif 5 in this_cycle_results:
                                    this_cycle_result = 5
                                elif 6 in this_cycle_results:
                                    this_cycle_result = 6
                                elif 2 in this_cycle_results:
                                    this_cycle_result = 2
                                elif 4 in this_cycle_results:
                                    this_cycle_result = 4
                                elif 3 in this_cycle_results:
                                    this_cycle_result = 3
                                else:
                                    this_cycle_result = 7
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

    return d_rev, simple_cycles, simple_cycles_semantics

def get_cycle_basis_and_semantic_2_semansimplified_4extractingboundary(output_points, output_edges):
    # Dictionary mapping output points to indices
    d = {}
    for output_point_index, output_point in enumerate(output_points):
        d[output_point] = output_point_index  # Cannot handle duplicate points here; NMS cannot be removed
    d_rev = {}
    for output_point_index, output_point in enumerate(output_points):
        d_rev[output_point_index] = output_point  # Cannot handle duplicate points here; NMS cannot be removed
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
    # print('breakpoint1', simple_cycles)
    bridges = list(nx.bridges(G))
    # Handling virtual edges: remove them from edge set and treat each resulting connected component separately
    for b in bridges:
        if (d_rev[b[0]], d_rev[b[1]]) in output_edges:
            output_edges.remove((d_rev[b[0]], d_rev[b[1]]))
            es.remove((b[0], b[1]))
            G.remove_edge(b[0], b[1])
        if (d_rev[b[1]], d_rev[b[0]]) in output_edges:
            output_edges.remove((d_rev[b[1]], d_rev[b[0]]))
            es.remove((b[1], b[0]))
            G.remove_edge(b[1], b[0])
    # After removing bridge edges inspect remaining connected components (only isolated nodes or cycles) and traverse cycles
    connected_components = list(nx.connected_components(G))


    for c in connected_components:
        if len(c) == 1:
            pass
        else:
            simple_cycles_c = []
            simple_cycles_number_c = []
            simple_cycle_semantics_c = []
            # print(c) # {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23}
            # Collect corresponding point/edge subset
            output_points_c = [p for p in output_points if d[p] in c]
            output_edges_c = [e for e in output_edges if d[e[0]] in c or d[e[1]] in c]  # Fixed edge set (not removed)
            output_edges_c_copy_for_traversing = copy.deepcopy(output_edges_c)  # Traversal working edge set; edges removed to reduce complexity
            # print(output_points_c)
            # print(output_edges_c)

            # Method to enumerate all CCW simple cycles in this component
            # Treat indices in d as local point ids within the component
            # Iterate over undirected edges in output_edges_c
            # For each edge choose the smaller-index endpoint as initial; last_point starts there; current_point is the larger-index endpoint
            # Collect all neighbor edges of the current point and compute their outgoing angles
            # Compute the reverse (outgoing) direction of the edge from last_point into current_point (angle in standard polar [0, 2pi))
            # From that angle rotate CCW and pick the last encountered neighbor edge (max CCW rotation)
            # Use the opposite endpoint of that chosen edge as next_point
            # When next_point equals the initial point we obtain a cycle of the form [p0, p1, ..., pn-1, p0]
            # After identifying a cycle, remove all edges with pi < p(i+1) (including the current one) from the traversing set and continue
            # Advance: last_point <- current_point, current_point <- next_point

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
                        # Previous point (last_point)
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
                        # The point after the initial point (to decide termination of while loop)
                        next_initial_point = copy.deepcopy(current_point)
                        next_initial_point_number = copy.deepcopy(current_point_number)
                        # Next point placeholder
                        next_point = None
                        next_point_number = None
                        # Stop when next_point becomes the second vertex (next_initial_point)
                        while_count = 0
                        while next_point != next_initial_point and while_count < 1000:
                            # Gather all neighbor edges of current_point
                            relevant_edges = []
                            for edge in output_edges_c:
                                if (edge[0] == current_point or edge[1] == current_point) and (not (edge[0] == current_point and edge[1] == current_point)):
                                    relevant_edges.append(edge)
                            # Compute outgoing angles for each neighbor edge
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
                            # Compute reverse direction (outgoing) from last_point->current_point and its angle
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
                            # Rotate CCW from that angle to find the last neighbor edge (max rotation)
                            # Record the untouched angular interval (the interior angle) along the way
                            # Interior angle semantics replaced here with full semantics
                            rotate_deltas_counterclockwise = []
                            # Record interior angle interval, CCW from previous angle to next angle
                            interior_angles = []
                            for relevant_edge_degree in relevant_edges_degree:
                                rotate_delta = rotate_degree_counterclockwise_from_counter_degree(
                                    vec_from_current_point_to_last_point_degree, relevant_edge_degree)
                                rotate_deltas_counterclockwise.append(rotate_delta)
                                interior_angles.append((relevant_edge_degree, vec_from_current_point_to_last_point_degree))
                            # print(rotate_deltas_counterclockwise)
                            # Index of maximal rotation
                            max_rotate_index = rotate_deltas_counterclockwise.index(max(rotate_deltas_counterclockwise))
                            # Retrieve corresponding interior angle
                            interior_angle_counterclockwise = interior_angles[max_rotate_index]
                            # Determine semantic region
                            # Collect all semantics of current point ordered by quadrants
                            # current_point_semantic = [current_point[3], current_point[2], current_point[5],
                            #                           current_point[4], ]
                            current_point_semantic = [current_point[3], current_point[2], current_point[5],
                                                      current_point[4], current_point[6], current_point[7],
                                                      current_point[8]]
                            # Compute portion of quadrants covered by this CCW interior angle
                            # Method: angle from smaller degree rotating CCW to larger degree covers quadrants
                            # If smaller degree is the source direction keep coverage; if it's the target subtract from 90
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
                            # Accumulate semantics for this vertex
                            simple_cycle_semantics.append(current_point_semantic_valid)

                            # Edge chosen by maximal rotation
                            max_rotate_edge = relevant_edges[max_rotate_index]
                            # Determine next point
                            if max_rotate_edge[0] == current_point:
                                next_point = max_rotate_edge[1]
                                next_point_number = d[next_point]
                            elif max_rotate_edge[1] == current_point:
                                next_point = max_rotate_edge[0]
                                next_point_number = d[next_point]
                            else:
                                assert 0
                            # Update last/current pointers and append current_point to simple_cycle
                            last_point = current_point
                            last_point_number = current_point_number
                            current_point = next_point
                            current_point_number = next_point_number
                            simple_cycle.append(current_point)
                            simple_cycle_number.append(current_point_number)
                            while_count += 1
                        if len(simple_cycle) > 800:
                            continue
                        # (Optionally) append initial point for edge deletion (unused here)
                        # simple_cycle.append(initial_point)
                        # simple_cycle_number.append(initial_point_number)
                        # Remove all edges with pi < p(i+1) from traversing set
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
                        # Store cycle (CCW area check omitted; negative would indicate outermost)
                        polygon_counterclockwise = [(int(p[0]), -int(p[1])) for p in simple_cycle]
                        polygon_counterclockwise.pop(-1)
                        # print('poly_area(polygon_counterclockwise)', poly_area(polygon_counterclockwise))

                        # if poly_area(polygon_counterclockwise) > 0:
                        if 1:
                            simple_cycles_c.append(simple_cycle)
                            simple_cycles_number_c.append(simple_cycle_number)
                            # Majority semantic (skip largest outer cycle); tally semantics for this cycle
                            semantic_result = {}
                            for semantic_label in range(0, 7):
                                semantic_result[semantic_label] = 0
                            for everypoint_semantic in simple_cycle_semantics:
                                for _ in range(0, 7):
                                    if _ in everypoint_semantic:
                                        semantic_result[_] += 1

                            # print(semantic_result)
                            # If top counts tie choose uniformly at random (label 13 excluded in original variant)
                            this_cycle_semantic = sorted(semantic_result.items(), key=lambda d: d[1], reverse=True)
                            this_cycle_result = None
                            if this_cycle_semantic[0][1] > this_cycle_semantic[1][1]:
                                # Unique highest vote wins
                                this_cycle_result = this_cycle_semantic[0][0]
                            else:
                                # Collect all labels sharing highest vote count and sample uniformly
                                this_cycle_results = [i[0] for i in this_cycle_semantic if
                                                      i[1] == this_cycle_semantic[0][1]]
                                if 6 in this_cycle_results:
                                    if poly_area(polygon_counterclockwise) > 0:
                                        this_cycle_results.remove(6)
                                        this_cycle_result = this_cycle_results[random.randint(0, len(this_cycle_results) - 1)]
                                    else:
                                        this_cycle_result = 6
                                else:
                                    this_cycle_result = this_cycle_results[random.randint(0, len(this_cycle_results) - 1)]
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

    return d_rev, simple_cycles, simple_cycles_semantics

def edges_to_coordinates(edges_array, vertices_list):
    n = int(np.sqrt(len(edges_array)))  # Derive vertex count from flattened adjacency (n^2 length)
    edges_coordinates = []
    # Scan edge array indices where value == 1
    for idx, value in enumerate(edges_array):
        if value == 1:
            # Compute endpoint indices
            vertex1_idx, vertex2_idx = divmod(idx, n)
            # Look up vertex coordinates
            vertex1 = vertices_list[vertex1_idx]
            vertex2 = vertices_list[vertex2_idx]
            # Represent edge as coordinate pair
            edge_coordinates = (vertex1, vertex2)
            edges_coordinates.append(edge_coordinates)
    return edges_coordinates

# merge points
def get_near_corners(points_array, merge_threshold):
    points = points_array.reshape(-1, 2)
    n = len(points)
    distance_matrix = np.ones((n, n)) * 999999
    for i in range(n):
        for j in range(n):
            distance_matrix[i, j] = np.max(np.abs(points[i] - points[j]))
    edges = np.argwhere(distance_matrix < merge_threshold)
    edges = edges[edges[:, 0] != edges[:, 1]]
    graph = nx.Graph()
    graph.add_edges_from(edges)
    components = list(nx.connected_components(graph))
    return components

def merge_array_elements(array, full_indices_list, random_indices_list):
    merged_elements = []
    for i in range(len(array)):
        if i in full_indices_list and i not in random_indices_list:
            continue
        else:
            merged_elements.append(array[i])
    return np.array(merged_elements)

def graph_level_feature_analysis_for_cvae():
    import pandas as pd

    s0n = []
    s1n = []
    s2n = []
    s3n = []
    s4n = []
    s5n = []
    cn = []
    rn = []
    en = []
    ln = []
    test_files = os.listdir('../datasets/rplang-v3-bubble-diagram/test')
    for test_file in tqdm(test_files):
        test_graph = np.load('../datasets/rplang-v3-bubble-diagram/test/' + test_file, allow_pickle=True).item()

        s0n.append(test_graph['semantics'].count(0))
        s1n.append(test_graph['semantics'].count(1))
        s2n.append(test_graph['semantics'].count(2))
        s3n.append(test_graph['semantics'].count(3))
        s4n.append(test_graph['semantics'].count(4))
        s5n.append(test_graph['semantics'].count(5))

        cn.append(test_graph['corner_number'])
        rn.append(len(test_graph['polygons']))
        en.append(np.sum(np.triu(np.array(test_graph['adjacency_matrix']))))

    # Count simple cycles of various lengths via adjacency matrix traces
        circles_of_length_3 = np.trace(np.linalg.matrix_power(np.array(test_graph['adjacency_matrix']), 3)) / 6
        circles_of_length_4 = np.trace(np.linalg.matrix_power(np.array(test_graph['adjacency_matrix']), 4)) / 8
        circles_of_length_5 = np.trace(np.linalg.matrix_power(np.array(test_graph['adjacency_matrix']), 5)) / 10
        circles_of_length_6 = np.trace(np.linalg.matrix_power(np.array(test_graph['adjacency_matrix']), 6)) / 12
        circles_of_length_7 = np.trace(np.linalg.matrix_power(np.array(test_graph['adjacency_matrix']), 7)) / 14
        circles_of_length_8 = np.trace(np.linalg.matrix_power(np.array(test_graph['adjacency_matrix']), 8)) / 16
        # loopn = circles_of_length_3 + circles_of_length_4 + circles_of_length_5 + circles_of_length_6 + circles_of_length_7 + circles_of_length_8
        loopn = circles_of_length_3
        # print(len(test_graph['polygons']))
        # print(circles_of_length_3)
        # print(circles_of_length_4)
        # print(circles_of_length_5)
        # print(circles_of_length_6)
        # print(circles_of_length_7)
        # print(circles_of_length_8)

        ln.append(loopn)

    train_files = os.listdir('../datasets/rplang-v3-bubble-diagram/train')
    for train_file in tqdm(train_files):
        train_graph = np.load('../datasets/rplang-v3-bubble-diagram/train/' + train_file, allow_pickle=True).item()

        s0n.append(train_graph['semantics'].count(0))
        s1n.append(train_graph['semantics'].count(1))
        s2n.append(train_graph['semantics'].count(2))
        s3n.append(train_graph['semantics'].count(3))
        s4n.append(train_graph['semantics'].count(4))
        s5n.append(train_graph['semantics'].count(5))

        cn.append(train_graph['corner_number'])
        rn.append(len(train_graph['polygons']))
        en.append(np.sum(np.triu(np.array(train_graph['adjacency_matrix']))))

    # Count simple cycles of various lengths via adjacency matrix traces
        circles_of_length_3 = np.trace(np.linalg.matrix_power(np.array(train_graph['adjacency_matrix']), 3)) / 6
        circles_of_length_4 = np.trace(np.linalg.matrix_power(np.array(train_graph['adjacency_matrix']), 4)) / 8
        circles_of_length_5 = np.trace(np.linalg.matrix_power(np.array(train_graph['adjacency_matrix']), 5)) / 10
        circles_of_length_6 = np.trace(np.linalg.matrix_power(np.array(train_graph['adjacency_matrix']), 6)) / 12
        circles_of_length_7 = np.trace(np.linalg.matrix_power(np.array(train_graph['adjacency_matrix']), 7)) / 14
        circles_of_length_8 = np.trace(np.linalg.matrix_power(np.array(train_graph['adjacency_matrix']), 8)) / 16
        # loopn = circles_of_length_3 + circles_of_length_4 + circles_of_length_5 + circles_of_length_6 + circles_of_length_7 + circles_of_length_8
        loopn = circles_of_length_3
        ln.append(loopn)

    val_files = os.listdir('../datasets/rplang-v3-bubble-diagram/val')
    for val_file in tqdm(val_files):
        val_graph = np.load('../datasets/rplang-v3-bubble-diagram/val/' + val_file, allow_pickle=True).item()

        s0n.append(val_graph['semantics'].count(0))
        s1n.append(val_graph['semantics'].count(1))
        s2n.append(val_graph['semantics'].count(2))
        s3n.append(val_graph['semantics'].count(3))
        s4n.append(val_graph['semantics'].count(4))
        s5n.append(val_graph['semantics'].count(5))

        cn.append(val_graph['corner_number'])
        rn.append(len(val_graph['polygons']))
        en.append(np.sum(np.triu(np.array(val_graph['adjacency_matrix']))))

    # Count simple cycles of various lengths via adjacency matrix traces
        circles_of_length_3 = np.trace(np.linalg.matrix_power(np.array(val_graph['adjacency_matrix']), 3)) / 6
        circles_of_length_4 = np.trace(np.linalg.matrix_power(np.array(val_graph['adjacency_matrix']), 4)) / 8
        circles_of_length_5 = np.trace(np.linalg.matrix_power(np.array(val_graph['adjacency_matrix']), 5)) / 10
        circles_of_length_6 = np.trace(np.linalg.matrix_power(np.array(val_graph['adjacency_matrix']), 6)) / 12
        circles_of_length_7 = np.trace(np.linalg.matrix_power(np.array(val_graph['adjacency_matrix']), 7)) / 14
        circles_of_length_8 = np.trace(np.linalg.matrix_power(np.array(val_graph['adjacency_matrix']), 8)) / 16
        # loopn = circles_of_length_3 + circles_of_length_4 + circles_of_length_5 + circles_of_length_6 + circles_of_length_7 + circles_of_length_8
        loopn = circles_of_length_3
        ln.append(loopn)

    df = pd.DataFrame(
        {
            'corner_number': cn,
            'room_number': rn,
            'edge_number': en,
            'living_room_number': s0n,
            'bedroom_number': s1n,
            'kitchen_number': s2n,
            'bathroom_number': s3n,
            'balcony_number': s4n,
            'storage_number': s5n,
            'loop_number': ln,
        }
    )
    # Compute Pearson correlation matrix
    correlation_matrix = df.corr()
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print('--------------------------------------------------------------------------------------------------')
        print(correlation_matrix)

    # Compute Spearman rank correlation matrix
    spearman_corr_matrix = df.corr(method='spearman')
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print('--------------------------------------------------------------------------------------------------')
        print(spearman_corr_matrix)

    # Compute Kendall rank correlation matrix
    kendall_corr_matrix = df.corr(method='kendall')
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print('--------------------------------------------------------------------------------------------------')
        print(kendall_corr_matrix)
    print(ln)
    assert 0