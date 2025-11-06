import os
import json
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
import torch
torch.set_printoptions(threshold=np.inf, linewidth=999999)
np.set_printoptions(threshold=np.inf, linewidth=999999)



 
class lifull(Dataset):
    def __init__(self, mode):
    # Semantics dictionary
        self.semantics_dict = {
            'living_room': 1,
            'kitchen': 2,
            'bedroom': 3,
            'bathroom': 4,
            'restroom': 5,
            'balcony': 6,
            'closet': 7,
            'corridor': 8,
            'washing_room': 9,
            'PS': 10,
            'outside': 11,
            'wall': 12,
            'no_type': 0
        }
        if mode == 'train':
            self.data = self.load_data('../datasets/lifulldata/annot_json/instances_train.json', '../datasets/lifulldata/annot_npy')
        elif mode == 'val':
            self.data = self.load_data('../datasets/lifulldata/annot_json/instances_val.json', '../datasets/lifulldata/annot_npy')
        elif mode == 'test':
            self.data = self.load_data('../datasets/lifulldata/annot_json/instances_test.json', '../datasets/lifulldata/annot_npy')
        

    def load_data(self, json_path, npy_dir):
    # Read JSON file
        with open(json_path, 'r') as file:
            data = json.load(file)
        
    # Initialize integrated data structure
        integrated_data = {}

    # Integrate image information
        for image_info in tqdm(data['images']):
            image_id = image_info['id']
            integrated_data[image_id] = {
                'annotations': [],
                'npy_data': None
            }

    # Integrate annotation information
        for annotation in tqdm(data['annotations']):
            image_id = annotation['image_id']
            if image_id in integrated_data:
                integrated_data[image_id]['annotations'].append({
                    'point': annotation['point'],
                    'semantic': annotation['semantic']
                })
        
            # Provided annotation data
            annotations = integrated_data[image_id]['annotations']
            
            # Initialize point set and semantics set
            point_set = []
            semantic_set = []
            
            # Iterate through annotation entries
            for annotation in annotations:
                point = annotation['point']
                semantics = annotation['semantic']
                
                # Append point to set
                point_set.append(point)
                
                # Create multi-hot encoded vector
                semantic_vector = [0] * len(self.semantics_dict)  # Initialize zero vector
                for semantic in semantics:
                    if semantic in self.semantics_dict:
                        semantic_index = self.semantics_dict[semantic]
                        semantic_vector[semantic_index] = 1  # Set corresponding index to 1
                
                # Append semantic vector to collection
                semantic_set.append(semantic_vector)
            assert len(point_set) == len(semantic_set)
            corners_with_semantics = np.concatenate(((np.array(point_set) - 256) / 256, np.array(semantic_set)), axis=1) # Only divide by 256 here to match rplan data scale
            corners_with_semantics_pad = np.zeros((53 - len(corners_with_semantics), 15))
            corners_with_semantics = np.concatenate((corners_with_semantics, corners_with_semantics_pad), axis=0)
            integrated_data[image_id]['corners_with_semantics'] = corners_with_semantics

        for image_id in integrated_data.keys():
            integrated_data[image_id].pop('annotations', None)
    # Integrate npy data
        for image_id in integrated_data.keys():
            npy_path = os.path.join(npy_dir, f"{image_id}.npy")
            if os.path.exists(npy_path):
                npy_data = np.load(npy_path, allow_pickle=True).item()
                # Remove unnecessary 'quatree' key
                npy_data.pop('quatree', None)
                integrated_data[image_id]['npy_data'] = npy_data
                # Collect all points and assign indices
                points_npy = list(npy_data.keys())
                point_indices = {point: idx for idx, point in enumerate(points_npy)}
                
                # Initialize adjacency matrix
                n = len(points_npy)
                adjacency_matrix = np.zeros((53, 53), dtype=np.uint8)
                adjacency_matrix[:n, :n] = 1
                edges = np.zeros((53, 53), dtype=np.uint8)

                # Populate adjacency matrix
                for point, neighbors in npy_data.items():
                    for neighbor in neighbors:
                        if neighbor != (-1, -1):  # Ignore placeholder non-existent neighbor
                            point_idx = point_indices[point]
                            neighbor_idx = point_indices[neighbor]
                            edges[point_idx, neighbor_idx] = 1
                edges = edges.reshape(2809, 1)
                integrated_data[image_id]['adjacency_matrix'] = adjacency_matrix
                integrated_data[image_id]['edges'] = edges
                pdm = np.zeros((53, 1), dtype=np.uint8)
                pdm[:n, :] = 1
                integrated_data[image_id]['padding_mask'] = pdm
        for image_id in integrated_data.keys():
            integrated_data[image_id].pop('npy_data', None)

        
        return integrated_data
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_id = list(self.data.keys())[idx]
        d = self.data[image_id]
        corners_withsemantics_simplified = d['corners_with_semantics']
        corners_padding_mask = d['padding_mask']
        edges = d['edges']
        global_attn_matrix = d['adjacency_matrix'].astype(bool)
        
        return corners_withsemantics_simplified, global_attn_matrix, corners_padding_mask, edges
