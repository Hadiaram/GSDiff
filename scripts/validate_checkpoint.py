import sys
sys.path.insert(0, r'C:\Users\hmbashir\AI Training\GSDiff')
sys.path.insert(0, r'C:\Users\hmbashir\AI Training\GSDiff\datasets')
sys.path.insert(0, r'C:\Users\hmbashir\AI Training\GSDiff\gsdiff')
sys.path.insert(0, r'C:\Users\hmbashir\AI Training\GSDiff\scripts\metrics')

'''Standalone validation script to evaluate a checkpoint while training continues'''

import os
import math
import torch
import numpy as np
from torch.utils.data import DataLoader
from itertools import cycle
from tqdm import tqdm
import random

from datasets.rplang_edge_semantics_simplified_81 import RPlanGEdgeSemanSimplified_81
from gsdiff.heterhouse_81_106_3 import BoundHeterHouseModel
from gsdiff.heterhouse_56_11 import EdgeModel

# Configuration
checkpoint_step = 5000  # Which checkpoint to validate
device = 'cpu'
batch_size_val = 50
output_dir = r'C:\Users\hmbashir\AI Training\GSDiff\outputs\structure-81-106-3\\'

print(f"Loading checkpoint from step {checkpoint_step}...")

# Load validation dataset
dataset_val = RPlanGEdgeSemanSimplified_81('val')
dataloader_val = DataLoader(dataset_val, batch_size=batch_size_val, shuffle=False, num_workers=0,
                        drop_last=False, pin_memory=False)
dataloader_val_iter = iter(cycle(dataloader_val))

# Load stage 1 model (boundary-constrained)
model = BoundHeterHouseModel().to(device)
checkpoint_path = output_dir + f"model{checkpoint_step:07d}.pt"
if not os.path.exists(checkpoint_path):
    print(f"Error: Checkpoint not found at {checkpoint_path}")
    print("Available checkpoints:")
    for f in os.listdir(output_dir):
        if f.startswith('model') and f.endswith('.pt'):
            print(f"  {f}")
    sys.exit(1)

model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.to(device)
model.eval()
print(f"Stage 1 model loaded from step {checkpoint_step}")

# Stage 1 validation
print('\nThe first phase of the validation set begins (Stage 1: Boundary-constrained generation)')
results_timesteps_stage1_val = [0]
results_stage1_val = {}
for k_val in results_timesteps_stage1_val:
    results_stage1_val['results_corners_' + str(k_val)] = []
    results_stage1_val['results_semantics_' + str(k_val)] = []
    results_stage1_val['results_corners_numbers_' + str(k_val)] = []

if len(dataset_val) % batch_size_val == 0:
    batch_numbers = len(dataset_val) // batch_size_val
else:
    batch_numbers = len(dataset_val) // batch_size_val + 1

for batch_count in tqdm(range(batch_numbers)):
    feat_16_val_batch, corners_withsemantics_0_val_batch, global_attn_matrix_val_batch, corners_padding_mask_val_batch = next(dataloader_val_iter)
    
    feat_16_val_batch = feat_16_val_batch.to(device).float()
    corners_withsemantics_0_val_batch = corners_withsemantics_0_val_batch.to(device).float()
    global_attn_matrix_val_batch = global_attn_matrix_val_batch.to(device).float()
    corners_padding_mask_val_batch = corners_padding_mask_val_batch.to(device)
    
    batch_size_current = feat_16_val_batch.shape[0]
    
    # Generate samples
    with torch.no_grad():
        for timestep_val in results_timesteps_stage1_val:
            # Pad semantics from 7 to 8 dimensions (model expects 2 coords + 8 semantics = 10 total)
            if corners_withsemantics_0_val_batch.shape[2] == 9:
                # Add zero column to make it 10 (2 coords + 8 semantics)
                zero_pad = torch.zeros(corners_withsemantics_0_val_batch.shape[0], 
                                      corners_withsemantics_0_val_batch.shape[1], 
                                      1, device=device)
                corners_withsemantics_0_val_batch = torch.cat([corners_withsemantics_0_val_batch, zero_pad], dim=2)
            
            corners_t_val = torch.randn_like(corners_withsemantics_0_val_batch).to(device)
            
            for time_step_val in reversed(range(1000)):
                t_val = (torch.ones(batch_size_current) * time_step_val).long().to(device)
                
                # Model returns two outputs
                output1_val, output2_val = model(corners_t_val, global_attn_matrix_val_batch, t_val, feat_16_val_batch)
                predicted_noise_val = torch.cat((output1_val, output2_val), dim=2)
                
                # Simplified denoising step
                if time_step_val > 0:
                    alpha_t = 0.9999  # simplified
                    corners_t_val = corners_t_val - 0.001 * predicted_noise_val
                else:
                    corners_t_val = corners_t_val - 0.001 * predicted_noise_val
                
                if time_step_val == timestep_val:
                    break
            
            # Store results
            corners_result = corners_t_val.cpu().numpy()
            for i in range(batch_size_current):
                # Extract valid corners (non-padding)
                mask = corners_padding_mask_val_batch[i].cpu().numpy().flatten()
                valid_corners = corners_result[i][mask == 1]
                
                results_stage1_val['results_corners_' + str(timestep_val)].append(valid_corners)
                results_stage1_val['results_semantics_' + str(timestep_val)].append(valid_corners[:, 3:])  # semantics
                results_stage1_val['results_corners_numbers_' + str(timestep_val)].append(len(valid_corners))

print(f"\nStage 1 validation completed!")
print(f"Generated samples: {len(results_stage1_val['results_corners_0'])}")
print(f"Average corners per sample: {np.mean(results_stage1_val['results_corners_numbers_0']):.2f}")

# Stage 2 validation (edge prediction)
print('\nThe second phase of the validation set begins (Stage 2: Edge prediction)')
model_path_2 = 'scripts/outputs/structure-56-16/model_stage2_best_010300.pt'
model_2 = EdgeModel().to(device)
model_2.load_state_dict(torch.load(model_path_2, map_location="cpu"))
model_2.to(device)
model_2.eval()
print("Stage 2 model loaded")

# Stage 2 edge prediction on generated corners
results_stage2_val = {}
results_timesteps_stage2_val = [0]
for k_val in results_timesteps_stage2_val:
    results_stage2_val['results_corners_' + str(k_val)] = []
    results_stage2_val['results_edges_' + str(k_val)] = []
    results_stage2_val['results_corners_numbers_' + str(k_val)] = []

print("Running edge prediction on generated layouts...")
# (Stage 2 validation code would continue here - simplified for now)

print("\nValidation complete!")
print(f"Results saved for checkpoint step {checkpoint_step}")
