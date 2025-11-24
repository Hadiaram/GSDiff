import sys
import os

# Add project paths
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'datasets'))
sys.path.insert(0, os.path.join(project_root, 'gsdiff'))

import torch
import numpy as np
from torch.utils.data import DataLoader
from datasets.rplang_edge_semantics_simplified_81 import RPlanGEdgeSemanSimplified_81
from gsdiff.heterhouse_81_106_3 import BoundHeterHouseModel
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches

"""
Stage 1 Testing Script - Boundary-Constrained Corner Generation (150 corners)
Tests the trained Stage 1 model to verify corner generation is working.
"""

# Configuration
checkpoint_path = 'outputs/structure-81-106-3/model_best.pt'
device = 'cuda:0'
num_test_samples = 10  # Number of samples to generate and visualize
output_dir = 'test_outputs/stage1_only/'
os.makedirs(output_dir, exist_ok=True)

# Diffusion parameters (matching training)
diffusion_steps = 1000
import math
alpha_bar = lambda t: math.cos((t) / 1.000 * math.pi / 2) ** 2
betas = []
max_beta = 0.999
for i in range(diffusion_steps):
    t1 = i / diffusion_steps
    t2 = (i + 1) / diffusion_steps
    betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
betas = np.array(betas, dtype=np.float64)
alphas = 1.0 - betas
alphas_cumprod = np.cumprod(alphas)
sqrt_alphas_cumprod = np.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - alphas_cumprod)
sqrt_recip_alphas_cumprod = np.sqrt(1.0 / alphas_cumprod)
sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / alphas_cumprod - 1)
alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
posterior_mean_coef1 = betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
posterior_mean_coef2 = (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod)

print("="*70)
print("Stage 1 Testing - Boundary-Constrained Corner Generation")
print("="*70)
print(f"Checkpoint: {checkpoint_path}")
print(f"Device: {device}")
print(f"Output Directory: {output_dir}")
print(f"Number of test samples: {num_test_samples}")
print()

# Load model
print("Loading Stage 1 model...")
model = BoundHeterHouseModel().to(device)
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()
print("Model loaded successfully!")

# Load test dataset
print("\nLoading test dataset...")
dataset_test = RPlanGEdgeSemanSimplified_81('test')
dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=0)
print(f"Test dataset size: {len(dataset_test)}")

# Generate samples
print(f"\nGenerating {num_test_samples} samples...")
results = []

with torch.no_grad():
    for idx, (feat_16, corners_gt, global_attn, padding_mask) in enumerate(tqdm(dataloader_test)):
        if idx >= num_test_samples:
            break

        feat_16 = feat_16.to(device).float()
        corners_gt = corners_gt.to(device).float()
        global_attn = global_attn.to(device)
        padding_mask = padding_mask.to(device)

        # IMPORTANT: Add padding mask as 10th dimension (model was trained this way!)
        # (bs, 150, 9) + (bs, 150, 1) -> (bs, 150, 10)
        corners_gt_with_mask = torch.cat((corners_gt, (1 - padding_mask).type(corners_gt.dtype)), dim=2)

        # Start from random noise (10 dimensions now)
        corners_t = torch.randn_like(corners_gt_with_mask).to(device)

        # Reverse diffusion process
        for time_step in reversed(range(diffusion_steps)):
            t = torch.tensor([time_step], device=device).long()

            # Model prediction
            output1, output2 = model(corners_t, global_attn, t, feat_16)
            predicted_noise = torch.cat((output1, output2), dim=2)

            # Denoising step (DDPM with clamp trick - matching training!)
            if time_step > 0:
                # Predict x0
                pred_x0 = (corners_t - sqrt_one_minus_alphas_cumprod[time_step] * predicted_noise) / sqrt_alphas_cumprod[time_step]

                # Apply clamp trick (matching training behavior)
                pred_x0_coord = torch.clamp(pred_x0[:, :, 0:2], min=-1, max=1)  # Clamp coordinates
                pred_x0_seman = (pred_x0[:, :, 2:] >= 0.5).float()  # Binarize semantics
                pred_x0 = torch.cat((pred_x0_coord, pred_x0_seman), dim=2)

                # Get posterior mean
                mean = (posterior_mean_coef1[time_step] * pred_x0 +
                       posterior_mean_coef2[time_step] * corners_t)

                # Add noise
                noise = torch.randn_like(corners_t)
                corners_t = mean + torch.sqrt(torch.tensor(posterior_variance[time_step], device=device)) * noise
            else:
                # Final step - no noise
                pred_x0 = (corners_t - sqrt_one_minus_alphas_cumprod[time_step] * predicted_noise) / sqrt_alphas_cumprod[time_step]

                # Apply clamp trick on final output
                pred_x0_coord = torch.clamp(pred_x0[:, :, 0:2], min=-1, max=1)
                pred_x0_seman = (pred_x0[:, :, 2:] >= 0.5).float()
                corners_t = torch.cat((pred_x0_coord, pred_x0_seman), dim=2)

        # Extract results (remove the padding mask dimension we added)
        corners_pred = corners_t[0, :, :9].cpu().numpy()  # (150, 9) - remove 10th dim
        corners_gt_np = corners_gt[0].cpu().numpy()  # (150, 9)
        padding_mask_np = padding_mask[0].cpu().numpy().flatten()

        # Get valid corners (non-padded)
        valid_pred = corners_pred[padding_mask_np == 1]
        valid_gt = corners_gt_np[padding_mask_np == 1]

        results.append({
            'predicted': valid_pred,
            'ground_truth': valid_gt,
            'sample_idx': idx
        })

print(f"\nGenerated {len(results)} samples!")

# Visualize results
print("\nCreating visualizations...")
room_colors = {
    0: 'lightblue',    # Living room
    1: 'lightgreen',   # Bedroom
    2: 'lightyellow',  # Kitchen
    3: 'lightcoral',   # Bathroom
    4: 'lightgray',    # Balcony
    5: 'wheat',        # Closet
    6: 'white'         # Other
}

for i, result in enumerate(results):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

    # Ground truth
    gt_corners = result['ground_truth']
    gt_coords = gt_corners[:, :2]  # (N, 2)
    gt_semantics = gt_corners[:, 2:]  # (N, 7)

    ax1.set_xlim(-1.1, 1.1)
    ax1.set_ylim(-1.1, 1.1)
    ax1.set_aspect('equal')
    ax1.set_title(f'Ground Truth (Sample {result["sample_idx"]})\n{len(gt_coords)} corners', fontsize=14)
    ax1.grid(True, alpha=0.3)

    # Plot corners with room type colors
    for j, (coord, sem) in enumerate(zip(gt_coords, gt_semantics)):
        room_type = np.argmax(sem)
        color = room_colors.get(room_type, 'gray')
        ax1.scatter(coord[0], coord[1], c=color, s=100, edgecolors='black', linewidths=1.5, zorder=3)
        ax1.text(coord[0]+0.05, coord[1]+0.05, str(j), fontsize=8)

    # Predicted
    pred_corners = result['predicted']
    pred_coords = pred_corners[:, :2]
    pred_semantics = pred_corners[:, 2:]

    ax2.set_xlim(-1.1, 1.1)
    ax2.set_ylim(-1.1, 1.1)
    ax2.set_aspect('equal')
    ax2.set_title(f'Generated (Sample {result["sample_idx"]})\n{len(pred_coords)} corners', fontsize=14)
    ax2.grid(True, alpha=0.3)

    for j, (coord, sem) in enumerate(zip(pred_coords, pred_semantics)):
        room_type = np.argmax(sem)
        color = room_colors.get(room_type, 'gray')
        ax2.scatter(coord[0], coord[1], c=color, s=100, edgecolors='black', linewidths=1.5, zorder=3)
        ax2.text(coord[0]+0.05, coord[1]+0.05, str(j), fontsize=8)

    # Add legend
    legend_elements = [patches.Patch(facecolor=color, edgecolor='black', label=name)
                      for name, color in [('Living', 'lightblue'), ('Bedroom', 'lightgreen'),
                                         ('Kitchen', 'lightyellow'), ('Bathroom', 'lightcoral'),
                                         ('Balcony', 'lightgray'), ('Closet', 'wheat')]]
    ax2.legend(handles=legend_elements, loc='upper right', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'sample_{i:03d}.png'), dpi=150, bbox_inches='tight')
    plt.close()

print(f"\nVisualization complete! Results saved to: {output_dir}")

# Print statistics
print("\n" + "="*70)
print("Generation Statistics:")
print("="*70)
for i, result in enumerate(results):
    pred_count = len(result['predicted'])
    gt_count = len(result['ground_truth'])
    print(f"Sample {i}: Generated {pred_count} corners (GT: {gt_count})")

print("\n" + "="*70)
print("Stage 1 Testing Complete!")
print("="*70)
print(f"\nResults saved to: {output_dir}")
print("Review the visualizations to check if corner generation is working properly.")
print("\nWhat to look for:")
print("  - Are corners distributed across the layout?")
print("  - Do room type colors match reasonable patterns?")
print("  - Is the number of corners similar to ground truth?")
print("\nOnce Stage 1 looks good, you can train Stage 2 to add wall connections.")
