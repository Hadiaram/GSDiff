import sys
import os

# Add project paths to Python path (cross-platform)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'datasets'))
sys.path.insert(0, os.path.join(project_root, 'gsdiff'))

import math
import torch
from torch.optim import AdamW, SGD
from torch.utils.data import DataLoader
from datasets.rplang_edge_semantics_simplified_81_withedges import RPlanGEdgeSemanSimplified_81_WithEdges
from gsdiff.heterhouse_56_32_150corners import BoundEdgeModel_150Corners
from gsdiff.utils import *
from itertools import cycle

'''
Stage 2 Edge Prediction Training for Boundary-Constrained Generation with 150 Corners
This script trains an edge prediction model that takes corners from Stage 1 and predicts wall connections.
'''

lr = 1e-4
weight_decay = 1e-5
total_steps = float("inf")  # Will use early stopping
batch_size = 4  # Smaller batch size for 150x150=22500 edges vs 53x53=2809
device = 'cuda:0'

'''Create output directory'''
output_dir = 'outputs/structure-56-150corners-edge/'
os.makedirs(output_dir, exist_ok=True)

# Save training configuration
config_text = f"""
Stage 2 Edge Training Configuration (150 Corners)
================================================
Learning Rate: {lr}
Weight Decay: {weight_decay}
Batch Size: {batch_size}
Device: {device}
Max Corners: 150
Edge Dimensions: 150x150 = 22,500 edges
Model: BoundEdgeModel
Dataset: RPlanGEdgeSemanSimplified_81_WithEdges
Output Directory: {output_dir}
"""
with open(os.path.join(output_dir, 'training_config.txt'), 'w') as f:
    f.write(config_text)

print(config_text)

'''Neural Network'''
model = BoundEdgeModel_150Corners().to(device)
print(f'Total params: {sum(p.numel() for p in model.parameters()):,}')

'''Data'''
print("\nLoading training dataset...")
dataset_train = RPlanGEdgeSemanSimplified_81_WithEdges('train')
dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=4,
                              drop_last=True, pin_memory=True)  # Optimized for Linux/GPU
dataloader_train_iter = iter(cycle(dataloader_train))

print("Loading validation dataset...")
dataset_val = RPlanGEdgeSemanSimplified_81_WithEdges('val')
dataloader_val = DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=4,
                            drop_last=False, pin_memory=True)  # Optimized for Linux/GPU
dataloader_val_iter = iter(cycle(dataloader_val))

'''Optimizer'''
optimizer = AdamW(list(model.parameters()), lr=lr, weight_decay=weight_decay)

'''Training State'''
step = 0
loss_curve = []
val_metrics = []

# Learning rate reduction and early stopping settings
lr_reduce_patience = 5
lr_reduce_count = 0
stop_patience = 20
stop_count = 0
best_Acc = 0
current_Acc = 0
interval = 100  # Validation every 100 steps

from scipy.stats import truncnorm

def truncated_normal(tensor, mu, sigma, lower, upper, dtype, device):
    """Generates truncated Gaussian distribution samples"""
    with torch.no_grad():
        size = tensor.shape
        tmp = truncnorm.rvs((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma, size=size)
        tmp = torch.as_tensor(tmp, dtype=dtype, device=device)
        tensor.copy_(tmp)
    return tensor

print("\n" + "="*60)
print("Starting Stage 2 Edge Training (150 Corners)")
print("="*60 + "\n")

while step < total_steps:
    '''Training step'''
    feat_16, corners_withsemantics, global_attn_matrix, corners_padding_mask, edges = next(dataloader_train_iter)

    feat_16 = feat_16.to(device).float()  # (bs, c=1024, h=16, w=16)
    corners_withsemantics = corners_withsemantics.to(device).clamp(-1, 1)
    global_attn_matrix = global_attn_matrix.to(device)
    corners_padding_mask = corners_padding_mask.to(device)
    edges = edges.to(device)

    corners = corners_withsemantics[:, :, :2]
    semantics = corners_withsemantics[:, :, 2:]

    # Add random perturbations to corners and semantics (data augmentation)
    # This helps the model generalize to imperfect Stage 1 predictions
    corners_perturbed = truncated_normal(corners.clone(), 0, 0.05, -1, 1, torch.float32, device)

    # Randomly zero out some semantic values (dropout augmentation)
    semantics_mask = torch.rand_like(semantics) > 0.02  # 2% dropout
    semantics_perturbed = semantics * semantics_mask.float()

    '''Forward pass'''
    model.train()
    edges_pred_logits, _ = model(corners_perturbed, global_attn_matrix, corners_padding_mask,
                                   semantics_perturbed, feat_16)

    # Binary cross-entropy loss for edge prediction
    # edges_pred_logits: (bs, 22500, 2) -> softmax -> (bs, 22500, 2)
    # edges: (bs, 22500, 1) -> ground truth
    edges_gt = edges.squeeze(-1).long()  # (bs, 22500)
    edge_mask = global_attn_matrix.reshape(corners.shape[0], -1)  # (bs, 22500)

    loss = F.cross_entropy(edges_pred_logits[edge_mask], edges_gt[edge_mask])

    '''Backward pass'''
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
    optimizer.step()

    loss_curve.append(loss.item())

    '''Validation'''
    if step % interval == 0:
        model.eval()
        val_losses = []
        val_accs = []

        with torch.no_grad():
            for val_idx in range(min(100, len(dataset_val))):  # Validate on 100 samples
                feat_16_val, corners_val, global_attn_val, padding_mask_val, edges_val = next(dataloader_val_iter)

                feat_16_val = feat_16_val.to(device).float()
                corners_val = corners_val.to(device).clamp(-1, 1)
                global_attn_val = global_attn_val.to(device)
                padding_mask_val = padding_mask_val.to(device)
                edges_val = edges_val.to(device)

                corners_coords_val = corners_val[:, :, :2]
                semantics_val = corners_val[:, :, 2:]

                edges_pred_val, _ = model(corners_coords_val, global_attn_val, padding_mask_val,
                                          semantics_val, feat_16_val)

                edges_gt_val = edges_val.squeeze(-1).long()
                edge_mask_val = global_attn_val.reshape(corners_val.shape[0], -1)

                val_loss = F.cross_entropy(edges_pred_val[edge_mask_val], edges_gt_val[edge_mask_val])
                val_losses.append(val_loss.item())

                # Calculate accuracy
                edges_pred_class = edges_pred_val.argmax(dim=-1)
                acc = (edges_pred_class[edge_mask_val] == edges_gt_val[edge_mask_val]).float().mean()
                val_accs.append(acc.item())

        current_Acc = sum(val_accs) / len(val_accs)
        avg_val_loss = sum(val_losses) / len(val_losses)
        avg_train_loss = sum(loss_curve[-interval:]) / len(loss_curve[-interval:])

        print(f'Step {step:07d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {current_Acc:.4f}')

        val_metrics.append({
            'step': step,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'val_acc': current_Acc
        })

        # Save best model
        if current_Acc > best_Acc:
            best_Acc = current_Acc
            torch.save(model.state_dict(), os.path.join(output_dir, 'model_best.pt'))
            print(f'  -> New best model saved! Accuracy: {best_Acc:.4f}')
            stop_count = 0
            lr_reduce_count = 0
        else:
            stop_count += 1
            lr_reduce_count += 1

        # Learning rate reduction
        if lr_reduce_count >= lr_reduce_patience:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.5
            print(f'  -> Learning rate reduced to {optimizer.param_groups[0]["lr"]:.6f}')
            lr_reduce_count = 0

        # Early stopping
        if stop_count >= stop_patience:
            print(f'\nEarly stopping at step {step}. Best accuracy: {best_Acc:.4f}')
            break

        # Periodic checkpoint
        if step % 5000 == 0 and step > 0:
            torch.save(model.state_dict(), os.path.join(output_dir, f'model{step:07d}.pt'))
            print(f'  -> Checkpoint saved at step {step}')

    step += 1

print("\n" + "="*60)
print("Training Complete!")
print(f"Best validation accuracy: {best_Acc:.4f}")
print(f"Total steps: {step}")
print(f"Model saved to: {output_dir}")
print("="*60)
