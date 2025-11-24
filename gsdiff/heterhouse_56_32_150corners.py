import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F


'''Boundary-constrained edge prediction Transformer adapted for 150 corners and 8 semantic dimensions'''


class PositionEmbeddingSine(nn.Module):
    """
    Position embedding for CNN features
    """
    def __init__(self, normalize=True):
        super().__init__()
        self.normalize = normalize

    def forward(self, x):
        not_mask = ~(torch.zeros(x.shape[0], x.shape[2], x.shape[3]).to(x.device).to(torch.bool))
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = (y_embed - 0.5) / (y_embed[:, -1:, :] + eps) * (2 * math.pi)
            x_embed = (x_embed - 0.5) / (x_embed[:, :, -1:] + eps) * (2 * math.pi)
        dim_t = torch.arange(128, dtype=torch.float32, device=x.device)
        dim_t = 10000 ** (2 * (dim_t // 2) / 128)
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model):
        super().__init__()
        self.d_model = d_model
        self.heads = heads
        self.d_subspace = d_model // heads
        self.WQ = nn.Linear(d_model, d_model)
        self.WV = nn.Linear(d_model, d_model)
        self.WK = nn.Linear(d_model, d_model)
        self.fusion = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask):
        batch_size = k.shape[0]
        k = self.WK(k).reshape(batch_size, -1, self.heads, self.d_subspace).transpose(1, 2)
        q = self.WQ(q).reshape(batch_size, -1, self.heads, self.d_subspace).transpose(1, 2)
        v = self.WV(v).reshape(batch_size, -1, self.heads, self.d_subspace).transpose(1, 2)
        mask = mask[:, None, :, :]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_subspace)
        scores = scores.masked_fill(mask == False, -1e9)
        scores = F.softmax(scores, dim=-1)
        scores = torch.matmul(scores, v)
        concat = scores.transpose(1, 2).reshape(batch_size, -1, self.d_model)
        output = self.fusion(concat)
        return output


class TransformerLayer(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.edge_norm = nn.InstanceNorm1d(d_model)
        self.edge_global_attn = MultiHeadAttention(4, d_model)
        self.cross_attn = MultiHeadAttention(4, d_model)
        self.edge_feedforward = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model)
        )

    def forward(self, edges, global_attn_matrix, x, cross_attn_mask):
        '''Edge-to-edge attention'''
        edges_normed1 = self.edge_norm(edges)

        # Create edge attention matrix from global attention
        global_attn_matrix_flatten = global_attn_matrix.reshape(edges.shape[0], -1)
        global_mat_columns = global_attn_matrix_flatten[:, :, None].repeat(1, 1, edges.shape[1])
        global_mat_rows = global_attn_matrix_flatten[:, None, :].repeat(1, edges.shape[1], 1)
        edges_global_matrix = torch.logical_and(global_mat_columns, global_mat_rows)
        edges_attn_matrix = edges_global_matrix

        global_attn = self.edge_global_attn(edges_normed1, edges_normed1, edges_normed1, edges_attn_matrix)
        edges = edges + global_attn

        '''Cross attention from CNN features'''
        edges_normed2 = self.edge_norm(edges)
        cross_attn = self.cross_attn(edges_normed2, x, x, cross_attn_mask)
        edges = edges + cross_attn

        '''Feedforward'''
        edges_normed3 = self.edge_norm(edges)
        edges = edges + self.edge_feedforward(edges_normed3)

        return edges


class BoundEdgeModel_150Corners(nn.Module):
    """
    Edge prediction model adapted for 150 corners and 8 semantic dimensions
    """
    def __init__(self):
        super().__init__()
        self.d_model = 256

        self.transformer_layers = nn.Sequential(
            TransformerLayer(self.d_model),
            TransformerLayer(self.d_model),
            TransformerLayer(self.d_model),
            TransformerLayer(self.d_model),
            TransformerLayer(self.d_model),
            TransformerLayer(self.d_model),
            TransformerLayer(self.d_model),
            TransformerLayer(self.d_model),
            TransformerLayer(self.d_model),
            TransformerLayer(self.d_model),
            TransformerLayer(self.d_model),
            TransformerLayer(self.d_model),
        )

        # CHANGED: 8 semantic dimensions instead of 7
        self.semantics_embedding = nn.Linear(7, self.d_model // 2)  # Keep at 7 - dataset will provide 7

        self.edges_MLP = nn.Linear(self.d_model, 2)

        self.lambdas_MLP = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.SiLU(),
            nn.Linear(self.d_model, self.d_model),
            nn.SiLU(),
            nn.Linear(self.d_model, 1)
        )

        self.proj16 = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=(1, 1)),
            nn.InstanceNorm2d(num_features=256, eps=1e-05, affine=True)
        )
        self.sinopos = PositionEmbeddingSine()

    def forward(self, corners, global_attn_matrix, corners_padding_mask, semantics, feat_16):
        """
        Args:
            corners: (bs, 150, 2) - corner coordinates
            global_attn_matrix: (bs, 150, 150) - attention mask
            corners_padding_mask: (bs, 150, 1) - padding mask
            semantics: (bs, 150, 7) - semantic features
            feat_16: (bs, 1024, 16, 16) - CNN features

        Returns:
            edges_pred_logits: (bs, 22500, 2) - edge predictions (22500 = 150x150)
            lambdas: (bs, 22500, 1) - edge confidences
        """
        batch_size = corners.shape[0]

        '''Create edge embeddings from corner pairs'''
        # Edge coordinates - combine all pairs of corners
        edge_coords_padding_mask = global_attn_matrix.reshape(corners.shape[0], -1, 1)
        edge_coords1 = ((corners[:, :, None, :].repeat(1, 1, corners.shape[1], 1)
                         .reshape(corners.shape[0], -1, corners.shape[2]) * 128 + 128) *
                        edge_coords_padding_mask).float()
        edge_coords2 = ((corners[:, None, :, :].repeat(1, corners.shape[1], 1, 1)
                         .reshape(corners.shape[0], -1, corners.shape[2]) * 128 + 128) *
                        edge_coords_padding_mask).float()

        # Edge semantics - combine all pairs of semantics
        edge_semans1 = ((semantics[:, :, None, :].repeat(1, 1, semantics.shape[1], 1)
                         .reshape(semantics.shape[0], -1, semantics.shape[2])) *
                        edge_coords_padding_mask).float()
        edge_semans2 = ((semantics[:, None, :, :].repeat(1, semantics.shape[1], 1, 1)
                         .reshape(semantics.shape[0], -1, semantics.shape[2])) *
                        edge_coords_padding_mask).float()

        edge_semans = (edge_semans1 + edge_semans2) / 2

        # Embed edge features
        edge_coords = torch.cat([edge_coords1, edge_coords2], dim=2)
        edges_embedding = torch.cat([edge_coords, self.semantics_embedding(edge_semans)], dim=2)

        '''Process CNN features'''
        x16 = self.proj16(feat_16) + self.sinopos(feat_16)
        x16 = x16.flatten(2).permute(0, 2, 1)

        # Create cross-attention mask
        cross_attn_mask = torch.ones(edges_embedding.shape[0], edges_embedding.shape[1], x16.shape[1],
                                      dtype=torch.bool, device=edges_embedding.device)

        '''Transformer layers'''
        edges_out = self.transformer_layers[0](edges_embedding, global_attn_matrix, x16, cross_attn_mask)
        for layer in self.transformer_layers[1:]:
            edges_out = layer(edges_out, global_attn_matrix, x16, cross_attn_mask)

        '''Predict edges and confidences'''
        edges_pred_logits = self.edges_MLP(edges_out)
        lambdas = self.lambdas_MLP(edges_out)

        return edges_pred_logits, lambdas
