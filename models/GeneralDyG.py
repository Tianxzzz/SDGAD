import numpy as np
import torch
import torch.nn as nn

from models.modules import TimeEncoder
from utils.utils import NeighborSampler


class GeneralDyG(nn.Module):
    def __init__(
        self,
        node_raw_features: np.ndarray,
        edge_raw_features: np.ndarray,
        neighbor_sampler: NeighborSampler,
        time_feat_dim: int,
        num_layers: int,
        num_heads: int,
        num_neighbors: int,
        dropout: float,
        device: str = "cpu",
    ):
        super(GeneralDyG, self).__init__()

        self.node_raw_features = torch.from_numpy(node_raw_features.astype(np.float32)).to(device)
        self.edge_raw_features = torch.from_numpy(edge_raw_features.astype(np.float32)).to(device)
        self.neighbor_sampler = neighbor_sampler
        self.time_feat_dim = time_feat_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_neighbors = num_neighbors
        self.dropout = dropout
        self.device = device

        self.node_feat_dim = self.node_raw_features.shape[1]
        self.edge_feat_dim = self.edge_raw_features.shape[1]

        self.time_encoder = TimeEncoder(time_dim=time_feat_dim, parameter_requires_grad=False)
        self.edge_time_mlp = nn.Sequential(
            nn.Linear(self.edge_feat_dim + self.time_feat_dim, self.node_feat_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.node_feat_dim, self.node_feat_dim),
            nn.ReLU(),
        )
        self.output_layer = nn.Sequential(
            nn.Linear(self.node_feat_dim * 2, self.node_feat_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def _gather_node_temporal_embeddings(
        self,
        node_ids: np.ndarray,
        node_interact_times: np.ndarray,
        edge_ids: np.ndarray,
        num_neighbors: int,
    ) -> torch.Tensor:
        neighbor_node_ids, neighbor_edge_ids, neighbor_times = self.neighbor_sampler.get_historical_neighbors(
            node_ids=node_ids, node_interact_times=node_interact_times, num_neighbors=num_neighbors
        )

        node_ids_tensor = torch.from_numpy(node_ids).long().to(self.device)
        edge_ids_tensor = torch.from_numpy(edge_ids).long().to(self.device)
        interact_times_tensor = torch.from_numpy(node_interact_times).float().to(self.device)
        neighbor_edge_ids_tensor = torch.from_numpy(neighbor_edge_ids).long().to(self.device)
        neighbor_node_ids_tensor = torch.from_numpy(neighbor_node_ids).long().to(self.device)
        neighbor_times_tensor = torch.from_numpy(neighbor_times).float().to(self.device)

        node_features = self.node_raw_features[node_ids_tensor]
        neighbor_edge_features = self.edge_raw_features[neighbor_edge_ids_tensor]

        time_diffs = interact_times_tensor.unsqueeze(1) - neighbor_times_tensor
        neighbor_time_features = self.time_encoder(time_diffs)

        neighbor_embeddings = torch.cat([neighbor_edge_features, neighbor_time_features], dim=-1)
        neighbor_embeddings = self.edge_time_mlp(neighbor_embeddings)

        valid_neighbor_mask = (neighbor_node_ids_tensor > 0).float().unsqueeze(-1).to(self.device)
        neighbor_embeddings = neighbor_embeddings * valid_neighbor_mask

        neighbor_count = valid_neighbor_mask.sum(dim=1).clamp_min(1e-6)
        aggregated_neighbors = neighbor_embeddings.sum(dim=1) / neighbor_count

        combined = torch.cat([node_features, aggregated_neighbors], dim=-1)
        node_embeddings = self.output_layer(combined)

        return node_embeddings

    def compute_src_dst_node_temporal_embeddings(
        self,
        src_node_ids: np.ndarray,
        dst_node_ids: np.ndarray,
        node_interact_times: np.ndarray,
        edge_ids: np.ndarray,
        num_neighbors: int = 20,
        **kwargs,
    ):
        src_embeddings = self._gather_node_temporal_embeddings(
            node_ids=src_node_ids, node_interact_times=node_interact_times, edge_ids=edge_ids, num_neighbors=num_neighbors
        )
        dst_embeddings = self._gather_node_temporal_embeddings(
            node_ids=dst_node_ids, node_interact_times=node_interact_times, edge_ids=edge_ids, num_neighbors=num_neighbors
        )
        return src_embeddings, dst_embeddings

    def set_neighbor_sampler(self, neighbor_sampler: NeighborSampler):
        self.neighbor_sampler = neighbor_sampler
        if self.neighbor_sampler.sample_neighbor_strategy in ["uniform", "time_interval_aware"]:
            assert self.neighbor_sampler.seed is not None
            self.neighbor_sampler.reset_random_state()
