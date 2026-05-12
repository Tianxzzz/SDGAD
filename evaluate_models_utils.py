import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import logging
import time
import argparse
import os
import json

from models.EdgeBank import edge_bank_link_prediction
from utils.metrics import get_link_prediction_metrics, get_node_classification_metrics
from utils.utils import set_random_seed
from utils.utils import NegativeEdgeSampler, NeighborSampler
from utils.DataLoader import Data
from utils.loss import (
    get_logp,
    calculate_log_barrier_bg_spp_loss,
    calculate_log_barrier_bi_occ_loss,
    get_logp_boundary,
)


def evaluate_model_link_prediction(
    model_name: str,
    model: nn.Module,
    neighbor_sampler: NeighborSampler,
    evaluate_idx_data_loader: DataLoader,
    evaluate_neg_edge_sampler: NegativeEdgeSampler,
    evaluate_data: Data,
    loss_func: nn.Module,
    num_neighbors: int = 20,
    time_gap: int = 2000,
):
    """
    Original link prediction evaluation using NegativeEdgeSampler (used by train_link_prediction.py).
    """
    assert evaluate_neg_edge_sampler.seed is not None
    evaluate_neg_edge_sampler.reset_random_state()

    if model_name in ["DyRep", "TGAT", "TGN", "CAWN", "TCL", "GraphMixer", "DyGFormer"]:
        model[0].set_neighbor_sampler(neighbor_sampler)

    model.eval()

    if model_name in ["JODIE", "DyRep", "TGN"]:
        memory_bank = model["backbone"][0].memory_bank
        all_node_ids = np.arange(memory_bank.node_memories.shape[0])
        memory_bank.clear_node_raw_messages(node_ids=all_node_ids)

    with torch.no_grad():
        evaluate_losses, evaluate_metrics = [], []
        evaluate_idx_data_loader_tqdm = tqdm(evaluate_idx_data_loader, ncols=120)
        for batch_idx, evaluate_data_indices in enumerate(evaluate_idx_data_loader_tqdm):
            evaluate_data_indices = evaluate_data_indices.numpy()
            batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times, batch_edge_ids = (
                evaluate_data.src_node_ids[evaluate_data_indices],
                evaluate_data.dst_node_ids[evaluate_data_indices],
                evaluate_data.node_interact_times[evaluate_data_indices],
                evaluate_data.edge_ids[evaluate_data_indices],
            )

            if evaluate_neg_edge_sampler.negative_sample_strategy != "random":
                batch_neg_src_node_ids, batch_neg_dst_node_ids = evaluate_neg_edge_sampler.sample(
                    size=len(batch_src_node_ids),
                    batch_src_node_ids=batch_src_node_ids,
                    batch_dst_node_ids=batch_dst_node_ids,
                    current_batch_start_time=batch_node_interact_times[0],
                    current_batch_end_time=batch_node_interact_times[-1],
                )
            else:
                _, batch_neg_dst_node_ids = evaluate_neg_edge_sampler.sample(size=len(batch_src_node_ids))
                batch_neg_src_node_ids = batch_src_node_ids

            if model_name in ["TGAT", "CAWN", "TCL"]:
                batch_src_node_embeddings, batch_dst_node_embeddings = model[0].compute_src_dst_node_temporal_embeddings(
                    src_node_ids=batch_src_node_ids,
                    dst_node_ids=batch_dst_node_ids,
                    node_interact_times=batch_node_interact_times,
                    num_neighbors=num_neighbors,
                )
                batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = model[0].compute_src_dst_node_temporal_embeddings(
                    src_node_ids=batch_neg_src_node_ids,
                    dst_node_ids=batch_neg_dst_node_ids,
                    node_interact_times=batch_node_interact_times,
                    num_neighbors=num_neighbors,
                )
            elif model_name in ["JODIE", "DyRep", "TGN"]:
                batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = model[0].compute_src_dst_node_temporal_embeddings(
                    src_node_ids=batch_neg_src_node_ids,
                    dst_node_ids=batch_neg_dst_node_ids,
                    node_interact_times=batch_node_interact_times,
                    edge_ids=None,
                    edges_are_positive=False,
                    num_neighbors=num_neighbors,
                )
                batch_src_node_embeddings, batch_dst_node_embeddings = model[0].compute_src_dst_node_temporal_embeddings(
                    src_node_ids=batch_src_node_ids,
                    dst_node_ids=batch_dst_node_ids,
                    node_interact_times=batch_node_interact_times,
                    edge_ids=batch_edge_ids,
                    edges_are_positive=True,
                    num_neighbors=num_neighbors,
                )
            elif model_name in ["GraphMixer"]:
                batch_src_node_embeddings, batch_dst_node_embeddings = model[0].compute_src_dst_node_temporal_embeddings(
                    src_node_ids=batch_src_node_ids,
                    dst_node_ids=batch_dst_node_ids,
                    node_interact_times=batch_node_interact_times,
                    edge_ids=batch_edge_ids,
                    num_neighbors=num_neighbors,
                    time_gap=time_gap,
                )
                batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = model[0].compute_src_dst_node_temporal_embeddings(
                    src_node_ids=batch_neg_src_node_ids,
                    dst_node_ids=batch_neg_dst_node_ids,
                    node_interact_times=batch_node_interact_times,
                    edge_ids=batch_edge_ids,
                    num_neighbors=num_neighbors,
                    time_gap=time_gap,
                )
            elif model_name in ["DyGFormer"]:
                batch_src_node_embeddings, batch_dst_node_embeddings = model[0].compute_src_dst_node_temporal_embeddings(
                    src_node_ids=batch_src_node_ids,
                    dst_node_ids=batch_dst_node_ids,
                    node_interact_times=batch_node_interact_times,
                    edge_ids=batch_edge_ids,
                )
                batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = model[0].compute_src_dst_node_temporal_embeddings(
                    src_node_ids=batch_neg_src_node_ids,
                    dst_node_ids=batch_neg_dst_node_ids,
                    node_interact_times=batch_node_interact_times,
                    edge_ids=batch_edge_ids,
                )
            else:
                raise ValueError(f"Wrong value for model_name {model_name}!")

            positive_probabilities = model[1](input_1=batch_src_node_embeddings, input_2=batch_dst_node_embeddings).squeeze(dim=-1).sigmoid()
            negative_probabilities = model[1](input_1=batch_neg_src_node_embeddings, input_2=batch_neg_dst_node_embeddings).squeeze(dim=-1).sigmoid()

            predicts = torch.cat([positive_probabilities, negative_probabilities], dim=0)
            labels = torch.cat([torch.ones_like(positive_probabilities), torch.zeros_like(negative_probabilities)], dim=0)

            loss = loss_func(input=predicts, target=labels)

            evaluate_losses.append(loss.item())
            evaluate_metrics.append(get_link_prediction_metrics(predicts=predicts, labels=labels))

            evaluate_idx_data_loader_tqdm.set_description(
                f"evaluate for the {batch_idx + 1}-th batch, evaluate loss: {loss.item()}"
            )

    return evaluate_losses, evaluate_metrics


def evaluate_model_link_prediction_ad(
    model_name: str,
    model: nn.Module,
    neighbor_sampler: NeighborSampler,
    evaluate_idx_data_loader: DataLoader,
    evaluate_data: Data,
    loss_func: nn.Module,
    edge_raw_features: np.ndarray,
    num_neighbors: int = 20,
    time_gap: int = 2000,
    args: argparse.Namespace = None,
):
    if model_name in [
        "DyRep",
        "TGAT",
        "TGN",
        "CAWN",
        "TCL",
        "GraphMixer",
        "DyGFormer",
        "SODA",
        "SODA_cat",
        "GeneralDyG",
    ]:
        model["backbone"][0].set_neighbor_sampler(neighbor_sampler)

    device = args.device if args is not None else "cpu"
    model.eval()

    batch_eval_preds, batch_eval_labels = [], []
    evaluate_losses = []
    evaluate_idx_data_loader_tqdm = tqdm(evaluate_idx_data_loader, ncols=120)

    with torch.no_grad():
        for batch_idx, evaluate_data_indices in enumerate(evaluate_idx_data_loader_tqdm):
            evaluate_data_indices = evaluate_data_indices.numpy()
            batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times, batch_edge_ids = (
                evaluate_data.src_node_ids[evaluate_data_indices],
                evaluate_data.dst_node_ids[evaluate_data_indices],
                evaluate_data.node_interact_times[evaluate_data_indices],
                evaluate_data.edge_ids[evaluate_data_indices],
            )
            batch_labels = evaluate_data.labels[evaluate_data_indices]
            batch_raw_labels = evaluate_data.raw_labels[evaluate_data_indices]

            if model_name in ["TGAT", "CAWN", "TCL", "SODA", "SODA_cat", "GeneralDyG"]:
                batch_src_node_embeddings, batch_dst_node_embeddings = model["backbone"][0].compute_src_dst_node_temporal_embeddings(
                    src_node_ids=batch_src_node_ids,
                    dst_node_ids=batch_dst_node_ids,
                    node_interact_times=batch_node_interact_times,
                    edge_ids=batch_edge_ids,
                    num_neighbors=num_neighbors,
                )
            elif model_name in ["JODIE", "DyRep", "TGN"]:
                batch_src_node_embeddings, batch_dst_node_embeddings = model["backbone"][0].compute_src_dst_node_temporal_embeddings(
                    src_node_ids=batch_src_node_ids,
                    dst_node_ids=batch_dst_node_ids,
                    node_interact_times=batch_node_interact_times,
                    edge_ids=batch_edge_ids,
                    edges_are_positive=True,
                    num_neighbors=num_neighbors,
                )
            elif model_name in ["GraphMixer"]:
                batch_src_node_embeddings, batch_dst_node_embeddings = model["backbone"][0].compute_src_dst_node_temporal_embeddings(
                    src_node_ids=batch_src_node_ids,
                    dst_node_ids=batch_dst_node_ids,
                    node_interact_times=batch_node_interact_times,
                    edge_ids=batch_edge_ids,
                    num_neighbors=num_neighbors,
                    time_gap=time_gap,
                )
            elif model_name in ["DyGFormer"]:
                batch_src_node_embeddings, batch_dst_node_embeddings = model["backbone"][0].compute_src_dst_node_temporal_embeddings(
                    src_node_ids=batch_src_node_ids,
                    dst_node_ids=batch_dst_node_ids,
                    node_interact_times=batch_node_interact_times,
                    edge_ids=batch_edge_ids,
                )
            else:
                raise ValueError(f"Wrong value for model_name {model_name}!")

            batch_edge_features = torch.from_numpy(edge_raw_features[batch_edge_ids]).float().to(device)
            processed_edge_features = model["edge_feature_mlp"](batch_edge_features)
            if model_name == "SODA":
                edge_embeddings = torch.cat([batch_src_node_embeddings, batch_dst_node_embeddings], dim=1)
            elif model_name == "SODA_cat":
                edge_embeddings = batch_src_node_embeddings - batch_dst_node_embeddings
            else:
                edge_embeddings = torch.cat(
                    [batch_src_node_embeddings, batch_dst_node_embeddings, processed_edge_features], dim=1
                )
            dimension = edge_embeddings.shape[1]

            labels = torch.tensor([batch_labels], dtype=torch.float, device=edge_embeddings.device).squeeze(dim=0)
            raw_labels = torch.tensor([batch_raw_labels], dtype=torch.float, device=edge_embeddings.device).squeeze(dim=0)

            if args is None or getattr(args, "flow", "False") == "False":
                predicts = model["backbone"][1](input=edge_embeddings).squeeze(dim=-1).sigmoid()
                loss = loss_func(input=predicts, target=labels)
            else:
                log_sigmoid = torch.nn.LogSigmoid()
                proj_edge_embeddings = model["backbone"][1](edge_embeddings)

                z, log_jac_det = model["backbone"][2](proj_edge_embeddings)
                logps = get_logp(dimension, z, log_jac_det)
                logps_norm = logps - torch.max(logps)

                b_n, b_a = get_logp_boundary(
                    logps_norm, labels.cpu().numpy(), pos_beta=args.pos_beta, margin_tau=args.margin_tau
                )

                loss_ml = torch.mean(-log_sigmoid(logps[labels == 0]))

                scores = torch.exp(logps_norm)
                predicts = 1 - scores

                loss_n_con, loss_a_con = calculate_log_barrier_bg_spp_loss(
                    logps, labels.cpu().numpy(), (b_n, b_a)
                )
                loss_occ = calculate_log_barrier_bi_occ_loss(
                    proj_edge_embeddings, labels, args.r_max, args.r_coff
                )
                loss = loss_ml + args.lamda1 * (loss_n_con + loss_a_con) + args.lamda2 * loss_occ

            batch_eval_preds.append(predicts)
            batch_eval_labels.append(raw_labels)
            evaluate_losses.append(loss.item())

            evaluate_idx_data_loader_tqdm.set_description(
                f"evaluate for the {batch_idx + 1}-th batch, evaluate loss: {loss.item()}"
            )

    return evaluate_losses, batch_eval_preds, batch_eval_labels
