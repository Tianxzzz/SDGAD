import logging
import time
import sys
import os
from tqdm import tqdm
import numpy as np
import warnings
import shutil
import json
import torch
import torch.nn as nn

from models.TGAT import TGAT
from models.MemoryModel import MemoryModel, compute_src_dst_node_time_shifts
from models.CAWN import CAWN
from models.TCL import TCL
from models.SODA import SODA
from models.SODA_cat import SODA_cat
from models.GraphMixer import GraphMixer
from models.DyGFormer import DyGFormer
from models.GeneralDyG import GeneralDyG
from models.modules import ADLayer, flow_model
from utils.loss import (
    get_logp,
    get_logp_a,
    get_logp_boundary,
    calculate_orthogonal_regularizer,
    calculate_log_barrier_bg_spp_loss,
    calculate_bg_spp_loss,
    calculate_bi_occ_loss,
    calculate_log_barrier_bi_occ_loss,
    calculate_log_barrier_occ_loss,
)
from utils.utils import set_random_seed, convert_to_gpu, get_parameter_sizes, create_optimizer
from utils.utils import get_neighbor_sampler, NegativeEdgeSampler
from evaluate_models_utils import evaluate_model_link_prediction_ad as evaluate_model_link_prediction
from utils.metrics import get_anomaly_detection_metrics
from utils.DataLoader import get_idx_data_loader, get_link_prediction_data, Data
from utils.EarlyStopping import EarlyStopping
from utils.load_configs import get_anomaly_detection_args


class EdgeFeatureMLP(torch.nn.Module):
    def __init__(self, edge_feat_dim: int, node_feat_dim: int):
        super(EdgeFeatureMLP, self).__init__()
        self.mlp = torch.nn.Linear(edge_feat_dim, node_feat_dim)

    def forward(self, edge_features):
        return self.mlp(edge_features)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    args = get_anomaly_detection_args(is_evaluation=False)

    node_raw_features, edge_raw_features, full_data, train_data, val_data, test_data = get_link_prediction_data(
        dataset_name=args.dataset_name, val_ratio=args.val_ratio, test_ratio=args.test_ratio
    )

    if hasattr(args, "few_shot_k") and args.few_shot_k is not None:
        k = int(args.few_shot_k)
        anom_idx = np.where(train_data.labels == 1)[0]
        normal_idx = np.where(train_data.labels == 0)[0]
        if anom_idx.size <= k:
            selected_anom_idx = anom_idx
        else:
            np.random.seed(42)
            selected_anom_idx = np.random.choice(anom_idx, size=k, replace=False)
        selected_idx = np.concatenate([normal_idx, selected_anom_idx])
        selected_idx = np.sort(selected_idx)
        train_data = Data(
            src_node_ids=train_data.src_node_ids[selected_idx],
            dst_node_ids=train_data.dst_node_ids[selected_idx],
            node_interact_times=train_data.node_interact_times[selected_idx],
            edge_ids=train_data.edge_ids[selected_idx],
            labels=train_data.labels[selected_idx],
        )
        print(f"Few-shot setting: kept {selected_anom_idx.size} anomalies", flush=True)

    train_neighbor_sampler = get_neighbor_sampler(
        data=train_data, sample_neighbor_strategy=args.sample_neighbor_strategy, time_scaling_factor=args.time_scaling_factor, seed=0
    )
    full_neighbor_sampler = get_neighbor_sampler(
        data=full_data, sample_neighbor_strategy=args.sample_neighbor_strategy, time_scaling_factor=args.time_scaling_factor, seed=1
    )

    train_idx_data_loader = get_idx_data_loader(
        indices_list=list(range(len(train_data.src_node_ids))), batch_size=args.batch_size, shuffle=False
    )
    val_idx_data_loader = get_idx_data_loader(
        indices_list=list(range(len(val_data.src_node_ids))), batch_size=args.batch_size, shuffle=False
    )
    test_idx_data_loader = get_idx_data_loader(
        indices_list=list(range(len(test_data.src_node_ids))), batch_size=args.batch_size, shuffle=False
    )

    val_metric_all_runs, test_metric_all_runs = [], []

    for run in range(args.num_runs):
        set_random_seed(seed=run)
        args.seed = run
        if args.few_shot_k is not None:
            args.save_model_name = f"{args.model_name}_flow_{args.flow}_fewshot_{args.few_shot_k}_seed{args.seed}"
        else:
            args.save_model_name = f"{args.model_name}_flow_{args.flow}_seed{args.seed}"

        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        os.makedirs(f"./logs/{args.model_name}/{args.dataset_name}/{args.save_model_name}/", exist_ok=True)
        fh = logging.FileHandler(
            f"./logs/{args.model_name}/{args.dataset_name}/{args.save_model_name}/{str(time.time())}.log"
        )
        fh.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setLevel(logging.WARNING)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        logger.addHandler(fh)
        logger.addHandler(ch)

        run_start_time = time.time()
        logger.info(f"********** Run {run + 1} starts. **********")
        logger.info(f"configuration is {args}")

        if args.model_name == "TGAT":
            dynamic_backbone = TGAT(
                node_raw_features=node_raw_features,
                edge_raw_features=edge_raw_features,
                neighbor_sampler=train_neighbor_sampler,
                time_feat_dim=args.time_feat_dim,
                num_layers=args.num_layers,
                num_heads=args.num_heads,
                dropout=args.dropout,
                device=args.device,
            )
        elif args.model_name in ["JODIE", "DyRep", "TGN"]:
            src_node_mean_time_shift, src_node_std_time_shift, dst_node_mean_time_shift_dst, dst_node_std_time_shift = compute_src_dst_node_time_shifts(
                train_data.src_node_ids, train_data.dst_node_ids, train_data.node_interact_times
            )
            dynamic_backbone = MemoryModel(
                node_raw_features=node_raw_features,
                edge_raw_features=edge_raw_features,
                neighbor_sampler=train_neighbor_sampler,
                time_feat_dim=args.time_feat_dim,
                model_name=args.model_name,
                num_layers=args.num_layers,
                num_heads=args.num_heads,
                dropout=args.dropout,
                src_node_mean_time_shift=src_node_mean_time_shift,
                src_node_std_time_shift=src_node_std_time_shift,
                dst_node_mean_time_shift_dst=dst_node_mean_time_shift_dst,
                dst_node_std_time_shift=dst_node_std_time_shift,
                device=args.device,
            )
        elif args.model_name == "CAWN":
            dynamic_backbone = CAWN(
                node_raw_features=node_raw_features,
                edge_raw_features=edge_raw_features,
                neighbor_sampler=train_neighbor_sampler,
                time_feat_dim=args.time_feat_dim,
                position_feat_dim=args.position_feat_dim,
                walk_length=args.walk_length,
                num_walk_heads=args.num_walk_heads,
                dropout=args.dropout,
                device=args.device,
            )
        elif args.model_name == "TCL":
            dynamic_backbone = TCL(
                node_raw_features=node_raw_features,
                edge_raw_features=edge_raw_features,
                neighbor_sampler=train_neighbor_sampler,
                time_feat_dim=args.time_feat_dim,
                num_layers=args.num_layers,
                num_heads=args.num_heads,
                num_depths=args.num_neighbors + 1,
                dropout=args.dropout,
                device=args.device,
            )
        elif args.model_name == "SODA":
            dynamic_backbone = SODA(
                node_raw_features=node_raw_features,
                edge_raw_features=edge_raw_features,
                neighbor_sampler=train_neighbor_sampler,
                time_feat_dim=args.time_feat_dim,
                num_layers=args.num_layers,
                num_heads=args.num_heads,
                num_depths=args.num_neighbors + 1,
                dropout=args.dropout,
                device=args.device,
            )
        elif args.model_name == "SODA_cat":
            dynamic_backbone = SODA_cat(
                node_raw_features=node_raw_features,
                edge_raw_features=edge_raw_features,
                neighbor_sampler=train_neighbor_sampler,
                time_feat_dim=args.time_feat_dim,
                num_layers=args.num_layers,
                num_heads=args.num_heads,
                num_depths=args.num_neighbors + 1,
                dropout=args.dropout,
                device=args.device,
            )
        elif args.model_name == "GraphMixer":
            dynamic_backbone = GraphMixer(
                node_raw_features=node_raw_features,
                edge_raw_features=edge_raw_features,
                neighbor_sampler=train_neighbor_sampler,
                time_feat_dim=args.time_feat_dim,
                num_tokens=args.num_neighbors,
                num_layers=args.num_layers,
                dropout=args.dropout,
                device=args.device,
            )
        elif args.model_name == "DyGFormer":
            dynamic_backbone = DyGFormer(
                node_raw_features=node_raw_features,
                edge_raw_features=edge_raw_features,
                neighbor_sampler=train_neighbor_sampler,
                time_feat_dim=args.time_feat_dim,
                channel_embedding_dim=args.channel_embedding_dim,
                patch_size=args.patch_size,
                num_layers=args.num_layers,
                num_heads=args.num_heads,
                dropout=args.dropout,
                max_input_sequence_length=args.max_input_sequence_length,
                device=args.device,
            )
        elif args.model_name == "GeneralDyG":
            dynamic_backbone = GeneralDyG(
                node_raw_features=node_raw_features,
                edge_raw_features=edge_raw_features,
                neighbor_sampler=train_neighbor_sampler,
                time_feat_dim=args.time_feat_dim,
                num_layers=args.num_layers,
                num_heads=args.num_heads,
                num_neighbors=args.num_neighbors,
                dropout=args.dropout,
                device=args.device,
            )
        else:
            raise ValueError(f"Wrong value for model_name {args.model_name}!")

        edge_feature_mlp = EdgeFeatureMLP(
            edge_feat_dim=edge_raw_features.shape[1], node_feat_dim=node_raw_features.shape[1]
        ).to(args.device)

        final_embedding_dim = 2 * node_raw_features.shape[1] if args.model_name in ["SODA", "SODA_cat"] else 3 * node_raw_features.shape[1]

        if args.flow == "False":
            detector = ADLayer(input_dim=final_embedding_dim, hidden_dim=node_raw_features.shape[1], output_dim=1)
            model = nn.ModuleDict(
                {
                    "backbone": nn.Sequential(dynamic_backbone, detector),
                    "edge_feature_mlp": edge_feature_mlp,
                }
            )
        else:
            projection_layer = nn.Sequential(nn.Linear(final_embedding_dim, final_embedding_dim))
            detector = flow_model(in_channels=final_embedding_dim)
            model = nn.ModuleDict(
                {
                    "backbone": nn.Sequential(dynamic_backbone, projection_layer, detector),
                    "edge_feature_mlp": edge_feature_mlp,
                }
            )

        logger.info(f"model -> {model}")
        logger.info(
            f"model name: {args.model_name}, #parameters: {get_parameter_sizes(model) * 4} B, "
            f"{get_parameter_sizes(model) * 4 / 1024} KB, {get_parameter_sizes(model) * 4 / 1024 / 1024} MB."
        )
        optimizer = create_optimizer(model=model, optimizer_name=args.optimizer, learning_rate=args.learning_rate, weight_decay=args.weight_decay)

        model = convert_to_gpu(model, device=args.device)
        save_model_folder = f"./saved_models/{args.model_name}/{args.dataset_name}/{args.save_model_name}/"
        shutil.rmtree(save_model_folder, ignore_errors=True)
        os.makedirs(save_model_folder, exist_ok=True)

        early_stopping = EarlyStopping(
            patience=args.patience,
            save_model_folder=save_model_folder,
            save_model_name=args.save_model_name,
            logger=logger,
            model_name=args.model_name,
        )

        loss_func = nn.BCELoss()

        for epoch in range(args.num_epochs):
            model.train()
            if args.model_name in [
                "JODIE",
                "DyRep",
                "TGAT",
                "TGN",
                "CAWN",
                "TCL",
                "SODA",
                "SODA_cat",
                "GraphMixer",
                "DyGFormer",
                "GeneralDyG",
            ]:
                model["backbone"][0].set_neighbor_sampler(train_neighbor_sampler)
            if args.model_name in ["JODIE", "DyRep", "TGN"]:
                model["backbone"][0].memory_bank.__init_memory_bank__()

            train_losses, train_metrics = [], []
            batch_train_preds, batch_train_labels = [], []
            train_idx_data_loader_tqdm = tqdm(train_idx_data_loader, ncols=120)
            for batch_idx, train_data_indices in enumerate(train_idx_data_loader_tqdm):
                train_data_indices = train_data_indices.numpy()
                batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times, batch_edge_ids, batch_labels = (
                    train_data.src_node_ids[train_data_indices],
                    train_data.dst_node_ids[train_data_indices],
                    train_data.node_interact_times[train_data_indices],
                    train_data.edge_ids[train_data_indices],
                    train_data.labels[train_data_indices],
                )
                if args.model_name in ["TGAT", "CAWN", "TCL", "SODA", "SODA_cat", "GeneralDyG"]:
                    embedding_kwargs = {
                        "src_node_ids": batch_src_node_ids,
                        "dst_node_ids": batch_dst_node_ids,
                        "node_interact_times": batch_node_interact_times,
                        "edge_ids": batch_edge_ids,
                        "num_neighbors": args.num_neighbors,
                    }
                    batch_src_node_embeddings, batch_dst_node_embeddings = model["backbone"][0].compute_src_dst_node_temporal_embeddings(
                        **embedding_kwargs
                    )
                elif args.model_name in ["JODIE", "DyRep", "TGN"]:
                    batch_src_node_embeddings, batch_dst_node_embeddings = model["backbone"][0].compute_src_dst_node_temporal_embeddings(
                        src_node_ids=batch_src_node_ids,
                        dst_node_ids=batch_dst_node_ids,
                        node_interact_times=batch_node_interact_times,
                        edge_ids=batch_edge_ids,
                        edges_are_positive=True,
                        num_neighbors=args.num_neighbors,
                    )
                elif args.model_name == "GraphMixer":
                    batch_src_node_embeddings, batch_dst_node_embeddings = model["backbone"][0].compute_src_dst_node_temporal_embeddings(
                        src_node_ids=batch_src_node_ids,
                        dst_node_ids=batch_dst_node_ids,
                        node_interact_times=batch_node_interact_times,
                        edge_ids=batch_edge_ids,
                        num_neighbors=args.num_neighbors,
                        time_gap=args.time_gap,
                    )
                elif args.model_name == "DyGFormer":
                    batch_src_node_embeddings, batch_dst_node_embeddings = model["backbone"][0].compute_src_dst_node_temporal_embeddings(
                        src_node_ids=batch_src_node_ids,
                        dst_node_ids=batch_dst_node_ids,
                        node_interact_times=batch_node_interact_times,
                        edge_ids=batch_edge_ids,
                    )
                else:
                    raise ValueError(f"Wrong value for model_name {args.model_name}!")

                batch_edge_features = torch.from_numpy(edge_raw_features[batch_edge_ids]).float().to(args.device)
                current_edge_features = model["edge_feature_mlp"](batch_edge_features)
                if args.model_name not in ["SODA", "SODA_cat"]:
                    edge_embeddings = torch.cat(
                        [batch_src_node_embeddings, batch_dst_node_embeddings, current_edge_features], dim=1
                    )
                elif args.model_name == "SODA":
                    edge_embeddings = torch.cat([batch_src_node_embeddings, batch_dst_node_embeddings], dim=1)
                else:
                    edge_embeddings = batch_src_node_embeddings - batch_dst_node_embeddings
                dimension = edge_embeddings.shape[1]

                labels = torch.tensor([batch_labels], dtype=torch.float, device=edge_embeddings.device).squeeze(dim=0)

                if args.flow == "False":
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

                train_losses.append(loss.item())
                batch_train_preds.append(predicts)
                batch_train_labels.append(labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_idx_data_loader_tqdm.set_description(
                    f"Epoch: {epoch + 1}, train for the {batch_idx + 1}-th batch, train loss: {loss.item()}"
                )
                if args.model_name in ["JODIE", "DyRep", "TGN"]:
                    model["backbone"][0].memory_bank.detach_memory_bank()

            train_preds = torch.cat(batch_train_preds, dim=0)
            train_labels = torch.cat(batch_train_labels, dim=0)
            train_metrics_epoch = get_anomaly_detection_metrics(predicts=train_preds, labels=train_labels)

            val_memory_backup = None
            if args.model_name in ["JODIE", "DyRep", "TGN"]:
                val_memory_backup = model["backbone"][0].memory_bank.backup_memory_bank()

            val_losses, val_preds, val_labels = evaluate_model_link_prediction(
                model_name=args.model_name,
                model=model,
                neighbor_sampler=full_neighbor_sampler,
                evaluate_idx_data_loader=val_idx_data_loader,
                evaluate_data=val_data,
                loss_func=loss_func,
                edge_raw_features=edge_raw_features,
                num_neighbors=args.num_neighbors,
                time_gap=args.time_gap,
                args=args,
            )
            val_preds_cat = torch.cat(val_preds, dim=0)
            val_labels_cat = torch.cat(val_labels, dim=0)
            val_metrics_epoch = get_anomaly_detection_metrics(
                predicts=val_preds_cat, labels=(val_labels_cat != 0).float()
            )

            if args.model_name in ["JODIE", "DyRep", "TGN"]:
                model["backbone"][0].memory_bank.reload_memory_bank(val_memory_backup)

            logger.info(
                f"Epoch: {epoch + 1}, learning rate: {optimizer.param_groups[0]['lr']}, train loss: {np.mean(train_losses):.4f}"
            )
            for metric_name, metric_val in train_metrics_epoch.items():
                logger.info(f"train {metric_name}, {metric_val:.4f}")
            logger.info(f"validate loss: {np.mean(val_losses):.4f}")
            for metric_name, metric_val in val_metrics_epoch.items():
                logger.info(f"validate {metric_name}, {metric_val:.4f}")

            if (epoch + 1) % args.test_interval_epochs == 0:
                test_memory_backup = None
                if args.model_name in ["JODIE", "DyRep", "TGN"]:
                    test_memory_backup = model["backbone"][0].memory_bank.backup_memory_bank()

                test_losses, test_preds, test_labels = evaluate_model_link_prediction(
                    model_name=args.model_name,
                    model=model,
                    neighbor_sampler=full_neighbor_sampler,
                    evaluate_idx_data_loader=test_idx_data_loader,
                    evaluate_data=test_data,
                    loss_func=loss_func,
                    edge_raw_features=edge_raw_features,
                    num_neighbors=args.num_neighbors,
                    time_gap=args.time_gap,
                    args=args,
                )
                test_preds_cat = torch.cat(test_preds, dim=0)
                test_labels_cat = torch.cat(test_labels, dim=0)
                test_metrics_epoch = get_anomaly_detection_metrics(
                    predicts=test_preds_cat, labels=(test_labels_cat != 0).float()
                )

                if args.model_name in ["JODIE", "DyRep", "TGN"]:
                    model["backbone"][0].memory_bank.reload_memory_bank(test_memory_backup)

                logger.info(f"test loss: {np.mean(test_losses):.4f}")
                for metric_name, metric_val in test_metrics_epoch.items():
                    logger.info(f"test {metric_name}, {metric_val:.4f}")

            val_metric_indicator = [(metric_name, metric_val, True) for metric_name, metric_val in val_metrics_epoch.items()]
            early_stop = early_stopping.step(val_metric_indicator, model)
            if early_stop:
                break

        early_stopping.load_checkpoint(model)
        logger.info(f"get final performance on dataset {args.dataset_name}...")

        if args.model_name not in ["JODIE", "DyRep", "TGN"]:
            val_memory_backup = None
            if args.model_name in ["JODIE", "DyRep", "TGN"]:
                val_memory_backup = model["backbone"][0].memory_bank.backup_memory_bank()

            val_losses, val_preds, val_labels = evaluate_model_link_prediction(
                model_name=args.model_name,
                model=model,
                neighbor_sampler=full_neighbor_sampler,
                evaluate_idx_data_loader=val_idx_data_loader,
                evaluate_data=val_data,
                loss_func=loss_func,
                edge_raw_features=edge_raw_features,
                num_neighbors=args.num_neighbors,
                time_gap=args.time_gap,
                args=args,
            )
            val_preds_cat = torch.cat(val_preds, dim=0)
            val_labels_cat = torch.cat(val_labels, dim=0)
            val_metrics_epoch = get_anomaly_detection_metrics(
                predicts=val_preds_cat, labels=(val_labels_cat != 0).float()
            )

            if args.model_name in ["JODIE", "DyRep", "TGN"]:
                model["backbone"][0].memory_bank.reload_memory_bank(val_memory_backup)

        test_memory_backup = None
        if args.model_name in ["JODIE", "DyRep", "TGN"]:
            test_memory_backup = model["backbone"][0].memory_bank.backup_memory_bank()

        test_losses, test_preds, test_labels = evaluate_model_link_prediction(
            model_name=args.model_name,
            model=model,
            neighbor_sampler=full_neighbor_sampler,
            evaluate_idx_data_loader=test_idx_data_loader,
            evaluate_data=test_data,
            loss_func=loss_func,
            edge_raw_features=edge_raw_features,
            num_neighbors=args.num_neighbors,
            time_gap=args.time_gap,
            args=args,
        )
        test_preds_cat = torch.cat(test_preds, dim=0)
        test_labels_cat = torch.cat(test_labels, dim=0)
        test_metrics_epoch = get_anomaly_detection_metrics(
            predicts=test_preds_cat, labels=(test_labels_cat != 0).float()
        )

        if args.model_name in ["JODIE", "DyRep", "TGN"]:
            model["backbone"][0].memory_bank.reload_memory_bank(test_memory_backup)

        val_metric_dict, test_metric_dict = {}, {}
        if args.model_name not in ["JODIE", "DyRep", "TGN"]:
            logger.info(f"validate loss: {np.mean(val_losses):.4f}")
            for metric_name, metric_val in val_metrics_epoch.items():
                logger.info(f"validate {metric_name}, {metric_val:.4f}")
                val_metric_dict[metric_name] = metric_val

        logger.info(f"test loss: {np.mean(test_losses):.4f}")
        for metric_name, metric_val in test_metrics_epoch.items():
            logger.info(f"test {metric_name}, {metric_val:.4f}")
            test_metric_dict[metric_name] = metric_val

        single_run_time = time.time() - run_start_time
        logger.info(f"Run {run + 1} cost {single_run_time:.2f} seconds.")

        if args.model_name not in ["JODIE", "DyRep", "TGN"]:
            val_metric_all_runs.append(val_metric_dict)
        test_metric_all_runs.append(test_metric_dict)

        if run < args.num_runs - 1:
            logger.removeHandler(fh)
            logger.removeHandler(ch)

        if args.model_name not in ["JODIE", "DyRep", "TGN"]:
            result_json = {
                "validate metrics": {metric_name: f"{val_metric_dict[metric_name]:.4f}" for metric_name in val_metric_dict},
                "test metrics": {metric_name: f"{test_metric_dict[metric_name]:.4f}" for metric_name in test_metric_dict},
            }
        else:
            result_json = {
                "test metrics": {metric_name: f"{test_metric_dict[metric_name]:.4f}" for metric_name in test_metric_dict},
            }
        result_json = json.dumps(result_json, indent=4)

        save_result_folder = f"./saved_results/{args.model_name}/{args.dataset_name}"
        os.makedirs(save_result_folder, exist_ok=True)
        save_result_path = os.path.join(save_result_folder, f"{args.save_model_name}.json")
        with open(save_result_path, "w") as file:
            file.write(result_json)
        try:
            save_scores_path = os.path.join(save_result_folder, f"{args.save_model_name}_test_scores.npz")
            np.savez_compressed(
                save_scores_path,
                scores=test_preds_cat.detach().cpu().numpy(),
                labels=test_labels_cat.detach().cpu().numpy(),
                timestamps=test_data.node_interact_times,
                model_name=args.model_name,
                dataset_name=args.dataset_name,
                flow=args.flow,
                few_shot_k=(-1 if args.few_shot_k is None else int(args.few_shot_k)),
                seed=int(args.seed),
            )
            logger.info(f"Saved test scores for plotting at {save_scores_path}")
        except Exception as e:
            logger.warning(f"Failed to save test scores for plotting: {e}")

    logger.info(f"metrics over {args.num_runs} runs:")
    if val_metric_all_runs:
        for metric_name in val_metric_all_runs[0].keys():
            values = [m[metric_name] for m in val_metric_all_runs]
            logger.info(f"validate {metric_name}, {values}")
            logger.info(
                f"average validate {metric_name}, {np.mean(values):.4f} ± {np.std(values, ddof=1):.4f}"
            )
    for metric_name in test_metric_all_runs[0].keys():
        values = [m[metric_name] for m in test_metric_all_runs]
        logger.info(f"test {metric_name}, {values}")
        logger.info(
            f"average test {metric_name}, {np.mean(values):.4f} ± {np.std(values, ddof=1):.4f}"
        )

    sys.exit()
