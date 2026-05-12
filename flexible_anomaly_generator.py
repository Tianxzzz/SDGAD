

import os
import sys
import json
import argparse
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings


current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)



ANOMALY_ABBREVIATIONS = {
    'temporal': 't',
    'structural': 's', 
    'contextual': 'c',
    'temporal-structural-contextual': 'tsc',
    'mixed': 'm'
}

class FlexibleAnomalyGenerator:

    
    def __init__(self, dataset_name: str, val_ratio: float = 0.2, test_ratio: float = 0.4,
                 seed: int = 123, enable_contextual: bool = False):

        self.dataset_name = dataset_name
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed
        

        np.random.seed(seed)
        
        edge_features_path = f"./processed_data/{dataset_name}/ml_{dataset_name}.npy"
        self.has_edge_features = os.path.exists(edge_features_path)
        
        if enable_contextual and not self.has_edge_features:
            print(f"⚠️ Warning: Dataset {dataset_name} has no edge features, disabling contextual anomalies")
            self.enable_contextual = False
        else:
            self.enable_contextual = enable_contextual
            
        print(f"Edge features enabled: {self.enable_contextual}")
        
        self.df = None
        self.edge_features = None
        self.node_features = None
        self.contextual_anomaly_features = []
        
    def load_and_split_data(self):

        print(f"Loading and splitting {self.dataset_name} dataset...")
        

        csv_path = f"./processed_data/{self.dataset_name}/ml_{self.dataset_name}.csv"
        self.df = pd.read_csv(csv_path)
        
        if self.has_edge_features:
            edge_features_path = f"./processed_data/{self.dataset_name}/ml_{self.dataset_name}.npy"
            self.edge_features = np.load(edge_features_path)
            print(f"Loaded edge features: {self.edge_features.shape}")
            
  
        node_features_path = f"./processed_data/{self.dataset_name}/ml_{self.dataset_name}_node.npy"
        if os.path.exists(node_features_path):
            self.node_features = np.load(node_features_path)
            
        print(f"Dataset loaded: {len(self.df)} total edges")
        print(f"Node range: {self.df['u'].min()} to {self.df['u'].max()}")
        
    
        timestamps = self.df['ts'].values
        val_threshold = np.quantile(timestamps, 1 - self.val_ratio - self.test_ratio)
        test_threshold = np.quantile(timestamps, 1 - self.test_ratio)
        
        self.train_df = self.df[self.df['ts'] <= val_threshold].copy()
        self.val_df = self.df[(self.df['ts'] > val_threshold) & (self.df['ts'] <= test_threshold)].copy()
        self.test_df = self.df[self.df['ts'] > test_threshold].copy()
        
        print("Data splits:")
        print(f"  train: {len(self.train_df)} edges")
        print(f"  val: {len(self.val_df)} edges")
        print(f"  test: {len(self.test_df)} edges")
        
    def _compute_cosine_distance(self, vec1: np.ndarray, vec2: np.ndarray) -> np.ndarray:
   
        if vec1.ndim == 1:
            vec1 = vec1.reshape(1, -1)
        if vec2.ndim == 1:
            vec2 = vec2.reshape(1, -1)

        norm1 = np.linalg.norm(vec1, axis=1, keepdims=True)
        norm2 = np.linalg.norm(vec2, axis=1, keepdims=True)
        

        norm1 = np.where(norm1 == 0, 1e-8, norm1)
        norm2 = np.where(norm2 == 0, 1e-8, norm2)
        
        similarity = np.sum(vec1 * vec2, axis=1, keepdims=True) / (norm1 * norm2.T)
        
        similarity = np.clip(similarity, -1, 1)
        
        return 1 - similarity
    
    def _sample_contextually_inconsistent_attributes(self, reference_features: np.ndarray,
                                                   all_features: np.ndarray,
                                                   sample_size: int = 10) -> np.ndarray:

        if sample_size > len(all_features):
            sample_size = len(all_features)

        max_candidates = min(1000, len(all_features))
        candidate_indices = np.random.choice(len(all_features), max_candidates, replace=False)
        candidate_features = all_features[candidate_indices]
        
        distances = self._compute_cosine_distance(reference_features, candidate_features)
        
        if distances.ndim > 1 and distances.shape[1] > 1:
            distances = distances.flatten()[:max_candidates]
        else:
            distances = distances.flatten()
            
        top_indices = np.argsort(distances)[-sample_size:]
        selected_indices = candidate_indices[top_indices]
        
        return all_features[selected_indices]
    
    def generate_temporal_anomalies(self, edges_df: pd.DataFrame, num_anomalies: int, label_value: int = 1) -> pd.DataFrame:
        if num_anomalies == 0:
            return pd.DataFrame(columns=edges_df.columns)
            
        sampled_edges = edges_df.sample(n=min(num_anomalies, len(edges_df)))
        
        anomaly_data = sampled_edges.copy()
        
        all_timestamps = edges_df['ts'].values
        anomaly_data['ts'] = np.random.choice(all_timestamps, size=len(anomaly_data))
        
        anomaly_data['label'] = label_value
        
        return anomaly_data
    
    def generate_structural_anomalies(self, edges_df: pd.DataFrame, num_anomalies: int, label_value: int = 1) -> pd.DataFrame:
        if num_anomalies == 0:
            return pd.DataFrame(columns=edges_df.columns)
            
        all_nodes = set(edges_df['u'].unique()) | set(edges_df['i'].unique())
        all_nodes = list(all_nodes)
        
        existing_edges = set(zip(edges_df['u'], edges_df['i']))
        
        anomaly_data = []
        attempts = 0
        max_attempts = num_anomalies * 10
        
        while len(anomaly_data) < num_anomalies and attempts < max_attempts:
            u = np.random.choice(all_nodes)
            i = np.random.choice(all_nodes)
            
            if (u, i) not in existing_edges and u != i:
                ts = np.random.choice(edges_df['ts'].values)
                
                anomaly_data.append({
                    'u': u,
                    'i': i, 
                    'ts': ts,
                    'label': label_value,
                    'idx': len(edges_df) + len(anomaly_data)
                })
                
            attempts += 1
            
        return pd.DataFrame(anomaly_data)
    
    def generate_contextual_anomalies(self, edges_df: pd.DataFrame, num_anomalies: int, label_value: int = 1) -> pd.DataFrame:
        if not self.enable_contextual or self.edge_features is None or num_anomalies == 0:
            return pd.DataFrame(columns=edges_df.columns)
            
        anomaly_edges = edges_df.sample(n=min(num_anomalies, len(edges_df)))
        
        anomaly_data = []
        
        for i, (_, edge_row) in enumerate(anomaly_edges.iterrows()):
            edge_idx = edge_row['idx']
            if edge_idx < len(self.edge_features):
                original_features = self.edge_features[edge_idx].reshape(1, -1)
            else:
                original_features = np.random.randn(1, self.edge_features.shape[1])
            
            inconsistent_features = self._sample_contextually_inconsistent_attributes(
                original_features,
                self.edge_features,
                sample_size=1
            )
            
            anomaly_edge = edge_row.copy()
            anomaly_edge['label'] = label_value
            
            if len(inconsistent_features) > 0:
                self.contextual_anomaly_features.append(inconsistent_features[0])
            else:
                self.contextual_anomaly_features.append(np.random.randn(self.edge_features.shape[1]))
            
            anomaly_edge['anomaly_features'] = True
            anomaly_data.append(anomaly_edge)
            
        return pd.DataFrame(anomaly_data)
    
    def generate_mixed_anomalies(self, edges_df: pd.DataFrame, num_anomalies: int, label_value: int = 1) -> pd.DataFrame:
        if num_anomalies == 0:
            return pd.DataFrame(columns=edges_df.columns)

        nodes = list(set(edges_df['u']).union(edges_df['i']))
        existing_edges = set(zip(edges_df['u'], edges_df['i']))
        timestamps = edges_df['ts'].values

        anomaly_data = []
        attempts = 0
        max_attempts = num_anomalies * 10

        while len(anomaly_data) < num_anomalies and attempts < max_attempts:
            u = np.random.choice(nodes)
            i = np.random.choice(nodes)
            if u == i or (u, i) in existing_edges:
                attempts += 1
                continue

            ts = np.random.choice(timestamps)
            anomaly_data.append(
                {
                    "u": u,
                    "i": i,
                    "ts": ts,
                    "label": label_value,
                    "idx": len(edges_df) + len(anomaly_data),
                }
            )
            existing_edges.add((u, i))
            attempts += 1

        return pd.DataFrame(anomaly_data)
    
    def apply_anomaly_config(self, anomaly_config: Dict) -> pd.DataFrame:
        print("Applying anomaly configuration...")
        
        all_datasets = []
        
        for split_name, split_df in [('train', self.train_df), ('val', self.val_df), ('test', self.test_df)]:
            print(f"\nProcessing {split_name} set:")
            print(f"  Normal edges: {len(split_df)}")
            
            split_anomalies = anomaly_config.get(split_name, [])
            split_anomaly_data = []
            
            if not split_anomalies:
                print(f"  No anomalies configured for {split_name}")
            else:
                for anomaly_spec in split_anomalies:
                    anomaly_type = anomaly_spec['type']
                    anomaly_ratio = anomaly_spec['ratio']
                    label_value = anomaly_spec.get('label', 1)
                    num_anomalies = int(len(split_df) * anomaly_ratio)
                    
                    print(
                        f"  Generating {num_anomalies} {anomaly_type} anomalies "
                        f"(ratio: {anomaly_ratio}, label: {label_value})"
                    )
                    
                    if anomaly_type == 'temporal':
                        anomalies = self.generate_temporal_anomalies(
                            split_df, num_anomalies, label_value=label_value
                        )
                    elif anomaly_type == 'structural':
                        anomalies = self.generate_structural_anomalies(
                            split_df, num_anomalies, label_value=label_value
                        )
                    elif anomaly_type == 'contextual':
                        anomalies = self.generate_contextual_anomalies(
                            split_df, num_anomalies, label_value=label_value
                        )
                    elif anomaly_type == 'mixed':
                        temporal_ratio = anomaly_spec.get('temporal_ratio', 0.5)
                        anomalies = self.generate_mixed_anomalies(
                            split_df,
                            num_anomalies,
                            temporal_ratio=temporal_ratio,
                            label_value=label_value,
                        )
                    else:
                        print(f"  Unknown anomaly type: {anomaly_type}")
                        continue
                    
                    if len(anomalies) > 0:
                        split_anomaly_data.append(anomalies)
                        print(f"  Generated {len(anomalies)} {anomaly_type} anomalies")
            
            split_data = [split_df]
            if split_anomaly_data:
                split_data.extend(split_anomaly_data)
            
            combined_split = pd.concat(split_data, ignore_index=True)       
            combined_split = combined_split.sort_values('ts').reset_index(drop=True)
            print(f"  Total {split_name} edges after adding anomalies: {len(combined_split)}")
            
            all_datasets.append(combined_split)
        
        final_dataset = pd.concat(all_datasets, ignore_index=True)
        
        final_dataset['idx'] = range(len(final_dataset))
        
        normal_count = len(final_dataset[final_dataset['label'] == 0])
        anomaly_count = len(final_dataset[final_dataset['label'] != 0])
        label_counts = final_dataset['label'].value_counts().sort_index()
        
        print(f"Final dataset statistics:")
        print(f"  Total edges: {len(final_dataset)}")
        print(f"  Normal edges: {normal_count}")
        print(f"  Anomalous edges: {anomaly_count}")
        print(f"  Anomaly percentage: {anomaly_count/len(final_dataset)*100:.2f}%")
        print(f"  Label distribution:")
        for label_value, count in label_counts.items():
            label_name = "normal" if label_value == 0 else f"anomaly_{int(label_value)}"
            print(f"    {label_name} ({label_value}): {count}")
        
        return final_dataset
    
    def save_dataset(self, dataset: pd.DataFrame, output_name: str,
                    output_dir: str = "./processed_data_with_anomalies"):
        
        dataset_dir = os.path.join(output_dir, output_name)
        os.makedirs(dataset_dir, exist_ok=True)
        
        csv_path = os.path.join(dataset_dir, f"ml_{output_name}.csv")
        dataset.to_csv(csv_path, index=False)
        print(f"Saved dataset to: {csv_path}")
        
        if self.has_edge_features and self.edge_features is not None:
            new_size = len(dataset)
            original_size = len(self.edge_features)
            
            if new_size > original_size:
                additional_features = np.random.randn(new_size - original_size, self.edge_features.shape[1])
                new_features = np.vstack([self.edge_features, additional_features])
            else:
                new_features = self.edge_features[:new_size].copy()
            
            if hasattr(self, 'contextual_anomaly_features') and self.contextual_anomaly_features:
                anomaly_indices = dataset[dataset['label'] != 0].index.tolist()
                for i, anomaly_idx in enumerate(anomaly_indices):
                    if i < len(self.contextual_anomaly_features):
                        new_features[anomaly_idx] = self.contextual_anomaly_features[i]
                    else:
                        new_features[anomaly_idx] = np.random.randn(self.edge_features.shape[1])
            else:
                anomaly_indices = dataset[dataset['label'] != 0].index.tolist()
                new_features[anomaly_indices] = np.random.randn(len(anomaly_indices), self.edge_features.shape[1])
            
            edge_features = new_features
            print(f"Using original edge features: {self.edge_features.shape}")
        else:
            edge_features = np.random.randn(len(dataset), 100)
            print(f"Generated dummy edge features: {edge_features.shape}")
        
        if self.node_features is not None:
            node_features = self.node_features.copy()
            print(f"Using original node features: {self.node_features.shape}")
        else:
            max_node_id = max(dataset['u'].max(), dataset['i'].max())
            node_features = np.random.randn(max_node_id + 1, edge_features.shape[1])
            print(f"Generated dummy node features: {node_features.shape}")
        
        edge_features_path = os.path.join(dataset_dir, f"ml_{output_name}.npy")
        np.save(edge_features_path, edge_features)
        print(f"Saved edge features to: {edge_features_path}")
        
        node_features_path = os.path.join(dataset_dir, f"ml_{output_name}_node.npy")
        np.save(node_features_path, node_features)
        print(f"Saved node features to: {node_features_path}")

def create_example_config(filename: str = "anomaly_config_example.json"):

    config = {
        "dataset": "uci",
        "output_name": "uci_mixed_anomalies",
        "val_ratio": 0.2,
        "test_ratio": 0.4,
        "seed": 123,
        "enable_contextual": False,
        "anomalies": {
            "train": [],
            "val": [
                {
                    "type": "temporal",
                    "ratio": 0.05
                }
            ],
            "test": [
                {
                    "type": "structural", 
                    "ratio": 0.03
                },
                {
                    "type": "mixed",
                    "ratio": 0.02,
                    "temporal_ratio": 0.6
                }
            ]
        }
    }
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"Created example config file: {filename}")

def main():
    parser = argparse.ArgumentParser(description='Flexible anomaly data generator')
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--create-example', action='store_true', help='Create example configuration file')
    
    args = parser.parse_args()
    
    if args.create_example:
        create_example_config()
        return
    
    if not args.config:
        print("Please provide configuration file path or use --create-example to create example configuration")
        return
    
    with open(args.config, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    print("Flexible anomaly data generator")
    print("=" * 60)
    print(f"Configuration file: {args.config}")
    print(f"Dataset: {config['dataset']}")
    print(f"Output name: {config['output_name']}")
    print(f"Random seed: {config['seed']}")
    
    generator = FlexibleAnomalyGenerator(
        dataset_name=config['dataset'],
        val_ratio=config.get('val_ratio', 0.2),
        test_ratio=config.get('test_ratio', 0.4),
        seed=config.get('seed', 123),
        enable_contextual=config.get('enable_contextual', False)
    )
    
    generator.load_and_split_data()
    
    final_dataset = generator.apply_anomaly_config(config['anomalies'])
    
    generator.save_dataset(final_dataset, config['output_name'])
    
    print(f"\nSuccessfully generated dataset: {config['output_name']}")
    print(f"Output directory: ./processed_data_with_anomalies/{config['output_name']}")
    
    print(f"\nUsage:")
    print(f"1. Copy dataset:")
    print(f"   cp -r ./processed_data_with_anomalies/{config['output_name']} processed_data/")
    print(f"2. Train model:")
    print(f"   python train_link_AD.py --dataset_name {config['output_name']} --model_name TGAT")

if __name__ == "__main__":
    main()
