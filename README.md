# SDGAD
Code for ICML2026 paper “Learning Discriminative and Generalizable Anomaly Detector for Dynamic Graph with Limited Supervision”

# Datasets and Preprocessing
Most of the used original datasets can be downloaded [**here**](https://zenodo.org/records/7213796#.Y1cO6y8r30o). Please download them and put them in DG_data folder.

We can run preprocess_data/preprocess_data.py for pre-processing the datasets. For example, to preprocess the Wikipedia dataset, we can run the following commands:
```bash
cd preprocess_data/
python preprocess_data.py  --dataset_name wikipedia
```


