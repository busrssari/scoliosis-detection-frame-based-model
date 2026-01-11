# Transfer Learning Guide for OpenGait-Finetune

This README documents the process of transfer learning using the ScoNet model on a custom dataset, including dataset preparation, base model details, and the fine-tuning configuration.

## 1. Dataset Creation

The dataset creation process involves converting raw video data into the PKL format required by OpenGait.

### Scripts
The main script for generating the dataset is located at:
- `datasets/our-dataset/create_pkl_file/create-pkl.py`

### Usage
To generate the dataset, run the python script (ensure you have the necessary dependencies installed, such as YOLOv8 if used for segmentation as implied by `yolov8m-seg.pt` in the same directory):

```bash
python datasets/our-dataset/create_pkl_file/create-pkl.py
```

### Dataset Structure
- **Output Directory:** The generated dataset is expected to be at `datasets/our-dataset/Scoliosis1K_PKL_Output_30fps`.
- **Partition File:** The train/test split is defined in `datasets/our-dataset/Scoliosis1K_Transfer_30fps.json`.

## 2. Base Model

We utilize a pre-trained **ScoNet** model as the base for transfer learning. This model was likely pre-trained on a larger or related dataset (e.g., Scoliosis1K).

- **Architecture:** ScoNet (Backbone: ResNet9)
- **Pre-trained Checkpoint:** `output/Scoliosis1K/ScoNet/ScoNet_Sensitive_Scratch/checkpoints/ScoNet_Sensitive_Scratch-20000.pt`

### Training the Base Model
If you need to train the base model from scratch:

```bash
python -m torch.distributed.launch --nproc_per_node=1 opengait/main.py --cfgs configs/sconet/sconet_scoliosis1k.yaml --phase train
```

## 3. Transfer Learning (Fine-Tuning)

The transfer learning process involves fine-tuning the pre-trained ScoNet model on the target dataset (`OurDataset_FineTune`).

### Configuration
The configuration file for this process is:
- **Config File:** `configs/sconet/sconet_finetune_weightedx2.yaml`

### Key Settings
- **Strategy:** Fine-tuning with manual class weights to handle imbalance (emphasizing the patient class).
- **Loss Function:** 
  - CrossEntropyLoss (Weight: 1.0, Scale: 16)
  - TripletLoss (Weight: 0.0 - Disabled for fine-tuning in this config)
  - **Class Weights:** [1.0, 2.0] (Normal=1.0, Patient=2.0)
- **Learning Rate:** 0.0001
- **Optimizer:** SGD
- **Restore:** Parameters from the base model checkpoint are restored.

## 4. How to Run Training

To start the transfer learning (training) process, use the following command:

```bash
python -m torch.distributed.launch --nproc_per_node=1 opengait/main.py --cfgs configs/sconet/sconet_finetune_weightedx2.yaml --phase train
```

## 5. Model Output

- **Checkpoints & Logs:** The training results will be saved to:
  `output/OurDataset_FineTune/ScoNet/Sconet_Finetune_Weighted1x2`

## 6. Testing / Evaluation

After training is complete (or to test a specific checkpoint), use the following command to run the evaluation phase. This will evaluate the model on the test set defined in the partition file.

```bash
python -m torch.distributed.launch --nproc_per_node=1 opengait/main.py --cfgs configs/sconet/test_configs/test_finetune_weight2.yaml --phase test
```

### Important Notes for Testing
- **Config:** `configs/sconet/test_configs/test_finetune_weight2.yaml`
- **Restore Hint:** By default, the config executes `restore_hint` logic. Ensure `restore_hint` in the config points to the checkpoint you want to test, or matches the logic for automatically finding the latest checkpoint if set to `0` or a specific iteration.
- **Evaluation Metric:** The model uses Euclidean distance (`euc`) for metric evaluation.
- **Test Dataset:** Matches `test_dataset_name` in the config, which points to `OurDataset_FineTune`.