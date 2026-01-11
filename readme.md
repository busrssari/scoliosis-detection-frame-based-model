# 🏥 Scoliosis Detection from Gait Videos

This project provides a complete deep learning pipeline for detecting scoliosis from walking (gait) videos. It utilizes **YOLOv8** for silhouette extraction and a **ResNet18 + GRU** architecture for spatiotemporal classification, enhanced by an **Ensemble Voting** strategy.

## 🌟 Key Features
- **Robust Preprocessing**: Automatic subject tracking, segmentation, and alignment using YOLOv8-Seg.
- **Hybrid Architecture**: ResNet18 (Spatial Features) + Bi-Directional GRU (Temporal Features).
- **Ensemble Learning**: Uses **5-Seed Soft Voting** to achieve high reliability and mitigate single-model variance.
- **Medical Metrics**: Optimized for high **Recall (Sensitivity)** to minimize missed diagnoses.

## 📂 Project Structure

```
scoliosis-detection/
├── models/
│   ├── restnet.py            # Main Training & Evaluation Script (Ensemble)
│   └── ...
├── prepare-dataset/
│   ├── prepare_dataset-files.py  # Dataset Generator (Video -> PKL)
│   └── splits.json               # Train/Val/Test Split Definitions
├── checkpoints/
│   └── resnet_pro_best/      # Saved Model Checkpoints (Auto-generated)
└── README.md
```

## 🚀 Getting Started

### 1. Prerequisites
Ensure you have the necessary Python packages installed (PyTorch, Ultralytics, OpenCV, sklearn, etc.).
```bash
conda activate opengait
```

### 2. Dataset Preparation
The first step is to process raw videos into normalized silhouette sequences.

1.  **Define Splits**: Update `prepare-dataset/splits.json` with your video paths.
2.  **Generate Dataset**:
    ```bash
    python prepare-dataset/prepare_dataset-files.py
    ```
    *   **Output**: Generates `dataset_unified_64.pkl` containing the processed tensors (64x64 resolution).

### 3. Training & Evaluation
We use an **Ensemble approach** to train the model. This script trains 5 separate instances (Seeds 42-46) and then evaluates their combined performance.

*   **Run Training**:
    ```bash
    python models/restnet.py
    ```

*   **Process**:
    1.  **Training**: Trains 5 models using Focal Loss and Partial Freezing.
    2.  **Saving**: Saves the best checkpoint for each seed to `checkpoints/resnet_pro_best/`.
    3.  **Evaluation**: Automatically runs **Soft Voting Ensemble** on the Test Set.

## 📊 Results & Methodology

### Why Ensemble?
During our experiments, we observed a "Validation-Test Mismatch" where the best single model on the Validation set (Seed 42) underperformed on the Test set, while another seed (Seed 46) performed better. 
To resolve this and ensure scientific reliability, we use **Soft Voting Ensemble** (averaging probabilities of 5 models). This method proved to be:
*   **More Robust**: Accuracy ~83.18%
*   **High Sensitivity**: Recall 0.881

### Performance Metrics
| Metric | Score |
|:---|:---|
| **Accuracy** | **83.18%** |
| **Recall** | **0.881** |
| **Precision** | 0.825 |
| **F1 Score** | 0.852 |

## 🛠️ Configuration
*   **Dataset Path**: Ensure `CONFIG['dataset_path']` in `models/restnet.py` matches your generated `.pkl` file location.
*   **Hyperparameters**: You can adjust `batch_size`, `learning_rate`, (`0.0005`), and `epochs` in the `models/restnet.py` file.
