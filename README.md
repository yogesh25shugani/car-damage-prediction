# 🚗 Car Damage Prediction | Deep Learning

An AI-powered vehicle damage classification application built using **PyTorch, ResNet50, Transfer Learning, and Streamlit**.

The application analyzes vehicle images and classifies them into six damage categories based on vehicle orientation and damage condition.

<p align="center">
  <a href="https://car-damage-predicti0n.streamlit.app/"><strong>🚀 Live Demo</strong></a> |
  <a href="https://github.com/yogesh25shugani/car-damage-prediction"><strong>GitHub Repository</strong></a>
</p>

---

## 1. Project Overview

Manual vehicle damage inspection can be time-consuming and subjective. This project explores how deep learning can assist vehicle inspection by automatically classifying uploaded vehicle images.

The solution uses a pretrained ResNet50 model with transfer learning to identify six vehicle damage categories.

### Key Features

- Upload vehicle images through an interactive Streamlit interface.
- Classify images into six damage categories.
- Display prediction confidence.
- Visualize probability distribution across all categories.
- Validate uploaded image formats, file sizes, and dimensions.
- Load and cache the trained model for inference.
- Deploy the application using Streamlit Community Cloud.

## 2. Damage Categories

The model supports six classification categories:

| Category | Description |
|---|---|
| F_Breakage | Front-side breakage |
| F_Crushed | Front-side crushed damage |
| F_Normal | Normal front-side condition |
| R_Breakage | Rear-side breakage |
| R_Crushed | Rear-side crushed damage |
| R_Normal | Normal rear-side condition |

## 3. Dataset Overview

The dataset contains 2,300 vehicle images distributed across six classes.

| Class | Images |
|---|---:|
| F_Breakage | 500 |
| F_Crushed | 400 |
| F_Normal | 500 |
| R_Breakage | 300 |
| R_Crushed | 300 |
| R_Normal | 300 |
| **Total** | **2,300** |

### Dataset Split

A stratified dataset split was used to preserve class proportions.

| Dataset | Images | Percentage |
|---|---:|---:|
| Training | 1,610 | 70% |
| Validation | 345 | 15% |
| Testing | 345 | 15% |
| **Total** | **2,300** | **100%** |

The dataset is not included in the public repository.

## 4. Model Architecture

**ResNet50 with Transfer Learning**

The project uses ResNet50 pretrained on ImageNet.

The original classification layer was replaced with a custom classification head:

- Dropout: 0.3
- Fully connected layer: 2,048 input features to 6 output classes.
- Layer 1, Layer 2, and Layer 3: Frozen in the selected experiment.
- Layer 4: Fine-tuned.
- Classification loss: Cross-Entropy Loss.
- Optimizer: Adam.

### Image Preprocessing

- Resize images to 224 × 224.
- Convert images to RGB.
- Convert images to tensors.
- Normalize using ImageNet mean and standard deviation.

Training augmentation includes horizontal flipping, rotation, and color jitter.

## 5. Experimentation and Model Selection

Multiple experiments were conducted to evaluate different training configurations.

| Experiment | Approach | Validation Accuracy | Macro F1 |
|---|---|---:|---:|
| Baseline | Adam + Standard Cross-Entropy | 73.33% | 67.35% |
| Experiment 1 | AdamW + Differential Learning Rates | 68.99% | 63.99% |
| Experiment 2 | Adam + Macro-F1 Checkpoint Selection | **82.03%** | **79.62%** |
| Experiment 3 | Class-Weighted Cross-Entropy | 75.94% | 73.97% |
| Experiment 4 | Layer 3 + Layer 4 Fine-Tuning | 72.17% | 70.66% |

**Experiment 2 was selected as the final model** based on its validation performance.

## 6. Final Model Evaluation

The selected Experiment 2 model was evaluated on the held-out test dataset.

| Metric | Test Performance |
|---|---:|
| Accuracy | **81.16%** |
| Macro Precision | 80.57% |
| Macro Recall | 79.72% |
| Macro F1-Score | **79.52%** |
| Weighted F1-Score | 80.54% |

### Class-wise Performance

| Class | Precision | Recall | F1-Score |
|---|---:|---:|---:|
| F_Breakage | 0.88 | 0.93 | 0.90 |
| F_Crushed | 0.83 | 0.58 | 0.69 |
| F_Normal | 0.80 | 0.93 | 0.86 |
| R_Breakage | 0.73 | 0.84 | 0.78 |
| R_Crushed | 0.71 | 0.60 | 0.65 |
| R_Normal | 0.89 | 0.89 | 0.89 |

The model performs well across several categories, while crushed-damage categories remain more challenging due to visual similarities with other conditions.

## 7. Technology Stack

| Area | Technologies |
|---|---|
| Programming Language | Python |
| Deep Learning | PyTorch |
| Model Architecture | ResNet50 |
| Computer Vision | Torchvision, Pillow |
| Machine Learning Utilities | Scikit-learn, NumPy |
| Data Visualization | Streamlit charts |
| Web Application | Streamlit |
| Version Control | Git, GitHub |
| Deployment | Streamlit Community Cloud |

## 8. Project Structure

```text
car-damage-prediction/
│
├── app/
│   └── app.py
│
├── artifacts/
│
├── configs/
│
├── dataset/                 # Excluded from Git
│
├── notebooks/
│   └── 01_dataset_exploration.ipynb
│
├── reports/
│
├── src/
│   └── car_damage/
│       ├── __init__.py
│       ├── data.py
│       ├── model.py
│       ├── predict.py
│       └── train.py
│
├── tests/
│
├── .gitignore
├── requirements.txt
└── README.md
```

## 9. Run Locally

### Prerequisites

- Python
- Git
- pip

### Clone the repository

```bash
git clone https://github.com/yogesh25shugani/car-damage-prediction.git

cd car-damage-prediction
```

### Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the application

```bash
streamlit run app/app.py
```

The application automatically downloads the trained Experiment 2 checkpoint from the GitHub Release when a local checkpoint is unavailable.

## 10. Trained Model

The selected trained model is hosted separately using GitHub Releases.

- Model: ResNet50
- Checkpoint: `best_model_exp2.pth`
- Release: [v1.0.0](https://github.com/yogesh25shugani/car-damage-prediction/releases/tag/v1.0.0)

## 11. Future Improvements

- Improve classification performance for crushed-damage categories.
- Expand the dataset with more diverse vehicle images.
- Investigate duplicate and near-duplicate image leakage across dataset splits.
- Add explainability using Grad-CAM.
- Explore object detection and damage localization.
- Add automated model testing and monitoring.
- Evaluate model performance on external vehicle images.

## 12. Limitations and Disclaimer

This project is an experimental deep learning application for vehicle damage classification.

Predictions depend on image quality, visibility, and similarity to the training dataset. Model confidence should not be interpreted as a guarantee of prediction correctness.

The application is intended for informational and educational purposes and should not replace professional vehicle inspection or insurance assessment.

---

## Author

**Yogesh Shugani**

- GitHub: https://github.com/yogesh25shugani
- LinkedIn: https://www.linkedin.com/in/yogesh-shugani-b43442123/

If you find this project useful, consider giving the repository a ⭐.
