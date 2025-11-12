# Breast-Cancer-Multimodal-Transformer
A unified deep learning framework integrating clinical, genomic, and lifestyle data using denoising autoencoders, attention-based fusion, and a Transformer backbone for joint breast cancer recurrence classification and survival prediction, achieving high accuracy, interpretability, and robust cross-dataset generalization.
## Key Features
- **Unified Framework:**: Performs both recurrence classification and survival prediction in a single model. 
- **Multi-Modal Fusion**: Integrates clinical, genomic, and lifestyle data.  
- **Autoencoder Compression**: Reduces noise and dimensionality while preserving discriminative information.
- **Attention Mechanism**: Learns patient-specific modality importance for improved interpretability.
- **Contrastive Learning**: Enhances representation separability across recurrence types.

## Applications
- **Clinical Decision Support:**
- **Precision Oncology**
- **AI-Driven Prognostics**
- **Healthcare Research**
- **Survival Analysis**

## Prerequisites 
- **Python 3.7 or higher**
- **Required libraries**: numpy pandas scikit-learn matplotlib scipy fuzzy / skfuzzy tensorflow / keras joblib, matplotlib, seaborn, optuna,shap, captum
- **Deep Learning Framework**: PyTorch 2.0+ (or TensorFlow equivalent if adapted)
- **Hardware Requirements:** GPU-enabled system (e.g., NVIDIA RTX 3090/4090 or equivalent). Minimum 16 GB RAM (recommended: 32 GB+)

## Evaluation Metrics
- **Classification Metrics**
  Accuracy (%) – Measures overall correctness of recurrence type prediction.
  Precision (%) – Indicates reliability of positive predictions (reduces false positives).
  Recall (%) – Measures the model’s ability to detect true recurrence cases (reduces false negatives).
  F1-Score (%) – Harmonic mean of precision and recall for balanced evaluation.
  AUC-ROC – Evaluates class separability across recurrence risk categories.

- **Survival Analysis Metrics**
  C-Index (Concordance Index) – Quantifies how well predicted survival times align with actual outcomes.
  Kaplan–Meier (KM) Curves – Visualize survival probability over time for risk groups.
  Log-Rank Test (p-value) – Tests statistical significance between high- and low-risk survival groups.
  Brier Score – Measures calibration accuracy of predicted survival probabilities.

Time-Dependent AUC (10-year AUC) – Evaluates discrimination of survival predictions over a fixed horizon.
- **True Positives (TP):** Number of students correctly predicted as “passed".
- **True Negatives (TN):** Number of students correctly predicted as “failed.”
- **False Positives (FP):** Number of students incorrectly predicted as “passed.”
- **False Negatives (FN):** Number of students incorrectly predicted as “failed.”

## How to Run
<p>Step 1: Set Up the Environment
Install Python:
Make sure Python 3.7 or higher is installed on your system.
Install Required Libraries:
Open a terminal or command prompt and run:
pip install numpy pandas scikit-learn matplotlib seaborn scipy scikit-fuzzy tensorflow
(Ensure all required libraries listed in requirements.txt are installed.)</p>

<p>Step 2: Prepare the Dataset
Use a Dataset:
The dataset should be in CSV format.
Include features (input columns) and labels (target column).
Preprocess Your Dataset:
Ensure there are no missing values.
Encode categorical variables if needed (e.g., one-hot encoding).
Normalize or scale numeric features for consistency.
Place your dataset in the /data folder for convenience.</p>


<p>Step 3: Execute the Code
Run Feature Selection smartfs:
python smartFS.py
Run SmartHIVE:
python smartHIVE.py
Train and Evaluate Classifiers :
python evaluate.py
Output:
Iteration details during feature selection.
Selected features for each iteration.
Performance metrics: Accuracy, TPR, TNR, Confusion Matrix.
Optimal feature subset indices for downstream analysis.</p>

<p>Step 4: Modify Parameters (Optional)
Adjust Parameters:
Open the relevant Python script  in a text editor.

Modify parameters to fit your dataset, for example:
max_iterations = 10        # Number of iterations
initial_threshold = 0.5    # Threshold for feature selection

Use Your Dataset:
Replace any synthetic dataset code with:

import pandas as pd

data = pd.read_csv("data/your_dataset.csv")

X = data.iloc[:, :-1].values  # Features

y = data.iloc[:, -1].values   # Labels

Rerun the Script:
python feature_selection.py</p>

<p>Step 5: Analyze Results
Optimal Features:
The scripts output indices of selected features.
Use these indices to extract the most relevant features from your dataset.
Performance Metrics:
Check console output for Accuracy, True Positive Rate (TPR), True Negative Rate (TNR), and Confusion Matrix.
Optional: Calculate F1-score, Precision, or Recall if needed.
Optional Enhancements:
Save results to a file: .csv or .txt for reporting.</p>
