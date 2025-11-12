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
- **Classification Metrics**: 
  Accuracy (%) – Measures overall correctness of recurrence type prediction.
  Precision (%) – Indicates reliability of positive predictions (reduces false positives).
  Recall (%) – Measures the model’s ability to detect true recurrence cases (reduces false negatives).
  F1-Score (%) – Harmonic mean of precision and recall for balanced evaluation.
  AUC-ROC – Evaluates class separability across recurrence risk categories.

- **Survival Analysis Metrics**: 
  C-Index (Concordance Index) – Quantifies how well predicted survival times align with actual outcomes.
  Kaplan–Meier (KM) Curves – Visualize survival probability over time for risk groups.
  Log-Rank Test (p-value) – Tests statistical significance between high- and low-risk survival groups.
  Brier Score – Measures calibration accuracy of predicted survival probabilities.Time-Dependent AUC (10-year AUC).

## How to Run
1. Clone the Repository
git clone https://github.com/<your-username>/breast-cancer-multimodal-transformer.git  
cd breast-cancer-multimodal-transformer

2. Set Up a Python Environment
Create and activate a new environment (recommended):
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate

3. Install Dependencies
Install all required libraries:
pip install -r requirements.txt

4. Download Datasets
Download and organize datasets used in the study:
METABRIC Dataset
GSE2034
GSE2990
BCSC
Breast Cancer Coimbra Dataset

5. Preprocess Data
Run preprocessing to clean, normalize, and prepare datasets:
python scripts/preprocess_data.py
This step handles:
Missing value imputation
Normalization (Z-score scaling)
Feature encoding
Autoencoder-based dimensionality reduction

6. Train the Model
Train the unified multi-modal Transformer:
python train_model.py
This script:
Loads preprocessed data
Trains autoencoders and Transformer encoder
Optimizes classification, survival, reconstruction, and contrastive losses
Saves the best model under /models/checkpoints/
Training parameters can be adjusted in config.yaml (batch size, epochs, learning rate, etc.).

7. Evaluate the Model
After training, evaluate performance on the test sets:
python evaluate_model.py
Outputs include:
Accuracy, Precision, Recall, F1-score
C-index, Kaplan–Meier curves, Log-Rank test
SHAP explainability plots

8. Visualize and Interpret Results
Generate interpretability and visualization plots:
python visualize_results.py
This script creates:
SHAP-based feature importance charts
Kaplan–Meier survival plots
Training vs. validation accuracy/loss graphs

9. Predict New Patient Data
To predict recurrence risk and survival probability for a new patient:
python predict_patient.py --input patient_data.csv
Output includes predicted recurrence category, log-risk score, and survival probability.

10. Optional: Hyperparameter Optimization
To optimize model parameters:
python tune_hyperparameters.py
Uses Optuna for cross-validation tuning across multiple datasets.

- **Outputs Generated**
  <li>results/metrics.csv — summary of all performance metrics</li>
  <li>results/plots/ — accuracy/loss and survival graphs.</li> 
  <li>models/best_model.pt — saved Transformer model.</li>
  <li>logs/training.log — training progress and loss details.</li> 
