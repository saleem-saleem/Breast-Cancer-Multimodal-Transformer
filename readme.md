# EduFeatOpt
A hybrid feature selection framework combining SmartFS (filter-based) and SmartHive (ACO–GA wrapper-based) for student performance prediction. Enhances accuracy and reduces redundancy using UCI and real educational datasets validated with SVM, KNN, and BPNN classifiers.
## Key Features
- **Hybrid Feature Selection Framework**: Integrates filter (SmartFS) and wrapper (SmartHive) methods for optimal feature subset selection. 
- **Dual Optimization Mechanism**: Combines Ant Colony Optimization (ACO) for global search and Genetic Algorithm (GA) for local refinement.  
- **SmartFS Module**: Uses modified Chi-Square (MChi), ReliefF (MRel), and CFS to ensure relevance and minimize redundancy.
- **SmartHive Module**: Employs ACO–GA optimization with SVM classifier feedback for high-accuracy feature refinement.
- **Real-World Educational Application**: Enables early identification of at-risk students.
Supports data-driven educational interventions and decision-making.


## Applications
- **Early Identification of At-Risk Students**
- **Personalized Learning and Adaptive Education**
- **Data-Driven Educational Interventions**
- **Enhanced Decision-Making for Institutions**
- **Academic Policy and Resource Planning**

## Prerequisites 
- **Python 3.7 or higher**
- **Required libraries**: numpy pandas scikit-learn matplotlib scipy fuzzy / skfuzzy tensorflow / keras joblib 

## Evaluation Metrics
<p>The performance of the student performance prediction models is evaluated using standard classification metrics derived from the confusion matrix:</p>

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
