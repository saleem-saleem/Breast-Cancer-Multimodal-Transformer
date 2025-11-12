
import os
import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

# ------------------------------------------------------------
# Utility Functions
# ------------------------------------------------------------

def load_dataset(file_path: str) -> pd.DataFrame:
    """
    Load dataset from a CSV/TSV file.
    """
    ext = os.path.splitext(file_path)[-1].lower()
    if ext == ".csv":
        df = pd.read_csv(file_path)
    elif ext in [".tsv", ".txt"]:
        df = pd.read_csv(file_path, sep="\t")
    else:
        raise ValueError("Unsupported file format. Use CSV or TSV.")
    print(f"[INFO] Loaded dataset: {file_path}  →  Shape: {df.shape}")
    return df


def identify_columns(df: pd.DataFrame):
    """
    Automatically detect numeric and categorical columns.
    """
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    print(f"[INFO] Numeric columns: {len(numeric_cols)}, Categorical columns: {len(categorical_cols)}")
    return numeric_cols, categorical_cols


def preprocess(df: pd.DataFrame, label_col: str = None):
    """
    Preprocess the dataset with imputation, scaling, and encoding.
    Returns: X (features), y (target if provided)
    """
    numeric_cols, categorical_cols = identify_columns(df)

    # Separate target column if specified
    if label_col and label_col in df.columns:
        y = df[label_col]
        X = df.drop(columns=[label_col])
    else:
        X, y = df.copy(), None

    # Define transformers
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    # Apply transformations
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols)
        ]
    )

    X_processed = preprocessor.fit_transform(X)
    feature_names = (
        numeric_cols +
        list(preprocessor.named_transformers_["cat"]["encoder"].get_feature_names_out(categorical_cols))
        if categorical_cols else numeric_cols
    )

    X_processed = pd.DataFrame(X_processed, columns=feature_names)

    print(f"[INFO] After preprocessing → Shape: {X_processed.shape}")
    return X_processed, y


def split_and_save(X, y=None, out_dir="processed_data", test_size=0.2):
    """
    Split into train/test and save processed datasets.
    """
    os.makedirs(out_dir, exist_ok=True)

    if y is not None:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
        X_train.to_csv(os.path.join(out_dir, "X_train.csv"), index=False)
        X_test.to_csv(os.path.join(out_dir, "X_test.csv"), index=False)
        y_train.to_csv(os.path.join(out_dir, "y_train.csv"), index=False)
        y_test.to_csv(os.path.join(out_dir, "y_test.csv"), index=False)
    else:
        X.to_csv(os.path.join(out_dir, "X_processed.csv"), index=False)

    print(f"[INFO] Processed data saved in: {out_dir}")


# ------------------------------------------------------------
# Main Execution Block
# ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Preprocess multi-modal breast cancer datasets.")
    parser.add_argument("--input", type=str, required=True, help="Path to input dataset (CSV/TSV).")
    parser.add_argument("--label", type=str, default=None, help="Name of target column (if available).")
    parser.add_argument("--out_dir", type=str, default="processed_data", help="Output directory for processed files.")
    args = parser.parse_args()

    df = load_dataset(args.input)
    X, y = preprocess(df, args.label)
    split_and_save(X, y, args.out_dir)


if __name__ == "__main__":
    main()
