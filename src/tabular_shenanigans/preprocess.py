from datetime import datetime, timezone
from pathlib import Path
import zipfile

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def _find_competition_zip(competition_slug: str) -> Path:
    data_dir = Path("data") / competition_slug
    zip_files = sorted(data_dir.glob("*.zip"))
    if not zip_files:
        raise ValueError(f"No competition zip found in {data_dir}")
    return zip_files[0]


def _read_csv_from_zip(zip_path: Path, member_name: str) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as archive:
        if member_name not in archive.namelist():
            raise ValueError(f"Missing {member_name} in {zip_path}")
        with archive.open(member_name) as f:
            return pd.read_csv(f)


def _infer_target_column(train_df: pd.DataFrame, test_df: pd.DataFrame) -> str:
    candidate_columns = [col for col in train_df.columns if col not in test_df.columns]
    if len(candidate_columns) != 1:
        raise ValueError(f"Could not infer a single target column. Candidates: {candidate_columns}")
    return candidate_columns[0]


def run_preprocessing(competition_slug: str) -> Path:
    zip_path = _find_competition_zip(competition_slug)
    train_df = _read_csv_from_zip(zip_path, "train.csv")
    test_df = _read_csv_from_zip(zip_path, "test.csv")
    target_column = _infer_target_column(train_df, test_df)

    x_train_raw = train_df.drop(columns=[target_column])
    y_train = train_df[target_column]
    x_test_raw = test_df

    numeric_columns = x_train_raw.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = [col for col in x_train_raw.columns if col not in numeric_columns]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_columns),
            ("cat", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
    )

    x_train_processed = preprocessor.fit_transform(x_train_raw)
    x_test_processed = preprocessor.transform(x_test_raw)
    feature_names = preprocessor.get_feature_names_out().tolist()

    artifact_dir = Path("artifacts") / competition_slug / "preprocess"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    x_train_df = pd.DataFrame(x_train_processed, columns=feature_names)
    x_test_df = pd.DataFrame(x_test_processed, columns=feature_names)

    x_train_df.to_csv(artifact_dir / "X_train_processed.csv", index=False)
    x_test_df.to_csv(artifact_dir / "X_test_processed.csv", index=False)
    y_train.to_frame(name=target_column).to_csv(artifact_dir / "y_train.csv", index=False)

    summary_df = pd.DataFrame(
        [
            {"metric": "generated_at_utc", "value": datetime.now(timezone.utc).isoformat()},
            {"metric": "target_column", "value": target_column},
            {"metric": "train_rows", "value": int(x_train_df.shape[0])},
            {"metric": "train_cols", "value": int(x_train_df.shape[1])},
            {"metric": "test_rows", "value": int(x_test_df.shape[0])},
            {"metric": "test_cols", "value": int(x_test_df.shape[1])},
            {"metric": "numeric_feature_count", "value": len(numeric_columns)},
            {"metric": "categorical_feature_count", "value": len(categorical_columns)},
        ]
    )
    summary_df.to_csv(artifact_dir / "preprocess_summary.csv", index=False)

    print(f"Preprocessed train shape: {x_train_df.shape[0]} rows x {x_train_df.shape[1]} cols")
    print(f"Preprocessed test shape: {x_test_df.shape[0]} rows x {x_test_df.shape[1]} cols")

    return artifact_dir
