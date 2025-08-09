from pathlib import Path
from typing import Union

import yaml
from pydantic import BaseModel


class Config(BaseModel):
    """General parameters.

    Attributes:
        competition_name (str): Name of the Kaggle competition.
        data_path (Path): Path to the directory containing data files.
        submission_file (str): Name of the submission file.
        target_column (str): Name of the target column in the dataset.
    """

    competition_name: str
    data_path: Path
    submission_file: str
    target_column: str


def _convert_paths(data: dict) -> dict:
    """Recursively convert all dictionary values containing 'path' to Path objects."""
    for key, value in data.items():
        if isinstance(value, dict):
            data[key] = _convert_paths(value)
        elif isinstance(value, str) and "path" in key.lower():
            data[key] = Path(value)
    return data


def load_config(path: Union[str, Path]) -> Config:
    """Load configuration from YAML file."""
    with open(path, "r") as f:
        config = yaml.safe_load(f)

    # Convert all 'path' strings to Path objects
    config = _convert_paths(config)

    # Instantiate Config class
    config = Config(**config)
    print(f"Config loaded from {path}.")
    return config
