import os
import pathlib
import zipfile

import kaggle
import pandas as pd
from sqlalchemy import create_engine, inspect, text

from .config import Config


def initialize_project(config: Config) -> None:
    competition_name = config.competition_name
    root = pathlib.Path(__file__).parent.parent.parent
    db_uri = f"sqlite:///{root / 'db' / config.db}"
    _initialize_competition_table(competition_name, db_uri)
    _initialize_dataset_table(competition_name, db_uri)
    _download_competition_data(competition_name, db_uri)
    return db_uri


def _download_competition_data(
    competition_name: str,
    db_uri: str,
) -> None:
    """
    Downloads the competition data from Kaggle, extracts it,
    inserts train and test data into the database.

    Args:
        competition_name (str): The name of the Kaggle competition.
        db_uri (str): The URI of the database to connect to.
    """
    engine = create_engine(db_uri)
    tmpdir = pathlib.Path.cwd() / "tmp"
    tmpfile = tmpdir / f"{competition_name}.zip"

    print(f"Downloading data for competition '{competition_name}'...")
    kaggle.api.competition_download_files(
        competition=competition_name, path=tmpdir, quiet=True
    )

    # Extract the zip file
    zf = zipfile.ZipFile(tmpfile)
    with zf.open("train.csv") as f:
        train = pd.read_csv(f).convert_dtypes()
    with zf.open("test.csv") as f:
        test = pd.read_csv(f).convert_dtypes()
    tmpfile.unlink()
    tmpdir.rmdir()

    with engine.begin() as connection:
        train.to_sql(
            competition_name + "-train", connection, if_exists="replace", index=False
        )
        test.to_sql(
            competition_name + "-test", connection, if_exists="replace", index=False
        )
    return


def _initialize_competition_table(
    competition_name: str,
    db_uri: str,
) -> None:
    """
    Initializes the database for the competition by creating and/or populating necessary tables.

    Args:
        competition_name (str): The name of the Kaggle competition.
        db_uri (str): The URI of the database to connect to.
    """

    engine = create_engine(db_uri)
    inspector = inspect(engine)

    # check if table 'competitions' exists
    if not inspector.has_table("competitions"):
        with open("db/migrations/0_0_create_competitions_table.sql", "r") as f:
            create_competitions_table = text(f.read())
        with engine.begin() as connection:
            print("Creating table 'competitions' in the database.")
            connection.execute(create_competitions_table)

    else:
        print("Table 'competitions' already exists in the database.")

    # check if competition already exists in 'competitions' table
    with engine.begin() as connection:
        result = connection.execute(
            text("SELECT COUNT(*) FROM competitions WHERE name = :name"),
            {"name": competition_name},
        )
        count = result.scalar()
        if count > 0:
            print(f"Competition '{competition_name}' already exists in the database.")
            return

        else:
            print(f"Adding competition '{competition_name}' to the database.")
            connection.execute(
                text(f"INSERT INTO competitions (name) VALUES ('{competition_name}')")
            )
            return


def _initialize_dataset_table(
    competition_name: str,
    db_uri: str,
) -> None:
    """
    Initializes the dataset table for the competition by creating and/or populating necessary tables.

    Args:
        competition_name (str): The name of the Kaggle competition.
        db_uri (str): The URI of the database to connect to.
    """
    train_table = competition_name + "-train"
    test_table = competition_name + "-test"
    engine = create_engine(db_uri)
    inspector = inspect(engine)

    # check if table 'datasets' exists
    if not inspector.has_table("datasets"):
        with open("db/migrations/0_1_create_datasets_table.sql", "r") as f:
            create_datasets_table = text(f.read())
        with engine.begin() as connection:
            print("Creating table 'datasets' in the database.")
            connection.execute(create_datasets_table)

    else:
        print("Table 'datasets' already exists in the database.")

    # fetch competition_id
    with engine.begin() as connection:
        result = connection.execute(
            text(
                f"SELECT competition_id FROM competitions WHERE name = '{competition_name}'"
            )
        )
        competition_id = result.scalar()

    # check if raw data already exists in 'datasets' table
    with engine.begin() as connection:
        result = connection.execute(
            text(
                f"SELECT COUNT(*) FROM datasets WHERE competition_id = \
                {competition_id} AND table_name = '{train_table}'",
            )
        )
        count = result.scalar()
        if count > 0:
            print(f"Dataset '{train_table}' already exists in the 'datasets' table.")

        else:
            print(f"Adding '{train_table}' to the 'datasets' table.")
            connection.execute(
                text(
                    f"INSERT INTO datasets (competition_id, table_name) \
                    VALUES ({competition_id}, '{train_table}')",
                )
            )

        result = connection.execute(
            text(
                f"SELECT COUNT(*) FROM datasets WHERE competition_id = \
                {competition_id} AND table_name = '{test_table}'",
            )
        )
        count = result.scalar()
        if count > 0:
            print(f"Dataset '{test_table}' already exists in the 'datasets' table.")

        else:
            print(f"Adding '{test_table}' to the 'datasets' table.")
            connection.execute(
                text(
                    f"INSERT INTO datasets (competition_id, table_name) \
                    VALUES ({competition_id}, '{test_table}')",
                )
            )
    return
