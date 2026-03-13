import pandas as pd

from tabular_shenanigans.feature_recipes.base import FeatureRecipeDefinition
from tabular_shenanigans.feature_recipes.playground_series_s6e3 import (
    S6E3_ABLATION_FEATURE_RECIPES,
    S6E3_V1_FEATURE_RECIPE,
    S6E3_V2_FEATURE_RECIPE,
    S6E3_V3_FEATURE_RECIPE,
)

IDENTITY_FEATURE_RECIPE_ID = "fr0"


def _identity_recipe(
    x_train_raw: pd.DataFrame,
    x_test_raw: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    return x_train_raw.copy(), x_test_raw.copy()


FEATURE_RECIPE_REGISTRY = {
    IDENTITY_FEATURE_RECIPE_ID: FeatureRecipeDefinition(
        recipe_id=IDENTITY_FEATURE_RECIPE_ID,
        recipe_name="Identity",
        recipe_description="No engineered feature changes; pass raw modeled features through unchanged.",
        transform=_identity_recipe,
    ),
    S6E3_V1_FEATURE_RECIPE.recipe_id: S6E3_V1_FEATURE_RECIPE,
    S6E3_V2_FEATURE_RECIPE.recipe_id: S6E3_V2_FEATURE_RECIPE,
    S6E3_V3_FEATURE_RECIPE.recipe_id: S6E3_V3_FEATURE_RECIPE,
    **{definition.recipe_id: definition for definition in S6E3_ABLATION_FEATURE_RECIPES},
}


def get_supported_feature_recipe_ids() -> list[str]:
    return sorted(FEATURE_RECIPE_REGISTRY)


def resolve_feature_recipe_id(recipe_id: str) -> str:
    if recipe_id in FEATURE_RECIPE_REGISTRY:
        return recipe_id

    supported_recipe_ids = get_supported_feature_recipe_ids()
    raise ValueError(
        f"Feature recipe id '{recipe_id}' is not supported. Supported values: {supported_recipe_ids}"
    )


def get_feature_recipe_definition(recipe_id: str) -> FeatureRecipeDefinition:
    resolved_recipe_id = resolve_feature_recipe_id(recipe_id)
    return FEATURE_RECIPE_REGISTRY[resolved_recipe_id]


def _validate_transformed_frames(
    recipe_id: str,
    x_train_raw: pd.DataFrame,
    x_test_raw: pd.DataFrame,
    transformed_train: pd.DataFrame,
    transformed_test: pd.DataFrame,
) -> None:
    if not isinstance(transformed_train, pd.DataFrame) or not isinstance(transformed_test, pd.DataFrame):
        raise ValueError(f"Feature recipe '{recipe_id}' must return pandas DataFrame objects for train and test.")
    if transformed_train.shape[0] != x_train_raw.shape[0]:
        raise ValueError(f"Feature recipe '{recipe_id}' changed the train row count.")
    if transformed_test.shape[0] != x_test_raw.shape[0]:
        raise ValueError(f"Feature recipe '{recipe_id}' changed the test row count.")
    if not transformed_train.index.equals(x_train_raw.index):
        raise ValueError(f"Feature recipe '{recipe_id}' changed the train row index.")
    if not transformed_test.index.equals(x_test_raw.index):
        raise ValueError(f"Feature recipe '{recipe_id}' changed the test row index.")
    if transformed_train.columns.tolist() != transformed_test.columns.tolist():
        raise ValueError(
            f"Feature recipe '{recipe_id}' must produce the same feature columns for train and test."
        )
    if transformed_train.columns.duplicated().any():
        duplicate_columns = transformed_train.columns[transformed_train.columns.duplicated()].tolist()
        raise ValueError(f"Feature recipe '{recipe_id}' produced duplicate feature columns: {duplicate_columns}")
    if transformed_train.shape[1] == 0:
        raise ValueError(f"Feature recipe '{recipe_id}' produced zero feature columns.")


def apply_feature_recipe(
    recipe_id: str,
    x_train_raw: pd.DataFrame,
    x_test_raw: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    definition = get_feature_recipe_definition(recipe_id)
    transformed_train, transformed_test = definition.transform(
        x_train_raw=x_train_raw.copy(),
        x_test_raw=x_test_raw.copy(),
    )
    _validate_transformed_frames(
        recipe_id=definition.recipe_id,
        x_train_raw=x_train_raw,
        x_test_raw=x_test_raw,
        transformed_train=transformed_train,
        transformed_test=transformed_test,
    )
    return transformed_train, transformed_test
