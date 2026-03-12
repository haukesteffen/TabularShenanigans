from dataclasses import dataclass
from typing import Callable

import pandas as pd

FeatureRecipeTransform = Callable[[pd.DataFrame, pd.DataFrame], tuple[pd.DataFrame, pd.DataFrame]]


@dataclass(frozen=True)
class FeatureRecipeDefinition:
    recipe_id: str
    recipe_name: str
    recipe_description: str
    transform: FeatureRecipeTransform
