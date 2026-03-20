from tabular_shenanigans.representations.compilation import (
    CompiledRepresentation,
    compile_representation,
)
from tabular_shenanigans.representations.feature_schema import (
    ResolvedFeatureSchema,
    resolve_feature_schema,
    resolve_feature_types,
)
from tabular_shenanigans.representations.fitted import FittedRepresentation
from tabular_shenanigans.representations.registry import (
    REPRESENTATION_REGISTRY,
    get_representation_definition,
    resolve_representation_id,
)
from tabular_shenanigans.representations.types import (
    FitMode,
    FittedOutputAdapter,
    FittedStep,
    OutputAdapter,
    OutputContract,
    RepresentationDefinition,
    RepresentationStep,
)

# Register competition-specific representations
import tabular_shenanigans.representations.playground_series_s6e3 as _s6e3  # noqa: F401

__all__ = [
    "CompiledRepresentation",
    "FitMode",
    "FittedOutputAdapter",
    "FittedRepresentation",
    "FittedStep",
    "OutputAdapter",
    "OutputContract",
    "REPRESENTATION_REGISTRY",
    "RepresentationDefinition",
    "RepresentationStep",
    "ResolvedFeatureSchema",
    "compile_representation",
    "get_representation_definition",
    "resolve_feature_schema",
    "resolve_feature_types",
    "resolve_representation_id",
]
