from tabular_shenanigans.representations.output_adapters import (
    DenseArrayAdapter,
    NativeFrameAdapter,
    SparseCSRAdapter,
)
from tabular_shenanigans.representations.steps import (
    FrequencyEncodeStep,
    KBinsStep,
    MedianImputeStep,
    NativeCategoricalStep,
    OneHotEncodeStep,
    OrdinalEncodeStep,
    StandardizeStep,
)
from tabular_shenanigans.representations.types import RepresentationDefinition

REPRESENTATION_REGISTRY: dict[str, RepresentationDefinition] = {}


def _register(definition: RepresentationDefinition) -> None:
    REPRESENTATION_REGISTRY[definition.representation_id] = definition


def _build_base_representations() -> None:
    numeric_configs = {
        "median": ("MedianImpute", MedianImputeStep),
        "standardize": ("Standardize", StandardizeStep),
        "kbins": ("KBins", KBinsStep),
    }
    categorical_configs = {
        "ordinal": ("Ordinal", OrdinalEncodeStep, DenseArrayAdapter),
        "onehot": ("OneHot", OneHotEncodeStep, DenseArrayAdapter),
        "frequency": ("Frequency", FrequencyEncodeStep, DenseArrayAdapter),
        "native": ("Native", NativeCategoricalStep, NativeFrameAdapter),
    }

    for num_id, (num_name, num_step_cls) in numeric_configs.items():
        for cat_id, (cat_name, cat_step_cls, adapter_cls) in categorical_configs.items():
            representation_id = f"{num_id}-{cat_id}"
            representation_name = f"{num_name}{cat_name}"

            num_step = num_step_cls()
            cat_step = cat_step_cls()
            adapter = adapter_cls()

            # For onehot + kbins with sparse, use SparseCSRAdapter
            if cat_id == "onehot" and num_id == "kbins":
                adapter = SparseCSRAdapter()
                num_step = KBinsStep(sparse_output=True)
                cat_step = OneHotEncodeStep(sparse_output=True)

            _register(
                RepresentationDefinition(
                    representation_id=representation_id,
                    representation_name=representation_name,
                    steps=(num_step, cat_step),
                    output_adapter=adapter,
                    numeric_preprocessor_id=num_id,
                    categorical_preprocessor_id=cat_id,
                )
            )


_build_base_representations()


def resolve_representation_id(representation_id: str) -> str:
    if representation_id in REPRESENTATION_REGISTRY:
        return representation_id
    supported_ids = sorted(REPRESENTATION_REGISTRY)
    raise ValueError(
        f"Representation id '{representation_id}' is not supported. "
        f"Supported values: {supported_ids}"
    )


def get_representation_definition(representation_id: str) -> RepresentationDefinition:
    resolved_id = resolve_representation_id(representation_id)
    return REPRESENTATION_REGISTRY[resolved_id]
