from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse

from tabular_shenanigans.representations.definition import (
    build_representation_id,
    normalize_representation_config,
    representation_has_cuml_compatible_numerics,
    representation_has_dense_numeric,
    representation_has_frequency_categorical,
    representation_has_native_categorical,
    representation_has_native_numeric,
    representation_has_native_tabular,
    representation_has_sparse_numeric,
    resolve_representation_matrix_output_kind,
    resolve_representation_output_kinds,
    validate_representation_definition,
)
from tabular_shenanigans.representations.feature_schema import ResolvedFeatureSchema
from tabular_shenanigans.representations.operators import build_feature_generator, build_feature_pruner
from tabular_shenanigans.representations.types import (
    FeatureBundle,
    FeatureGenerator,
    FeaturePruner,
    MaterializedRepresentation,
    MatrixOutputKind,
    OutputKind,
    RepresentationComponentSpec,
    RepresentationConfigLike,
)


GENERIC_PREPROCESSING_BACKEND = "feature_bundle"


@dataclass(frozen=True)
class FittedRepresentation:
    compiled_representation: "CompiledRepresentation"
    fitted_generators: tuple[object, ...]
    fitted_pruners: tuple[object, ...]

    @property
    def representation_id(self) -> str:
        return self.compiled_representation.representation_id

    @property
    def matrix_output_kind(self) -> MatrixOutputKind:
        return self.compiled_representation.matrix_output_kind

    @property
    def preprocessing_backend(self) -> str:
        return self.compiled_representation.preprocessing_backend

    def transform(self, X: pd.DataFrame) -> object:
        bundle = FeatureBundle(blocks=tuple(generator.transform(X) for generator in self.fitted_generators))
        for pruner in self.fitted_pruners:
            bundle = pruner.transform(bundle)
        return materialize_feature_bundle(
            bundle,
            matrix_output_kind=self.matrix_output_kind,
            preprocessing_backend=self.preprocessing_backend,
        ).values


@dataclass(frozen=True)
class CompiledRepresentation:
    representation_id: str
    operators: tuple[RepresentationComponentSpec, ...]
    pruner_components: tuple[RepresentationComponentSpec, ...]
    feature_schema: ResolvedFeatureSchema
    feature_generators: tuple[FeatureGenerator, ...]
    feature_pruners: tuple[FeaturePruner, ...]
    matrix_output_kind_override: MatrixOutputKind | None = None
    preprocessing_backend: str = GENERIC_PREPROCESSING_BACKEND

    @property
    def output_kinds(self) -> frozenset[OutputKind]:
        return resolve_representation_output_kinds(self.operators)

    @property
    def has_dense_numeric(self) -> bool:
        return representation_has_dense_numeric(self.operators)

    @property
    def has_sparse_numeric(self) -> bool:
        return representation_has_sparse_numeric(self.operators)

    @property
    def has_native_tabular(self) -> bool:
        return representation_has_native_tabular(self.operators)

    @property
    def has_native_categorical(self) -> bool:
        return representation_has_native_categorical(self.operators)

    @property
    def has_native_numeric(self) -> bool:
        return representation_has_native_numeric(self.operators)

    @property
    def has_frequency_categorical(self) -> bool:
        return representation_has_frequency_categorical(self.operators)

    @property
    def has_cuml_compatible_numerics(self) -> bool:
        return representation_has_cuml_compatible_numerics(self.operators)

    @property
    def matrix_output_kind(self) -> MatrixOutputKind:
        if self.matrix_output_kind_override is not None:
            return self.matrix_output_kind_override
        return resolve_representation_matrix_output_kind(self.operators)

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> FittedRepresentation:
        fitted_generators = []
        for generator in self.feature_generators:
            y_arg = y_train if generator.fit_mode == "supervised" else None
            fitted_generators.append(generator.fit(X_train, y_arg))

        bundle = FeatureBundle(blocks=tuple(generator.transform(X_train) for generator in fitted_generators))
        fitted_pruners = []
        for pruner in self.feature_pruners:
            y_arg = y_train if pruner.fit_mode == "supervised" else None
            fitted_pruner = pruner.fit(bundle, y_arg)
            bundle = fitted_pruner.transform(bundle)
            fitted_pruners.append(fitted_pruner)

        return FittedRepresentation(
            compiled_representation=self,
            fitted_generators=tuple(fitted_generators),
            fitted_pruners=tuple(fitted_pruners),
        )


def materialize_feature_bundle(
    bundle: FeatureBundle,
    *,
    matrix_output_kind: MatrixOutputKind,
    preprocessing_backend: str = GENERIC_PREPROCESSING_BACKEND,
) -> MaterializedRepresentation:
    dense_frame = bundle.dense_frame()
    native_frame = bundle.native_frame()
    sparse_matrix = bundle.sparse_matrix()

    if matrix_output_kind == "native_frame":
        frames = [frame for frame in (native_frame, dense_frame) if not frame.empty]
        if not frames:
            values = pd.DataFrame()
        elif len(frames) == 1:
            values = frames[0]
        else:
            values = pd.concat(frames, axis=1)
        return MaterializedRepresentation(
            matrix_output_kind="native_frame",
            values=values,
            preprocessing_backend=preprocessing_backend,
        )

    if matrix_output_kind == "sparse_csr":
        matrices = []
        if not dense_frame.empty:
            matrices.append(sparse.csr_matrix(dense_frame.to_numpy(dtype=float)))
        if sparse_matrix is not None:
            matrices.append(sparse.csr_matrix(sparse_matrix))
        if not matrices:
            values = sparse.csr_matrix((dense_frame.shape[0], 0), dtype=float)
        elif len(matrices) == 1:
            values = matrices[0]
        else:
            values = sparse.hstack(matrices, format="csr")
        return MaterializedRepresentation(
            matrix_output_kind="sparse_csr",
            values=values,
            preprocessing_backend=preprocessing_backend,
        )

    dense_values = dense_frame.to_numpy(dtype=float) if not dense_frame.empty else None
    if sparse_matrix is not None:
        sparse_as_dense = sparse_matrix.toarray()
        if dense_values is not None:
            values = np.hstack([dense_values, sparse_as_dense])
        else:
            values = sparse_as_dense
    elif dense_values is not None:
        values = dense_values
    else:
        values = np.empty((0, 0), dtype=float)
    return MaterializedRepresentation(
        matrix_output_kind="dense_array",
        values=values,
        preprocessing_backend=preprocessing_backend,
    )


def compile_representation(
    representation: RepresentationConfigLike,
    feature_schema: ResolvedFeatureSchema,
    x_train_sample: pd.DataFrame,
    matrix_output_kind_override: MatrixOutputKind | None = None,
) -> CompiledRepresentation:
    operators, pruners = normalize_representation_config(representation)
    validate_representation_definition(operators, pruners)

    feature_generators = tuple(
        build_feature_generator(
            component_id=component.component_id,
            params=component.params,
            feature_schema=feature_schema,
            X_sample=x_train_sample,
        )
        for component in operators
    )
    feature_pruners = tuple(
        build_feature_pruner(
            component_id=component.component_id,
            params=component.params,
        )
        for component in pruners
    )

    return CompiledRepresentation(
        representation_id=build_representation_id(operators, pruners),
        operators=operators,
        pruner_components=pruners,
        feature_schema=feature_schema,
        feature_generators=feature_generators,
        feature_pruners=feature_pruners,
        matrix_output_kind_override=matrix_output_kind_override,
    )
