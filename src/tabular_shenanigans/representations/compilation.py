from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse

from tabular_shenanigans.representations.feature_schema import ResolvedFeatureSchema
from tabular_shenanigans.representations.operators import build_feature_generator, build_feature_pruner
from tabular_shenanigans.representations.types import (
    FeatureBundle,
    FeatureGenerator,
    FeaturePruner,
    MaterializedRepresentation,
    RepresentationContract,
    RepresentationSpec,
)


GENERIC_PREPROCESSING_BACKEND = "feature_bundle"


@dataclass(frozen=True)
class FittedRepresentation:
    representation_id: str
    fitted_generators: tuple[object, ...]
    fitted_pruners: tuple[object, ...]
    contract: RepresentationContract

    def transform(self, X: pd.DataFrame) -> object:
        bundle = FeatureBundle(blocks=tuple(generator.transform(X) for generator in self.fitted_generators))
        for pruner in self.fitted_pruners:
            bundle = pruner.transform(bundle)
        return materialize_feature_bundle(bundle, self.contract).values


@dataclass(frozen=True)
class CompiledRepresentation:
    representation_spec: RepresentationSpec
    feature_schema: ResolvedFeatureSchema
    generators: tuple[FeatureGenerator, ...]
    pruners: tuple[FeaturePruner, ...]
    contract: RepresentationContract
    matrix_output_kind: str
    preprocessing_backend: str

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> FittedRepresentation:
        fitted_generators = []
        for generator in self.generators:
            y_arg = y_train if generator.fit_mode == "supervised" else None
            fitted_generators.append(generator.fit(X_train, y_arg))

        bundle = FeatureBundle(blocks=tuple(generator.transform(X_train) for generator in fitted_generators))
        fitted_pruners = []
        for pruner in self.pruners:
            y_arg = y_train if pruner.fit_mode == "supervised" else None
            fitted_pruner = pruner.fit(bundle, y_arg)
            bundle = fitted_pruner.transform(bundle)
            fitted_pruners.append(fitted_pruner)

        return FittedRepresentation(
            representation_id=self.representation_spec.representation_id,
            fitted_generators=tuple(fitted_generators),
            fitted_pruners=tuple(fitted_pruners),
            contract=self.contract,
        )


def materialize_feature_bundle(
    bundle: FeatureBundle,
    contract: RepresentationContract,
) -> MaterializedRepresentation:
    dense_frame = bundle.dense_frame()
    native_frame = bundle.native_frame()
    sparse_matrix = bundle.sparse_matrix()

    if contract.matrix_output_kind == "native_frame":
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
            preprocessing_backend=GENERIC_PREPROCESSING_BACKEND,
        )

    if contract.matrix_output_kind == "sparse_csr":
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
            preprocessing_backend=GENERIC_PREPROCESSING_BACKEND,
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
        preprocessing_backend=GENERIC_PREPROCESSING_BACKEND,
    )


def compile_representation(
    representation_spec: RepresentationSpec,
    feature_schema: ResolvedFeatureSchema,
    x_train_sample: pd.DataFrame,
    representation_contract: RepresentationContract,
) -> CompiledRepresentation:
    generators = tuple(
        build_feature_generator(
            component_id=component.component_id,
            params=component.params,
            feature_schema=feature_schema,
            X_sample=x_train_sample,
        )
        for component in representation_spec.operators
    )
    pruners = tuple(
        build_feature_pruner(
            component_id=component.component_id,
            params=component.params,
        )
        for component in representation_spec.pruners
    )
    return CompiledRepresentation(
        representation_spec=representation_spec,
        feature_schema=feature_schema,
        generators=generators,
        pruners=pruners,
        contract=representation_contract,
        matrix_output_kind=representation_contract.matrix_output_kind,
        preprocessing_backend=GENERIC_PREPROCESSING_BACKEND,
    )
