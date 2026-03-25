from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

import pandas as pd
from scipy import sparse

FitMode = Literal["stateless", "unsupervised", "supervised"]
OutputKind = Literal["dense_numeric", "sparse_numeric", "native_tabular"]
MatrixOutputKind = Literal["dense_array", "sparse_csr", "native_frame"]


@dataclass(frozen=True)
class RepresentationComponentSpec:
    component_id: str
    params: dict[str, object]


class RepresentationComponentConfigLike(Protocol):
    id: str

    def params(self) -> dict[str, object]: ...


class RepresentationConfigLike(Protocol):
    operators: object
    pruners: object


@dataclass(frozen=True)
class FeatureBlock:
    block_id: str
    output_kind: OutputKind
    values: pd.DataFrame | sparse.csr_matrix
    feature_names: tuple[str, ...]


@dataclass(frozen=True)
class MaterializedRepresentation:
    matrix_output_kind: MatrixOutputKind
    values: object
    preprocessing_backend: str


class FittedFeatureGenerator(Protocol):
    block_id: str
    output_kind: OutputKind

    def transform(self, X: pd.DataFrame) -> FeatureBlock: ...


class FeatureGenerator(Protocol):
    operator_id: str
    fit_mode: FitMode
    output_kind: OutputKind

    def fit(self, X: pd.DataFrame, y: pd.Series | None) -> FittedFeatureGenerator: ...


class FittedFeaturePruner(Protocol):
    def transform(self, bundle: "FeatureBundle") -> "FeatureBundle": ...


class FeaturePruner(Protocol):
    pruner_id: str
    fit_mode: FitMode

    def fit(self, bundle: "FeatureBundle", y: pd.Series | None) -> FittedFeaturePruner: ...


@dataclass(frozen=True)
class FeatureBundle:
    blocks: tuple[FeatureBlock, ...]

    @property
    def dense_blocks(self) -> tuple[FeatureBlock, ...]:
        return tuple(block for block in self.blocks if block.output_kind == "dense_numeric")

    @property
    def sparse_blocks(self) -> tuple[FeatureBlock, ...]:
        return tuple(block for block in self.blocks if block.output_kind == "sparse_numeric")

    @property
    def native_blocks(self) -> tuple[FeatureBlock, ...]:
        return tuple(block for block in self.blocks if block.output_kind == "native_tabular")

    def dense_frame(self) -> pd.DataFrame:
        frames = [block.values for block in self.dense_blocks if isinstance(block.values, pd.DataFrame)]
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, axis=1)

    def native_frame(self) -> pd.DataFrame:
        frames = [block.values for block in self.native_blocks if isinstance(block.values, pd.DataFrame)]
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, axis=1)

    def sparse_matrix(self) -> sparse.csr_matrix | None:
        matrices = [block.values for block in self.sparse_blocks if sparse.issparse(block.values)]
        if not matrices:
            return None
        if len(matrices) == 1:
            return sparse.csr_matrix(matrices[0])
        return sparse.hstack(matrices, format="csr")

    def drop_dense_columns(self, column_names: set[str]) -> "FeatureBundle":
        if not column_names:
            return self

        next_blocks: list[FeatureBlock] = []
        for block in self.blocks:
            if block.output_kind != "dense_numeric" or not isinstance(block.values, pd.DataFrame):
                next_blocks.append(block)
                continue

            kept_columns = [column for column in block.values.columns if column not in column_names]
            if not kept_columns:
                continue
            next_blocks.append(
                FeatureBlock(
                    block_id=block.block_id,
                    output_kind=block.output_kind,
                    values=block.values.loc[:, kept_columns],
                    feature_names=tuple(kept_columns),
                )
            )

        return FeatureBundle(blocks=tuple(next_blocks))
