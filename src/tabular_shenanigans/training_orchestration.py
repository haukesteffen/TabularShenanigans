import time
import traceback
from dataclasses import dataclass
from typing import Literal

from tabular_shenanigans.config import AppConfig, BlendCandidateConfig, ModelCandidateConfig
from tabular_shenanigans.data import CompetitionDatasetContext
from tabular_shenanigans.mlflow_store import candidate_run_exists
from tabular_shenanigans.train import run_training_workflow


@dataclass(frozen=True)
class CandidateBatchResult:
    candidate_index: int
    candidate_id: str
    candidate_type: str
    status: Literal["trained", "screened", "skipped", "failed"]
    run_id: str | None
    wall_seconds: float
    error: str | None = None
    metric_mean: float | None = None
    metric_std: float | None = None


@dataclass(frozen=True)
class TrainingBatchSummary:
    results: list[CandidateBatchResult]

    @property
    def total_candidates(self) -> int:
        return len(self.results)

    @property
    def trained_count(self) -> int:
        return sum(result.status == "trained" for result in self.results)

    @property
    def screened_count(self) -> int:
        return sum(result.status == "screened" for result in self.results)

    @property
    def skipped_count(self) -> int:
        return sum(result.status == "skipped" for result in self.results)

    @property
    def failed_count(self) -> int:
        return sum(result.status == "failed" for result in self.results)


def run_training_batch(
    config: AppConfig,
    dataset_context: CompetitionDatasetContext,
    candidate_id: str | None = None,
    index: int | None = None,
    skip_existing: bool = False,
    screening: bool = False,
) -> TrainingBatchSummary:
    selected_candidate_indices = config.resolve_candidate_indices(
        candidate_id=candidate_id,
        index=index,
        require_explicit=False,
    )

    if screening:
        explicit_selection = candidate_id is not None or index is not None
        if explicit_selection:
            for candidate_index in selected_candidate_indices:
                candidate = config.get_candidate(candidate_index)
                if isinstance(candidate, BlendCandidateConfig):
                    raise ValueError(
                        f"Blend candidates are not supported in screening mode. "
                        f"candidate_index={candidate_index + 1} is a blend candidate."
                    )
        else:
            model_indices = []
            for candidate_index in selected_candidate_indices:
                candidate = config.get_candidate(candidate_index)
                if isinstance(candidate, BlendCandidateConfig):
                    print(
                        f"Skipping candidate_index={candidate_index + 1}: "
                        "blend candidates are not supported in screening mode."
                    )
                    continue
                model_indices.append(candidate_index)
            selected_candidate_indices = model_indices

    mode_label = "Screening" if screening else "Training"
    print(
        f"{mode_label} batch starting: "
        f"selected_candidates={len(selected_candidate_indices)}, "
        f"configured_candidates={config.candidate_count}, "
        f"skip_existing={skip_existing}"
    )

    results: list[CandidateBatchResult] = []
    for batch_position, candidate_index in enumerate(selected_candidate_indices, start=1):
        candidate_config = config.with_candidate_index(candidate_index, screening=screening)
        candidate = candidate_config.experiment.candidate
        resolved_candidate_id = candidate_config.resolved_candidate_id

        optimization_override = None
        if screening and isinstance(candidate, ModelCandidateConfig):
            if candidate.optimization is None and config.screening is not None and config.screening.optimization is not None:
                optimization_override = config.screening.optimization

        print(
            f"[{batch_position}/{len(selected_candidate_indices)}] "
            f"candidate_index={candidate_index + 1}, "
            f"candidate_id={resolved_candidate_id}, "
            f"candidate_type={candidate.candidate_type}"
        )

        started = time.perf_counter()
        try:
            if skip_existing and candidate_run_exists(candidate_config, resolved_candidate_id):
                print(
                    "Candidate skipped: "
                    f"candidate_index={candidate_index + 1}, "
                    f"candidate_id={resolved_candidate_id}, "
                    "reason=existing_mlflow_run"
                )
                results.append(
                    CandidateBatchResult(
                        candidate_index=candidate_index + 1,
                        candidate_id=resolved_candidate_id,
                        candidate_type=candidate.candidate_type,
                        status="skipped",
                        run_id=None,
                        wall_seconds=0.0,
                    )
                )
                continue
            candidate_run = run_training_workflow(
                config=candidate_config,
                dataset_context=dataset_context,
                optimization_override=optimization_override,
            )
        except Exception as exc:
            wall_seconds = time.perf_counter() - started
            print(
                f"{mode_label} candidate failed: "
                f"candidate_index={candidate_index + 1}, "
                f"candidate_id={resolved_candidate_id}, "
                f"error={exc}"
            )
            traceback.print_exc()
            results.append(
                CandidateBatchResult(
                    candidate_index=candidate_index + 1,
                    candidate_id=resolved_candidate_id,
                    candidate_type=candidate.candidate_type,
                    status="failed",
                    run_id=None,
                    wall_seconds=wall_seconds,
                    error=str(exc),
                )
            )
            continue

        wall_seconds = time.perf_counter() - started

        metric_mean = candidate_run.metric_mean
        metric_std = candidate_run.metric_std

        status = "screened" if screening else "trained"
        completion_line = (
            f"{mode_label} candidate complete: "
            f"candidate_index={candidate_index + 1}, "
            f"candidate_id={resolved_candidate_id}, "
            f"mlflow_run_id={candidate_run.run_id}, "
            f"wall_seconds={wall_seconds:.2f}"
        )
        if metric_mean is not None:
            completion_line = (
                f"{completion_line}, "
                f"{config.competition.primary_metric}={metric_mean:.6f}"
            )
        print(completion_line)

        results.append(
            CandidateBatchResult(
                candidate_index=candidate_index + 1,
                candidate_id=resolved_candidate_id,
                candidate_type=candidate.candidate_type,
                status=status,
                run_id=candidate_run.run_id,
                wall_seconds=wall_seconds,
                metric_mean=metric_mean,
                metric_std=metric_std,
            )
        )

    return TrainingBatchSummary(results=results)


def print_training_batch_summary(
    config: AppConfig,
    summary: TrainingBatchSummary,
    screening: bool = False,
) -> None:
    mode_label = "Screening" if screening else "Training"
    counts = (
        f"total={summary.total_candidates}, "
        f"{'screened' if screening else 'trained'}="
        f"{summary.screened_count if screening else summary.trained_count}, "
        f"skipped={summary.skipped_count}, "
        f"failed={summary.failed_count}"
    )
    print(f"{mode_label} batch summary: {counts}")
    for result in summary.results:
        summary_line = (
            f"candidate_index={result.candidate_index}, "
            f"candidate_id={result.candidate_id}, "
            f"candidate_type={result.candidate_type}, "
            f"status={result.status}, "
            f"wall_seconds={result.wall_seconds:.2f}"
        )
        if result.metric_mean is not None and result.metric_std is not None:
            summary_line = (
                f"{summary_line}, "
                f"{config.competition.primary_metric}={result.metric_mean:.6f}, "
                f"std={result.metric_std:.6f}"
            )
        if result.run_id is not None:
            summary_line = f"{summary_line}, mlflow_run_id={result.run_id}"
        if result.error is not None:
            summary_line = f"{summary_line}, error={result.error}"
        print(summary_line)
