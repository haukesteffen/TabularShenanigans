import time
import traceback
from dataclasses import dataclass
from typing import Literal

from tabular_shenanigans.config import AppConfig
from tabular_shenanigans.data import CompetitionDatasetContext
from tabular_shenanigans.mlflow_store import candidate_run_exists
from tabular_shenanigans.train import run_training_workflow


@dataclass(frozen=True)
class CandidateBatchResult:
    candidate_index: int
    candidate_id: str
    candidate_type: str
    status: Literal["trained", "skipped", "failed"]
    run_id: str | None
    wall_seconds: float
    error: str | None = None


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
) -> TrainingBatchSummary:
    selected_candidate_indices = config.resolve_candidate_indices(
        candidate_id=candidate_id,
        index=index,
        require_explicit=False,
    )
    print(
        "Training batch starting: "
        f"selected_candidates={len(selected_candidate_indices)}, "
        f"configured_candidates={config.candidate_count}, "
        f"skip_existing={skip_existing}"
    )

    results: list[CandidateBatchResult] = []
    for batch_position, candidate_index in enumerate(selected_candidate_indices, start=1):
        candidate_config = config.with_candidate_index(candidate_index)
        candidate = candidate_config.experiment.candidate
        resolved_candidate_id = candidate_config.resolved_candidate_id
        print(
            f"[{batch_position}/{len(selected_candidate_indices)}] "
            f"candidate_index={candidate_index + 1}, "
            f"candidate_id={resolved_candidate_id}, "
            f"candidate_type={candidate.candidate_type}"
        )

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

        started = time.perf_counter()
        try:
            candidate_run = run_training_workflow(
                config=candidate_config,
                dataset_context=dataset_context,
            )
        except Exception as exc:
            wall_seconds = time.perf_counter() - started
            print(
                "Candidate failed: "
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
        print(
            "Candidate complete: "
            f"candidate_index={candidate_index + 1}, "
            f"candidate_id={resolved_candidate_id}, "
            f"mlflow_run_id={candidate_run.run_id}, "
            f"wall_seconds={wall_seconds:.2f}"
        )
        results.append(
            CandidateBatchResult(
                candidate_index=candidate_index + 1,
                candidate_id=resolved_candidate_id,
                candidate_type=candidate.candidate_type,
                status="trained",
                run_id=candidate_run.run_id,
                wall_seconds=wall_seconds,
            )
        )

    return TrainingBatchSummary(results=results)


def print_training_batch_summary(summary: TrainingBatchSummary) -> None:
    print(
        "Training batch summary: "
        f"total={summary.total_candidates}, "
        f"trained={summary.trained_count}, "
        f"skipped={summary.skipped_count}, "
        f"failed={summary.failed_count}"
    )
    for result in summary.results:
        summary_line = (
            f"candidate_index={result.candidate_index}, "
            f"candidate_id={result.candidate_id}, "
            f"candidate_type={result.candidate_type}, "
            f"status={result.status}, "
            f"wall_seconds={result.wall_seconds:.2f}"
        )
        if result.run_id is not None:
            summary_line = f"{summary_line}, mlflow_run_id={result.run_id}"
        if result.error is not None:
            summary_line = f"{summary_line}, error={result.error}"
        print(summary_line)
