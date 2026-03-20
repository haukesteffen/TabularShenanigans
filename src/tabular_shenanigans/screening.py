from pathlib import Path
import time
import tempfile
import traceback
from dataclasses import dataclass
from typing import Literal

import yaml

from tabular_shenanigans.config import AppConfig
from tabular_shenanigans.cv import is_higher_better
from tabular_shenanigans.data import CompetitionDatasetContext
from tabular_shenanigans.mlflow_store import download_candidate_manifest
from tabular_shenanigans.train import run_training_workflow


@dataclass(frozen=True)
class ScreeningBatchResult:
    candidate_index: int
    candidate_id: str
    status: Literal["screened", "failed"]
    run_id: str | None
    wall_seconds: float
    metric_mean: float | None = None
    metric_std: float | None = None
    error: str | None = None


@dataclass(frozen=True)
class ScreeningBatchSummary:
    results: list[ScreeningBatchResult]

    @property
    def total_candidates(self) -> int:
        return len(self.results)

    @property
    def screened_count(self) -> int:
        return sum(result.status == "screened" for result in self.results)

    @property
    def failed_count(self) -> int:
        return sum(result.status == "failed" for result in self.results)


def _metric_from_manifest(manifest: dict[str, object]) -> tuple[float, float]:
    cv_summary = manifest.get("cv_summary")
    if not isinstance(cv_summary, dict):
        raise ValueError("Candidate manifest cv_summary must be a mapping.")
    return float(cv_summary["metric_mean"]), float(cv_summary["metric_std"])


def _print_screening_promotion_snippet(config: AppConfig, summary: ScreeningBatchSummary) -> None:
    if config.screening is None:
        raise ValueError("screening is not configured in config.yaml.")
    successful_results = [result for result in summary.results if result.status == "screened"]
    if not successful_results:
        print("Screening promotion summary skipped: no successful screening runs.")
        return

    reverse_sort = is_higher_better(config.competition.primary_metric)
    ranked_results = sorted(
        successful_results,
        key=lambda result: result.metric_mean if result.metric_mean is not None else float("-inf"),
        reverse=reverse_sort,
    )
    top_results = ranked_results[: min(config.screening.promote_top_k, len(ranked_results))]

    print("Screening ranking:")
    for rank, result in enumerate(top_results, start=1):
        print(
            f"  [{rank}] candidate_index={result.candidate_index}, "
            f"candidate_id={result.candidate_id}, "
            f"{config.competition.primary_metric}={result.metric_mean:.6f}, "
            f"std={result.metric_std:.6f}, "
            f"mlflow_run_id={result.run_id}"
        )

    promoted_candidates = [
        config.get_screening_candidate(result.candidate_index - 1).model_dump(mode="python", exclude_none=True)
        for result in top_results
    ]
    snippet = yaml.safe_dump(
        {"experiment": {"candidates": promoted_candidates}},
        sort_keys=False,
        default_flow_style=False,
    ).strip()
    print("Suggested canonical candidate snippet:")
    print(snippet)


def run_screening_batch(
    config: AppConfig,
    dataset_context: CompetitionDatasetContext,
    candidate_id: str | None = None,
    index: int | None = None,
) -> ScreeningBatchSummary:
    if config.screening is None:
        raise ValueError("screening is not configured in config.yaml.")

    selected_candidate_indices = config.resolve_screening_candidate_indices(
        candidate_id=candidate_id,
        index=index,
        require_explicit=False,
    )
    print(
        "Screening batch starting: "
        f"selected_candidates={len(selected_candidate_indices)}, "
        f"configured_candidates={config.screening_candidate_count}, "
        f"promote_top_k={config.screening.promote_top_k}"
    )

    results: list[ScreeningBatchResult] = []
    for batch_position, candidate_index in enumerate(selected_candidate_indices, start=1):
        screening_config = config.with_screening_candidate_index(candidate_index)
        resolved_candidate_id = screening_config.resolved_candidate_id
        print(
            f"[{batch_position}/{len(selected_candidate_indices)}] "
            f"screening_candidate_index={candidate_index + 1}, "
            f"candidate_id={resolved_candidate_id}"
        )

        started = time.perf_counter()
        try:
            candidate_run = run_training_workflow(
                config=screening_config,
                dataset_context=dataset_context,
            )
            with tempfile.TemporaryDirectory(prefix="tabular-shenanigans-screening-manifest-") as temp_dir:
                manifest = download_candidate_manifest(
                    config=screening_config,
                    run_id=candidate_run.run_id,
                    destination_dir=Path(temp_dir),
                )
            metric_mean, metric_std = _metric_from_manifest(manifest)
        except Exception as exc:
            wall_seconds = time.perf_counter() - started
            print(
                "Screening candidate failed: "
                f"candidate_index={candidate_index + 1}, "
                f"candidate_id={resolved_candidate_id}, "
                f"error={exc}"
            )
            traceback.print_exc()
            results.append(
                ScreeningBatchResult(
                    candidate_index=candidate_index + 1,
                    candidate_id=resolved_candidate_id,
                    status="failed",
                    run_id=None,
                    wall_seconds=wall_seconds,
                    error=str(exc),
                )
            )
            continue

        wall_seconds = time.perf_counter() - started
        print(
            "Screening candidate complete: "
            f"candidate_index={candidate_index + 1}, "
            f"candidate_id={resolved_candidate_id}, "
            f"mlflow_run_id={candidate_run.run_id}, "
            f"{config.competition.primary_metric}={metric_mean:.6f}, "
            f"wall_seconds={wall_seconds:.2f}"
        )
        results.append(
            ScreeningBatchResult(
                candidate_index=candidate_index + 1,
                candidate_id=resolved_candidate_id,
                status="screened",
                run_id=candidate_run.run_id,
                wall_seconds=wall_seconds,
                metric_mean=metric_mean,
                metric_std=metric_std,
            )
        )

    return ScreeningBatchSummary(results=results)


def print_screening_batch_summary(config: AppConfig, summary: ScreeningBatchSummary) -> None:
    print(
        "Screening batch summary: "
        f"total={summary.total_candidates}, "
        f"screened={summary.screened_count}, "
        f"failed={summary.failed_count}"
    )
    for result in summary.results:
        summary_line = (
            f"candidate_index={result.candidate_index}, "
            f"candidate_id={result.candidate_id}, "
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
    _print_screening_promotion_snippet(config=config, summary=summary)
