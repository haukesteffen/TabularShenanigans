import argparse
import json
import subprocess
import sys
import time
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import yaml
from mlflow import MlflowClient

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "config.yaml"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "reports" / "benchmark_checkpoints"
sys.path.insert(0, str(REPO_ROOT / "src"))

from tabular_shenanigans.config import load_config


@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    model_family: str
    compute_target: str
    gpu_backend: str | None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the #183 GPU checkpoint benchmark matrix.")
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Directory where the benchmark session report and logs should be written.",
    )
    return parser


def _load_yaml(path: Path) -> dict[str, object]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a top-level mapping in {path}.")
    return payload


def _write_yaml(path: Path, payload: dict[str, object]) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _set_nested(payload: dict[str, object], path: list[str], value: object) -> None:
    cursor = payload
    for key in path[:-1]:
        next_value = cursor.get(key)
        if not isinstance(next_value, dict):
            next_value = {}
            cursor[key] = next_value
        cursor = next_value
    cursor[path[-1]] = value


def _build_case_config(
    base_config: dict[str, object],
    case: BenchmarkCase,
    tracking_uri: str,
) -> dict[str, object]:
    payload = json.loads(json.dumps(base_config))
    _set_nested(payload, ["experiment", "candidate", "candidate_type"], "model")
    _set_nested(payload, ["experiment", "candidate", "model_family"], case.model_family)
    _set_nested(payload, ["experiment", "candidate", "numeric_preprocessor"], "standardize")
    _set_nested(payload, ["experiment", "candidate", "categorical_preprocessor"], "frequency")
    _set_nested(payload, ["experiment", "candidate", "model_params"], {})
    _set_nested(payload, ["experiment", "candidate", "optimization", "enabled"], False)
    _set_nested(payload, ["experiment", "submit", "enabled"], False)
    _set_nested(payload, ["experiment", "tracking", "tracking_uri"], tracking_uri)
    _set_nested(payload, ["experiment", "runtime", "compute_target"], case.compute_target)
    if case.gpu_backend is None:
        with suppress(KeyError):
            experiment = payload["experiment"]
            if isinstance(experiment, dict):
                runtime = experiment.get("runtime")
                if isinstance(runtime, dict):
                    runtime.pop("gpu_backend", None)
    else:
        _set_nested(payload, ["experiment", "runtime", "gpu_backend"], case.gpu_backend)
    return payload


def _start_gpu_monitor(log_path: Path) -> subprocess.Popen[str] | None:
    command = [
        "nvidia-smi",
        "--query-gpu=timestamp,utilization.gpu,memory.used",
        "--format=csv,noheader,nounits",
        "-l",
        "1",
    ]
    try:
        handle = log_path.open("w", encoding="utf-8")
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        handle.close()
        return process
    except FileNotFoundError:
        return None


def _stop_gpu_monitor(process: subprocess.Popen[str] | None) -> None:
    if process is None:
        return
    process.terminate()
    with suppress(subprocess.TimeoutExpired):
        process.wait(timeout=5)
        return
    process.kill()
    process.wait(timeout=5)


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return payload


def _parse_gpu_samples(path: Path) -> dict[str, float | int | None]:
    if not path.exists():
        return {
            "sample_count": 0,
            "gpu_util_mean": None,
            "gpu_util_peak": None,
            "memory_used_peak_mb": None,
        }

    gpu_utils: list[float] = []
    memory_used: list[float] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 3:
            continue
        with suppress(ValueError):
            gpu_utils.append(float(parts[1]))
            memory_used.append(float(parts[2]))
    if not gpu_utils:
        return {
            "sample_count": 0,
            "gpu_util_mean": None,
            "gpu_util_peak": None,
            "memory_used_peak_mb": None,
        }
    return {
        "sample_count": len(gpu_utils),
        "gpu_util_mean": sum(gpu_utils) / len(gpu_utils),
        "gpu_util_peak": max(gpu_utils),
        "memory_used_peak_mb": max(memory_used) if memory_used else None,
    }


def _find_run(client: MlflowClient, experiment_name: str, candidate_id: str):
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' was not created.")
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.candidate_id = '{candidate_id}'",
        max_results=1,
    )
    if len(runs) != 1:
        raise ValueError(f"Expected one run for candidate_id '{candidate_id}', got {len(runs)}.")
    return runs[0]


def _download_manifest(client: MlflowClient, run_id: str, target_dir: Path) -> dict[str, object]:
    manifest_path = Path(
        client.download_artifacts(
            run_id=run_id,
            path="candidate/candidate.json",
            dst_path=str(target_dir),
        )
    )
    return _read_json(manifest_path)


def _format_float(value: object, digits: int = 3) -> str:
    if isinstance(value, int | float):
        return f"{float(value):.{digits}f}"
    return "-"


def _build_markdown_report(session_summary: dict[str, object]) -> str:
    rows = [
        "| Case | Full wall s | CV preprocess s | CV fit s | CV predict s | Artifact s | Other s | CV score | GPU util mean | GPU mem peak MB | Residency |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for case_result in session_summary["cases"]:
        runtime_profile = case_result.get("runtime_profile") or {}
        residency = runtime_profile.get("first_fold_residency") or {}
        train_residency = "-"
        if isinstance(residency, dict):
            train_processed = residency.get("train_processed")
            if isinstance(train_processed, dict):
                train_residency = str(train_processed.get("residency", "-"))
        rows.append(
            "| "
            + " | ".join(
                [
                    str(case_result["case_id"]),
                    _format_float(case_result.get("process_wall_seconds")),
                    _format_float(runtime_profile.get("cv_preprocess_wall_seconds")),
                    _format_float(runtime_profile.get("cv_fit_wall_seconds")),
                    _format_float(runtime_profile.get("cv_predict_wall_seconds")),
                    _format_float(runtime_profile.get("artifact_staging_wall_seconds")),
                    _format_float(case_result.get("other_wall_seconds_estimate")),
                    _format_float(case_result.get("cv_score_mean"), 6),
                    _format_float(case_result.get("gpu_util_mean")),
                    _format_float(case_result.get("memory_used_peak_mb"), 1),
                    train_residency,
                ]
            )
            + " |"
        )

    lines = [
        f"# GPU Benchmark Checkpoint",
        "",
        f"- Generated at: {session_summary['generated_at_utc']}",
        f"- Competition: {session_summary['competition_slug']}",
        f"- Feature recipe: {session_summary['feature_recipe_id']}",
        f"- Tracking URI root: `{session_summary['tracking_uri']}`",
        "",
        "## Results",
        "",
        *rows,
        "",
        "## Notes",
        "",
        "- `Full wall s` is the end-to-end subprocess wall time for `uv run python main.py train`.",
        "- `Other s` is the remainder after subtracting captured training-context, CV-stage, and artifact staging timings from the full subprocess wall time.",
        "- `Residency` reports the first fold train matrix residency after preprocessing and any explicit GPU-native promotion.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    args = build_parser().parse_args()
    output_root = Path(args.output_root)
    session_dir = output_root / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    session_dir.mkdir(parents=True, exist_ok=False)
    tracking_dir = session_dir / "mlruns"
    tracking_uri = tracking_dir.resolve().as_uri()

    base_config = _load_yaml(CONFIG_PATH)
    original_config_text = CONFIG_PATH.read_text(encoding="utf-8")
    base_loaded_config = load_config(str(CONFIG_PATH))
    cases = [
        BenchmarkCase("logreg_cpu", "logistic_regression", "cpu", None),
        BenchmarkCase("logreg_gpu_patch", "logistic_regression", "gpu", "patch"),
        BenchmarkCase("logreg_gpu_native", "logistic_regression", "gpu", "native"),
        BenchmarkCase("xgboost_cpu", "xgboost", "cpu", None),
        BenchmarkCase("xgboost_gpu_patch", "xgboost", "gpu", "patch"),
        BenchmarkCase("xgboost_gpu_native", "xgboost", "gpu", "native"),
    ]
    client = MlflowClient(tracking_uri=tracking_uri)
    session_cases: list[dict[str, object]] = []

    try:
        for case in cases:
            case_dir = session_dir / case.case_id
            case_dir.mkdir()
            case_config = _build_case_config(
                base_config=base_config,
                case=case,
                tracking_uri=tracking_uri,
            )
            _write_yaml(CONFIG_PATH, case_config)
            resolved_config = load_config(str(CONFIG_PATH))
            gpu_log_path = case_dir / "gpu_samples.csv"
            monitor = None
            if case.compute_target == "gpu":
                monitor = _start_gpu_monitor(gpu_log_path)

            stdout_path = case_dir / "train.stdout.log"
            stderr_path = case_dir / "train.stderr.log"
            started = time.perf_counter()
            try:
                with stdout_path.open("w", encoding="utf-8") as stdout_handle, stderr_path.open(
                    "w",
                    encoding="utf-8",
                ) as stderr_handle:
                    subprocess.run(
                        ["uv", "run", "python", "main.py", "train"],
                        cwd=REPO_ROOT,
                        stdout=stdout_handle,
                        stderr=stderr_handle,
                        text=True,
                        check=True,
                    )
            finally:
                _stop_gpu_monitor(monitor)
            process_wall_seconds = time.perf_counter() - started

            run = _find_run(
                client=client,
                experiment_name=resolved_config.competition.slug,
                candidate_id=resolved_config.resolved_candidate_id,
            )
            manifest = _download_manifest(client=client, run_id=run.info.run_id, target_dir=case_dir)
            runtime_profile = manifest.get("runtime_profile")
            if not isinstance(runtime_profile, dict):
                raise ValueError(f"Run for {case.case_id} did not produce runtime_profile.")
            gpu_metrics = _parse_gpu_samples(gpu_log_path)
            other_wall_seconds_estimate = (
                process_wall_seconds
                - float(runtime_profile.get("prepare_training_context_wall_seconds", 0.0))
                - float(runtime_profile.get("cv_stage_wall_seconds", 0.0))
                - float(runtime_profile.get("artifact_staging_wall_seconds", 0.0))
            )
            session_cases.append(
                {
                    "case_id": case.case_id,
                    "model_family": case.model_family,
                    "compute_target": case.compute_target,
                    "gpu_backend": case.gpu_backend,
                    "candidate_id": resolved_config.resolved_candidate_id,
                    "run_id": run.info.run_id,
                    "process_wall_seconds": process_wall_seconds,
                    "other_wall_seconds_estimate": other_wall_seconds_estimate,
                    "cv_score_mean": run.data.metrics.get("cv_score_mean"),
                    "runtime_profile": runtime_profile,
                    **gpu_metrics,
                }
            )
    finally:
        CONFIG_PATH.write_text(original_config_text, encoding="utf-8")

    session_summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "competition_slug": base_loaded_config.competition.slug,
        "feature_recipe_id": base_loaded_config.experiment.candidate.feature_recipe_id,
        "tracking_uri": tracking_uri,
        "cases": session_cases,
    }
    (session_dir / "benchmark_summary.json").write_text(
        json.dumps(session_summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (session_dir / "benchmark_report.md").write_text(
        _build_markdown_report(session_summary),
        encoding="utf-8",
    )
    print(json.dumps(session_summary, indent=2, sort_keys=True))
    print(f"Benchmark report: {session_dir / 'benchmark_report.md'}")


if __name__ == "__main__":
    main()
