import argparse
import json
import platform
import shutil
import subprocess
import sys
import time
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path

import yaml
from mlflow import MlflowClient

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "config.yaml"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "reports" / "gpu_target_validation"
DEFAULT_BINARY_TEMPLATE_PATH = REPO_ROOT / "config.binary.example.yaml"
DEFAULT_REGRESSION_TEMPLATE_PATH = REPO_ROOT / "config.regression.example.yaml"
REQUIRED_ARTIFACT_PATHS = frozenset(
    {
        "candidate",
        "candidate/candidate.json",
        "candidate/fold_metrics.csv",
        "candidate/oof_predictions.csv",
        "candidate/test_predictions.csv",
        "config",
        "config/runtime_config.json",
        "context",
        "context/competition.json",
        "context/folds.csv",
        "logs",
        "logs/runtime.log",
    }
)
PACKAGE_VERSION_NAMES = (
    "numpy",
    "pandas",
    "scikit-learn",
    "mlflow-skinny",
    "cudf-cu12",
    "cuml-cu12",
    "xgboost",
    "catboost",
    "lightgbm",
)
sys.path.insert(0, str(REPO_ROOT / "src"))


def _find_uv() -> str:
    uv_path = shutil.which("uv")
    if uv_path:
        return uv_path
    for candidate in [Path.home() / ".local/bin/uv", Path("/usr/local/bin/uv")]:
        if candidate.exists():
            return str(candidate)
    raise FileNotFoundError("uv not found on PATH or in ~/.local/bin. Install from https://docs.astral.sh/uv/")

from tabular_shenanigans.config import load_config
from tabular_shenanigans.runtime_execution import (
    NATIVE_GPU_BACKEND,
    PATCH_GPU_BACKEND,
    detect_runtime_capabilities,
)


@dataclass(frozen=True)
class ValidationCase:
    case_id: str
    template_config_path: Path
    model_family: str
    numeric_preprocessor: str
    categorical_preprocessor: str
    gpu_backend: str
    expected_preprocessing_backend: str
    notes: str

    @property
    def expected_resolved_gpu_backend(self) -> str:
        return NATIVE_GPU_BACKEND if self.gpu_backend == "native" else PATCH_GPU_BACKEND

    @property
    def expected_rapids_hooks_installed(self) -> bool:
        return self.expected_resolved_gpu_backend == PATCH_GPU_BACKEND


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the #193 target-host GPU smoke validation matrix."
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Directory where the timestamped validation session should be written.",
    )
    parser.add_argument(
        "--cases",
        default=None,
        help="Optional comma-separated case IDs to run. Defaults to the full target-host matrix.",
    )
    parser.add_argument(
        "--cv-splits",
        type=int,
        default=3,
        help="CV split count for the smoke session. Defaults to 3 for faster target-host validation.",
    )
    parser.add_argument(
        "--skip-sync",
        action="store_true",
        help="Skip `uv sync --extra boosters --extra gpu` before running the matrix.",
    )
    parser.add_argument(
        "--skip-lightgbm-install",
        action="store_true",
        help="Skip `./scripts/install_lightgbm_cuda.sh` before running the matrix.",
    )
    return parser


def _default_cases() -> list[ValidationCase]:
    return [
        ValidationCase(
            case_id="binary_logistic_native_frequency_standardize",
            template_config_path=DEFAULT_BINARY_TEMPLATE_PATH,
            model_family="logistic_regression",
            numeric_preprocessor="standardize",
            categorical_preprocessor="frequency",
            gpu_backend="native",
            expected_preprocessing_backend="gpu_native_frequency",
            notes="Validates explicit gpu_native logistic regression on the repo-owned frequency backend.",
        ),
        ValidationCase(
            case_id="binary_logistic_patch_frequency_kbins",
            template_config_path=DEFAULT_BINARY_TEMPLATE_PATH,
            model_family="logistic_regression",
            numeric_preprocessor="kbins",
            categorical_preprocessor="frequency",
            gpu_backend="patch",
            expected_preprocessing_backend="gpu_patch",
            notes="Validates the retained RAPIDS hook path for logistic regression.",
        ),
        ValidationCase(
            case_id="binary_xgboost_native_frequency_standardize",
            template_config_path=DEFAULT_BINARY_TEMPLATE_PATH,
            model_family="xgboost",
            numeric_preprocessor="standardize",
            categorical_preprocessor="frequency",
            gpu_backend="native",
            expected_preprocessing_backend="gpu_native_frequency",
            notes="Validates the explicit gpu_native XGBoost path.",
        ),
        ValidationCase(
            case_id="binary_xgboost_patch_ordinal_kbins",
            template_config_path=DEFAULT_BINARY_TEMPLATE_PATH,
            model_family="xgboost",
            numeric_preprocessor="kbins",
            categorical_preprocessor="ordinal",
            gpu_backend="patch",
            expected_preprocessing_backend="gpu_patch",
            notes="Validates the remaining registered GPU patch XGBoost tuple.",
        ),
        ValidationCase(
            case_id="binary_catboost_native_native_median",
            template_config_path=DEFAULT_BINARY_TEMPLATE_PATH,
            model_family="catboost",
            numeric_preprocessor="median",
            categorical_preprocessor="native",
            gpu_backend="native",
            expected_preprocessing_backend="cpu_native_frame",
            notes="Validates explicit gpu_native CatBoost with native categorical preprocessing.",
        ),
        ValidationCase(
            case_id="regression_ridge_native_frequency_standardize",
            template_config_path=DEFAULT_REGRESSION_TEMPLATE_PATH,
            model_family="ridge",
            numeric_preprocessor="standardize",
            categorical_preprocessor="frequency",
            gpu_backend="native",
            expected_preprocessing_backend="gpu_native_frequency",
            notes="Validates the explicit cuML ridge backend from #177.",
        ),
        ValidationCase(
            case_id="regression_elasticnet_native_frequency_standardize",
            template_config_path=DEFAULT_REGRESSION_TEMPLATE_PATH,
            model_family="elasticnet",
            numeric_preprocessor="standardize",
            categorical_preprocessor="frequency",
            gpu_backend="native",
            expected_preprocessing_backend="gpu_native_frequency",
            notes="Validates the explicit cuML elasticnet backend from #177.",
        ),
        ValidationCase(
            case_id="binary_random_forest_native_onehot_kbins",
            template_config_path=DEFAULT_BINARY_TEMPLATE_PATH,
            model_family="random_forest",
            numeric_preprocessor="kbins",
            categorical_preprocessor="onehot",
            gpu_backend="native",
            expected_preprocessing_backend="gpu_cuml",
            notes="Validates gpu_native random forest through the dense cuML preprocessing path.",
        ),
        ValidationCase(
            case_id="regression_random_forest_native_frequency_standardize",
            template_config_path=DEFAULT_REGRESSION_TEMPLATE_PATH,
            model_family="random_forest",
            numeric_preprocessor="standardize",
            categorical_preprocessor="frequency",
            gpu_backend="native",
            expected_preprocessing_backend="gpu_native_frequency",
            notes="Validates gpu_native random forest through the frequency preprocessing path.",
        ),
        ValidationCase(
            case_id="regression_lightgbm_native_onehot_median",
            template_config_path=DEFAULT_REGRESSION_TEMPLATE_PATH,
            model_family="lightgbm",
            numeric_preprocessor="median",
            categorical_preprocessor="onehot",
            gpu_backend="native",
            expected_preprocessing_backend="cpu_sklearn",
            notes="Validates the explicit LightGBM CUDA adapter on the sparse onehot boundary.",
        ),
        # #205: linear models with onehot categorical
        ValidationCase(
            case_id="binary_logistic_native_onehot_standardize",
            template_config_path=DEFAULT_BINARY_TEMPLATE_PATH,
            model_family="logistic_regression",
            numeric_preprocessor="standardize",
            categorical_preprocessor="onehot",
            gpu_backend="native",
            expected_preprocessing_backend="gpu_cuml",
            notes="Validates #205 — logistic regression onehot registered in GPU_SUPPORT_REGISTRY.",
        ),
        ValidationCase(
            case_id="regression_ridge_native_onehot_kbins",
            template_config_path=DEFAULT_REGRESSION_TEMPLATE_PATH,
            model_family="ridge",
            numeric_preprocessor="kbins",
            categorical_preprocessor="onehot",
            gpu_backend="native",
            expected_preprocessing_backend="gpu_cuml",
            notes="Validates #205 — ridge onehot registered in GPU_SUPPORT_REGISTRY.",
        ),
        ValidationCase(
            case_id="regression_elasticnet_native_onehot_median",
            template_config_path=DEFAULT_REGRESSION_TEMPLATE_PATH,
            model_family="elasticnet",
            numeric_preprocessor="median",
            categorical_preprocessor="onehot",
            gpu_backend="native",
            expected_preprocessing_backend="gpu_cuml",
            notes="Validates #205 — elasticnet onehot registered in GPU_SUPPORT_REGISTRY.",
        ),
        # #206 xgboost+onehot cases intentionally omitted: registry entries will be removed
        # in #209 (sparse CSR / XGBoost GPU incompatibility).
    ]


def _resolve_cases(case_filter: str | None) -> list[ValidationCase]:
    cases_by_id = {case.case_id: case for case in _default_cases()}
    if case_filter is None:
        return list(cases_by_id.values())

    requested_case_ids = [case_id.strip() for case_id in case_filter.split(",") if case_id.strip()]
    if not requested_case_ids:
        raise ValueError("--cases must include at least one non-empty case ID.")

    unknown_case_ids = [case_id for case_id in requested_case_ids if case_id not in cases_by_id]
    if unknown_case_ids:
        raise ValueError(
            f"Unknown validation case IDs: {unknown_case_ids}. Supported case IDs: {sorted(cases_by_id)}"
        )
    return [cases_by_id[case_id] for case_id in requested_case_ids]


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
    *,
    base_config: dict[str, object],
    case: ValidationCase,
    tracking_uri: str,
    cv_splits: int,
) -> dict[str, object]:
    payload = json.loads(json.dumps(base_config))
    _set_nested(payload, ["competition", "cv", "n_splits"], cv_splits)
    _set_nested(payload, ["experiment", "tracking", "tracking_uri"], tracking_uri)
    _set_nested(payload, ["experiment", "submit", "enabled"], False)
    _set_nested(payload, ["experiment", "runtime", "compute_target"], "gpu")
    _set_nested(payload, ["experiment", "runtime", "gpu_backend"], case.gpu_backend)
    _set_nested(
        payload,
        ["experiment", "candidates"],
        [
            {
                "candidate_type": "model",
                "feature_recipe_id": "fr0",
                "model_family": case.model_family,
                "numeric_preprocessor": case.numeric_preprocessor,
                "categorical_preprocessor": case.categorical_preprocessor,
                "model_params": {},
                "optimization": {"enabled": False},
            }
        ],
    )
    return payload


def _run_command(
    *,
    command: list[str],
    stdout_path: Path,
    stderr_path: Path,
) -> dict[str, object]:
    started = time.perf_counter()
    with stdout_path.open("w", encoding="utf-8") as stdout_handle, stderr_path.open(
        "w",
        encoding="utf-8",
    ) as stderr_handle:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            stdout=stdout_handle,
            stderr=stderr_handle,
            text=True,
            check=False,
        )
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "wall_seconds": time.perf_counter() - started,
    }


def _capture_command(command: list[str]) -> dict[str, object]:
    try:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError as exc:
        return {
            "command": command,
            "returncode": 127,
            "stdout": "",
            "stderr": str(exc),
        }

    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def _package_version(name: str) -> str | None:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return None


def _collect_environment_snapshot() -> dict[str, object]:
    capabilities = detect_runtime_capabilities()
    return {
        "validation_path": "repo_owned_install",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python_version": platform.python_version(),
        },
        "runtime_capabilities": capabilities.to_dict(),
        "package_versions": {name: _package_version(name) for name in PACKAGE_VERSION_NAMES},
        "uv_version": _capture_command([_find_uv(), "--version"]),
        "uname": _capture_command(["uname", "-a"]),
        "os_release": {
            "path": "/etc/os-release",
            "content": Path("/etc/os-release").read_text(encoding="utf-8")
            if Path("/etc/os-release").exists()
            else None,
        },
        "nvidia_smi_summary": _capture_command(
            [
                "nvidia-smi",
                "--query-gpu=index,name,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ]
        ),
        "nvidia_smi_full": _capture_command(["nvidia-smi"]),
    }


def _start_gpu_monitor(log_path: Path) -> subprocess.Popen[str] | None:
    command = [
        "nvidia-smi",
        "--query-gpu=timestamp,utilization.gpu,memory.used",
        "--format=csv,noheader,nounits",
        "-lms",
        "500",
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


def _collect_artifact_paths(client: MlflowClient, run_id: str, artifact_path: str = "") -> set[str]:
    collected: set[str] = set()
    for entry in client.list_artifacts(run_id, artifact_path):
        collected.add(entry.path)
        if entry.is_dir:
            collected.update(_collect_artifact_paths(client, run_id, entry.path))
    return collected


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return payload


def _download_artifact(client: MlflowClient, run_id: str, artifact_path: str, target_dir: Path) -> Path:
    return Path(client.download_artifacts(run_id=run_id, path=artifact_path, dst_path=str(target_dir)))


def _count_csv_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        row_count = sum(1 for _ in handle)
    return max(row_count - 1, 0)


def _validate_case_result(
    *,
    case: ValidationCase,
    run,
    manifest: dict[str, object],
    artifact_paths: set[str],
    runtime_log_path: Path,
    oof_predictions_path: Path,
    test_predictions_path: Path,
) -> list[str]:
    failures: list[str] = []
    tags = run.data.tags
    params = run.data.params
    metrics = run.data.metrics

    if run.info.status != "FINISHED":
        failures.append(f"MLflow run status is {run.info.status}, expected FINISHED.")

    if tags.get("runtime_resolved_compute_target") != "gpu":
        failures.append(
            "MLflow tag runtime_resolved_compute_target "
            f"is {tags.get('runtime_resolved_compute_target')!r}, expected 'gpu'."
        )

    if tags.get("runtime_resolved_gpu_backend") != case.expected_resolved_gpu_backend:
        failures.append(
            "MLflow tag runtime_resolved_gpu_backend "
            f"is {tags.get('runtime_resolved_gpu_backend')!r}, "
            f"expected {case.expected_resolved_gpu_backend!r}."
        )

    if tags.get("runtime_preprocessing_backend") != case.expected_preprocessing_backend:
        failures.append(
            "MLflow tag runtime_preprocessing_backend "
            f"is {tags.get('runtime_preprocessing_backend')!r}, "
            f"expected {case.expected_preprocessing_backend!r}."
        )

    if params.get("runtime__gpu_available") != "True":
        failures.append(
            f"MLflow param runtime__gpu_available is {params.get('runtime__gpu_available')!r}, expected 'True'."
        )

    if params.get("runtime__resolved_gpu_backend") != case.expected_resolved_gpu_backend:
        failures.append(
            "MLflow param runtime__resolved_gpu_backend "
            f"is {params.get('runtime__resolved_gpu_backend')!r}, "
            f"expected {case.expected_resolved_gpu_backend!r}."
        )

    if params.get("runtime__preprocessing_backend") != case.expected_preprocessing_backend:
        failures.append(
            "MLflow param runtime__preprocessing_backend "
            f"is {params.get('runtime__preprocessing_backend')!r}, "
            f"expected {case.expected_preprocessing_backend!r}."
        )

    expected_hooks_value = str(case.expected_rapids_hooks_installed)
    if params.get("runtime__rapids_hooks_installed") != expected_hooks_value:
        failures.append(
            "MLflow param runtime__rapids_hooks_installed "
            f"is {params.get('runtime__rapids_hooks_installed')!r}, "
            f"expected {expected_hooks_value!r}."
        )

    missing_artifacts = sorted(REQUIRED_ARTIFACT_PATHS - artifact_paths)
    if missing_artifacts:
        failures.append(f"Missing required MLflow artifacts: {missing_artifacts}")

    if not runtime_log_path.exists() or runtime_log_path.stat().st_size == 0:
        failures.append("Downloaded logs/runtime.log is missing or empty.")

    if _count_csv_rows(oof_predictions_path) == 0:
        failures.append("candidate/oof_predictions.csv did not contain any data rows.")

    if _count_csv_rows(test_predictions_path) == 0:
        failures.append("candidate/test_predictions.csv did not contain any data rows.")

    runtime_execution = manifest.get("runtime_execution")
    if not isinstance(runtime_execution, dict):
        failures.append("candidate manifest did not contain runtime_execution metadata.")
    else:
        if runtime_execution.get("resolved_compute_target") != "gpu":
            failures.append(
                "candidate manifest runtime_execution.resolved_compute_target "
                f"is {runtime_execution.get('resolved_compute_target')!r}, expected 'gpu'."
            )
        if runtime_execution.get("resolved_gpu_backend") != case.expected_resolved_gpu_backend:
            failures.append(
                "candidate manifest runtime_execution.resolved_gpu_backend "
                f"is {runtime_execution.get('resolved_gpu_backend')!r}, "
                f"expected {case.expected_resolved_gpu_backend!r}."
            )
        if runtime_execution.get("rapids_hooks_installed") != case.expected_rapids_hooks_installed:
            failures.append(
                "candidate manifest runtime_execution.rapids_hooks_installed "
                f"is {runtime_execution.get('rapids_hooks_installed')!r}, "
                f"expected {case.expected_rapids_hooks_installed!r}."
            )

    if manifest.get("preprocessing_backend") != case.expected_preprocessing_backend:
        failures.append(
            "candidate manifest preprocessing_backend "
            f"is {manifest.get('preprocessing_backend')!r}, "
            f"expected {case.expected_preprocessing_backend!r}."
        )

    runtime_profile = manifest.get("runtime_profile")
    if not isinstance(runtime_profile, dict):
        failures.append("candidate manifest did not contain runtime_profile.")

    if metrics.get("cv_score_mean") is None:
        failures.append("MLflow metric cv_score_mean is missing.")

    return failures


def _build_bootstrap_result(
    *,
    step_id: str,
    skipped: bool,
    command_result: dict[str, object] | None = None,
) -> dict[str, object]:
    result = {
        "step_id": step_id,
        "status": "skipped" if skipped else "passed",
        "command": None,
        "returncode": None,
        "stdout_path": None,
        "stderr_path": None,
        "wall_seconds": None,
    }
    if command_result is None:
        return result
    result.update(command_result)
    result["status"] = "passed" if command_result["returncode"] == 0 else "failed"
    return result


def _format_float(value: object, digits: int = 3) -> str:
    if isinstance(value, int | float):
        return f"{float(value):.{digits}f}"
    return "-"


def _build_markdown_report(session_summary: dict[str, object]) -> str:
    bootstrap_rows = [
        "| Step | Status | Wall s |",
        "| --- | --- | ---: |",
    ]
    for bootstrap_result in session_summary["bootstrap"]:
        bootstrap_rows.append(
            "| "
            + " | ".join(
                [
                    str(bootstrap_result["step_id"]),
                    str(bootstrap_result["status"]),
                    _format_float(bootstrap_result.get("wall_seconds")),
                ]
            )
            + " |"
        )

    case_rows = [
        "| Case | Status | Competition | Model | GPU backend | Preprocessing | CV score | GPU util mean | GPU mem peak MB |",
        "| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: |",
    ]
    for case_result in session_summary["cases"]:
        case_rows.append(
            "| "
            + " | ".join(
                [
                    str(case_result["case_id"]),
                    str(case_result["status"]),
                    str(case_result.get("competition_slug", "-")),
                    str(case_result.get("model_family", "-")),
                    str(case_result.get("resolved_gpu_backend", "-")),
                    str(case_result.get("resolved_preprocessing_backend", "-")),
                    _format_float(case_result.get("cv_score_mean"), 6),
                    _format_float(case_result.get("gpu_util_mean")),
                    _format_float(case_result.get("memory_used_peak_mb"), 1),
                ]
            )
            + " |"
        )

    environment = session_summary.get("environment") or {}
    runtime_capabilities = environment.get("runtime_capabilities") or {}
    package_versions = environment.get("package_versions") or {}
    nvidia_smi_summary = environment.get("nvidia_smi_summary") or {}

    lines = [
        "# Target GPU Smoke Validation",
        "",
        f"- Generated at: {session_summary['generated_at_utc']}",
        f"- Validation path: {environment.get('validation_path', 'repo_owned_install')}",
        f"- Tracking URI root: `{session_summary['tracking_uri']}`",
        f"- Output directory: `{session_summary['session_dir']}`",
        "",
        "## Bootstrap",
        "",
        *bootstrap_rows,
        "",
        "## Environment",
        "",
        f"- Platform: {environment.get('platform', {})}",
        f"- Runtime capabilities: {runtime_capabilities}",
        f"- Package versions: {package_versions}",
        f"- nvidia-smi summary: {nvidia_smi_summary.get('stdout', '-')}",
        "",
        "## Cases",
        "",
        *case_rows,
        "",
        "## Findings",
        "",
    ]

    finding_lines: list[str] = []
    if session_summary.get("fatal_error") is not None:
        finding_lines.append(f"- Fatal error: {session_summary['fatal_error']}")

    for bootstrap_result in session_summary["bootstrap"]:
        if bootstrap_result["status"] == "failed":
            finding_lines.append(
                "- Bootstrap step "
                f"`{bootstrap_result['step_id']}` failed with return code {bootstrap_result['returncode']}."
            )

    for case_result in session_summary["cases"]:
        failures = case_result.get("failures") or []
        if failures:
            finding_lines.append(f"- `{case_result['case_id']}`: {'; '.join(str(item) for item in failures)}")

    if not finding_lines:
        finding_lines.append("- No validation failures were recorded.")

    lines.extend(finding_lines)
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "- This report covers the target-host install/runtime smoke step for `#193` only.",
            "- It is intentionally separate from the broader parity/performance audit tracked in `#182`.",
            "- Until the Linux image path in `#173` lands, this validation runs against the repo-owned host install path.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = build_parser().parse_args()
    if args.cv_splits < 2:
        raise ValueError("--cv-splits must be >= 2.")

    output_root = Path(args.output_root)
    session_dir = output_root / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    session_dir.mkdir(parents=True, exist_ok=False)
    tracking_dir = session_dir / "mlruns"
    tracking_uri = tracking_dir.resolve().as_uri()
    cases = _resolve_cases(args.cases)
    client = MlflowClient(tracking_uri=tracking_uri)
    template_cache: dict[Path, dict[str, object]] = {}
    original_config_text = CONFIG_PATH.read_text(encoding="utf-8") if CONFIG_PATH.exists() else None
    fatal_error: str | None = None
    bootstrap_results: list[dict[str, object]] = []
    case_results: list[dict[str, object]] = []
    environment_snapshot: dict[str, object] | None = None

    try:
        bootstrap_dir = session_dir / "bootstrap"
        bootstrap_dir.mkdir()

        if args.skip_sync:
            bootstrap_results.append(_build_bootstrap_result(step_id="uv_sync", skipped=True))
        else:
            sync_result = _run_command(
                command=[_find_uv(), "sync", "--extra", "boosters", "--extra", "gpu"],
                stdout_path=bootstrap_dir / "uv_sync.stdout.log",
                stderr_path=bootstrap_dir / "uv_sync.stderr.log",
            )
            bootstrap_results.append(
                _build_bootstrap_result(step_id="uv_sync", skipped=False, command_result=sync_result)
            )
            if sync_result["returncode"] != 0:
                fatal_error = "uv sync failed; target-host validation could not continue."

        if fatal_error is None:
            if args.skip_lightgbm_install:
                bootstrap_results.append(
                    _build_bootstrap_result(step_id="lightgbm_cuda_install", skipped=True)
                )
            else:
                lightgbm_result = _run_command(
                    command=["bash", "scripts/install_lightgbm_cuda.sh"],
                    stdout_path=bootstrap_dir / "lightgbm_cuda_install.stdout.log",
                    stderr_path=bootstrap_dir / "lightgbm_cuda_install.stderr.log",
                )
                bootstrap_results.append(
                    _build_bootstrap_result(
                        step_id="lightgbm_cuda_install",
                        skipped=False,
                        command_result=lightgbm_result,
                    )
                )

        environment_snapshot = _collect_environment_snapshot()

        if fatal_error is None:
            for case in cases:
                case_dir = session_dir / case.case_id
                case_dir.mkdir()
                base_config = template_cache.get(case.template_config_path)
                if base_config is None:
                    base_config = _load_yaml(case.template_config_path)
                    template_cache[case.template_config_path] = base_config

                case_config = _build_case_config(
                    base_config=base_config,
                    case=case,
                    tracking_uri=tracking_uri,
                    cv_splits=args.cv_splits,
                )
                _write_yaml(CONFIG_PATH, case_config)
                resolved_config = load_config(str(CONFIG_PATH))
                gpu_log_path = case_dir / "gpu_samples.csv"
                monitor = _start_gpu_monitor(gpu_log_path)

                command_result: dict[str, object]
                started = time.perf_counter()
                try:
                    command_result = _run_command(
                        command=[_find_uv(), "run", "python", "main.py", "train"],
                        stdout_path=case_dir / "train.stdout.log",
                        stderr_path=case_dir / "train.stderr.log",
                    )
                finally:
                    _stop_gpu_monitor(monitor)

                gpu_metrics = _parse_gpu_samples(gpu_log_path)
                case_result = {
                    "case_id": case.case_id,
                    "notes": case.notes,
                    "status": "passed" if command_result["returncode"] == 0 else "failed",
                    "task_type": resolved_config.competition.task_type,
                    "competition_slug": resolved_config.competition.slug,
                    "feature_recipe_id": resolved_config.experiment.candidate.feature_recipe_id,
                    "model_family": case.model_family,
                    "numeric_preprocessor": case.numeric_preprocessor,
                    "categorical_preprocessor": case.categorical_preprocessor,
                    "requested_gpu_backend": case.gpu_backend,
                    "expected_resolved_gpu_backend": case.expected_resolved_gpu_backend,
                    "expected_preprocessing_backend": case.expected_preprocessing_backend,
                    "candidate_id": resolved_config.resolved_candidate_id,
                    "process_wall_seconds": time.perf_counter() - started,
                    "stdout_path": command_result["stdout_path"],
                    "stderr_path": command_result["stderr_path"],
                    "returncode": command_result["returncode"],
                    "failures": [],
                    **gpu_metrics,
                }

                if command_result["returncode"] != 0:
                    case_result["failures"] = [
                        "Training command failed. See stdout/stderr logs in the case directory."
                    ]
                    case_results.append(case_result)
                    continue

                try:
                    run = _find_run(
                        client=client,
                        experiment_name=resolved_config.competition.slug,
                        candidate_id=resolved_config.resolved_candidate_id,
                    )
                    artifact_paths = _collect_artifact_paths(client=client, run_id=run.info.run_id)
                    downloads_dir = case_dir / "downloaded"
                    downloads_dir.mkdir()
                    manifest_path = _download_artifact(
                        client=client,
                        run_id=run.info.run_id,
                        artifact_path="candidate/candidate.json",
                        target_dir=downloads_dir,
                    )
                    runtime_log_path = _download_artifact(
                        client=client,
                        run_id=run.info.run_id,
                        artifact_path="logs/runtime.log",
                        target_dir=downloads_dir,
                    )
                    oof_predictions_path = _download_artifact(
                        client=client,
                        run_id=run.info.run_id,
                        artifact_path="candidate/oof_predictions.csv",
                        target_dir=downloads_dir,
                    )
                    test_predictions_path = _download_artifact(
                        client=client,
                        run_id=run.info.run_id,
                        artifact_path="candidate/test_predictions.csv",
                        target_dir=downloads_dir,
                    )
                    manifest = _read_json(manifest_path)
                    failures = _validate_case_result(
                        case=case,
                        run=run,
                        manifest=manifest,
                        artifact_paths=artifact_paths,
                        runtime_log_path=runtime_log_path,
                        oof_predictions_path=oof_predictions_path,
                        test_predictions_path=test_predictions_path,
                    )
                    case_result.update(
                        {
                            "run_id": run.info.run_id,
                            "mlflow_status": run.info.status,
                            "resolved_gpu_backend": run.data.tags.get("runtime_resolved_gpu_backend"),
                            "resolved_preprocessing_backend": run.data.tags.get(
                                "runtime_preprocessing_backend"
                            ),
                            "cv_score_mean": run.data.metrics.get("cv_score_mean"),
                            "artifact_paths": sorted(artifact_paths),
                            "failures": failures,
                        }
                    )
                    if failures:
                        case_result["status"] = "failed"
                except Exception as exc:
                    case_result["status"] = "failed"
                    case_result["failures"] = [str(exc)]

                case_results.append(case_result)
    except Exception as exc:
        fatal_error = str(exc)
    finally:
        if original_config_text is None:
            with suppress(FileNotFoundError):
                CONFIG_PATH.unlink()
        else:
            CONFIG_PATH.write_text(original_config_text, encoding="utf-8")

    session_summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "tracking_uri": tracking_uri,
        "session_dir": str(session_dir),
        "bootstrap": bootstrap_results,
        "environment": environment_snapshot,
        "cases": case_results,
        "fatal_error": fatal_error,
    }
    summary_path = session_dir / "validation_summary.json"
    report_path = session_dir / "validation_report.md"
    summary_path.write_text(
        json.dumps(session_summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    report_path.write_text(_build_markdown_report(session_summary), encoding="utf-8")

    print(json.dumps(session_summary, indent=2, sort_keys=True))
    print(f"Validation report: {report_path}")

    any_bootstrap_failures = any(result["status"] == "failed" for result in bootstrap_results)
    any_case_failures = any(case_result["status"] != "passed" for case_result in case_results)
    if fatal_error is not None or any_bootstrap_failures or any_case_failures:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
