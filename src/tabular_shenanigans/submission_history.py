import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pandas as pd

SUBMISSION_EVENT_COLUMNS = [
    "submission_event_id",
    "submitted_at_utc",
    "competition_slug",
    "candidate_id",
    "candidate_type",
    "config_fingerprint",
    "feature_recipe_id",
    "preprocessing_scheme_id",
    "model_registry_key",
    "estimator_name",
    "cv_metric_name",
    "cv_metric_mean",
    "cv_metric_std",
    "submission_path",
    "submit_message",
    "submit_response_message",
]

SUBMISSION_SCORE_COLUMNS = [
    "observed_at_utc",
    "submission_event_id",
    "competition_slug",
    "candidate_id",
    "kaggle_submitted_at",
    "kaggle_file_name",
    "kaggle_description",
    "kaggle_status",
    "public_score",
    "private_score",
    "observation_source",
]

SUBMISSION_SCORE_DEDUP_COLUMNS = [
    "submission_event_id",
    "kaggle_submitted_at",
    "kaggle_file_name",
    "kaggle_status",
    "public_score",
    "private_score",
]

SUBMISSION_EVENT_ID_PATTERN = re.compile(r"(?:^|\s\|\s)submit=(?P<submission_event_id>[a-z0-9_-]+)(?:\s\|\s|$)")


@dataclass(frozen=True)
class SubmissionEvent:
    submission_event_id: str
    submitted_at_utc: str
    competition_slug: str
    candidate_id: str
    candidate_type: str
    config_fingerprint: str | None
    feature_recipe_id: str | None
    preprocessing_scheme_id: str | None
    model_registry_key: str
    estimator_name: str
    cv_metric_name: str
    cv_metric_mean: float
    cv_metric_std: float
    submission_path: str
    submit_message: str
    submit_response_message: str

    def to_row(self) -> dict[str, object]:
        return {
            "submission_event_id": self.submission_event_id,
            "submitted_at_utc": self.submitted_at_utc,
            "competition_slug": self.competition_slug,
            "candidate_id": self.candidate_id,
            "candidate_type": self.candidate_type,
            "config_fingerprint": self.config_fingerprint,
            "feature_recipe_id": self.feature_recipe_id,
            "preprocessing_scheme_id": self.preprocessing_scheme_id,
            "model_registry_key": self.model_registry_key,
            "estimator_name": self.estimator_name,
            "cv_metric_name": self.cv_metric_name,
            "cv_metric_mean": self.cv_metric_mean,
            "cv_metric_std": self.cv_metric_std,
            "submission_path": self.submission_path,
            "submit_message": self.submit_message,
            "submit_response_message": self.submit_response_message,
        }


@dataclass(frozen=True)
class SubmissionScoreObservation:
    observed_at_utc: str
    submission_event_id: str
    competition_slug: str
    candidate_id: str
    kaggle_submitted_at: str
    kaggle_file_name: str
    kaggle_description: str
    kaggle_status: str
    public_score: float | None
    private_score: float | None
    observation_source: str

    def to_row(self) -> dict[str, object]:
        return {
            "observed_at_utc": self.observed_at_utc,
            "submission_event_id": self.submission_event_id,
            "competition_slug": self.competition_slug,
            "candidate_id": self.candidate_id,
            "kaggle_submitted_at": self.kaggle_submitted_at,
            "kaggle_file_name": self.kaggle_file_name,
            "kaggle_description": self.kaggle_description,
            "kaggle_status": self.kaggle_status,
            "public_score": self.public_score,
            "private_score": self.private_score,
            "observation_source": self.observation_source,
        }


@dataclass(frozen=True)
class SubmissionRefreshResult:
    competition_slug: str
    submission_score_ledger_path: Path
    tracked_submission_event_count: int
    matched_submission_event_count: int
    appended_observation_count: int
    scanned_remote_submission_count: int
    observation_source: str


@dataclass(frozen=True)
class KaggleSubmissionRecord:
    kaggle_submitted_at: str
    kaggle_file_name: str
    kaggle_description: str
    kaggle_status: str
    public_score: float | None
    private_score: float | None


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_submission_event_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dt%H%M%S%f")
    return f"sub_{timestamp}_{uuid4().hex[:6]}"


def submission_event_ledger_path(competition_slug: str) -> Path:
    return Path("artifacts") / competition_slug / "submissions.csv"


def submission_score_ledger_path(competition_slug: str) -> Path:
    return Path("artifacts") / competition_slug / "submission_scores.csv"


def extract_submission_event_id(kaggle_description: str) -> str | None:
    match = SUBMISSION_EVENT_ID_PATTERN.search(kaggle_description)
    if match is None:
        return None
    return match.group("submission_event_id")


def _read_ledger(ledger_path: Path, required_columns: list[str]) -> pd.DataFrame:
    ledger_df = pd.read_csv(ledger_path)
    missing_columns = [column for column in required_columns if column not in ledger_df.columns]
    if missing_columns:
        raise ValueError(
            f"Ledger {ledger_path} is missing required columns {missing_columns}. "
            f"Present columns: {ledger_df.columns.tolist()}"
        )
    return ledger_df


def _merged_columns(*column_groups: list[str]) -> list[str]:
    merged_columns: list[str] = []
    for group in column_groups:
        for column in group:
            if column not in merged_columns:
                merged_columns.append(column)
    return merged_columns


def _write_ledger_rows(
    ledger_path: Path,
    required_columns: list[str],
    row_dicts: list[dict[str, object]],
) -> None:
    row_df = pd.DataFrame(row_dicts)
    if ledger_path.exists():
        existing_df = _read_ledger(ledger_path=ledger_path, required_columns=required_columns)
        merged_columns = _merged_columns(existing_df.columns.tolist(), required_columns, row_df.columns.tolist())
        combined_df = pd.concat(
            [
                existing_df.reindex(columns=merged_columns),
                row_df.reindex(columns=merged_columns),
            ],
            ignore_index=True,
            sort=False,
        )
        combined_df.to_csv(ledger_path, index=False)
        return

    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    merged_columns = _merged_columns(required_columns, row_df.columns.tolist())
    row_df.reindex(columns=merged_columns).to_csv(ledger_path, index=False)


def append_submission_event(submission_event: SubmissionEvent) -> Path:
    ledger_path = submission_event_ledger_path(submission_event.competition_slug)
    if ledger_path.exists():
        existing_df = _read_ledger(ledger_path=ledger_path, required_columns=SUBMISSION_EVENT_COLUMNS)
        existing_event_ids = set(existing_df["submission_event_id"].astype(str))
        if submission_event.submission_event_id in existing_event_ids:
            raise ValueError(
                "Submission ledger already contains submission_event_id "
                f"'{submission_event.submission_event_id}'."
            )
    _write_ledger_rows(
        ledger_path=ledger_path,
        required_columns=SUBMISSION_EVENT_COLUMNS,
        row_dicts=[submission_event.to_row()],
    )
    return ledger_path


def _row_signature(row: dict[str, object], signature_columns: list[str]) -> tuple[object, ...]:
    normalized_values: list[object] = []
    for column in signature_columns:
        value = row.get(column)
        if pd.isna(value):
            normalized_values.append(None)
            continue
        normalized_values.append(value)
    return tuple(normalized_values)


def append_submission_score_observations(
    competition_slug: str,
    observations: list[SubmissionScoreObservation],
) -> tuple[Path, int]:
    ledger_path = submission_score_ledger_path(competition_slug)
    if not observations:
        return ledger_path, 0

    existing_signatures: set[tuple[object, ...]] = set()
    if ledger_path.exists():
        existing_df = _read_ledger(ledger_path=ledger_path, required_columns=SUBMISSION_SCORE_COLUMNS)
        for row in existing_df[SUBMISSION_SCORE_DEDUP_COLUMNS].to_dict(orient="records"):
            existing_signatures.add(_row_signature(row, SUBMISSION_SCORE_DEDUP_COLUMNS))

    new_rows: list[dict[str, object]] = []
    for observation in observations:
        row = observation.to_row()
        row_signature = _row_signature(row, SUBMISSION_SCORE_DEDUP_COLUMNS)
        if row_signature in existing_signatures:
            continue
        existing_signatures.add(row_signature)
        new_rows.append(row)

    if not new_rows:
        return ledger_path, 0

    _write_ledger_rows(
        ledger_path=ledger_path,
        required_columns=SUBMISSION_SCORE_COLUMNS,
        row_dicts=new_rows,
    )
    return ledger_path, len(new_rows)


def _parse_optional_float(raw_value: object) -> float | None:
    if raw_value is None:
        return None
    if isinstance(raw_value, str):
        normalized = raw_value.strip()
        if normalized == "" or normalized.lower() in {"none", "nan"}:
            return None
        return float(normalized)
    return float(raw_value)


def _iter_kaggle_submissions(competition_slug: str, page_size: int = 100):
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        from kagglesdk.competitions.types.competition_api_service import ApiListSubmissionsRequest
        from kagglesdk.competitions.types.competition_enums import SubmissionGroup, SubmissionSortBy
    except ImportError as exc:
        raise ImportError(
            "Submission refresh requires the Kaggle CLI Python package and its SDK dependencies."
        ) from exc

    api = KaggleApi()
    api.authenticate()
    page_token = ""

    with api.build_kaggle_client() as kaggle:
        while True:
            request = ApiListSubmissionsRequest()
            request.competition_name = competition_slug
            request.group = SubmissionGroup.SUBMISSION_GROUP_ALL
            request.sort_by = SubmissionSortBy.SUBMISSION_SORT_BY_DATE
            request.page = -1
            request.page_token = page_token
            request.page_size = page_size

            response = kaggle.competitions.competition_api_client.list_submissions(request)
            submissions = getattr(response, "submissions", None) or []
            for submission in submissions:
                yield KaggleSubmissionRecord(
                    kaggle_submitted_at=str(getattr(submission, "date", "") or ""),
                    kaggle_file_name=str(getattr(submission, "file_name", "") or ""),
                    kaggle_description=str(getattr(submission, "description", "") or ""),
                    kaggle_status=str(getattr(submission, "status", "") or ""),
                    public_score=_parse_optional_float(getattr(submission, "public_score", None)),
                    private_score=_parse_optional_float(getattr(submission, "private_score", None)),
                )

            next_page_token = str(getattr(response, "next_page_token", "") or "")
            if not next_page_token:
                return
            page_token = next_page_token


def refresh_submission_scores(
    competition_slug: str,
    target_submission_event_ids: set[str] | None = None,
    observation_source: str = "refresh_submissions",
) -> SubmissionRefreshResult:
    event_ledger_path = submission_event_ledger_path(competition_slug)
    score_ledger_path = submission_score_ledger_path(competition_slug)
    if not event_ledger_path.exists():
        return SubmissionRefreshResult(
            competition_slug=competition_slug,
            submission_score_ledger_path=score_ledger_path,
            tracked_submission_event_count=0,
            matched_submission_event_count=0,
            appended_observation_count=0,
            scanned_remote_submission_count=0,
            observation_source=observation_source,
        )

    submission_event_df = _read_ledger(
        ledger_path=event_ledger_path,
        required_columns=SUBMISSION_EVENT_COLUMNS,
    )
    submission_events_by_id = {
        str(row["submission_event_id"]): row
        for row in submission_event_df.to_dict(orient="records")
    }
    tracked_submission_event_ids = set(submission_events_by_id)
    if target_submission_event_ids is not None:
        unknown_submission_event_ids = sorted(target_submission_event_ids - tracked_submission_event_ids)
        if unknown_submission_event_ids:
            raise ValueError(
                "Submission refresh got unknown submission_event_id values. "
                f"Unknown IDs: {unknown_submission_event_ids}"
            )
        tracked_submission_event_ids = set(target_submission_event_ids)

    if not tracked_submission_event_ids:
        return SubmissionRefreshResult(
            competition_slug=competition_slug,
            submission_score_ledger_path=score_ledger_path,
            tracked_submission_event_count=0,
            matched_submission_event_count=0,
            appended_observation_count=0,
            scanned_remote_submission_count=0,
            observation_source=observation_source,
        )

    observed_at_utc = utc_now_iso()
    matched_submission_event_ids: set[str] = set()
    scanned_remote_submission_count = 0
    observations: list[SubmissionScoreObservation] = []

    for kaggle_submission in _iter_kaggle_submissions(competition_slug=competition_slug):
        scanned_remote_submission_count += 1
        submission_event_id = extract_submission_event_id(kaggle_submission.kaggle_description)
        if submission_event_id is None or submission_event_id not in tracked_submission_event_ids:
            continue

        event_row = submission_events_by_id[submission_event_id]
        matched_submission_event_ids.add(submission_event_id)
        observations.append(
            SubmissionScoreObservation(
                observed_at_utc=observed_at_utc,
                submission_event_id=submission_event_id,
                competition_slug=competition_slug,
                candidate_id=str(event_row["candidate_id"]),
                kaggle_submitted_at=kaggle_submission.kaggle_submitted_at,
                kaggle_file_name=kaggle_submission.kaggle_file_name,
                kaggle_description=kaggle_submission.kaggle_description,
                kaggle_status=kaggle_submission.kaggle_status,
                public_score=kaggle_submission.public_score,
                private_score=kaggle_submission.private_score,
                observation_source=observation_source,
            )
        )
        if target_submission_event_ids is not None and matched_submission_event_ids == tracked_submission_event_ids:
            break

    _, appended_observation_count = append_submission_score_observations(
        competition_slug=competition_slug,
        observations=observations,
    )
    return SubmissionRefreshResult(
        competition_slug=competition_slug,
        submission_score_ledger_path=score_ledger_path,
        tracked_submission_event_count=len(tracked_submission_event_ids),
        matched_submission_event_count=len(matched_submission_event_ids),
        appended_observation_count=appended_observation_count,
        scanned_remote_submission_count=scanned_remote_submission_count,
        observation_source=observation_source,
    )
