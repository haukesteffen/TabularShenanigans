import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from tabular_shenanigans.cv import is_higher_better

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
    submission_file_name: str
    submit_message: str
    submit_response_message: str

    def to_dict(self) -> dict[str, object]:
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
            "submission_file_name": self.submission_file_name,
            "submit_message": self.submit_message,
            "submit_response_message": self.submit_response_message,
        }

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "SubmissionEvent":
        return cls(
            submission_event_id=str(value["submission_event_id"]),
            submitted_at_utc=str(value["submitted_at_utc"]),
            competition_slug=str(value["competition_slug"]),
            candidate_id=str(value["candidate_id"]),
            candidate_type=str(value["candidate_type"]),
            config_fingerprint=str(value["config_fingerprint"]) if value.get("config_fingerprint") is not None else None,
            feature_recipe_id=str(value["feature_recipe_id"]) if value.get("feature_recipe_id") is not None else None,
            preprocessing_scheme_id=(
                str(value["preprocessing_scheme_id"]) if value.get("preprocessing_scheme_id") is not None else None
            ),
            model_registry_key=str(value["model_registry_key"]),
            estimator_name=str(value["estimator_name"]),
            cv_metric_name=str(value["cv_metric_name"]),
            cv_metric_mean=float(value["cv_metric_mean"]),
            cv_metric_std=float(value["cv_metric_std"]),
            submission_file_name=str(value["submission_file_name"]),
            submit_message=str(value["submit_message"]),
            submit_response_message=str(value["submit_response_message"]),
        )


@dataclass(frozen=True)
class SubmissionScoreObservation:
    observed_at_utc: str
    submission_event_id: str
    kaggle_submitted_at: str
    kaggle_file_name: str
    kaggle_description: str
    kaggle_status: str
    public_score: float | None
    private_score: float | None
    observation_source: str

    def to_dict(self) -> dict[str, object]:
        return {
            "observed_at_utc": self.observed_at_utc,
            "submission_event_id": self.submission_event_id,
            "kaggle_submitted_at": self.kaggle_submitted_at,
            "kaggle_file_name": self.kaggle_file_name,
            "kaggle_description": self.kaggle_description,
            "kaggle_status": self.kaggle_status,
            "public_score": self.public_score,
            "private_score": self.private_score,
            "observation_source": self.observation_source,
        }

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "SubmissionScoreObservation":
        return cls(
            observed_at_utc=str(value["observed_at_utc"]),
            submission_event_id=str(value["submission_event_id"]),
            kaggle_submitted_at=str(value["kaggle_submitted_at"]),
            kaggle_file_name=str(value["kaggle_file_name"]),
            kaggle_description=str(value["kaggle_description"]),
            kaggle_status=str(value["kaggle_status"]),
            public_score=float(value["public_score"]) if value.get("public_score") is not None else None,
            private_score=float(value["private_score"]) if value.get("private_score") is not None else None,
            observation_source=str(value["observation_source"]),
        )


@dataclass(frozen=True)
class SubmissionRefreshResult:
    competition_slug: str
    tracked_candidate_count: int
    tracked_submission_event_count: int
    matched_submission_event_count: int
    updated_candidate_count: int
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


@dataclass(frozen=True)
class CandidateSubmissionHistory:
    events: list[SubmissionEvent]
    observations: list[SubmissionScoreObservation]

    @classmethod
    def empty(cls) -> "CandidateSubmissionHistory":
        return cls(events=[], observations=[])

    @classmethod
    def from_path(cls, history_path: Path) -> "CandidateSubmissionHistory":
        payload = json.loads(history_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Submission history must be a JSON object: {history_path}")
        raw_events = payload.get("events", [])
        raw_observations = payload.get("observations", [])
        if not isinstance(raw_events, list) or not isinstance(raw_observations, list):
            raise ValueError(f"Submission history must contain list fields 'events' and 'observations': {history_path}")
        return cls(
            events=[SubmissionEvent.from_dict(event) for event in raw_events],
            observations=[SubmissionScoreObservation.from_dict(observation) for observation in raw_observations],
        )

    def write(self, history_path: Path) -> None:
        history_path.write_text(
            json.dumps(
                {
                    "events": [event.to_dict() for event in self.events],
                    "observations": [observation.to_dict() for observation in self.observations],
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    def with_submission_event(self, submission_event: SubmissionEvent) -> "CandidateSubmissionHistory":
        existing_event_ids = {event.submission_event_id for event in self.events}
        if submission_event.submission_event_id in existing_event_ids:
            raise ValueError(
                "Submission history already contains submission_event_id "
                f"'{submission_event.submission_event_id}'."
            )
        return CandidateSubmissionHistory(
            events=[*self.events, submission_event],
            observations=self.observations,
        )

    def with_submission_observations(
        self,
        observations: list[SubmissionScoreObservation],
    ) -> tuple["CandidateSubmissionHistory", int]:
        existing_signatures = {
            (
                observation.submission_event_id,
                observation.kaggle_submitted_at,
                observation.kaggle_file_name,
                observation.kaggle_status,
                observation.public_score,
                observation.private_score,
            )
            for observation in self.observations
        }
        appended_observations: list[SubmissionScoreObservation] = []
        for observation in observations:
            signature = (
                observation.submission_event_id,
                observation.kaggle_submitted_at,
                observation.kaggle_file_name,
                observation.kaggle_status,
                observation.public_score,
                observation.private_score,
            )
            if signature in existing_signatures:
                continue
            existing_signatures.add(signature)
            appended_observations.append(observation)
        if not appended_observations:
            return self, 0
        return (
            CandidateSubmissionHistory(
                events=self.events,
                observations=[*self.observations, *appended_observations],
            ),
            len(appended_observations),
        )

    def get_event(self, submission_event_id: str) -> SubmissionEvent | None:
        for event in self.events:
            if event.submission_event_id == submission_event_id:
                return event
        return None

    def get_observations(self, submission_event_id: str) -> list[SubmissionScoreObservation]:
        return [
            observation
            for observation in self.observations
            if observation.submission_event_id == submission_event_id
        ]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_submission_event_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dt%H%M%S%f")
    return f"sub_{timestamp}_{uuid4().hex[:6]}"


def extract_submission_event_id(kaggle_description: str) -> str | None:
    match = SUBMISSION_EVENT_ID_PATTERN.search(kaggle_description)
    if match is None:
        return None
    return match.group("submission_event_id")


def _parse_optional_float(raw_value: object) -> float | None:
    if raw_value is None:
        return None
    if isinstance(raw_value, str):
        normalized = raw_value.strip()
        if normalized == "" or normalized.lower() in {"none", "nan"}:
            return None
        return float(normalized)
    return float(raw_value)


def iter_kaggle_submissions(competition_slug: str, page_size: int = 100):
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


def build_submission_score_metrics(
    primary_metric: str,
    history: CandidateSubmissionHistory,
) -> dict[str, float | int | None]:
    public_observations = [observation for observation in history.observations if observation.public_score is not None]
    private_observations = [observation for observation in history.observations if observation.private_score is not None]
    higher_is_better = is_higher_better(primary_metric)

    latest_public = public_observations[-1].public_score if public_observations else None
    latest_private = private_observations[-1].private_score if private_observations else None
    best_public = None
    best_private = None
    if public_observations:
        public_scores = [observation.public_score for observation in public_observations if observation.public_score is not None]
        best_public = max(public_scores) if higher_is_better else min(public_scores)
    if private_observations:
        private_scores = [observation.private_score for observation in private_observations if observation.private_score is not None]
        best_private = max(private_scores) if higher_is_better else min(private_scores)

    return {
        "submit_count": len(history.events),
        "latest_public_score": latest_public,
        "best_public_score": best_public,
        "latest_private_score": latest_private,
        "best_private_score": best_private,
    }
