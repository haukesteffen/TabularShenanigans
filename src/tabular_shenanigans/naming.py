import re

MODEL_CANDIDATE_ID_SEPARATOR = "--"
BLEND_CANDIDATE_ID_PATTERN = re.compile(r"^blend__v[1-9][0-9]*$")
TOKEN_PATTERN = re.compile(r"^[a-z0-9]+(?:_[a-z0-9]+)*$")
VERSION_TOKEN_PATTERN = re.compile(r"^v[1-9][0-9]*$")


def _ensure_competition_slug_not_repeated(candidate_id: str, competition_slug: str) -> None:
    if competition_slug in candidate_id:
        raise ValueError(
            "experiment.candidate.candidate_id must not repeat competition.slug because the competition already "
            f"scopes artifact paths. Remove '{competition_slug}' from '{candidate_id}'."
        )


def validate_model_candidate_id(
    candidate_id: str,
    competition_slug: str,
    feature_recipe_id: str,
    preprocessing_scheme_id: str,
) -> None:
    _ensure_competition_slug_not_repeated(candidate_id, competition_slug)

    expected_prefix = f"{feature_recipe_id}{MODEL_CANDIDATE_ID_SEPARATOR}{preprocessing_scheme_id}{MODEL_CANDIDATE_ID_SEPARATOR}"
    if not candidate_id.startswith(expected_prefix):
        raise ValueError(
            "Model candidate_id must follow "
            "<feature_recipe_id>--<preprocessing_scheme_id>--<variant_token>--vN and start with "
            f"'{expected_prefix}'. Got '{candidate_id}'."
        )

    suffix = candidate_id.removeprefix(expected_prefix)
    suffix_parts = suffix.split(MODEL_CANDIDATE_ID_SEPARATOR)
    if len(suffix_parts) != 2:
        raise ValueError(
            "Model candidate_id must end with exactly one variant token and one version token after the "
            f"feature recipe and preprocessing prefix. Got '{candidate_id}'."
        )

    variant_token, version_token = suffix_parts
    if not TOKEN_PATTERN.fullmatch(variant_token):
        raise ValueError(
            "Model candidate_id variant token must use lowercase letters, digits, and single underscores only. "
            f"Got '{variant_token}' in '{candidate_id}'."
        )
    if not VERSION_TOKEN_PATTERN.fullmatch(version_token):
        raise ValueError(
            "Model candidate_id must end with a version token like 'v1'. "
            f"Got '{version_token}' in '{candidate_id}'."
        )


def validate_blend_candidate_id(candidate_id: str, competition_slug: str) -> None:
    _ensure_competition_slug_not_repeated(candidate_id, competition_slug)
    if BLEND_CANDIDATE_ID_PATTERN.fullmatch(candidate_id):
        return
    raise ValueError(
        "Blend candidate_id must follow the simple sequential contract 'blend__vN'. "
        f"Got '{candidate_id}'."
    )
