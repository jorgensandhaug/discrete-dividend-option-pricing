"""Public paper-surface tests."""

from discrete_dividend_study.neural_spec import PROJECT_STUDY as CANONICAL_PROJECT_STUDY
from discrete_dividend_study.paper import PROJECT_MODELS, PROJECT_STUDY

_EXPECTED_MODEL_COUNT = 5


def test_public_paper_surface_owns_the_complete_frozen_study() -> None:
    """The concise public alias must not define a second scientific identity."""
    assert PROJECT_STUDY is CANONICAL_PROJECT_STUDY
    assert PROJECT_STUDY.models == PROJECT_MODELS
    assert len(PROJECT_MODELS) == _EXPECTED_MODEL_COUNT
