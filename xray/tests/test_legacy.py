"""
The hero plate is a real run of the transformers Trainer as released before the October 2024
gradient-accumulation fix (4.45.2), in a side environment (.venv-hf445, see scripts/setup.sh).
"""
from pathlib import Path

import pytest

from token_xray.scenarios import legacy_python, run_legacy_ga

TOL = 1e-6  # instrument precision floor (see test_xray.py)

pytestmark = pytest.mark.skipif(
    not Path(legacy_python()).exists(), reason=".venv-hf445 is missing: run scripts/setup.sh to create it"
)


@pytest.fixture(scope="module")
def legacy():
    return run_legacy_ga()


def test_runs_the_pre_fix_release(legacy):
    assert legacy["versions"]["transformers"] == "4.45.2"
    assert legacy["kind"] == "real"


def test_pre_fix_trainer_averages_micro_batch_means(legacy):
    """Detected (far from the token mean) and exactly the predicted mean-of-micro-batch-means weights."""
    assert legacy["maxDeviation"] > 0.05
    assert legacy["predictedDeviation"] < TOL
    assert legacy["minCos"] > 1 - 1e-9  # a pure re-weighting: the bug only changes the scale per micro-batch
