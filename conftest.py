"""Pytest configuration: gate `@pytest.mark.network` tests behind --run-network.

The bulk of `test_trans.py` runs offline by mocking yt-dlp / faster-whisper at
the seam. A small `TestTranscribeNetworkSmoke` block exercises the real stack
end-to-end and is opt-in: skipped by default, run with `pytest --run-network`.
"""

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--run-network",
        action="store_true",
        default=False,
        help="Run tests marked @pytest.mark.network (hits real services).",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-network"):
        return
    skip = pytest.mark.skip(reason="requires --run-network")
    for item in items:
        if "network" in item.keywords:
            item.add_marker(skip)
