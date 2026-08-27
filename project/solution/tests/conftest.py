"""Test bootstrap: import `final` with dummy credentials and a scratch log file.

`final` performs real work at import time (creates directories, opens a logging
FileHandler, constructs the Azure client and six agents). None of it touches the
network, so importing here is safe -- but the credentials have to parse and the
audit log has to point somewhere disposable, which is what this file arranges.
"""

import os
import sys
import tempfile
from pathlib import Path

SOLUTION_DIR = Path(__file__).resolve().parent.parent

os.environ.setdefault("AZURE_OPENAI_KEY", "unit-test-key")
os.environ.setdefault("URL", "https://unit-test.openai.azure.com/")
os.environ.setdefault(
    "AGENT_CHAT_LOG", str(Path(tempfile.gettempdir()) / "pytest_agent_chat.log")
)

sys.path.insert(0, str(SOLUTION_DIR))

import pytest  # noqa: E402

import final  # noqa: E402,F401


@pytest.fixture
def solution_dir() -> Path:
    return SOLUTION_DIR


@pytest.fixture
def marketing_csv() -> Path:
    return SOLUTION_DIR / "data" / "data-Marketing-1.csv"


@pytest.fixture
def bom_csv() -> Path:
    """data-Sensor-2.csv ships with a UTF-8 BOM on its single column header."""
    return SOLUTION_DIR / "data" / "data-Sensor-2.csv"
