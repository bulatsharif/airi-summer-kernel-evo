import pytest

from cute_harness.policy import UnsafeSourceError, validate_source


@pytest.mark.parametrize(
    "source",
    [
        "import subprocess\nsubprocess.run(['id'])",
        "import os\nos.system('id')",
        "import os as safe\nsafe.popen('id')",
        "from os import system as run\nrun('id')",
        "eval('1 + 1')",
        "object.__subclasses__()",
        "import ctypes",
        "import asyncio\nasyncio.create_subprocess_shell('id')",
    ],
)
def test_rejects_process_and_escape_apis(source: str) -> None:
    with pytest.raises(UnsafeSourceError):
        validate_source(source)


def test_allows_torch_code_and_environment_access() -> None:
    validate_source(
        """
import os
import torch

if __name__ == "__main__":
    print(os.environ.get("CUDA_VISIBLE_DEVICES"))
    x = torch.ones(8, device="cuda")
    print(x.sum())
"""
    )


def test_reports_syntax_errors() -> None:
    with pytest.raises(UnsafeSourceError, match="not valid Python"):
        validate_source("if")
