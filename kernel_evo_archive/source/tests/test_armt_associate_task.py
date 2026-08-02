from __future__ import annotations

import runpy
from pathlib import Path

import pytest
import torch


TASK = runpy.run_path(str(Path(__file__).resolve().parents[1] / "tasks" / "armt_associate" / "task.py"))
Model = TASK["Model"]


@pytest.mark.parametrize("use_denom", [False, True])
def test_armt_associate_reference_contract(use_denom: bool) -> None:
    """Keep a cheap CPU check for the task ABI used by the FP8 evolution config."""
    torch.manual_seed(7)
    model = Model(
        d_model=32,
        d_mem=8,
        n_heads=4,
        use_denom=use_denom,
        nu=2,
        batch_size=2,
    ).eval()
    hidden_states = torch.rand(2, 7, 32)
    original = hidden_states.clone()

    with torch.no_grad():
        output = model(hidden_states)
        repeated = model(hidden_states.clone())

    assert output.shape == hidden_states.shape
    assert output.dtype == hidden_states.dtype
    assert torch.isfinite(output).all()
    torch.testing.assert_close(output, repeated, rtol=0, atol=0)
    torch.testing.assert_close(hidden_states, original, rtol=0, atol=0)
