"""Fault injection: each deliberate fault must make the benchmark register fail,
which shows that a passing register is informative."""
import os
import types

import pytest

import beam_dsm
from benchmarks import register, register_example2

SOURCE = open(os.path.join(os.path.dirname(beam_dsm.__file__), "beam_dsm.py")).read()

FAULTS = {
    "UDL fixed-end moment sign flipped": ("        -w * L**2 / 12,", "         w * L**2 / 12,"),
    "point-load a and b swapped": ("-P * a * b**2 / L**2,", "-P * a**2 * b / L**2,"),
    "stiffness 6L term replaced by 4L": ("[12, 6*L, -12, 6*L],", "[12, 4*L, -12, 6*L],"),
    "reaction sign (K d - F -> K d + F)": ("K_global @ displacements\n        - F_global", "K_global @ displacements\n        + F_global"),
    "UDL loads scaled by 0.9": ("-w * L / 2,", "-0.9 * w * L / 2,"),
}


def _mutated(old, new):
    assert old in SOURCE, "fault target not found: the module text changed"
    module = types.ModuleType("beam_dsm_mutated")
    exec(compile(SOURCE.replace(old, new), "beam_dsm_mutated", "exec"), module.__dict__)
    return module


@pytest.mark.parametrize("name", list(FAULTS))
def test_fault_is_detected(name):
    old, new = FAULTS[name]
    module = _mutated(old, new)
    try:
        records = register(module)
    except Exception:
        return  # a crash also counts as detection
    assert any(not r["passed"] for r in records), f"fault not detected: {name}"


@pytest.mark.parametrize("name", list(FAULTS))
def test_fault_is_detected_by_second_example(name):
    """The second example is an independent detector: each fault must also fail its register."""
    old, new = FAULTS[name]
    module = _mutated(old, new)
    try:
        records = register_example2(module)
    except Exception:
        return
    assert any(not r["passed"] for r in records), f"fault not detected by example 2: {name}"
