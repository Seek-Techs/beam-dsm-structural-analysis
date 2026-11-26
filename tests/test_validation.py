"""Input validation and model-stability behaviour."""
import contextlib
import copy
import io

import numpy as np
import pytest

import beam_dsm as m


def run(beam):
    with contextlib.redirect_stdout(io.StringIO()):
        return m.run_beam_analysis(beam)


def test_primary_beam_runs():
    assert len(run(m.define_beam())["support_reactions"]) == 5


def test_default_ei_is_normalised():
    assert m.define_beam()["EI"] == 1.0


BAD = {
    "negative EI": lambda b: b.__setitem__("EI", -1.0),
    "nodes not increasing": lambda b: b.__setitem__("nodes", np.array([0, 5, 3, 10, 12.5, 14.1, 17, 22.0])),
    "wrong UDL count": lambda b: b.__setitem__("element_udl", np.array([1.0, 2.0])),
    "duplicate support": lambda b: b.__setitem__("support_nodes", [0, 1, 1, 6, 7]),
    "no supports": lambda b: b.__setitem__("support_nodes", []),
    "support index out of range": lambda b: b.__setitem__("support_nodes", [0, 1, 4, 6, 99]),
    "NaN nodal load": lambda b: b["point_loads"].__setitem__(2, float("nan")),
    "single support (mechanism)": lambda b: b.__setitem__("support_nodes", [0]),
}


@pytest.mark.parametrize("name", list(BAD))
def test_bad_input_is_rejected(name):
    beam = copy.deepcopy(m.define_beam())
    BAD[name](beam)
    with pytest.raises((ValueError, IndexError, TypeError)):
        run(beam)


def test_single_support_message_is_clear():
    beam = m.define_beam()
    beam["support_nodes"] = [0]
    with pytest.raises(ValueError, match="two vertical supports"):
        run(beam)


def test_upward_internal_point_load_is_rejected():
    beam = m.define_point_load_test_beam(4.0)
    beam["element_point_loads"][0]["P"] = -50.0
    with pytest.raises(ValueError):
        run(beam)


def test_two_supports_are_stable():
    beam = {"nodes": np.array([0.0, 10.0]), "element_udl": np.array([20.0]),
            "point_loads": {}, "support_nodes": [0, 1], "EI": 1.0}
    assert run(beam)["condition_number"] < 1e6
