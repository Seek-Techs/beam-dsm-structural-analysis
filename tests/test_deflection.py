"""Deflection against closed forms, and the mesh-refinement caveat (EI = 1, so deflections are normalised)."""
import contextlib
import io

import numpy as np
import pytest

import beam_dsm as m


def worst_deflection(beam):
    with contextlib.redirect_stdout(io.StringIO()):
        a = m.run_beam_analysis(beam)
    c = m.find_maximum_deflection_candidates(a["beam"], a["elements"], a["element_dofs"], a["displacements"])
    return min(c, key=lambda x: x["deflection"])


def ss(nodes, udl, loads):
    nodes = np.array(nodes, float)
    return {"nodes": nodes, "element_udl": np.full(len(nodes) - 1, float(udl)),
            "point_loads": loads, "support_nodes": [0, len(nodes) - 1], "EI": 1.0}


def test_central_point_load():
    w = worst_deflection(ss([0, 5, 10], 0, {1: 100.0}))
    assert w["deflection"] == pytest.approx(-100 * 10**3 / 48, rel=1e-9)


def test_udl_with_node_at_midspan():
    w = worst_deflection(ss([0, 2.5, 5, 7.5, 10], 20.0, {}))
    assert w["deflection"] == pytest.approx(-5 * 20 * 10**4 / 384, rel=1e-9)


def test_off_centre_point_load_maximum():
    P, L, a = 100.0, 10.0, 4.0
    ref = -P * a * (L**2 - a**2) ** 1.5 / (9 * np.sqrt(3) * L)
    w = worst_deflection(ss([0, 4, 10], 0, {1: P}))
    assert w["deflection"] == pytest.approx(ref, rel=1e-9)
    assert w["position"] == pytest.approx(L - np.sqrt((L**2 - a**2) / 3), rel=1e-9)


def test_single_element_udl_is_20_percent_low():
    """The documented limitation: cubic Hermite cannot follow the quartic UDL deflection."""
    w = worst_deflection(ss([0, 10], 20.0, {}))
    exact = -5 * 20 * 10**4 / 384
    assert w["deflection"] / exact == pytest.approx(0.8, rel=1e-6)


def test_refinement_recovers_exact_udl_deflection():
    out = m.find_converged_maximum_deflection(ss([0, 10], 20.0, {}), refinements=(1, 2, 4, 8, 16))
    assert out["converged"]
    assert out["deflection"] == pytest.approx(-5 * 20 * 10**4 / 384, rel=1e-6)


def test_primary_beam_original_mesh_understates_by_about_a_third():
    out = m.find_converged_maximum_deflection(m.define_beam(), refinements=(1, 2, 4, 8, 16, 32))
    assert out["converged"]
    assert out["original_mesh_deflection"] / out["deflection"] == pytest.approx(0.68, abs=0.01)
    assert out["position"] == pytest.approx(2.756, abs=0.01)


def test_refine_beam_preserves_total_load_and_supports():
    b = m.define_mixed_load_test_beam()
    r = m.refine_beam(b, 4)
    L0, L1 = np.diff(b["nodes"]), np.diff(r["nodes"])
    total = lambda beam, L: (beam["element_udl"] * L).sum() + sum(beam["point_loads"].values()) + sum(
        l["P"] for l in beam.get("element_point_loads", []))
    assert total(b, L0) == pytest.approx(total(r, L1))
    assert [r["nodes"][s] for s in r["support_nodes"]] == [b["nodes"][s] for s in b["support_nodes"]]
