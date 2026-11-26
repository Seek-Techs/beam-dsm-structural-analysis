"""Shear and moment diagrams and the moment extrema (Chapter 11)."""
import contextlib
import io

import numpy as np

import beam_dsm as m


def analysis():
    with contextlib.redirect_stdout(io.StringIO()):
        return m.run_beam_analysis(m.define_beam())


def test_shear_jump_at_each_support_equals_reaction():
    a = analysis()
    D = a["element_diagrams"]
    for node, R in zip(a["beam"]["support_nodes"], a["support_reactions"]):
        left = D[node - 1]["shear"][-1] if node > 0 else 0.0
        right = D[node]["shear"][0] if node < len(D) else 0.0
        assert abs(right - left - R) < 1e-9


def test_slope_of_moment_equals_shear():
    for d in analysis()["element_diagrams"]:
        x, V, M = d["x"], d["shear"], d["moment"]
        for k in range(len(x) - 1):
            if x[k + 1] - x[k] > 1e-9:
                assert abs((M[k + 1] - M[k]) / (x[k + 1] - x[k]) - 0.5 * (V[k] + V[k + 1])) < 1e-6


def test_moments_vanish_at_free_ends():
    D = analysis()["element_diagrams"]
    assert abs(D[0]["moment"][0]) < 1e-9 and abs(D[-1]["moment"][-1]) < 1e-9


def test_extrema_are_exact():
    a = analysis()
    assert abs(a["maximum_sagging"]["moment"] - 187.837610) < 5e-7
    assert abs(a["maximum_sagging"]["position"] - 2.50225) < 5e-6
    assert abs(a["maximum_hogging"]["moment"] + 181.366836) < 5e-7


def test_sampled_diagram_peak_understates_exact_peak_slightly():
    """Book Chapter 11 samples 50 points per element; the exact peak is slightly higher."""
    a = analysis()
    pos, shear, moment = m.recover_beam_diagrams(
        a["beam"], a["elements"], a["element_end_forces"], a["beam"]["element_udl"], []
    )
    gap = a["maximum_sagging"]["moment"] - np.max(moment)
    assert 0 <= gap < 0.1
