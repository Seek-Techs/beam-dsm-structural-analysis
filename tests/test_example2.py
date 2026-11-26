"""Second worked example (book Chapter 15) and the exact reference used to check it."""
import pytest

import beam_dsm
from benchmarks import register_example2
from reference_force_method import solve_reference

RECORDS = register_example2(beam_dsm)


def test_reference_simple_span():
    r = solve_reference([0, 10], [(0, 10, 20)], [])
    assert [float(x) for x in r["reactions"]] == [100.0, 100.0]
    assert float(r["moment"](5)) == 250.0
    assert float(r["deflection"](5)) == pytest.approx(-5 * 20 * 10**4 / 384, rel=1e-12)


def test_reference_two_equal_spans():
    r = solve_reference([0, 10, 20], [(0, 20, 20)], [])
    assert [float(x) for x in r["reactions"]] == [75.0, 250.0, 75.0]
    assert float(r["moment"](10)) == -250.0


def test_example2_all_checks_pass():
    failed = [r for r in RECORDS if not r["passed"]]
    assert not failed, "\n".join(f"{r['quantity']}: calc={r['calculated']} ref={r['reference']}" for r in failed)


def test_example2_register_size():
    assert len(RECORDS) == 30


def test_example2_all_reactions_upward():
    res = beam_dsm.run_beam_analysis(beam_dsm.define_second_example_beam())
    assert all(r > 0 for r in res["support_reactions"])


def test_example2_overhang_moment_is_statically_determinate():
    # 20 kN/m over 3 m plus 40 kN at the tip: M = 20*3*1.5 + 40*3 = 210 kN.m hogging, independent of EI and of the spans
    res = beam_dsm.run_beam_analysis(beam_dsm.define_second_example_beam())
    assert res["maximum_hogging"]["moment"] == pytest.approx(-210.0, abs=1e-9)


def test_example2_moments_do_not_depend_on_EI():
    a = beam_dsm.run_beam_analysis(beam_dsm.define_second_example_beam())
    b = beam_dsm.define_second_example_beam(); b["EI"] = 135000.0
    b = beam_dsm.run_beam_analysis(b)
    assert list(b["support_reactions"]) == pytest.approx(list(a["support_reactions"]), abs=1e-8)
    assert b["maximum_sagging"]["moment"] == pytest.approx(a["maximum_sagging"]["moment"], abs=1e-8)
    assert b["displacements"] == pytest.approx(a["displacements"] / 135000.0, abs=1e-12)
