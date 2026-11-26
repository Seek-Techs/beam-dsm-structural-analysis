"""Chapter 9: reactions and global equilibrium for the worked example."""
import contextlib
import io

import numpy as np

import beam_dsm as m


def analysis():
    with contextlib.redirect_stdout(io.StringIO()):
        return m.run_beam_analysis(m.define_beam())


def test_reactions_match_published_values():
    r = analysis()["support_reactions"]
    assert np.allclose(r, [150.135, 283.462, 102.389, 350.287, 163.727], atol=5e-4, rtol=0)


def test_force_and_moment_equilibrium():
    a = analysis()
    F, x, R = a["F_global"], a["beam"]["nodes"], a["support_reactions"]
    assert abs(R.sum() + F[0::2].sum()) < 1e-9
    applied = np.sum(x * F[0::2] + F[1::2])
    assert applied == np.float64(-12537.5) or abs(applied + 12537.5) < 1e-9
    assert abs((R * x[a["beam"]["support_nodes"]]).sum() + applied) < 1e-9


def test_free_dof_residuals_are_negligible():
    a = analysis()
    assert np.max(np.abs(a["reaction_vector"][a["free_dofs"]])) < 1e-10


def test_one_kn_reaction_fault_is_caught_by_global_equilibrium():
    a = analysis()
    a = dict(a, reaction_vector=a["reaction_vector"].copy())
    a["reaction_vector"][0] += 1.0
    with contextlib.redirect_stdout(io.StringIO()):
        checks = m.verify_engineering_results(a)
    assert not checks["Global equilibrium"]["passed"]
    assert not checks["Independent load check"]["passed"]
    assert checks["Element equilibrium"]["passed"]


def test_wrong_load_vector_is_caught_by_independent_check():
    """A self-consistent but wrong load vector (all loads x0.9, re-solved)."""
    a = analysis()
    K, F = a["K_global"], 0.9 * a["F_global"]
    K_r, F_r = m.reduce_system(K, F, a["free_dofs"])
    d = m.solve_beam_displacements(K_r, F_r, K.shape[0], a["free_dofs"])
    a = dict(a, F_global=F, displacements=d, reaction_vector=K @ d - F)
    with contextlib.redirect_stdout(io.StringIO()):
        checks = m.verify_engineering_results(a)
    assert checks["Global equilibrium"]["passed"]  # self-referential: cannot see the fault
    assert not checks["Independent load check"]["passed"]
