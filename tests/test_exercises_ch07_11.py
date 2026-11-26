"""Exercises and function-card tasks of Chapters 7 to 11: every answer quoted in Appendix B is checked against
constants worked out by hand (two-span beam A: 4 m + 4 m, 10 kN/m, supports at every node, EI = 1)."""
import numpy as np
import pytest

import exercise_solutions_ch07_11 as ex

T = 40.0 / 3.0  # wL^2/12 for w = 10, L = 4


def test_7_3_udl_nodal_loads():
    assert ex.ex7_3_udl_nodal_loads() == pytest.approx([-20, -T, -20, T], abs=1e-12)


def test_7_4_assembled_vector():
    assert ex.ex7_4_assembled_load_vector() == pytest.approx([-20, -T, -70, 0, -20, T], abs=1e-12)


def test_7_5_point_load_positions():
    off_centre, centre = ex.ex7_5_point_load_positions()
    assert off_centre == pytest.approx([-25.3125, -16.875, -4.6875, 5.625], abs=1e-12)
    assert centre == pytest.approx([-15, -15, -15, 15], abs=1e-12)
    assert off_centre[0] + off_centre[2] == pytest.approx(-30)


def test_7_6_sign_error():
    correct, wrong = ex.ex7_6_sign_error()
    assert correct == pytest.approx(-110.0) and wrong == pytest.approx(-50.0)


def test_8_3_reduced_system():
    restrained, free, K, F = ex.ex8_3_reduced_system()
    assert restrained.tolist() == [0, 2, 4] and free.tolist() == [1, 3, 5]
    assert K == pytest.approx(np.array([[1, 0.5, 0], [0.5, 2, 0.5], [0, 0.5, 1]]), abs=1e-12)
    assert F == pytest.approx([-T, 0, T], abs=1e-12)


def test_8_4_displacements():
    assert ex.ex8_4_displacements() == pytest.approx([0, -T, 0, 0, 0, T], abs=1e-9)


def test_8_5_simply_supported_8m():
    # theta = wL^3/(24EI) = 10*512/24 = 640/3 ; v = 5wL^4/(384EI) = 1600/3
    assert ex.ex8_5_simply_supported_8m() == pytest.approx([0, -640 / 3, -1600 / 3, 0, 0, 640 / 3], abs=1e-9)


def test_8_6_single_support_message():
    assert ex.ex8_6_single_support_message() == "At least two vertical supports are required for a stable beam."


def test_9_3_reactions():
    assert ex.ex9_3_reactions() == pytest.approx([15, 50, 15], abs=1e-9)


def test_9_4_moment_equilibrium():
    resisting, applied = ex.ex9_4_moment_about_left_end()
    assert resisting == pytest.approx(320.0) and applied == pytest.approx(320.0)


def test_9_5_middle_support_removed():
    assert ex.ex9_5_middle_support_removed() == pytest.approx([40, 40], abs=1e-9)


def test_9_6_doubled_ei():
    reactions, u1, u2 = ex.ex9_6_doubled_ei()
    assert reactions == pytest.approx([15, 50, 15], abs=1e-9)
    assert u2 == pytest.approx(u1 / 2, abs=1e-9)


def test_10_3_end_forces():
    assert ex.ex10_3_end_forces() == pytest.approx(np.array([[15, 0, 25, -20], [25, 20, 15, 0]]), abs=1e-9)


def test_10_4_element_equilibrium():
    vertical, moment, load, moment_load = ex.ex10_4_element_equilibrium()
    assert vertical == pytest.approx(40.0) and load == 40.0
    assert moment == pytest.approx(80.0) and moment_load == 80.0


def test_10_5_shear_jump():
    jump, reaction = ex.ex10_5_shear_jump_at_node_1()
    assert jump == pytest.approx(50.0) and reaction == pytest.approx(50.0)


def test_10_6_load_vector_forgotten():
    ku, vertical = ex.ex10_6_load_vector_forgotten()
    assert ku == pytest.approx([-5, -T, 5, -20 / 3], abs=1e-9)
    assert vertical == pytest.approx(0.0, abs=1e-12)


def test_11_3_extrema():
    sagging, hogging = ex.ex11_3_extrema()
    assert sagging["position"] == pytest.approx(1.5) and sagging["moment"] == pytest.approx(81 / 8 + 9 / 8)  # 9wL^2/128 = 11.25
    assert hogging["position"] == pytest.approx(4.0) and hogging["moment"] == pytest.approx(-20.0)


def test_11_4_moment_at_middle_support():
    assert ex.ex11_4_moment_at_middle_support() == -20.0


def test_11_5_single_span():
    reactions, sagging, hogging = ex.ex11_5_single_span()
    assert reactions == pytest.approx([40, 40]) and sagging["moment"] == pytest.approx(80.0) and sagging["position"] == pytest.approx(4.0)
    assert hogging["moment"] == pytest.approx(0.0)


def test_11_6_internal_point_load():
    reactions, sagging = ex.ex11_6_internal_point_load()
    assert reactions == pytest.approx([30, 10]) and sagging["moment"] == pytest.approx(60.0) and sagging["position"] == pytest.approx(2.0)


def test_card_tasks():
    assert ex.card7a_negative_udl() == pytest.approx([20, T, 20, -T])
    assert ex.card7b_negative_udl_message() == "UDL magnitudes must be finite and non-negative."
    assert ex.card8a_support_node_out_of_range() == "IndexError"
    assert ex.card8b_singular_matrix() == "Singular matrix"
    assert ex.card9a_duplicate_restraints_message() == "Restrained DOFs must not contain duplicates."
    assert ex.card10a_wrong_udl_length_message() == "UDL array must contain one value per element."
    assert ex.card11a_no_hogging()["moment"] == pytest.approx(0.0)


def test_even_numbers_for_simple_supports():
    # Chapter 8 check: the restrained numbers of a simple-support model are all even
    restrained = ex.ex8_3_reduced_system()[0]
    assert np.all(restrained % 2 == 0)
