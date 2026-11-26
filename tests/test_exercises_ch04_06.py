"""The pilot exercises (Chapters 4 to 6): every answer quoted in Appendix B is checked against hand-derived constants."""
import numpy as np
import pytest

import exercise_solutions_ch04_06 as ex


def test_4_3_numbering_and_mapping():
    dof_numbers, element_dofs = ex.ex4_3_dofs_and_mapping()
    assert dof_numbers.tolist() == [[0, 1], [2, 3], [4, 5], [6, 7]]
    assert element_dofs.tolist() == [[0, 1, 2, 3], [2, 3, 4, 5], [4, 5, 6, 7]]


def test_4_4_restrained_and_free():
    restrained, free = ex.ex4_4_restrained_and_free()
    assert restrained.tolist() == [0, 4]
    assert free.tolist() == [1, 2, 3, 5, 6, 7]


def test_4_5_extra_support():
    restrained, free = ex.ex4_5_extra_support()
    assert restrained.tolist() == [0, 2, 6, 8, 12, 14]
    assert len(free) == 16 - 6


def test_4_6_forgotten_factor_two():
    wrong, correct, wrongly_restrained, left_free = ex.ex4_6_forgotten_factor_two()
    assert wrong.tolist() == [0, 1, 4, 6, 7]
    assert correct.tolist() == [0, 2, 8, 12, 14]
    assert wrongly_restrained.tolist() == [1, 4, 6, 7]
    assert left_free.tolist() == [2, 8, 12, 14]


def test_5_3_matrix_by_hand():
    k, k64 = ex.ex5_3_hand_matrix()
    hand = np.array([[12, 24, -12, 24], [24, 64, -24, 32], [-12, -24, 12, -24], [24, 32, -24, 64]], float)
    assert np.allclose(k64, hand, atol=1e-12)
    assert k[0, 0] == pytest.approx(12 / 64)


def test_5_4_rigid_body_modes():
    translation, rotation, rank = ex.ex5_4_rigid_body_modes()
    assert np.allclose(translation, 0.0, atol=1e-12)
    assert np.allclose(rotation, 0.0, atol=1e-12)
    assert rank == 2


def test_5_5_doubling_the_length():
    k4, k8 = ex.ex5_5_doubling_the_length()
    assert k4[0, 0] == pytest.approx(0.1875) and k8[0, 0] == pytest.approx(0.0234375)
    assert k8[0, 0] / k4[0, 0] == pytest.approx(1 / 8)       # 12 EI / L^3
    assert k4[1, 1] == pytest.approx(1.0) and k8[1, 1] == pytest.approx(0.5)
    assert k8[1, 1] / k4[1, 1] == pytest.approx(1 / 2)       # 4 EI / L
    assert k8[0, 1] / k4[0, 1] == pytest.approx(1 / 4)       # 6 EI / L^2


def test_5_6_faulty_matrix_is_detected():
    symmetry_error, translation, rotation = ex.ex5_6_faulty_matrix()
    assert symmetry_error == pytest.approx(0.125)
    assert np.allclose(translation, 0.0, atol=1e-12)          # this check alone would not reveal the fault
    assert rotation[0] == pytest.approx(-0.125)


def test_6_3_assembly_by_hand():
    K, K64 = ex.ex6_3_assemble_two_elements()
    hand = np.array([
        [12, 24, -12, 24, 0, 0],
        [24, 64, -24, 32, 0, 0],
        [-12, -24, 24, 0, -12, 24],
        [24, 32, 0, 128, -24, 32],
        [0, 0, -12, -24, 12, -24],
        [0, 0, 24, 32, -24, 64]], float)
    assert np.allclose(K64, hand, atol=1e-12)
    assert np.allclose(K, K.T)
    assert np.allclose(K @ np.array([1, 0, 1, 0, 1, 0.0]), 0.0, atol=1e-12)
    assert np.allclose(K @ np.array([0, 1, 4, 1, 8, 1.0]), 0.0, atol=1e-12)


def test_6_5_wrong_mapping_passes_symmetry_but_fails_rigid_body():
    symmetry_error, translation, rotation = ex.ex6_5_wrong_mapping()
    assert symmetry_error == 0.0
    assert not np.allclose(translation, 0.0)
    assert not np.allclose(rotation, 0.0)
    assert translation[2] == pytest.approx(0.625)


def test_6_6_bandwidth_of_worked_beam():
    shape, half_bandwidth, nonzeros = ex.ex6_6_bandwidth()
    assert shape == (16, 16)
    assert half_bandwidth == 3          # an element couples DOFs at most three places apart
    assert nonzeros == 86


# ---------------------------------------------------------------- function-card tasks (Appendix B)
import beam_dsm as dsm


def test_card_4A_three_dofs_per_node_is_rejected():
    beam = dsm.define_beam()
    elements = dsm.generate_elements(beam)
    three_per_node = np.arange(len(beam["nodes"]) * 3).reshape(-1, 3)
    element_dofs = dsm.generate_element_dofs(elements, three_per_node)
    assert element_dofs.shape == (7, 6)
    with pytest.raises(ValueError, match="four DOF indices"):
        dsm.assemble_global_stiffness(beam, elements, element_dofs)


def test_card_4B_reversed_element_is_rejected():
    beam = dsm.define_beam()
    elements = dsm.generate_elements(beam).copy()
    elements[2] = [3, 2]
    dof_numbers = dsm.generate_dofs(beam)
    element_dofs = dsm.generate_element_dofs(elements, dof_numbers)
    with pytest.raises(ValueError, match="invalid length"):
        dsm.assemble_global_stiffness(beam, elements, element_dofs)


def test_card_5A_invalid_arguments_are_refused():
    with pytest.raises(ValueError, match="Element length must be positive and finite"):
        dsm.beam_element_stiffness(1.0, 0.0)
    with pytest.raises(ValueError, match="EI must be positive and finite"):
        dsm.beam_element_stiffness(-1.0, 4.0)


def test_card_6A_doubling_EI_doubles_K():
    beam = dsm.define_beam()
    elements = dsm.generate_elements(beam)
    element_dofs = dsm.generate_element_dofs(elements, dsm.generate_dofs(beam))
    K1 = dsm.assemble_global_stiffness(beam, elements, element_dofs)
    beam2 = dict(beam, EI=2.0)
    K2 = dsm.assemble_global_stiffness(beam2, elements, element_dofs)
    assert np.allclose(K2, 2 * K1, rtol=0, atol=1e-12)


def test_4_6_restrained_numbers_of_a_simple_support_model_are_even():
    wrong, correct, _, _ = ex.ex4_6_forgotten_factor_two()
    assert np.all(correct % 2 == 0)
    assert np.any(wrong % 2 == 1)
