"""Python solutions to the pilot exercises of Chapters 4, 5 and 6 (book Appendix B).

Each function returns the quantities that the Appendix quotes.  The tests in tests/test_exercises_ch04_06.py
compare them with constants worked out by hand, so a wrong solution cannot pass quietly.
"""
import numpy as np

import beam_dsm as dsm


# ---------------------------------------------------------------- Chapter 4
def ex4_3_dofs_and_mapping():
    """Exercise 4.3: nodes at 0, 4, 9 and 12 m (three elements)."""
    beam = {"nodes": np.array([0.0, 4.0, 9.0, 12.0]), "EI": 1.0, "support_nodes": [0, 2]}
    elements = dsm.generate_elements(beam)
    dof_numbers = dsm.generate_dofs(beam)
    element_dofs = dsm.generate_element_dofs(elements, dof_numbers)
    return dof_numbers, element_dofs


def ex4_4_restrained_and_free():
    """Exercise 4.4: the same beam supported at nodes 0 and 2 (x = 0 and 9 m)."""
    beam = {"nodes": np.array([0.0, 4.0, 9.0, 12.0]), "EI": 1.0, "support_nodes": [0, 2]}
    dof_numbers = dsm.generate_dofs(beam)
    return dsm.generate_boundary_conditions(beam, dof_numbers)


def ex4_5_extra_support():
    """Exercise 4.5: the worked beam with an additional support at node 3 (x = 10 m)."""
    beam = dsm.define_beam()
    beam["support_nodes"] = [0, 1, 3, 4, 6, 7]
    dof_numbers = dsm.generate_dofs(beam)
    return dsm.generate_boundary_conditions(beam, dof_numbers)


def ex4_6_forgotten_factor_two():
    """Exercise 4.6: restrained DOFs written as support_nodes instead of 2 * support_nodes."""
    support_nodes = [0, 1, 4, 6, 7]
    wrong = np.array(support_nodes)
    correct = 2 * np.array(support_nodes)
    wrongly_restrained = np.setdiff1d(wrong, correct)
    left_free = np.setdiff1d(correct, wrong)
    return wrong, correct, wrongly_restrained, left_free


# ---------------------------------------------------------------- Chapter 5
def ex5_3_hand_matrix():
    """Exercise 5.3: EI = 1, L = 4 m.  Returns the function result and 64 times it."""
    k = dsm.beam_element_stiffness(1.0, 4.0)
    return k, 64 * k


def ex5_4_rigid_body_modes():
    """Exercise 5.4: forces produced by a rigid translation and a rigid rotation of the element."""
    k = dsm.beam_element_stiffness(1.0, 4.0)
    translation = k @ np.array([1.0, 0.0, 1.0, 0.0])
    rotation = k @ np.array([0.0, 1.0, 4.0, 1.0])  # [v_i, theta_i, v_j, theta_j] = [0, 1, L, 1]
    return translation, rotation, np.linalg.matrix_rank(k)


def ex5_5_doubling_the_length():
    """Exercise 5.5: the same EI with L = 4 m and L = 8 m."""
    k4 = dsm.beam_element_stiffness(1.0, 4.0)
    k8 = dsm.beam_element_stiffness(1.0, 8.0)
    return k4, k8


def ex5_6_faulty_matrix():
    """Exercise 5.6: k[0, 1] written with 4L instead of 6L.  Returns the symmetry error and the rigid-body forces."""
    k = dsm.beam_element_stiffness(1.0, 4.0)
    bad = k.copy()
    bad[0, 1] = 4 * 4.0 / 4.0**3  # (EI/L^3) * 4L
    symmetry_error = np.abs(bad - bad.T).max()
    translation = bad @ np.array([1.0, 0.0, 1.0, 0.0])
    rotation = bad @ np.array([0.0, 1.0, 4.0, 1.0])
    return symmetry_error, translation, rotation


# ---------------------------------------------------------------- Chapter 6
def _two_element_beam():
    beam = {"nodes": np.array([0.0, 4.0, 8.0]), "EI": 1.0}
    elements = np.array([[0, 1], [1, 2]])
    dof_numbers = dsm.generate_dofs(beam)
    return beam, elements, dsm.generate_element_dofs(elements, dof_numbers)


def ex6_3_assemble_two_elements():
    """Exercises 6.3 and 6.4: two equal elements of 4 m, EI = 1.  Returns K and 64 K."""
    beam, elements, element_dofs = _two_element_beam()
    K = dsm.assemble_global_stiffness(beam, elements, element_dofs)
    return K, 64 * K


def ex6_5_wrong_mapping():
    """Exercise 6.5: element 2 mapped to [3, 2, 4, 5] (first two entries swapped)."""
    beam, elements, _ = _two_element_beam()
    bad_mapping = np.array([[0, 1, 2, 3], [3, 2, 4, 5]])
    K = dsm.assemble_global_stiffness(beam, elements, bad_mapping)
    symmetry_error = np.abs(K - K.T).max()
    translation = K @ np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
    rotation = K @ np.array([0.0, 1.0, 4.0, 1.0, 8.0, 1.0])
    return symmetry_error, translation, rotation


def ex6_6_bandwidth():
    """Exercise 6.6: half-bandwidth and fill of the global matrix of the worked beam."""
    beam = dsm.define_beam()
    elements = dsm.generate_elements(beam)
    dof_numbers = dsm.generate_dofs(beam)
    element_dofs = dsm.generate_element_dofs(elements, dof_numbers)
    K = dsm.assemble_global_stiffness(beam, elements, element_dofs)
    i, j = np.nonzero(np.abs(K) > 1e-12)
    return K.shape, int(np.abs(i - j).max()), int(len(i))
