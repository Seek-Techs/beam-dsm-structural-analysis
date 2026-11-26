"""Python solutions to the exercises of Chapters 7 to 11 (book Appendix B).

Each function returns the quantities that the Appendix quotes.  tests/test_exercises_ch07_11.py compares them with
constants worked out by hand, so a wrong solution cannot pass quietly.

Test beam A is a two-span beam: nodes at 0, 4 and 8 m, EI = 1, a UDL of 10 kN/m on both elements and a simple
support at every node.
"""
import contextlib
import io

import numpy as np

import beam_dsm as dsm


def beam_a(**changes):
    beam = {
        "nodes": np.array([0.0, 4.0, 8.0]),
        "element_udl": np.array([10.0, 10.0]),
        "point_loads": {},
        "support_nodes": [0, 1, 2],
        "EI": 1.0,
    }
    beam.update(changes)
    return beam


def run(beam):
    """Run the complete analysis without printing."""
    with contextlib.redirect_stdout(io.StringIO()):
        return dsm.run_beam_analysis(beam)


def _model(beam):
    elements = np.array([[i, i + 1] for i in range(len(beam["nodes"]) - 1)])
    dof_numbers = dsm.generate_dofs(beam)
    element_dofs = dsm.generate_element_dofs(elements, dof_numbers)
    return elements, dof_numbers, element_dofs


# ---------------------------------------------------------------- Chapter 7
def ex7_3_udl_nodal_loads():
    return dsm.beam_element_load(10.0, 4.0)


def ex7_4_assembled_load_vector():
    beam = beam_a(point_loads={1: 30.0})
    elements, _, element_dofs = _model(beam)
    return dsm.assemble_global_load(beam, elements, element_dofs)


def ex7_5_point_load_positions():
    return dsm.beam_element_point_load(30.0, 4.0, 1.0), dsm.beam_element_point_load(30.0, 4.0, 2.0)


def ex7_6_sign_error():
    """Total vertical load found with the correct and with the wrong sign for the 30 kN nodal load."""
    beam = beam_a(point_loads={1: 30.0})
    elements, _, element_dofs = _model(beam)
    correct = dsm.assemble_global_load(beam, elements, element_dofs)[0::2].sum()
    wrong = correct + 2 * 30.0
    return correct, wrong


# ---------------------------------------------------------------- Chapter 8
def ex8_3_reduced_system():
    r = run(beam_a())
    K_reduced, F_reduced = dsm.reduce_system(r["K_global"], r["F_global"], r["free_dofs"])
    return r["restrained_dofs"], r["free_dofs"], K_reduced, F_reduced


def ex8_4_displacements():
    return run(beam_a())["displacements"]


def ex8_5_simply_supported_8m():
    return run(beam_a(support_nodes=[0, 2]))["displacements"]


def ex8_6_single_support_message():
    beam = beam_a(support_nodes=[0])
    elements, dof_numbers, element_dofs = _model(beam)
    K = dsm.assemble_global_stiffness(beam, elements, element_dofs)
    F = dsm.assemble_global_load(beam, elements, element_dofs)
    restrained, free = dsm.generate_boundary_conditions(beam, dof_numbers)
    K_reduced, _ = dsm.reduce_system(K, F, free)
    try:
        dsm.validate_model_stability(K_reduced, beam["support_nodes"])
    except ValueError as error:
        return str(error)
    return None


# ---------------------------------------------------------------- Chapter 9
def ex9_3_reactions():
    return run(beam_a())["support_reactions"]


def ex9_4_moment_about_left_end():
    r = run(beam_a())
    x = r["beam"]["nodes"][r["beam"]["support_nodes"]]
    resisting = float(np.sum(r["support_reactions"] * x))
    applied = float(np.sum(r["beam"]["element_udl"] * np.diff(r["beam"]["nodes"]) * (r["beam"]["nodes"][:-1] + r["beam"]["nodes"][1:]) / 2))
    return resisting, applied


def ex9_5_middle_support_removed():
    return run(beam_a(support_nodes=[0, 2]))["support_reactions"]


def ex9_6_doubled_ei():
    r1, r2 = run(beam_a()), run(beam_a(EI=2.0))
    return r2["support_reactions"], r1["displacements"], r2["displacements"]


# ---------------------------------------------------------------- Chapter 10
def ex10_3_end_forces():
    return run(beam_a())["element_end_forces"]


def ex10_4_element_equilibrium():
    f = run(beam_a())["element_end_forces"][0]
    L, w = 4.0, 10.0
    return f[0] + f[2], f[2] * L + f[1] + f[3], w * L, w * L**2 / 2


def ex10_5_shear_jump_at_node_1():
    r = run(beam_a())
    f = r["element_end_forces"]
    return f[0, 2] + f[1, 0], r["support_reactions"][1]


def ex10_6_load_vector_forgotten():
    r = run(beam_a())
    elements, _, element_dofs = _model(r["beam"])
    k = dsm.beam_element_stiffness(1.0, 4.0)
    ku = k @ r["displacements"][element_dofs[0]]
    return ku, ku[0] + ku[2]


# ---------------------------------------------------------------- Chapter 11
def ex11_3_extrema():
    r = run(beam_a())
    return r["maximum_sagging"], r["maximum_hogging"]


def ex11_4_moment_at_middle_support():
    return 15.0 * 4.0 - 10.0 * 4.0**2 / 2


def ex11_5_single_span():
    beam = beam_a(nodes=np.array([0.0, 8.0]), element_udl=np.array([10.0]), support_nodes=[0, 1])
    r = run(beam)
    return r["support_reactions"], r["maximum_sagging"], r["maximum_hogging"]


def ex11_6_internal_point_load():
    beam = beam_a(nodes=np.array([0.0, 8.0]), element_udl=np.array([0.0]), support_nodes=[0, 1],
                  element_point_loads=[{"element": 0, "P": 40.0, "a": 2.0}])
    r = run(beam)
    return r["support_reactions"], r["maximum_sagging"]


# ---------------------------------------------------------------- function-card tasks
def card7a_negative_udl():
    return dsm.beam_element_load(-10.0, 4.0)


def card7b_negative_udl_message():
    beam = beam_a(element_udl=np.array([-10.0, 10.0]))
    elements, _, element_dofs = _model(beam)
    try:
        dsm.assemble_global_load(beam, elements, element_dofs)
    except ValueError as error:
        return str(error)


def card8a_support_node_out_of_range():
    beam = beam_a(support_nodes=[0, 1, 3])
    dof_numbers = dsm.generate_dofs(beam)
    try:
        dsm.generate_boundary_conditions(beam, dof_numbers)
    except IndexError as error:
        return type(error).__name__


def card8b_singular_matrix():
    try:
        dsm.solve_beam_displacements(np.array([[1.0, 1.0], [1.0, 1.0]]), np.array([1.0, 2.0]), 4, np.array([1, 3]))
    except np.linalg.LinAlgError as error:
        return str(error)


def card9a_duplicate_restraints_message():
    r = run(beam_a())
    try:
        dsm.calculate_support_reactions(r["K_global"], r["displacements"], r["F_global"], np.array([0, 0, 2, 4]))
    except ValueError as error:
        return str(error)


def card10a_wrong_udl_length_message():
    r = run(beam_a())
    try:
        dsm.recover_all_element_end_forces(r["beam"], r["elements"], r["element_dofs"], r["displacements"], np.array([10.0]))
    except ValueError as error:
        return str(error)


def card11a_no_hogging():
    return ex11_5_single_span()[2]
