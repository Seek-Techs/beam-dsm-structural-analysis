"""Assemble beam_dsm.py from (a) the book's validated functions, verbatim, (b) the notebook's final
functions, verbatim, and (c) a small amount of new code. Every function carries a provenance comment."""
import json,ast,re
OUT='../clean/beam_dsm.py'
nb=json.load(open('nb.ipynb')); MB=json.load(open('ms_blocks.json'))
ch9=open('ch9/ch9.md').read()
ch9_blocks=re.findall(r'```python[^\n]*\n(.*?)```',ch9,re.S)

def nb_fn(name,cell):
    src=''.join(nb['cells'][cell]['source']); t=ast.parse(src)
    n=[x for x in t.body if isinstance(x,ast.FunctionDef) and x.name==name][0]
    return ast.get_source_segment(src,n)
def book_fn(name,chapter):
    for b in MB:
        if b['chapter']!=chapter or b['occ']!=1: continue
        try: t=ast.parse(b['code'])
        except SyntaxError: continue
        for n in t.body:
            if isinstance(n,ast.FunctionDef) and n.name==name: return ast.get_source_segment(b['code'],n)
    raise KeyError((name,chapter))
def ch9_fn(name):
    for c in ch9_blocks:
        if f'def {name}' in c: return c.strip()
    raise KeyError(name)

parts=[]
def sec(title): parts.append(f'\n# {"="*76}\n# {title}\n# {"="*76}\n')
def add(src,origin,edit=None):
    if edit:
        for a,b in edit:
            assert a in src,(origin,a[:50]); src=src.replace(a,b,1)
    parts.append(f'# Source: {origin}\n{src}\n')

parts.append('''"""beam_dsm: continuous-beam analysis by the Direct Stiffness Method (educational engine).

Companion code for "Python for Structural Analysis: Continuous Beam Analysis Using the Direct
Stiffness Method".  Scope: linear-elastic Euler-Bernoulli beams, constant EI, vertical-only
supports (rotation free), element UDLs, nodal vertical forces and internal downward point loads.
It is an educational engine, not a commercial structural-analysis package.

Sign convention: loads positive downward in the input dictionary; displacement DOFs are
[v, theta] per node (upward positive); reactions are positive upward; bending moment is
positive for sagging.

Flexural rigidity: the worked example uses the NORMALISED value EI = 1.  For a beam of uniform EI,
reactions, shear forces and bending moments do not depend on EI; displacements scale as 1/EI, so
normalised displacements must not be reported as physical deflections.  (The hypothetical
section of the original problem, E = 25 GPa, 300 x 600 mm, has EI = 135 000 kN.m^2.)

Provenance: every function is marked "Source: Chapter N" (book text, verbatim), "Source: notebook
cell N" (Beam_Analysis_Exercise_3_2_Final.ipynb, verbatim) or "New".  Edits to verbatim code are
listed beside the function.
"""
import numpy as np

try:  # plotting is optional
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None
''')

sec('1. INPUT VALIDATION (book Chapter 16)')
for n in ['validate_nodes','validate_elements','calculate_element_length','validate_flexural_rigidity','validate_support_nodes','validate_element_udl','validate_point_loads','validate_internal_point_loads','validate_beam_input']:
    add(book_fn(n,16),'Chapter 16')

sec('2. ELEMENT MATRICES AND LOADS (book Chapters 5 and 7)')
add(book_fn('beam_element_stiffness',5),'Chapter 5')
add(book_fn('beam_element_load',7),'Chapter 7')
add(book_fn('beam_element_point_load',7),'Chapter 7')

sec('3. MODEL DEFINITION (notebook; EI changed to the normalised value)')
add(nb_fn('define_beam',53),'notebook cell 53',[
 ('    beam = {','    # Normalised flexural rigidity (EI = 1).  Not a physical section property.\n    beam = {'),
 ('"EI": 135000.0','"EI": 1.0')])
add(nb_fn('generate_elements',55),'notebook cell 55')
add(nb_fn('generate_dofs',56),'notebook cell 56')
add(nb_fn('generate_element_dofs',57),'notebook cell 57')

sec('4. ASSEMBLY (book Chapters 6 and 7)')
add(book_fn('assemble_global_stiffness',6),'Chapter 6')
add(book_fn('assemble_global_load',7),'Chapter 7')

sec('5. BOUNDARY CONDITIONS AND SOLUTION (notebook, now presented explicitly)')
add(nb_fn('generate_boundary_conditions',69),'notebook cell 69')
add(nb_fn('reduce_system',71),'notebook cell 71')
add(nb_fn('solve_beam_displacements',73),'notebook cell 73')
parts.append('''# Source: New
def validate_model_stability(K_reduced, support_nodes):
    """Reject models that are mechanisms (too few supports or a singular reduced matrix).

    The book's input validators do not detect an under-restrained beam; without this check
    the solver returns meaningless numbers (e.g. a single support gives displacements ~1e15).
    """
    if len(support_nodes) < 2:
        raise ValueError("At least two vertical supports are required for a stable beam.")
    rank = np.linalg.matrix_rank(K_reduced)
    if rank < K_reduced.shape[0]:
        raise ValueError(
            f"Reduced stiffness matrix is singular (rank {rank} of {K_reduced.shape[0]}): "
            "the beam is a mechanism."
        )
    return float(np.linalg.cond(K_reduced))
''')

sec('6. SUPPORT REACTIONS (book Chapter 9)')
add(ch9_fn('calculate_support_reactions'),'Chapter 9 (supplied text)')

sec('7. ELEMENT END FORCES (book Chapter 10)')
add(book_fn('calculate_element_end_forces',10),'Chapter 10')
add(book_fn('calculate_element_load_vector',10),'Chapter 10')
add(book_fn('recover_all_element_end_forces',10),'Chapter 10')

sec('8. SECTION FORCES, DIAGRAMS AND MOMENT EXTREMA (book Chapter 11; notebook diagram builder)')
add(book_fn('recover_element_section_forces',11),'Chapter 11')
add(book_fn('recover_beam_diagrams',11),'Chapter 11')
add(book_fn('find_beam_moment_extrema',11),'Chapter 11')
add(nb_fn('calculate_element_diagrams',80),'notebook cell 80 (exact diagram with jumps; used for plotting and continuity checks)')

sec('9. DEFLECTED SHAPE WITH MESH-REFINEMENT CHECK (notebook functions plus new refinement)')
add(nb_fn('beam_deflection_shape',91),'notebook cell 91')
add(nb_fn('calculate_deflected_shape',92),'notebook cell 92')
add(nb_fn('calculate_element_slope',96),'notebook cell 96')
add(nb_fn('find_maximum_deflection_candidates',97),'notebook cell 97')
parts.append('''# Source: New
def refine_beam(beam, n):
    """Return a copy of the beam with every element divided into n equal elements.

    Nodal loads and supports keep their positions.  An internal point load is moved to the
    sub-element that contains it, or becomes a nodal load if it falls on a new node.
    """
    if int(n) != n or n < 1:
        raise ValueError("n must be a positive integer.")
    n = int(n)
    x = np.asarray(beam["nodes"], dtype=float)
    nodes = [x[0]]
    udl = []
    for e in range(len(x) - 1):
        for k in range(1, n + 1):
            nodes.append(x[e] + (x[e + 1] - x[e]) * k / n)
            udl.append(float(beam["element_udl"][e]))
    point_loads = {n * node: P for node, P in beam["point_loads"].items()}
    internal = []
    for load in beam.get("element_point_loads", []):
        e, P, a = load["element"], load["P"], load["a"]
        h = (x[e + 1] - x[e]) / n
        k = min(int(a // h), n - 1)
        a_sub = a - k * h
        if abs(a_sub) < 1e-9 * h or abs(a_sub - h) < 1e-9 * h:
            node = n * e + k + (1 if abs(a_sub - h) < 1e-9 * h else 0)
            point_loads[node] = point_loads.get(node, 0.0) + P
        else:
            internal.append({"element": n * e + k, "P": P, "a": a_sub})
    refined = {
        "nodes": np.array(nodes),
        "element_udl": np.array(udl),
        "point_loads": point_loads,
        "support_nodes": [n * s for s in beam["support_nodes"]],
        "EI": beam["EI"],
    }
    if internal:
        refined["element_point_loads"] = internal
    return refined


# Source: New
def find_converged_maximum_deflection(beam, refinements=(1, 2, 4, 8, 16), tolerance=1e-3):
    """Maximum downward deflection with a mesh-refinement convergence check.

    Nodal displacements are exact, but the cubic Hermite shape inside a loaded element is
    only approximate (a UDL deflection is quartic).  The value from the original mesh can be
    tens of per cent too small, so it must not be quoted without refinement.

    Returns a dict with the value at each refinement, the relative change between the last
    two levels, a "converged" flag, and the final deflection and position.
    """
    history = []
    for n in refinements:
        analysis = run_beam_analysis(refine_beam(beam, n))
        candidates = find_maximum_deflection_candidates(
            analysis["beam"], analysis["elements"],
            analysis["element_dofs"], analysis["displacements"]
        )
        worst = min(candidates, key=lambda c: c["deflection"])
        history.append({"refinement": n, "deflection": worst["deflection"],
                        "position": worst["position"]})
    last, previous = history[-1]["deflection"], history[-2]["deflection"]
    change = abs(last - previous) / max(abs(last), 1e-300)
    return {"history": history, "relative_change": change,
            "converged": change <= tolerance,
            "deflection": last, "position": history[-1]["position"],
            "original_mesh_deflection": history[0]["deflection"]}
''')

sec('10. VERIFICATION (notebook suite; book validators; two added checks)')
vsrc=nb_fn('verify_engineering_results',156)
vedit=[
 ('        validate_beam_input(beam)\n        validate_element_point_loads(beam, elements)\n','        validate_beam_input(beam, elements)\n'),
 ('    # 7. NUMERICAL FINITENESS\n','''    # 7. INDEPENDENT LOAD CHECK (applied loads recomputed from the INPUTS)
    # ------------------------------------------------
    # Global equilibrium above reads the applied load from the assembled load
    # vector, so it cannot detect a wrong load vector.  This check can.

    element_lengths = np.diff(x)
    internal_loads = beam.get("element_point_loads", [])

    input_load = (
        float(np.sum(beam["element_udl"] * element_lengths))
        + float(sum(beam["point_loads"].values()))
        + float(sum(load["P"] for load in internal_loads))
    )

    input_moment = (
        float(np.sum(
            beam["element_udl"] * element_lengths
            * (x[:-1] + x[1:]) / 2
        ))
        + float(sum(P * x[node] for node, P in beam["point_loads"].items()))
        + float(sum(
            load["P"] * (x[elements[load["element"]][0]] + load["a"])
            for load in internal_loads
        ))
    )

    input_force_error = reaction_load - input_load
    input_moment_error = reaction_moment - input_moment

    input_scale = max(1.0, abs(input_load), abs(input_moment))

    checks["Independent load check"] = {
        "passed": (
            abs(input_force_error) <= tolerance * input_scale
            and abs(input_moment_error) <= tolerance * input_scale
        ),
        "detail": (
            f"Reactions minus loads from inputs: force {input_force_error:.3e} kN; "
            f"moment {input_moment_error:.3e} kN.m"
        )
    }

    # ------------------------------------------------
    # 8. NUMERICAL FINITENESS
'''),
]
add(vsrc,'notebook cell 156 (validator call changed to the book signature; check 7 added)',vedit)
add(book_fn('generate_verification_report',12),'Chapter 12')
add(book_fn('compare_results',19),'Chapter 19')

sec('11. ANALYSIS DRIVER')
parts.append('''# Source: notebook cell 120, rewritten to use the book's function names and validators
def run_beam_analysis(beam=None):
    """Run the complete analysis and return every intermediate and final result.

    Steps: validate, generate elements and DOFs, assemble K and F, apply boundary
    conditions, check stability, solve, calculate reactions, recover element end forces,
    build diagrams and find the moment extrema.
    """
    if beam is None:
        beam = define_beam()

    # 1. Validate (book Chapter 16) and generate elements
    elements = generate_elements(beam)
    validate_beam_input(beam, elements)
    dof_numbers = generate_dofs(beam)
    element_dofs = generate_element_dofs(elements, dof_numbers)

    # 2. Assemble the global system
    K_global = assemble_global_stiffness(beam, elements, element_dofs)
    F_global = assemble_global_load(beam, elements, element_dofs)

    # 3. Boundary conditions, stability check and solution
    restrained_dofs, free_dofs = generate_boundary_conditions(beam, dof_numbers)
    K_reduced, F_reduced = reduce_system(K_global, F_global, free_dofs)
    condition_number = validate_model_stability(K_reduced, beam["support_nodes"])
    displacements = solve_beam_displacements(
        K_reduced, F_reduced, K_global.shape[0], free_dofs
    )

    # 4. Reactions (Chapter 9)
    reaction_vector, support_reactions = calculate_support_reactions(
        K_global, displacements, F_global, restrained_dofs
    )

    # 5. Element end forces (Chapter 10)
    element_end_forces = recover_all_element_end_forces(
        beam, elements, element_dofs, displacements,
        beam["element_udl"], beam.get("element_point_loads", [])
    )

    # 6. Diagrams and extrema (Chapter 11)
    element_diagrams = calculate_element_diagrams(beam, elements, element_end_forces)
    maximum_sagging, maximum_hogging, moment_candidates = find_beam_moment_extrema(
        beam, elements, element_end_forces,
        beam["element_udl"], beam.get("element_point_loads", [])
    )

    return {
        "beam": beam,
        "elements": elements,
        "dof_numbers": dof_numbers,
        "element_dofs": element_dofs,
        "K_global": K_global,
        "F_global": F_global,
        "restrained_dofs": restrained_dofs,
        "free_dofs": free_dofs,
        "condition_number": condition_number,
        "displacements": displacements,
        "reaction_vector": reaction_vector,
        "support_reactions": support_reactions,
        "element_end_forces": element_end_forces,
        "element_diagrams": element_diagrams,
        "maximum_sagging": maximum_sagging,
        "maximum_hogging": maximum_hogging,
        "moment_candidates": moment_candidates,
    }
''')

sec('12. TEST BEAMS AND PLOTS')
add(nb_fn('define_test_beam',118),'notebook cell 118')
add(nb_fn('define_point_load_test_beam',127),'notebook cell 127')
add(nb_fn('define_multiple_point_load_beam',140),'notebook cell 140')
add(nb_fn('define_mixed_load_test_beam',145),'notebook cell 145')
add(nb_fn('plot_shear_force_diagram',136),'notebook cell 136')
add(nb_fn('plot_bending_moment_diagram',138),'notebook cell 138')
open(OUT,'w').write('\n'.join(parts))
print('written',OUT,len('\n'.join(parts).split('\n')),'lines')
