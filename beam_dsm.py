"""beam_dsm: continuous-beam analysis by the Direct Stiffness Method (educational engine).

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


# ============================================================================
# 1. INPUT VALIDATION (book Chapter 16)
# ============================================================================

# Source: Chapter 16
def validate_nodes(nodes):

    try:
        nodes = np.asarray(nodes, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "Node coordinates must be numeric."
        ) from error

    if nodes.ndim != 1:
        raise ValueError(
            "Node coordinates must be a one-dimensional array."
        )

    if len(nodes) < 2:
        raise ValueError(
            "At least two nodes are required."
        )

    if not np.all(np.isfinite(nodes)):
        raise ValueError(
            "Node coordinates must contain finite values."
        )

    if not np.all(np.diff(nodes) > 0):
        raise ValueError(
            "Node coordinates must be strictly increasing."
        )

    return nodes

# Source: Chapter 16
def validate_elements(elements, number_of_nodes):

    elements = np.asarray(elements)

    if elements.ndim != 2 or elements.shape[1] != 2:
        raise ValueError(
            "Each element must contain exactly two node indices."
        )

    if not np.issubdtype(elements.dtype, np.integer):
        raise ValueError(
            "Element connectivity must use integer node indices."
        )

    if number_of_nodes < 2:
        raise ValueError(
            "At least two nodes are required."
        )

    if np.any(elements < 0):
        raise ValueError(
            "Node indices cannot be negative."
        )

    if np.any(elements >= number_of_nodes):
        raise ValueError(
            "Element connectivity refers to a node that does not exist."
        )

    if np.any(elements[:, 0] >= elements[:, 1]):
        raise ValueError(
            "Each element must connect a left node to a right node."
        )

    return elements

# Source: Chapter 16
def calculate_element_length(nodes, element):

    nodes = validate_nodes(nodes)

    if len(element) != 2:
        raise ValueError(
            "An element must contain exactly two node indices."
        )

    i, j = element

    if not isinstance(i, (int, np.integer)):
        raise ValueError("The left node index must be an integer.")

    if not isinstance(j, (int, np.integer)):
        raise ValueError("The right node index must be an integer.")

    if i < 0 or j < 0 or i >= len(nodes) or j >= len(nodes):
        raise ValueError(
            "Element connectivity contains an invalid node index."
        )

    L = nodes[j] - nodes[i]

    if not np.isfinite(L) or L <= 0:
        raise ValueError(
            "Element length must be positive and finite."
        )

    return L

# Source: Chapter 16
def validate_flexural_rigidity(EI):

    if not np.isscalar(EI):
        raise ValueError(
            "The current model requires one scalar flexural rigidity."
        )

    if not np.isfinite(EI):
        raise ValueError(
            "Flexural rigidity must be finite."
        )

    if EI <= 0:
        raise ValueError(
            "Flexural rigidity must be greater than zero."
        )

    return float(EI)

# Source: Chapter 16
def validate_support_nodes(
    support_nodes,
    number_of_nodes
):

    support_nodes = np.asarray(support_nodes)

    if support_nodes.ndim != 1:
        raise ValueError(
            "Support nodes must be a one-dimensional array."
        )

    if not np.issubdtype(
        support_nodes.dtype,
        np.integer
    ):
        raise ValueError(
            "Support node indices must be integers."
        )

    if len(support_nodes) == 0:
        raise ValueError(
            "At least one support is required."
        )

    if np.any(support_nodes < 0):
        raise ValueError(
            "Support indices cannot be negative."
        )

    if np.any(support_nodes >= number_of_nodes):
        raise ValueError(
            "A support refers to a node that does not exist."
        )

    if len(np.unique(support_nodes)) != len(support_nodes):
        raise ValueError(
            "Duplicate support node indices were detected."
        )

    return support_nodes

# Source: Chapter 16
def validate_element_udl(
    element_udl,
    number_of_elements
):

    try:
        element_udl = np.asarray(
            element_udl,
            dtype=float
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            "UDL values must be numeric."
        ) from error

    if element_udl.ndim != 1:
        raise ValueError(
            "Element UDL data must be one-dimensional."
        )

    if len(element_udl) != number_of_elements:
        raise ValueError(
            "UDL data must contain one value per element."
        )

    if not np.all(np.isfinite(element_udl)):
        raise ValueError(
            "UDL values must be finite."
        )

    if np.any(element_udl < 0):
        raise ValueError(
            "UDL magnitudes must be non-negative under the current input convention."
        )

    return element_udl

# Source: Chapter 16
def validate_point_loads(
    point_loads,
    number_of_nodes
):

    if point_loads is None:
        point_loads = {}

    if not isinstance(point_loads, dict):
        raise ValueError(
            "Nodal point loads must be supplied as a dictionary."
        )

    for node, load in point_loads.items():

        if not isinstance(node, (int, np.integer)):
            raise ValueError(
                "Point-load node indices must be integers."
            )

        if node < 0 or node >= number_of_nodes:
            raise ValueError(
                "Point load refers to an invalid node."
            )

        if not np.isscalar(load):
            raise ValueError(
                "Point-load magnitude must be a scalar."
            )

        if not np.isfinite(load):
            raise ValueError(
                "Point-load magnitudes must be finite."
            )

        if load < 0:
            raise ValueError(
                "Point-load magnitudes must be non-negative under the current input convention."
            )

    return point_loads

# Source: Chapter 16
def validate_internal_point_loads(
    element_point_loads,
    elements,
    nodes
):

    if element_point_loads is None:
        element_point_loads = []

    if not isinstance(element_point_loads, (list, tuple)):
        raise ValueError(
            "Internal point loads must be supplied as a list or tuple."
        )

    number_of_elements = len(elements)

    for load in element_point_loads:

        if not isinstance(load, dict):
            raise ValueError(
                "Each internal point load must be a dictionary."
            )

        required_keys = {"element", "P", "a"}

        if not required_keys.issubset(load):
            raise ValueError(
                "Each internal point load requires element, P and a."
            )

        e = load["element"]
        P = load["P"]
        a = load["a"]

        if not isinstance(e, (int, np.integer)):
            raise ValueError(
                "Element index must be an integer."
            )

        if e < 0 or e >= number_of_elements:
            raise ValueError(
                "Internal point load refers to an invalid element."
            )

        if not np.isscalar(P) or not np.isfinite(P) or P < 0:
            raise ValueError(
                "Point-load magnitude must be finite and non-negative."
            )

        if not np.isscalar(a) or not np.isfinite(a):
            raise ValueError(
                "Point-load position must be a finite scalar."
            )

        i, j = elements[e]
        L = nodes[j] - nodes[i]

        if not 0 < a < L:
            raise ValueError(
                "Internal point load must lie strictly inside its element."
            )

    return element_point_loads

# Source: Chapter 16
def validate_beam_input(beam, elements):

    nodes = validate_nodes(beam["nodes"])

    elements = validate_elements(
        elements,
        len(nodes)
    )

    EI = validate_flexural_rigidity(
        beam["EI"]
    )

    support_nodes = validate_support_nodes(
        beam["support_nodes"],
        len(nodes)
    )

    element_udl = validate_element_udl(
        beam["element_udl"],
        len(elements)
    )

    point_loads = validate_point_loads(
        beam.get("point_loads", {}),
        len(nodes)
    )

    element_point_loads = validate_internal_point_loads(
        beam.get("element_point_loads", []),
        elements,
        nodes
    )

    validated_beam = {
        **beam,
        "nodes": nodes,
        "EI": EI,
        "support_nodes": support_nodes,
        "element_udl": element_udl,
        "point_loads": point_loads,
        "element_point_loads": element_point_loads
    }

    return validated_beam, elements


# ============================================================================
# 2. ELEMENT MATRICES AND LOADS (book Chapters 5 and 7)
# ============================================================================

# Source: Chapter 5
def beam_element_stiffness(EI, L):

    if not np.isfinite(EI) or EI <= 0:
        raise ValueError(
            "EI must be positive and finite."
        )

    if not np.isfinite(L) or L <= 0:
        raise ValueError(
            "Element length must be positive and finite."
        )

    k = (EI / L**3) * np.array([

        [12, 6*L, -12, 6*L],

        [6*L, 4*L**2, -6*L, 2*L**2],

        [-12, -6*L, 12, -6*L],

        [6*L, 2*L**2, -6*L, 4*L**2]

    ], dtype=float)

    return k

# Source: Chapter 7
def beam_element_load(w, L):

    return np.array([
        -w * L / 2,
        -w * L**2 / 12,
        -w * L / 2,
         w * L**2 / 12
    ])

# Source: Chapter 7
def beam_element_point_load(P, L, a):

    if not np.isfinite(P) or P <= 0:
        raise ValueError(
            "P must be a positive finite load magnitude."
        )

    if not np.isfinite(L) or L <= 0:
        raise ValueError(
            "L must be a positive finite element length."
        )

    if not np.isfinite(a) or not (0 < a < L):
        raise ValueError(
            "The point load must act strictly "
            "inside the element: 0 < a < L."
        )

    b = L - a

    return np.array([
        -P * b**2 * (3*a + b) / L**3,
        -P * a * b**2 / L**2,
        -P * a**2 * (a + 3*b) / L**3,
         P * a**2 * b / L**2
    ])


# ============================================================================
# 3. MODEL DEFINITION (notebook; EI changed to the normalised value)
# ============================================================================

# Source: notebook cell 53
def define_beam():

    # Normalised flexural rigidity (EI = 1).  Not a physical section property.
    beam = {
        "nodes": np.array([
            0.0, 6.0, 8.0, 10.0,
            12.5, 14.1, 17.0, 22.0
        ]),

        "element_udl": np.array([
            60.0, 0.0, 0.0, 0.0,
            20.0, 20.0, 80.0
        ]),

        "point_loads": {
            2: 40.0,
            3: 60.0,
            5: 100.0
        },

        "support_nodes": [0, 1, 4, 6, 7],

        "EI": 1.0
    }

    return beam

# Source: notebook cell 55
def generate_elements(beam):

    number_of_nodes = len(beam["nodes"])

    elements = np.column_stack((
        np.arange(number_of_nodes - 1),
        np.arange(1, number_of_nodes)
    ))

    return elements

# Source: notebook cell 56
def generate_dofs(beam):

    number_of_nodes = len(beam["nodes"])
    dofs_per_node = 2

    dof_numbers = np.arange(
        number_of_nodes * dofs_per_node
    ).reshape(number_of_nodes, dofs_per_node)

    return dof_numbers

# Source: notebook cell 57
def generate_element_dofs(elements, dof_numbers):

    element_dofs = []

    for element in elements:

        node_i = element[0]
        node_j = element[1]

        dofs = dof_numbers[[node_i, node_j]].flatten()

        element_dofs.append(dofs)

    return np.array(element_dofs)


# ============================================================================
# 4. ASSEMBLY (book Chapters 6 and 7)
# ============================================================================

# Source: Chapter 6
def assemble_global_stiffness(
    beam,
    elements,
    element_dofs
):
    nodes = np.asarray(
        beam["nodes"],
        dtype=float
    )

    elements = np.asarray(
        elements,
        dtype=int
    )

    element_dofs = np.asarray(
        element_dofs,
        dtype=int
    )

    EI = float(beam["EI"])

    # Validate beam properties
    if not np.isfinite(EI) or EI <= 0:
        raise ValueError(
            "EI must be a positive finite value."
        )

    if nodes.ndim != 1 or len(nodes) < 2:
        raise ValueError(
            "Beam nodes must be a one-dimensional "
            "array containing at least two nodes."
        )

    if not np.all(np.isfinite(nodes)):
        raise ValueError(
            "Node coordinates must be finite."
        )

    if np.any(np.diff(nodes) <= 0):
        raise ValueError(
            "Node coordinates must be strictly increasing."
        )

    number_of_nodes = len(nodes)
    number_of_dofs = number_of_nodes * 2

    # Validate element connectivity
    if (
        elements.ndim != 2
        or elements.shape[1] != 2
    ):
        raise ValueError(
            "Each element must contain two node indices."
        )

    if len(elements) == 0:
        raise ValueError(
            "At least one element is required."
        )

    if (
        np.any(elements < 0)
        or np.any(elements >= number_of_nodes)
    ):
        raise ValueError(
            "Element connectivity contains an "
            "invalid node index."
        )

    # Validate element DOF mapping
    if element_dofs.shape != (
        len(elements), 4
    ):
        raise ValueError(
            "element_dofs must have four DOF "
            "indices for each element."
        )

    if (
        np.any(element_dofs < 0)
        or np.any(element_dofs >= number_of_dofs)
    ):
        raise ValueError(
            "Element DOF mapping contains "
            "an invalid global DOF index."
        )

    # Initialise global stiffness matrix
    K_global = np.zeros((
        number_of_dofs,
        number_of_dofs
    ))

    # Assemble element stiffness contributions
    for e, element in enumerate(elements):

        i, j = element

        L = nodes[j] - nodes[i]

        if not np.isfinite(L) or L <= 0:
            raise ValueError(
                f"Element {e} has an invalid length."
            )

        k_element = beam_element_stiffness(
            EI,
            L
        )

        dofs = element_dofs[e]

        for local_i in range(4):

            for local_j in range(4):

                global_i = dofs[local_i]
                global_j = dofs[local_j]

                K_global[
                    global_i,
                    global_j
                ] += k_element[
                    local_i,
                    local_j
                ]

    return K_global

# Source: Chapter 7
def assemble_global_load(
    beam,
    elements,
    element_dofs
):

    nodes = np.asarray(
        beam["nodes"],
        dtype=float
    )

    elements = np.asarray(
        elements,
        dtype=int
    )

    element_dofs = np.asarray(
        element_dofs,
        dtype=int
    )

    element_udl = np.asarray(
        beam["element_udl"],
        dtype=float
    )

    point_loads = beam["point_loads"]

    element_point_loads = beam.get(
        "element_point_loads",
        []
    )

    number_of_nodes = len(nodes)
    number_of_dofs = number_of_nodes * 2

    # Basic geometry and input checks
    if (
        nodes.ndim != 1
        or len(nodes) < 2
        or not np.all(np.isfinite(nodes))
        or np.any(np.diff(nodes) <= 0)
    ):
        raise ValueError(
            "Beam nodes must be finite and "
            "strictly increasing."
        )

    if (
        elements.ndim != 2
        or elements.shape[1] != 2
        or len(elements) == 0
    ):
        raise ValueError(
            "Elements must contain two node indices."
        )

    if (
        np.any(elements < 0)
        or np.any(elements >= number_of_nodes)
    ):
        raise ValueError(
            "Invalid node index in element connectivity."
        )

    if element_dofs.shape != (
        len(elements), 4
    ):
        raise ValueError(
            "Invalid element DOF mapping shape."
        )

    if element_udl.shape != (len(elements),):
        raise ValueError(
            "There must be one UDL value per element."
        )

    if (
        not np.all(np.isfinite(element_udl))
        or np.any(element_udl < 0)
    ):
        raise ValueError(
            "UDL magnitudes must be finite "
            "and non-negative."
        )

    if not isinstance(point_loads, dict):
        raise ValueError(
            "point_loads must be a dictionary."
        )

    F_global = np.zeros(number_of_dofs)

    # 1. Assemble distributed loads
    for e, element in enumerate(elements):

        i, j = element

        L = nodes[j] - nodes[i]

        if L <= 0:
            raise ValueError(
                f"Element {e} has an invalid length."
            )

        w = element_udl[e]

        f_udl = beam_element_load(w, L)

        dofs = element_dofs[e]

        for local_i in range(4):

            global_i = dofs[local_i]

            F_global[global_i] += f_udl[local_i]

    # 2. Assemble nodal point loads
    for node, P in point_loads.items():

        if (
            not isinstance(node, (int, np.integer))
            or node < 0
            or node >= number_of_nodes
        ):
            raise ValueError(
                f"Invalid node index: {node}"
            )

        if not np.isfinite(P) or P < 0:
            raise ValueError(
                "Nodal load magnitudes must be "
                "finite and non-negative."
            )

        vertical_dof = 2 * node

        F_global[vertical_dof] -= P

    # 3. Assemble internal element point loads
    for load in element_point_loads:

        e = load["element"]
        P = load["P"]
        a = load["a"]

        if (
            not isinstance(e, (int, np.integer))
            or e < 0
            or e >= len(elements)
        ):
            raise ValueError(
                f"Invalid element index: {e}"
            )

        i, j = elements[e]

        L = nodes[j] - nodes[i]

        f_point = beam_element_point_load(
            P,
            L,
            a
        )

        dofs = element_dofs[e]

        for local_i in range(4):

            global_i = dofs[local_i]

            F_global[global_i] += f_point[local_i]

    return F_global


# ============================================================================
# 5. BOUNDARY CONDITIONS AND SOLUTION (notebook, now presented explicitly)
# ============================================================================

# Source: notebook cell 69
def generate_boundary_conditions(beam, dof_numbers):

    support_nodes = beam["support_nodes"]

    restrained_dofs = []

    for node in support_nodes:

        vertical_dof = dof_numbers[node, 0]

        restrained_dofs.append(vertical_dof)

    number_of_dofs = dof_numbers.size

    all_dofs = np.arange(number_of_dofs)

    free_dofs = np.setdiff1d(
        all_dofs,
        restrained_dofs
    )

    return np.array(restrained_dofs), free_dofs

# Source: notebook cell 71
def reduce_system(K_global, F_global, free_dofs):

    K_reduced = K_global[np.ix_(
        free_dofs,
        free_dofs
    )]

    F_reduced = F_global[free_dofs]

    return K_reduced, F_reduced

# Source: notebook cell 73
def solve_beam_displacements(
    K_reduced,
    F_reduced,
    number_of_dofs,
    free_dofs
):

    # Solve for unknown displacements
    displacements_free = np.linalg.solve(
        K_reduced,
        F_reduced
    )

    # Create the complete displacement vector
    displacements = np.zeros(number_of_dofs)

    # Insert the calculated free displacements
    displacements[free_dofs] = displacements_free

    return displacements

# Source: New
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


# ============================================================================
# 6. SUPPORT REACTIONS (book Chapter 9)
# ============================================================================

# Source: Chapter 9 (supplied text)
def calculate_support_reactions(
    K_global,
    displacements,
    F_global,
    restrained_dofs
):
    K_global = np.asarray(
        K_global,
        dtype=float
    )

    displacements = np.asarray(
        displacements,
        dtype=float
    )

    F_global = np.asarray(
        F_global,
        dtype=float
    )

    restrained_dofs = np.asarray(
        restrained_dofs,
        dtype=int
    )

    number_of_dofs = len(displacements)

    if K_global.shape != (
        number_of_dofs,
        number_of_dofs
    ):
        raise ValueError(
            "Global stiffness matrix has an invalid shape."
        )

    if F_global.shape != displacements.shape:
        raise ValueError(
            "Load vector and displacement vector "
            "must have matching dimensions."
        )

    if not np.all(np.isfinite(K_global)):
        raise ValueError(
            "Global stiffness matrix contains "
            "non-finite values."
        )

    if not np.all(np.isfinite(displacements)):
        raise ValueError(
            "Displacement vector contains "
            "non-finite values."
        )

    if not np.all(np.isfinite(F_global)):
        raise ValueError(
            "Load vector contains non-finite values."
        )

    if (
        np.any(restrained_dofs < 0)
        or np.any(restrained_dofs >= number_of_dofs)
    ):
        raise ValueError(
            "Restrained DOF index is out of range."
        )

    if len(np.unique(restrained_dofs)) != len(
        restrained_dofs
    ):
        raise ValueError(
            "Restrained DOFs must not contain duplicates."
        )

    reaction_vector = (
        K_global @ displacements
        - F_global
    )

    support_reactions = reaction_vector[
        restrained_dofs
    ]

    return reaction_vector, support_reactions


# ============================================================================
# 7. ELEMENT END FORCES (book Chapter 10)
# ============================================================================

# Source: Chapter 10
def calculate_element_end_forces(
    k_element,
    element_displacements,
    element_load_vector
):
    k_element = np.asarray(
        k_element,
        dtype=float
    )

    element_displacements = np.asarray(
        element_displacements,
        dtype=float
    )

    element_load_vector = np.asarray(
        element_load_vector,
        dtype=float
    )

    if k_element.shape != (4, 4):
        raise ValueError(
            "Element stiffness matrix must be 4x4."
        )

    if element_displacements.shape != (4,):
        raise ValueError(
            "Element displacement vector must contain four values."
        )

    if element_load_vector.shape != (4,):
        raise ValueError(
            "Element load vector must contain four values."
        )

    if not (
        np.all(np.isfinite(k_element))
        and np.all(np.isfinite(element_displacements))
        and np.all(np.isfinite(element_load_vector))
    ):
        raise ValueError(
            "Element data must contain finite values."
        )

    element_end_forces = (
        k_element @ element_displacements
        - element_load_vector
    )

    return element_end_forces

# Source: Chapter 10
def calculate_element_load_vector(
    e,
    beam,
    elements,
    element_udl,
    element_point_loads
):
    nodes = np.asarray(
        beam["nodes"],
        dtype=float
    )

    elements = np.asarray(
        elements,
        dtype=int
    )

    element_udl = np.asarray(
        element_udl,
        dtype=float
    )

    if e < 0 or e >= len(elements):
        raise IndexError(
            "Element index is out of range."
        )

    if element_udl.shape != (len(elements),):
        raise ValueError(
            "UDL array must contain one value per element."
        )

    if not np.all(np.isfinite(element_udl)):
        raise ValueError(
            "UDL values must be finite."
        )

    i, j = elements[e]

    L = nodes[j] - nodes[i]

    if L <= 0:
        raise ValueError(
            "Element length must be positive."
        )

    f_element = beam_element_load(
        element_udl[e],
        L
    )

    if element_point_loads is None:
        element_point_loads = []

    for load in element_point_loads:

        if load["element"] == e:

            f_point = beam_element_point_load(
                load["P"],
                L,
                load["a"]
            )

            f_element += f_point

    return f_element

# Source: Chapter 10
def recover_all_element_end_forces(
    beam,
    elements,
    element_dofs,
    displacements,
    element_udl,
    element_point_loads=None
):

    nodes = np.asarray(
        beam["nodes"],
        dtype=float
    )

    elements = np.asarray(
        elements,
        dtype=int
    )

    element_dofs = np.asarray(
        element_dofs,
        dtype=int
    )

    displacements = np.asarray(
        displacements,
        dtype=float
    )

    element_udl = np.asarray(
        element_udl,
        dtype=float
    )

    EI = float(beam["EI"])

    number_of_elements = len(elements)

    if not np.isfinite(EI) or EI <= 0:
        raise ValueError(
            "EI must be positive and finite."
        )

    if element_dofs.shape != (
        number_of_elements,
        4
    ):
        raise ValueError(
            "Element DOF mapping has an invalid shape."
        )

    if element_udl.shape != (
        number_of_elements,
    ):
        raise ValueError(
            "UDL array must contain one value per element."
        )

    if not np.all(np.isfinite(nodes)):
        raise ValueError(
            "Node coordinates must be finite."
        )

    if not np.all(np.isfinite(displacements)):
        raise ValueError(
            "Displacements must be finite."
        )

    if np.any(element_dofs < 0) or np.any(
        element_dofs >= len(displacements)
    ):
        raise ValueError(
            "Element DOF index is out of range."
        )

    all_element_end_forces = []

    for e, element in enumerate(elements):

        i, j = element

        if i < 0 or j >= len(nodes) or i >= j:
            raise ValueError(
                f"Invalid connectivity for element {e}."
            )

        L = nodes[j] - nodes[i]

        if L <= 0:
            raise ValueError(
                f"Element {e} has non-positive length."
            )

        k_element = beam_element_stiffness(
            EI,
            L
        )

        dofs = element_dofs[e]

        element_displacements = displacements[dofs]

        element_load_vector = (
            calculate_element_load_vector(
                e,
                beam,
                elements,
                element_udl,
                element_point_loads
            )
        )

        element_end_forces = (
            calculate_element_end_forces(
                k_element,
                element_displacements,
                element_load_vector
            )
        )

        all_element_end_forces.append(
            element_end_forces
        )

    return np.asarray(
        all_element_end_forces
    )


# ============================================================================
# 8. SECTION FORCES, DIAGRAMS AND MOMENT EXTREMA (book Chapter 11; notebook diagram builder)
# ============================================================================

# Source: Chapter 11
def recover_element_section_forces(
    e,
    beam,
    elements,
    element_end_forces,
    element_udl,
    element_point_loads=None,
    number_of_points=50
):

    nodes = np.asarray(
        beam["nodes"],
        dtype=float
    )

    elements = np.asarray(
        elements,
        dtype=int
    )

    element_end_forces = np.asarray(
        element_end_forces,
        dtype=float
    )

    element_udl = np.asarray(
        element_udl,
        dtype=float
    )

    if e < 0 or e >= len(elements):
        raise IndexError(
            "Element index is out of range."
        )

    if element_end_forces.shape != (
        len(elements), 4
    ):
        raise ValueError(
            "Element end-force array has an invalid shape."
        )

    if element_udl.shape != (len(elements),):
        raise ValueError(
            "UDL array must contain one value per element."
        )

    if not isinstance(number_of_points, int) or number_of_points < 2:
        raise ValueError(
            "number_of_points must be an integer of at least 2."
        )

    i, j = elements[e]

    if i < 0 or j >= len(nodes) or i >= j:
        raise ValueError(
            "Invalid element connectivity."
        )

    L = nodes[j] - nodes[i]

    if L <= 0:
        raise ValueError(
            "Element length must be positive."
        )

    w = element_udl[e]

    if not np.isfinite(w):
        raise ValueError(
            "UDL must be finite."
        )

    V_i = element_end_forces[e, 0]

    M_i = -element_end_forces[e, 1]

    if element_point_loads is None:
        element_point_loads = []

    loads = [
        load for load in element_point_loads
        if load["element"] == e
    ]

    for load in loads:

        P = float(load["P"])
        a = float(load["a"])

        if not np.isfinite(P) or P <= 0:
            raise ValueError(
                "Internal point load magnitude must be positive and finite."
            )

        if not np.isfinite(a) or not 0 < a < L:
            raise ValueError(
                "Internal point load position must lie strictly inside the element."
            )

    load_positions = np.array(
        [load["a"] for load in loads],
        dtype=float
    )

    x_samples = np.unique(
        np.concatenate((
            np.linspace(0.0, L, number_of_points),
            load_positions
        ))
    )

    local_x = []
    shear_values = []
    moment_values = []

    for x in x_samples:

        loads_before = [
            load for load in loads
            if load["a"] < x
        ]

        loads_at = [
            load for load in loads
            if np.isclose(
                load["a"],
                x,
                atol=1e-10,
                rtol=0.0
            )
        ]

        point_load_before = sum(
            load["P"] for load in loads_before
        )

        point_load_at = sum(
            load["P"] for load in loads_at
        )

        V_left = (
            V_i
            - w * x
            - point_load_before
        )

        M_x = (
            M_i
            + V_i * x
            - 0.5 * w * x**2
            - sum(
                load["P"] * (x - load["a"])
                for load in loads_before
            )
        )

        local_x.append(x)
        shear_values.append(V_left)
        moment_values.append(M_x)

        if point_load_at > 0:

            local_x.append(x)

            shear_values.append(
                V_left - point_load_at
            )

            moment_values.append(M_x)

    return (
        np.asarray(local_x),
        np.asarray(shear_values),
        np.asarray(moment_values)
    )

# Source: Chapter 11
def recover_beam_diagrams(
    beam,
    elements,
    element_end_forces,
    element_udl,
    element_point_loads=None,
    number_of_points=50
):

    all_positions = []
    all_shear = []
    all_moment = []

    for e, element in enumerate(elements):

        i, j = element

        x_start = beam["nodes"][i]

        local_x, shear, moment = (
            recover_element_section_forces(
                e,
                beam,
                elements,
                element_end_forces,
                element_udl,
                element_point_loads,
                number_of_points
            )
        )

        global_x = x_start + local_x

        all_positions.extend(global_x)
        all_shear.extend(shear)
        all_moment.extend(moment)

    return (
        np.asarray(all_positions),
        np.asarray(all_shear),
        np.asarray(all_moment)
    )

# Source: Chapter 11
def find_beam_moment_extrema(
    beam,
    elements,
    element_end_forces,
    element_udl,
    element_point_loads=None
):

    nodes = np.asarray(
        beam["nodes"],
        dtype=float
    )

    elements = np.asarray(
        elements,
        dtype=int
    )

    element_end_forces = np.asarray(
        element_end_forces,
        dtype=float
    )

    element_udl = np.asarray(
        element_udl,
        dtype=float
    )

    if element_point_loads is None:
        element_point_loads = []

    candidates = []

    for e, (i, j) in enumerate(elements):

        x_start = nodes[i]

        L = nodes[j] - nodes[i]

        w = element_udl[e]

        V_i = element_end_forces[e, 0]

        M_i = -element_end_forces[e, 1]

        loads = sorted(
            [
                load for load in element_point_loads
                if load["element"] == e
            ],
            key=lambda load: load["a"]
        )

        load_positions = [
            float(load["a"])
            for load in loads
        ]

        interval_boundaries = sorted(
            set([0.0, L] + load_positions)
        )

        local_candidates = list(
            interval_boundaries
        )

        for left, right in zip(
            interval_boundaries[:-1],
            interval_boundaries[1:]
        ):

            if w == 0:
                continue

            loads_before = [
                load for load in loads
                if load["a"] <= left
            ]

            load_sum = sum(
                load["P"]
                for load in loads_before
            )

            root = (V_i - load_sum) / w

            if left < root < right:
                local_candidates.append(root)

        for x in local_candidates:

            point_load_moment = sum(
                load["P"] * (x - load["a"])
                for load in loads
                if load["a"] < x
            )

            M_x = (
                M_i
                + V_i * x
                - 0.5 * w * x**2
                - point_load_moment
            )

            candidates.append({
                "position": x_start + x,
                "moment": M_x,
                "element": e + 1
            })

    maximum_sagging = max(
        candidates,
        key=lambda item: item["moment"]
    )

    maximum_hogging = min(
        candidates,
        key=lambda item: item["moment"]
    )

    return (
        maximum_sagging,
        maximum_hogging,
        candidates
    )

# Source: notebook cell 80 (exact diagram with jumps; used for plotting and continuity checks)
def calculate_element_diagrams(beam, elements, element_end_forces):

    x = beam["nodes"]
    element_udl = beam["element_udl"]

    element_diagrams = []

    for e, element in enumerate(elements):

        i, j = element

        L = x[j] - x[i]
        w = element_udl[e]

        f_end = element_end_forces[e]

        # Left-end section forces
        V0 = f_end[0]
        M0 = -f_end[1]

        # Find point loads acting inside this element
        point_loads = [
            load for load in beam.get("element_point_loads", [])
            if load["element"] == e
        ]

        # Sort point loads by their position
        point_loads = sorted(
            point_loads,
            key=lambda load: load["a"]
        )

        # Generate regular sample positions and include
        # every exact point-load position
        load_positions = np.array([
            load["a"] for load in point_loads
        ])

        x_local = np.unique(
            np.concatenate((
                np.linspace(0, L, 101),
                load_positions
            ))
        )

        diagram_x = []
        diagram_shear = []
        diagram_moment = []

        for s in x_local:

            # Identify loads located at this exact position
            loads_at_position = [
                load for load in point_loads
                if np.isclose(load["a"], s)
            ]

            # Calculate moment at this position
            M = (
                M0
                + V0 * s
                - (w * s**2) / 2
                - sum(
                    load["P"] * max(s - load["a"], 0)
                    for load in point_loads
                )
            )

            # Shear immediately before and after a point load
            if loads_at_position:

                V_before = (
                    V0
                    - w * s
                    - sum(
                        load["P"]
                        for load in point_loads
                        if load["a"] < s
                        and not np.isclose(load["a"], s)
                    )
                )

                V_after = (
                    V_before
                    - sum(load["P"] for load in loads_at_position)
                )

                # Duplicate the position to show the vertical jump
                diagram_x.extend([s, s])
                diagram_shear.extend([V_before, V_after])
                diagram_moment.extend([M, M])

            else:

                V = (
                    V0
                    - w * s
                    - sum(
                        load["P"]
                        for load in point_loads
                        if load["a"] < s
                    )
                )

                diagram_x.append(s)
                diagram_shear.append(V)
                diagram_moment.append(M)

        element_diagrams.append({
            "element": e + 1,
            "x": x[i] + np.array(diagram_x),
            "shear": np.array(diagram_shear),
            "moment": np.array(diagram_moment)
        })

    return element_diagrams


# ============================================================================
# 9. DEFLECTED SHAPE WITH MESH-REFINEMENT CHECK (notebook functions plus new refinement)
# ============================================================================

# Source: notebook cell 91
def beam_deflection_shape(v_i, theta_i, v_j, theta_j, L, x_local):

    # Normalised position within the element
    r = x_local / L

    # Cubic Hermite shape functions
    N1 = 1 - 3*r**2 + 2*r**3

    N2 = L * (r - 2*r**2 + r**3)

    N3 = 3*r**2 - 2*r**3

    N4 = L * (-r**2 + r**3)

    # Interpolated vertical displacement
    v = (
        N1 * v_i
        + N2 * theta_i
        + N3 * v_j
        + N4 * theta_j
    )

    return v

# Source: notebook cell 92
def calculate_deflected_shape(
    beam,
    elements,
    element_dofs,
    displacements,
    points_per_element=101
):

    x = beam["nodes"]

    deflection_results = []

    for e, element in enumerate(elements):

        i, j = element

        L = x[j] - x[i]

        # Obtain the four element DOFs
        dofs = element_dofs[e]

        d_element = displacements[dofs]

        v_i = d_element[0]
        theta_i = d_element[1]

        v_j = d_element[2]
        theta_j = d_element[3]

        # Positions within the element
        x_local = np.linspace(
            0, L, points_per_element
        )

        # Calculate interpolated deflection
        v = beam_deflection_shape(
            v_i,
            theta_i,
            v_j,
            theta_j,
            L,
            x_local
        )

        # Convert local positions to global positions
        x_global = x[i] + x_local

        deflection_results.append({
            "element": e + 1,
            "x": x_global,
            "deflection": v
        })

    return deflection_results

# Source: notebook cell 96
def calculate_element_slope(
    v_i, theta_i, v_j, theta_j, L, x_local
):

    r = x_local / L

    slope = (
        ((-6*r + 6*r**2) / L) * v_i
        + (1 - 4*r + 3*r**2) * theta_i
        + ((6*r - 6*r**2) / L) * v_j
        + (-2*r + 3*r**2) * theta_j
    )

    return slope

# Source: notebook cell 97
def find_maximum_deflection_candidates(
    beam,
    elements,
    element_dofs,
    displacements
):

    x = beam["nodes"]

    candidates = []

    for e, element in enumerate(elements):

        i, j = element

        L = x[j] - x[i]

        dofs = element_dofs[e]
        d_element = displacements[dofs]

        v_i = d_element[0]
        theta_i = d_element[1]

        v_j = d_element[2]
        theta_j = d_element[3]

        # Coefficients of the quadratic slope equation
        a = (
            6*(v_i - v_j)/L
            + 3*theta_i
            + 3*theta_j
        )

        b = (
            6*(v_j - v_i)/L
            - 4*theta_i
            - 2*theta_j
        )

        c = theta_i

        # Solve the slope equation
        if abs(a) < 1e-12:

            if abs(b) < 1e-12:
                roots = []
            else:
                roots = [-c / b]

        else:
            roots = np.roots([a, b, c])

        # Include element start and end positions
        local_positions = [0.0, L]

        # Keep real roots inside the element
        for root in roots:

            if abs(np.imag(root)) < 1e-10:

                r = float(np.real(root))

                if 0 < r < 1:
                    local_positions.append(r * L)

        # Calculate deflection at candidate positions
        for x_local in local_positions:

            v = beam_deflection_shape(
                v_i,
                theta_i,
                v_j,
                theta_j,
                L,
                x_local
            )

            candidates.append({
                "element": e + 1,
                "position": x[i] + x_local,
                "deflection": v
            })

    return candidates

# Source: New
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


# ============================================================================
# 10. VERIFICATION (notebook suite; book validators; two added checks)
# ============================================================================

# Source: notebook cell 156 (validator call changed to the book signature; check 7 added)
def verify_engineering_results(results, tolerance=1e-6):

    beam = results["beam"]
    elements = results["elements"]

    K_global = results["K_global"]
    F_global = results["F_global"]

    displacements = results["displacements"]
    reaction_vector = results["reaction_vector"]

    element_end_forces = results["element_end_forces"]
    element_diagrams = results["element_diagrams"]

    x = beam["nodes"]

    checks = {}

    # ------------------------------------------------
    # 1. INPUT VALIDATION
    # ------------------------------------------------

    try:
        validate_beam_input(beam, elements)

        checks["Input validation"] = {
            "passed": True,
            "detail": "All input validation checks passed."
        }

    except Exception as error:

        checks["Input validation"] = {
            "passed": False,
            "detail": str(error)
        }

    # ------------------------------------------------
    # 2. GLOBAL EQUILIBRIUM
    # ------------------------------------------------

    applied_load = -np.sum(F_global[0::2])

    reaction_load = sum(
        reaction_vector[2 * node]
        for node in beam["support_nodes"]
    )

    applied_moment = (
        np.sum(F_global[0::2] * x)
        + np.sum(F_global[1::2])
    )

    reaction_moment = sum(
        reaction_vector[2 * node] * x[node]
        for node in beam["support_nodes"]
    )

    force_error = reaction_load - applied_load
    moment_error = reaction_moment + applied_moment

    global_pass = (
        abs(force_error) <= tolerance
        and abs(moment_error) <= tolerance
    )

    checks["Global equilibrium"] = {
        "passed": global_pass,
        "detail": (
            f"Force error = {force_error:.3e} kN; "
            f"Moment error = {moment_error:.3e} kN.m"
        )
    }

    # ------------------------------------------------
    # 3. GLOBAL STIFFNESS MATRIX SYMMETRY
    # ------------------------------------------------

    symmetry_error = np.max(
        np.abs(K_global - K_global.T)
    )

    checks["Stiffness matrix symmetry"] = {
        "passed": symmetry_error <= tolerance,
        "detail": f"Maximum asymmetry = {symmetry_error:.3e}"
    }

    # ------------------------------------------------
    # 4. SUPPORT DISPLACEMENT CHECK
    # ------------------------------------------------

    restrained_dofs = results["restrained_dofs"]

    support_displacement_error = np.max(
        np.abs(displacements[restrained_dofs])
    )

    checks["Support displacement"] = {
        "passed": support_displacement_error <= tolerance,
        "detail": (
            f"Maximum restrained displacement = "
            f"{support_displacement_error:.3e}"
        )
    }

    # ------------------------------------------------
    # 5. ELEMENT EQUILIBRIUM
    # ------------------------------------------------

    element_errors = []

    for e, element in enumerate(elements):

        i, j = element

        L = x[j] - x[i]
        w = beam["element_udl"][e]

        f_end = element_end_forces[e]

        internal_loads = [
            load for load in beam.get("element_point_loads", [])
            if load["element"] == e
        ]

        total_point_load = sum(
            load["P"] for load in internal_loads
        )

        point_load_moment = sum(
            load["P"] * load["a"]
            for load in internal_loads
        )

        vertical_error = (
            f_end[0] + f_end[2]
            - w * L
            - total_point_load
        )

        moment_error_element = (
            f_end[1] + f_end[3]
            + f_end[2] * L
            - w * L**2 / 2
            - point_load_moment
        )

        element_errors.extend([
            abs(vertical_error),
            abs(moment_error_element)
        ])

    maximum_element_error = max(element_errors, default=0.0)

    checks["Element equilibrium"] = {
        "passed": maximum_element_error <= tolerance,
        "detail": (
            f"Maximum element equilibrium error = "
            f"{maximum_element_error:.3e}"
        )
    }

    # ------------------------------------------------
    # 6. BENDING MOMENT CONTINUITY
    # ------------------------------------------------

    moment_errors = []

    for e in range(len(element_diagrams) - 1):

        left_moment = element_diagrams[e]["moment"][-1]

        right_moment = element_diagrams[e + 1]["moment"][0]

        moment_errors.append(
            abs(left_moment - right_moment)
        )

    maximum_moment_error = max(moment_errors, default=0.0)

    checks["Bending moment continuity"] = {
        "passed": maximum_moment_error <= tolerance,
        "detail": (
            f"Maximum moment discontinuity = "
            f"{maximum_moment_error:.3e} kN.m"
        )
    }

    # ------------------------------------------------
    # 7. INDEPENDENT LOAD CHECK (applied loads recomputed from the INPUTS)
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
    # ------------------------------------------------

    finite_values = all(
        np.all(np.isfinite(array))
        for array in [
            K_global,
            F_global,
            displacements,
            reaction_vector,
            element_end_forces
        ]
    )

    checks["Numerical finiteness"] = {
        "passed": finite_values,
        "detail": (
            "All numerical results are finite."
            if finite_values
            else "NaN or infinite values detected."
        )
    }

    # ------------------------------------------------
    # FINAL VERIFICATION REPORT
    # ------------------------------------------------

    print("\nAUTOMATED ENGINEERING VERIFICATION")
    print("=" * 48)

    for name, result in checks.items():

        status = "PASS" if result["passed"] else "FAIL"

        print(f"{name}: {status}")
        print(f"  {result['detail']}")

    overall_pass = all(
        result["passed"] for result in checks.values()
    )

    print("-" * 48)

    if overall_pass:
        print("OVERALL ENGINEERING VERIFICATION: PASS")
    else:
        print("OVERALL ENGINEERING VERIFICATION: FAIL")

    return checks

# Source: Chapter 12
def generate_verification_report(checks):

    report = []

    for check_name, result in checks.items():

        status = (
            "PASS"
            if result["passed"]
            else "FAIL"
        )

        report.append({
            "check": check_name,
            "status": status,
            "detail": result["detail"],
            "error": result.get("error"),
            "tolerance": result.get("tolerance")
        })

    return report

# Source: Chapter 19
def compare_results(
    calculated,
    reference,
    atol=1e-8,
    rtol=1e-6
):

    values = [calculated, reference, atol, rtol]

    if not all(np.isscalar(value) for value in values):
        raise TypeError(
            "Inputs must be scalar values."
        )

    if not all(np.isfinite(value) for value in values):
        raise ValueError(
            "All inputs must be finite."
        )

    if atol < 0 or rtol < 0:
        raise ValueError(
            "Tolerances cannot be negative."
        )

    absolute_error = abs(
        calculated - reference
    )

    allowed_error = (
        atol + rtol * abs(reference)
    )

    passed = absolute_error <= allowed_error

    relative_error = (
        absolute_error / abs(reference)
        if reference != 0
        else None
    )

    return {
        "calculated": calculated,
        "reference": reference,
        "absolute_error": absolute_error,
        "relative_error": relative_error,
        "allowed_error": allowed_error,
        "passed": passed
    }


# ============================================================================
# 11. ANALYSIS DRIVER
# ============================================================================

# Source: notebook cell 120, rewritten to use the book's function names and validators
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


# ============================================================================
# 12. TEST BEAMS AND PLOTS
# ============================================================================

# Source: notebook cell 118
def define_test_beam():
    beam = {
        "nodes": np.array([
            0.0, 5.0, 10.0
        ]),

        "element_udl": np.array([
            20.0, 20.0
        ]),

        "point_loads": {},

        "element_point_loads": [
            {"element": 0, "P": 50.0, "a": 4.0}
        ],

        "support_nodes": [0, 2],

        "EI": 1.0
    }

    return beam

# Source: notebook cell 127
def define_point_load_test_beam(a):

    beam = {
        "nodes": np.array([0.0, 10.0]),
        "element_udl": np.array([0.0]),
        "point_loads": {},
        "element_point_loads": [
            {"element": 0, "P": 50.0, "a": a}
        ],
        "support_nodes": [0, 1],
        "EI": 1.0
    }

    return beam

# Source: notebook cell 140
def define_multiple_point_load_beam():

    beam = {
        "nodes": np.array([0.0, 10.0]),
        "element_udl": np.array([0.0]),
        "point_loads": {},
        "element_point_loads": [
            {"element": 0, "P": 40.0, "a": 3.0},
            {"element": 0, "P": 50.0, "a": 7.0}
        ],
        "support_nodes": [0, 1],
        "EI": 1.0
    }

    return beam

# Source: notebook cell 145
def define_mixed_load_test_beam():

    beam = {
        "nodes": np.array([0.0, 5.0, 10.0]),

        "element_udl": np.array([10.0, 10.0]),

        "point_loads": {
            1: 20.0
        },

        "element_point_loads": [
            {"element": 0, "P": 30.0, "a": 3.0},
            {"element": 1, "P": 40.0, "a": 2.0}
        ],

        "support_nodes": [0, 2],

        "EI": 1.0
    }

    return beam

# Source: notebook cell 136
def plot_shear_force_diagram(beam, element_diagrams):

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 5))

    # Plot the shear diagram for each element
    for diagram in element_diagrams:

        ax.plot(
            diagram["x"],
            diagram["shear"],
            linewidth=2,
            label=f"Element {diagram['element']}"
        )

        ax.fill_between(
            diagram["x"],
            diagram["shear"],
            0,
            alpha=0.12
        )

    # Connect shear values at internal element boundaries
    for e in range(len(element_diagrams) - 1):

        left_diagram = element_diagrams[e]
        right_diagram = element_diagrams[e + 1]

        x_boundary = left_diagram["x"][-1]

        V_left = left_diagram["shear"][-1]
        V_right = right_diagram["shear"][0]

        ax.plot(
            [x_boundary, x_boundary],
            [V_left, V_right],
            color="black",
            linewidth=2
        )

    # Add the left support reaction jump
    first_node = 0

    if first_node in beam["support_nodes"]:

        x_left = beam["nodes"][0]
        V_left = element_diagrams[0]["shear"][0]

        ax.plot(
            [x_left, x_left],
            [0, V_left],
            color="black",
            linewidth=2
        )

    # Add the right support reaction jump
    last_node = len(beam["nodes"]) - 1

    if last_node in beam["support_nodes"]:

        x_right = beam["nodes"][-1]
        V_right = element_diagrams[-1]["shear"][-1]

        ax.plot(
            [x_right, x_right],
            [V_right, 0],
            color="black",
            linewidth=2
        )

    ax.axhline(0, color="black", linewidth=1)

    ax.set_title("Shear Force Diagram")
    ax.set_xlabel("Beam Position (m)")
    ax.set_ylabel("Shear Force (kN)")

    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()

    plt.tight_layout()
    plt.show()

# Source: notebook cell 138
def plot_bending_moment_diagram(beam, element_diagrams):

    fig, ax = plt.subplots(figsize=(14, 6))

    for diagram in element_diagrams:

        ax.plot(
            diagram["x"],
            diagram["moment"],
            linewidth=2,
            label=f'Element {diagram["element"]}'
        )

    ax.axhline(
        0,
        color="black",
        linewidth=0.8
    )

    ax.set_title(
        "Bending Moment Diagram",
        fontsize=15,
        fontweight="bold"
    )

    ax.set_xlabel("Beam Position (m)")
    ax.set_ylabel("Bending Moment (kN.m)")

    ax.grid(True, linestyle="--", alpha=0.5)

    ax.legend()

    # Conventional structural diagram orientation
    ax.invert_yaxis()

    plt.tight_layout()
    plt.show()


# Second worked example (book Chapter 15): three spans and an overhang, with every load type.
def define_second_example_beam():
    """Supports A, B, C, D at x = 0, 6, 14, 20 m; free end at x = 23 m (3 m overhang).

    Loads: 25 kN/m on span AB; a 150 kN point load inside span BC, 3 m from B; 30 kN/m on the right-hand 3 m of
    span CD; 20 kN/m on the overhang; and a 40 kN load at the free end.  EI is normalised to 1.
    """
    return {
        "nodes": np.array([0.0, 3.0, 6.0, 10.0, 14.0, 17.0, 20.0, 23.0]),
        "element_udl": np.array([25.0, 25.0, 0.0, 0.0, 0.0, 30.0, 20.0]),
        "point_loads": {7: 40.0},
        "element_point_loads": [{"element": 2, "P": 150.0, "a": 3.0}],
        "support_nodes": [0, 2, 4, 6],
        "EI": 1.0,
    }
