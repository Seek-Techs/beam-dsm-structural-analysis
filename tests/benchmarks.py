"""Six-case benchmark register (book Chapter 19) as a function of the module under test.

Tolerances are fixed here, before any run:
  * closed-form hand solutions: absolute 1e-6 plus relative 1e-9;
  * the independent Three-Moment reference: absolute 1e-6 plus relative 1e-8;
  * values published to 3 decimals: half a unit of the last printed digit (5e-4).
Where a point load can sit at a node, each case is run twice: as an internal element load and
as a nodal load.
"""
import contextlib
import io

import numpy as np

from reference_three_moment import three_moment_reference

EXACT = dict(atol=1e-6, rtol=1e-9)
REFERENCE = dict(atol=1e-6, rtol=1e-8)


def _quiet(f, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return f(*args, **kwargs)


def _simple_beam(L, udl, element_point_loads=None, nodal=None, nodes=None, EI=1.0):
    nodes = np.array(nodes if nodes is not None else [0.0, L])
    ne = len(nodes) - 1
    beam = {"nodes": nodes, "element_udl": np.array([udl] * ne, float),
            "point_loads": nodal or {}, "support_nodes": [0, ne], "EI": EI}
    if element_point_loads:
        beam["element_point_loads"] = element_point_loads
    return beam


def _shear_at(res, xq):
    xs = np.concatenate([d["x"] for d in res["element_diagrams"]])
    vs = np.concatenate([d["shear"] for d in res["element_diagrams"]])
    idx = np.where(np.isclose(xs, xq))[0]
    return vs[idx[0]], vs[idx[-1]]


def register(m):
    """Run all six cases against module m; return a list of check records."""
    records = []

    def rec(test, case, rep, quantity, calc, ref, **tol):
        tol = tol or EXACT
        r = m.compare_results(float(calc), float(ref), **tol)
        r.update(test=test, case=case, representation=rep, quantity=quantity, tol=tol)
        records.append(r)

    def suite(test, case, rep, res):
        checks = _quiet(m.verify_engineering_results, res)
        records.append(dict(test=test, case=case, representation=rep,
                            quantity="Internal verification suite (%d checks)" % len(checks),
                            passed=all(v["passed"] for v in checks.values()),
                            calculated=None, reference=None, absolute_error=None,
                            allowed_error=None, tol="1e-6"))

    def simply_supported(test, case, rep, beam, RA, RB, Mmax, xM, extra=None):
        res = _quiet(m.run_beam_analysis, beam)
        R = res["support_reactions"]
        pk = res["maximum_sagging"]
        rec(test, case, rep, "Reaction A (kN)", R[0], RA)
        rec(test, case, rep, "Reaction B (kN)", R[1], RB)
        rec(test, case, rep, "Maximum sagging moment (kN.m)", pk["moment"], Mmax)
        rec(test, case, rep, "Location of maximum moment (m)", pk["position"], xM)
        suite(test, case, rep, res)
        if extra:
            extra(res)

    # Test 1 - central point load
    c = "Central point load, L = 10 m, P = 100 kN"
    simply_supported(1, c, "1 element, internal load",
                     _simple_beam(10, 0, [{"element": 0, "P": 100.0, "a": 5.0}]), 50, 50, 250, 5)
    simply_supported(1, c, "2 elements, nodal load",
                     _simple_beam(10, 0, nodal={1: 100.0}, nodes=[0, 5, 10]), 50, 50, 250, 5)
    # Test 2 - full-span UDL
    c = "Full-span UDL 20 kN/m"
    simply_supported(2, c, "1 element", _simple_beam(10, 20.0), 100, 100, 250, 5)
    simply_supported(2, c, "4 elements",
                     _simple_beam(10, 20.0, nodes=[0, 2.5, 5, 7.5, 10]), 100, 100, 250, 5)
    # Test 3 - eccentric load
    c = "Eccentric point load, P = 100 kN at 4 m"
    simply_supported(3, c, "1 element, internal load",
                     _simple_beam(10, 0, [{"element": 0, "P": 100.0, "a": 4.0}]), 60, 40, 240, 4)
    simply_supported(3, c, "2 elements, nodal load",
                     _simple_beam(10, 0, nodal={1: 100.0}, nodes=[0, 4, 10]), 60, 40, 240, 4)

    # Test 4 - two point loads
    c = "Two point loads, 40 kN at 3 m and 50 kN at 7 m"

    def shears4(res):
        before, after = _shear_at(res, 3.0)
        rec(4, c, "1 element, internal loads", "Shear before P1 (kN)", before, 43)
        rec(4, c, "1 element, internal loads", "Shear after P1 (kN)", after, 3)
        before, after = _shear_at(res, 7.0)
        rec(4, c, "1 element, internal loads", "Shear before P2 (kN)", before, 3)
        rec(4, c, "1 element, internal loads", "Shear after P2 (kN)", after, -47)

    simply_supported(4, c, "1 element, internal loads",
                     _simple_beam(10, 0, [{"element": 0, "P": 40.0, "a": 3.0},
                                          {"element": 0, "P": 50.0, "a": 7.0}]),
                     43, 47, 141, 7, extra=shears4)
    simply_supported(4, c, "3 elements, nodal loads",
                     _simple_beam(10, 0, nodal={1: 40.0, 2: 50.0}, nodes=[0, 3, 7, 10]),
                     43, 47, 141, 7)

    # Test 5 - UDL plus midspan point load
    c = "UDL 10 kN/m with 40 kN at midspan"

    def shears5(res):
        before, after = _shear_at(res, 5.0)
        rec(5, c, "1 element, internal load", "Shear before P (kN)", before, 20)
        rec(5, c, "1 element, internal load", "Shear after P (kN)", after, -20)

    simply_supported(5, c, "1 element, internal load",
                     _simple_beam(10, 10.0, [{"element": 0, "P": 40.0, "a": 5.0}]),
                     70, 70, 225, 5, extra=shears5)
    simply_supported(5, c, "2 elements, nodal load",
                     _simple_beam(10, 10.0, nodal={1: 40.0}, nodes=[0, 5, 10]), 70, 70, 225, 5)

    # Test 6 - original five-support beam
    c = "Original five-support continuous beam"
    M_ref, R_ref = three_moment_reference()
    for EI in (1.0, 135000.0):
        beam = m.define_beam()
        beam["EI"] = EI
        res = _quiet(m.run_beam_analysis, beam)
        R, ef, x = res["support_reactions"], res["element_end_forces"], beam["nodes"]
        rep = "7 elements, EI = %g" % EI
        for k, name in enumerate("ABCDE"):
            rec(6, c, rep, "Reaction %s vs independent Three-Moment (kN)" % name, R[k], R_ref[k], **REFERENCE)
        for name, value, ref in zip("BCD", (ef[0, 3], ef[3, 3], ef[5, 3]), M_ref):
            rec(6, c, rep, "Support moment %s vs independent Three-Moment (kN.m)" % name, value, ref, **REFERENCE)
        if EI == 1.0:
            for k, (name, ref) in enumerate(zip("ABCDE", (150.135, 283.462, 102.389, 350.287, 163.727))):
                rec(6, c, rep, "Reaction %s vs published value (kN)" % name, R[k], ref, atol=5e-4, rtol=0)
            rec(6, c, rep, "Maximum sagging moment vs published (kN.m)",
                res["maximum_sagging"]["moment"], 187.837610, atol=5e-7, rtol=0)
            rec(6, c, rep, "Maximum hogging moment vs published (kN.m)",
                res["maximum_hogging"]["moment"], -181.366836, atol=5e-7, rtol=0)
            L = np.diff(x)
            load = float((beam["element_udl"] * L).sum() + sum(beam["point_loads"].values()))
            moment = float((beam["element_udl"] * L * (x[:-1] + x[1:]) / 2).sum()
                           + sum(P * x[n] for n, P in beam["point_loads"].items()))
            rec(6, c, rep, "Total applied load from inputs (kN)", load, 1050.0)
            rec(6, c, rep, "Applied moment about A from inputs (kN.m)", moment, 12537.5)
            rec(6, c, rep, "Sum of reactions (kN)", R.sum(), 1050.0)
            rec(6, c, rep, "Reaction moment about A (kN.m)",
                float((R * x[beam["support_nodes"]]).sum()), 12537.5)
        suite(6, c, rep, res)
    return records


def summarise(records):
    groups = {}
    for r in records:
        groups.setdefault((r["test"], r["case"], r["representation"]), []).append(r)
    rows = []
    for (test, case, rep), items in groups.items():
        worst = max((x["absolute_error"] for x in items if x["absolute_error"] is not None), default=0.0)
        rows.append((test, case, rep, all(x["passed"] for x in items), len(items), worst))
    return rows


# ---------------------------------------------------------------------------------------------
# Second worked example (book Chapter 15): three spans with an overhang, checked against the exact
# force-method reference (reference_force_method.py).  Tolerances: absolute 1e-6 plus relative 1e-9,
# fixed before any run; the reference is exact rational arithmetic, so no looser tolerance is needed.
# ---------------------------------------------------------------------------------------------
from reference_force_method import solve_reference

EXAMPLE2 = dict(
    supports=[0, 6, 14, 20],
    udl=[(0, 6, 25), (17, 20, 30), (20, 23, 20)],
    points=[(9, 150), (23, 40)],
)


def register_example2(m):
    """Run the second example against module m; return check records (test number 7)."""
    records = []
    ref = solve_reference(**EXAMPLE2)
    res = _quiet(m.run_beam_analysis, m.define_second_example_beam())
    case, rep = "Three spans with overhang, all load types", "8 nodes, 7 elements"

    def rec(quantity, calc, expected, **tol):
        tol = tol or EXACT
        r = m.compare_results(float(calc), float(expected), **tol)
        r.update(test=7, case=case, representation=rep, quantity=quantity, tol=tol)
        records.append(r)

    R = res["support_reactions"]
    for name, calc, exact in zip("ABCD", R, ref["reactions"]):
        rec(f"Reaction {name} (kN)", calc, exact)
    total = sum(w * (b - a) for a, b, w in EXAMPLE2["udl"]) + sum(P for _, P in EXAMPLE2["points"])
    rec("Sum of reactions = total load (kN)", sum(R), total)

    # bending moment at the supports, from the element end forces: left end of element e is -q[1], right end is +q[3]
    forces = res["element_end_forces"]
    for name, element_left, x in (("B", 1, 6), ("C", 3, 14), ("D", 5, 20)):
        rec(f"Support moment {name} (kN.m)", forces[element_left][3], ref["moment"](x))

    # whole bending-moment diagram against the exact moment function
    worst = 0.0
    for d in res["element_diagrams"]:
        for x, mom in zip(d["x"], d["moment"]):
            worst = max(worst, abs(float(mom) - float(ref["moment"](x))))
    r = m.compare_results(worst, 0.0, atol=1e-6, rtol=0.0)
    r.update(test=7, case=case, representation=rep, quantity="Largest difference over the whole moment diagram (kN.m)",
             tol=dict(atol=1e-6, rtol=0.0))
    records.append(r)

    pk = res["maximum_sagging"]
    rec("Maximum sagging moment (kN.m)", pk["moment"], ref["moment"](9))
    rec("Location of maximum sagging moment (m)", pk["position"], 9)
    hg = res["maximum_hogging"]
    rec("Maximum hogging moment, at the overhang root (kN.m)", hg["moment"], ref["moment"](20))
    rec("Location of maximum hogging moment (m)", hg["position"], 20)

    # nodal displacements and rotations (normalised, EI = 1); exact for Euler-Bernoulli elements
    xs = [0, 3, 6, 10, 14, 17, 20, 23]
    D = res["displacements"]
    for i, x in enumerate(xs):
        rec(f"Displacement at x = {x} m (normalised)", D[2 * i], ref["deflection"](x))
        rec(f"Rotation at x = {x} m (normalised)", D[2 * i + 1], ref["rotation"](x))

    checks = _quiet(m.verify_engineering_results, res)
    records.append(dict(test=7, case=case, representation=rep,
                        quantity="Internal verification suite (%d checks)" % len(checks),
                        passed=all(v["passed"] for v in checks.values()),
                        calculated=None, reference=None, absolute_error=None, allowed_error=None, tol="1e-6"))
    return records
