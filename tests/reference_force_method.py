"""Independent reference for the second worked example: the force (flexibility) method with Macaulay
singular functions, in exact rational arithmetic (fractions.Fraction).

It shares no code and no method with beam_dsm (which uses the stiffness method) or with the Three-Moment
reference of the first example.  Unknowns: the support reactions and the initial rotation at x = 0.
Equations: vertical equilibrium, moment equilibrium, and zero deflection at every support except the first.
Sign convention: loads downward positive; M sagging positive; v upward positive; EI v'' = M.
"""
from fractions import Fraction as Fr


def _solve(A, b):
    n = len(b)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for c in range(n):
        p = next(r for r in range(c, n) if M[r][c] != 0)
        M[c], M[p] = M[p], M[c]
        M[c] = [v / M[c][c] for v in M[c]]
        for r in range(n):
            if r != c and M[r][c] != 0:
                f = M[r][c]
                M[r] = [a - f * bb for a, bb in zip(M[r], M[c])]
    return [M[i][n] for i in range(n)]


def _mac(x, s, n):
    """Macaulay bracket <x - s>^n."""
    return (x - s) ** n if x > s else Fr(0)


def solve_reference(supports, udl, points, EI=1):
    """supports: list of x; udl: list of (start, end, w); points: list of (x, P).  Returns a dict of exact results."""
    supports = [Fr(s) for s in supports]
    udl = [(Fr(a), Fr(b), Fr(w)) for a, b, w in udl]
    points = [(Fr(x), Fr(P)) for x, P in points]
    EI = Fr(EI)
    ns = len(supports)
    total = sum(w * (b - a) for a, b, w in udl) + sum(P for _, P in points)
    moment0 = sum(w * (b - a) * ((a + b) / 2) for a, b, w in udl) + sum(P * x for x, P in points)

    def v_loads(x):       # deflection (times EI) due to applied loads, without theta0*x term
        t = Fr(0)
        for xp, P in points:
            t -= P * _mac(x, xp, 3) / 6
        for a, b, w in udl:
            t -= w / 2 * (_mac(x, a, 4) - _mac(x, b, 4)) / 12
        return t

    # unknowns: R_0 .. R_{ns-1}, theta0
    A, rhs = [], []
    A.append([Fr(1)] * ns + [Fr(0)]); rhs.append(total)
    A.append(list(supports) + [Fr(0)]); rhs.append(moment0)
    for k in range(1, ns):
        xk = supports[k]
        A.append([_mac(xk, supports[j], 3) / 6 for j in range(ns)] + [xk * EI])
        rhs.append(-v_loads(xk))
    sol = _solve(A, rhs)
    R, theta0 = sol[:ns], sol[ns]

    def moment(x):
        x = Fr(x)
        m = sum(R[j] * _mac(x, supports[j], 1) for j in range(ns))
        m -= sum(P * _mac(x, xp, 1) for xp, P in points)
        m -= sum(w / 2 * (_mac(x, a, 2) - _mac(x, b, 2)) for a, b, w in udl)
        # a UDL that ends at b keeps acting only up to b: the second bracket cancels the load beyond b
        return m

    def deflection(x):
        x = Fr(x)
        return (theta0 * EI * x + sum(R[j] * _mac(x, supports[j], 3) / 6 for j in range(ns)) + v_loads(x)) / EI

    def rotation(x):
        x = Fr(x)
        t = theta0 * EI + sum(R[j] * _mac(x, supports[j], 2) / 2 for j in range(ns))
        t -= sum(P * _mac(x, xp, 2) / 2 for xp, P in points)
        t -= sum(w / 2 * (_mac(x, a, 3) - _mac(x, b, 3)) / 3 for a, b, w in udl)
        return t / EI

    return {"reactions": R, "moment": moment, "deflection": deflection, "rotation": rotation}
