"""Independent reference for the five-support beam: Clapeyron's three-moment equations.

Written from first principles and sharing no code with beam_dsm.  Each span's simple-beam
moment diagram is integrated numerically (not from closed-form tables), the three equations for
supports B, C and D are solved, and reactions follow from statics.  Requires NumPy >= 2.0.
"""
import numpy as np


def three_moment_reference():
    """Return (M, R): support moments at B, C, D (sagging +) and reactions at A..E (kN)."""

    supports=[0.0,6.0,12.5,17.0,22.0]                    # A..E
    udl={(0,6):60.0,(12.5,17.0):20.0,(17.0,22.0):80.0}   # kN/m over [a,b]
    pts=[(8.0,40.0),(10.0,60.0),(14.1,100.0)]            # (x, P kN)
    spans=list(zip(supports[:-1],supports[1:]))
    L=np.array([b-a for a,b in spans])

    def span_load_terms(a0,b0):
        """returns (6*A*abar/L from left end, 6*A*bbar/L from right end) of the simple-beam M diagram
           for ALL loads inside span [a0,b0], via numerical integration of the M diagram (not closed forms)."""
        Ls=b0-a0
        # simple-span reactions
        def simple_M(x):                       # x measured from a0
            R=0.0
            # reaction at left = sum(load*(distance from right))/L
            mom_about_right=0.0
            for (s,e),w in udl.items():
                lo,hi=max(s,a0),min(e,b0)
                if hi>lo: mom_about_right+=w*(hi-lo)*(b0-(lo+hi)/2)
            for xp,P in pts:
                if a0<xp<b0: mom_about_right+=P*(b0-xp)
            R=mom_about_right/Ls
            M=R*x
            for (s,e),w in udl.items():
                lo,hi=max(s,a0),min(e,b0)
                xx=a0+x
                if hi>lo and xx>lo: M-=w*(min(xx,hi)-lo)*(xx-(lo+min(xx,hi))/2)
            for xp,P in pts:
                if a0<xp<b0 and a0+x>xp: M-=P*(a0+x-xp)
            return M
        xs=np.linspace(0,Ls,200001)
        Ms=np.array([simple_M(x) for x in xs[::100]]) if False else None
        # vectorised version
        xx=xs
        R=0.0
        mom_r=0.0
        for (s,e),w in udl.items():
            lo,hi=max(s,a0),min(e,b0)
            if hi>lo: mom_r+=w*(hi-lo)*(b0-(lo+hi)/2)
        for xp,P in pts:
            if a0<xp<b0: mom_r+=P*(b0-xp)
        R=mom_r/Ls
        M=R*xx
        for (s,e),w in udl.items():
            lo,hi=max(s,a0),min(e,b0)
            if hi>lo:
                g=a0+xx
                cov=np.clip(g,lo,hi)-lo
                M-=w*cov*(g-(lo+np.clip(g,lo,hi))/2)
        for xp,P in pts:
            if a0<xp<b0:
                M-=P*np.clip(a0+xx-xp,0,None)
        A=np.trapezoid(M,xx)
        abar=np.trapezoid(M*xx,xx)/A if A else 0
        return 6*A*abar/Ls, 6*A*(Ls-abar)/Ls   # (from-left-end term, from-right-end term)

    terms=[span_load_terms(a,b) for a,b in spans]
    # three-moment: M_{i-1} L_i + 2 M_i (L_i+L_{i+1}) + M_{i+1} L_{i+1} = -(6A a/L)_i_right - (6A b/L)_{i+1}_left  (hogging negative)
    # with 6A*abar/L measured from LEFT end = terms[i][0] ; from RIGHT end = terms[i][1]
    n=len(spans)-1   # interior supports B,C,D
    Amat=np.zeros((n,n)); rhs=np.zeros(n)
    for r in range(n):
        i=r        # span to the left of interior support r+1 is span i, to right is span i+1
        Amat[r,r]=2*(L[i]+L[i+1])
        if r>0: Amat[r,r-1]=L[i]
        if r<n-1: Amat[r,r+1]=L[i+1]
        rhs[r]=-(terms[i][0]+terms[i+1][1])    # left span: centroid from its LEFT (far) end; right span: from its RIGHT (far) end
    M=np.linalg.solve(Amat,rhs)
    # reactions by statics on each span
    R=np.zeros(5)
    Mfull=np.concatenate([[0],M,[0]])
    for i,(a,b) in enumerate(spans):
        Ls=b-a
        mom_r=0.0; tot=0.0
        for (s,e),w in udl.items():
            lo,hi=max(s,a),min(e,b)
            if hi>lo: mom_r+=w*(hi-lo)*(b-(lo+hi)/2); tot+=w*(hi-lo)
        for xp,P in pts:
            if a<xp<b: mom_r+=P*(b-xp); tot+=P
        Rl_simple=mom_r/Ls; Rr_simple=tot-Rl_simple
        # continuity correction: support moments M (sagging +): V_left_end += (M_right - M_left)/L ... derive from moment about right end
        Rl=Rl_simple+(Mfull[i+1]-Mfull[i])/Ls*(-1)*(-1)
        # careful sign: beam segment, take moments about right end: Rl*Ls - mom_r + (-Mfull[i]) ... use M_left, M_right as internal sagging moments
        # M(a)=Mfull[i]  => at left end internal moment sagging+; M(b)=Mfull[i+1]
        # M(b)=Rl*Ls - mom_r + Mfull[i]   (shear-moment relation along span) => Rl=(Mfull[i+1]-Mfull[i]+mom_r)/Ls
        Rl=(Mfull[i+1]-Mfull[i]+mom_r)/Ls
        Rr=tot-Rl
        R[i]+=Rl; R[i+1]+=Rr

    return M, R
