#!/usr/bin/env python3
"""
evidence_closed_loop.py
Reference implementation for "A unified dynamical theory of catatonia recovery
from an evidence-coupled bistable cascade".

Numerical corroboration only; the complete proofs are in the manuscript
(Results and Methods). Nothing in this file is part of a proof.

MODEL
-----
State  r_k in [0,1], k = 0..3 = (pi_s, beta, pi_m, pi_v_fast):
    dr_0/dt = g(r_0,Q) [ + u(t) ]
    dr_k/dt = g(r_k,Q) + kappa_k (r_{k-1} - r_k),   k = 1..3,   kappa_0 = 0
Evidence  Q in [0,1] = pi_v_slow,  z = r_0 - a(Q):
    dQ/dt = eps [ chi_+(z) C(r) (1-Q) - rho chi_-(z) Q ],   C(r) = prod_{j=0}^{3} r_j
Basin  g(r,Q) = r(1-r)(r-a(Q)),  a(Q) = a0 - (a0-a1) Q^p,  0 < a1 < a0 < 1,  p > 1.

Evidence coordinate:  Q = 1 - exp[-(Lambda-Lambda0)/Lambda*], so Q = 1 corresponds to
Lambda -> inf (a COMPACTIFIED consolidated limit, not finite-time infinite precision).
The general solution is  Q(t) = 1 - (1-Q0) exp[-eps N(t)],  N(t) = int_0^t chi_+(z) C ds.

Selectors (Axiom E, class level): chi_+ >= 0 with chi_+ = 0 for z <= 0 and chi_+ > 0
for z > 0; chi_- symmetric; both continuous and bounded below by a positive constant on
every compact subset of their active half-line (used for the T3c2 and T5 lower bounds).
The theorems use only these sign, support, positivity and continuity properties, so any
selector in the class gives the same results. Smooth realisation here:
chi_+-(z) = exp(-1/z^2). The 1e-9 cutoff below is a floating-point implementation
detail (a ~2e-9 dead zone), not part of the theoretical selector.

STATUS (mirrors Table 2 of the manuscript, "Status of theoretical results")
---------------------------------------------------------------------------
  Proved, unconditional:  T0 positive invariance; T1 sign, convexity and degree;
    T2 recovered state E_rec stable (any rho); T3a frozen-evidence root trigger;
    T3c1 exact invariant switching manifold of the root {r0 = a(Q)} (all eps, rho).
  Proved for rho > 0:  T2 pathological state E_path stable. At rho = 0 the
    Q-eigenvalue is 0 and the corner joins a non-isolated equilibrium continuum
    {(0,0,0,0,Q)}, so it is Lyapunov but not asymptotically stable.
  Proved given the observation-model premise:  T4  Q = 1 - (1-Q0) exp(-eps N); for
    Q0 < 1 strictly increasing in N (not in time: C = 0 freezes it) and Q < 1 at
    every finite time.
  Proved under a SUFFICIENT coupling bound (not necessary):  T3b whole-layer
    propagation. Some non-zero coupling IS necessary (kappa = 0 does not propagate),
    but the analytic bound kappa_k(a1-eta) > M is only sufficient, and propagation is
    seen well below it.
  Proved from a RECOVERED configuration (not over arbitrary data):  T3c2 whole-layer
    outcome and absorption. The barrier r_k > a(Q) >= a1 gives C >= a1^4 > 0, so
    Q -> 1 is settled first, then r0 -> 1, then a triangular scalar asymptotically
    autonomous cascade F_k^+(x) = (1-x)[x(x-a1) + kappa_k] gives r_k -> 1. Absorption
    is not unconditional over all initial data: a root on the recovery side with
    downstream nodes in the pathological basin need not recover them at low coupling.
  Basin boundary:  in the general T1 to T4 regime {r0 = a(Q)} is only the exact root
    switching manifold and need not be the global basin boundary; under the T5
    strong-coupling condition its two invariant regions ARE the basins of E_rec and
    E_path, so S is their common boundary. Not claimed: that E_rec and E_path are the
    only EQUILIBRIA (S carries a continuum E_S(Q) = (a(Q),a(Q),a(Q),a(Q),Q)).
  Proved under STRONG coupling (rho > 0, kappa_k > K*):  T5 global two-outcome. Every
    off-S trajectory goes to E_rec if z(0) > 0 and to E_path if z(0) < 0; E_rec and
    E_path are the only asymptotically stable attractors off S.
    K* = max over a in [a1,a0] of (1-a+a^2)/3, attained at an endpoint by convexity,
    ~= 0.291 (the illustrative kappa = 0.6 satisfies it). Proof: (i) root convergence
    by logistic comparison; (ii) persistence lemma liminf r_k >= kappa_k c_{k-1} /
    (a0+kappa_k) > 0, needing only kappa > 0, hence liminf C > 0 and Q reaches its
    endpoint by selector continuity; (iii) triangular limiting scalar cascade with
    F_k strictly decreasing (kappa > K*) and a unique root, giving stagewise
    convergence. A coupling condition is NECESSARY: for all sufficiently small
    positive coupling a locally asymptotically stable mixed equilibrium exists by
    implicit-function continuation from kappa = 0, carried out on the invariant Q = 1
    face, where the four-node state block is lower-triangular with negative diagonal
    and the transverse Q-eigenvalue is -eps chi_+ (1-a1) C < 0. The threshold
    kappa < a1^2/4 is the first-stage low-root condition, not the whole-branch one.
Axiom E is a disclosed modelling commitment; clinical validity is empirical.

AXIOM E IS THE LOAD-BEARING IDEALISATION
----------------------------------------
The mutual exclusivity of the selectors (disjoint supports, both vanishing at z = 0) is
what makes S invariant at every timescale: on S both selectors are zero, the evidence is
frozen, and no trajectory crosses. The exactness of T3c1 and its eps-independence are
therefore CONSEQUENCES OF THAT CHOICE, not independent findings. The axiom was selected
for analytic tractability and is not derived from the observation model; the change-point
reading that motivates it is, in its standard form, a soft procedure. It is offered as a
falsifiable limit: the Table 1 row-4 ablation replaces the exclusive selectors by an
overlapping soft pair and the switching boundary becomes eps-dependent, approaching the
exact Q_c as the evidence timescale is made slow (0.5935 at eps = 0.02, 0.5820 at
eps = 0.005, against Q_c = 0.5774). A measured boundary moving systematically with the
rate of evidence accumulation would count against the idealisation.

ILLUSTRATIVE NUMBERS
--------------------
Every number this script prints (milestone and consolidation times, boundary locations,
the mixed-branch eigenvalues) is illustrative and depends on the selector shape and the
parameter choice. Because chi_+(z) = exp(-1/z^2) is extremely small for small z,
TIMESCALES IN PARTICULAR MUST NOT BE READ AS PREDICTED REAL TIMES. The theorems depend
only on the class properties stated in Axiom E and are unaffected by this dependence.

DISPLAY ITEMS
-------------
Running the script regenerates the display items in the order in which they appear in
the manuscript: Table 1 (structural dependence), Figure 1, Table 2 (status of
theoretical results), Table 3 (division of labour). ALL FOUR ARE DRAWN: Figure 1 and the
three tables are written as PNG files next to this script (figure1.png, table1.png,
table2.png, table3.png) before the integrations start, so they do not depend on the rest
of the run finishing. Each table PNG is drawn from the same row text and the same column
proportions as the ASCII version, so the drawn and the printed table cannot drift apart.
The ASCII tables and the figure legend are printed later, in manuscript order. In a
notebook the images are also displayed inline as they are written.

USAGE:  python3 evidence_closed_loop.py [--quick | --full]
                                        [--no-figure | --figure-only]
                                        [--figure-path PATH]
        --figure-only writes Figure 1 and Tables 1 to 3 as PNGs and exits, without
        running the verification.
        --figure-path sets the Figure 1 output file (default: figure1.png beside this
        script); the three table PNGs are written into the same directory. The directory
        is created if it does not exist.
        --no-figure skips all four images; the ASCII tables still print.
        Runtime is integration bound; quick is the default.
"""
import os
import sys
import time
import numpy as np
from scipy.integrate import solve_ivp

NS = 4
NAMES = ['pi_s', 'beta', 'pi_m', 'pi_v_fast']
KAPPA = np.array([0.0, 0.6, 0.6, 0.6])
EPS, RHO = 0.02, 1.0
A0, A1, P = 0.60, 0.15, 2.0
QUICK = '--quick' in sys.argv or '--full' not in sys.argv
FIGURE = '--no-figure' not in sys.argv
FIGURE_ONLY = '--figure-only' in sys.argv


def _default_figure_path():
    """Absolute path for Figure 1: next to this script unless --figure-path says
    otherwise, so the file cannot land in whatever working directory an editor or a
    notebook happens to be using."""
    for i, a in enumerate(sys.argv):
        if a == '--figure-path' and i + 1 < len(sys.argv):
            return os.path.abspath(sys.argv[i + 1])
        if a.startswith('--figure-path='):
            return os.path.abspath(a.split('=', 1)[1])
    try:
        here = os.path.dirname(os.path.abspath(__file__))
    except NameError:                      # pasted into a REPL / exec'd
        here = os.getcwd()
    return os.path.join(here, 'figure1.png')


def _in_notebook():
    """True in Jupyter or Colab, where a saved file is not visible by itself and where
    sys.exit() would surface as a SystemExit traceback."""
    try:
        from IPython import get_ipython
        ip = get_ipython()
        if ip is None:
            return False
        return hasattr(ip, 'kernel') or 'colab' in type(ip).__module__
    except Exception:
        return False


def _show_inline(path):
    """In a notebook, display the PNG that was just written. The saved file is the one
    displayed, so what the reader sees is exactly the figure of the manuscript. Outside a
    notebook this does nothing."""
    if not _in_notebook():
        return False
    try:
        from IPython.display import Image, display
        display(Image(filename=path))
        return True
    except Exception as exc:
        print(f"     [Figure 1] saved, but could not be displayed inline ({type(exc).__name__}: {exc});"
              f" open {path} directly")
        return False


def a_of_Q(Q, a0=A0, a1=A1, p=P):
    return a0 - (a0 - a1) * np.clip(Q, 0, 1) ** p


def chi_p(z):
    return float(np.exp(-1.0 / z ** 2)) if z > 1e-9 else 0.0   # 1e-9: float detail, not theory


def chi_m(z):
    return float(np.exp(-1.0 / z ** 2)) if z < -1e-9 else 0.0


def rhs(t, x, kappa=KAPPA, eps=EPS, rho=RHO, a0=A0, a1=A1, p=P, u=None):
    r = np.clip(x[:NS], 0, 1)
    Q = min(max(x[NS], 0.0), 1.0)
    a = a_of_Q(Q, a0, a1, p)
    z = r[0] - a
    dr = np.zeros(NS)
    dr[0] = r[0] * (1 - r[0]) * (r[0] - a) + (u(t) if u else 0.0)
    for k in range(1, NS):
        dr[k] = r[k] * (1 - r[k]) * (r[k] - a) + kappa[k] * (r[k - 1] - r[k])
    C = np.prod(r)
    dQ = eps * (chi_p(z) * C * (1 - Q) - rho * chi_m(z) * Q)
    return np.concatenate([dr, [dQ]])


def integ(x0, T, **kw):
    s = solve_ivp(lambda t, x: rhs(t, x, **kw), [0, T], x0, max_step=0.5, rtol=1e-9, atol=1e-11)
    if not s.success:
        raise RuntimeError('solver failed: ' + s.message)
    return s.y[:, -1]


# ---------------------------------------------------------------- proved (unconditional)
def check_T0():
    n = 5000 if QUICK else 20000
    rng = np.random.default_rng(0); bad = 0
    for _ in range(n):
        x = rng.random(NS + 1); k = rng.integers(NS)
        x[k] = 0.0
        if rhs(0, x)[k] < -1e-12: bad += 1
        x[k] = 1.0
        if rhs(0, x)[k] > 1e-12: bad += 1
        xp = x.copy(); xp[NS] = 0.0
        if rhs(0, xp)[NS] < -1e-12: bad += 1
        xp[NS] = 1.0
        if rhs(0, xp)[NS] > 1e-12: bad += 1
    print(f"[T0 proved] invariance of [0,1]^5: {bad} violations in {n} samples -> "
          f"{'PASS' if bad == 0 else 'FAIL'}")


def check_T1():
    print("[T1 proved] evidence-mediated acceleration: dr/dQ>0, convexity d2r/dQ2>0; degree a''/a'=(p-1)/Q:")
    for Q in (0.25, 0.5, 0.75):
        ap = -P * (A0 - A1) * Q ** (P - 1); app = -P * (P - 1) * (A0 - A1) * Q ** (P - 2)
        print(f"        Q={Q}: a''/a'={app/ap:.4f}  (p-1)/Q={(P-1)/Q:.4f}")
    print(f"     absolute curvature p(p-1)(a0-a1)r(1-r)Q^(p-2): {'constant in Q' if P == 2 else 'p-dependent'} "
          f"for p={P} (low-Q dominance is RELATIVE)")


def check_T2():
    print("[T2] recovered stable for ANY rho; pathological needs rho>0:")
    xE = np.ones(NS + 1)
    lam_rec = -EPS * chi_p(1 - A1)
    print(f"     E_rec: dQ={rhs(0, xE)[NS]:.1e} (exact); Q-eigenvalue=-eps*chi+(1-a1)={lam_rec:.2e} (<0, any rho)")
    xP = np.zeros(NS + 1)
    print(f"     E_path: dQ={rhs(0, xP)[NS]:.1e} (exact)")
    for rho in (1.0, 0.0):
        h = 1e-7
        lam = (rhs(0, np.array([0, 0, 0, 0, h]), rho=rho)[NS]) / h
        tag = '<0 hyperbolic' if lam < -1e-9 else '=0 NON-hyperbolic'
        print(f"     E_path rho={rho}: Q-eigenvalue={lam:.2e} ({tag})")


def check_T3a():
    print("[T3a proved] frozen-evidence ROOT trigger (root-only criterion r0(T)<0.15):")
    for Delta in (0.55, 0.50):
        r0a = 1.0 - Delta; Qc = ((A0 - r0a) / (A0 - A1)) ** (1.0 / P)
        lo, hi = 0.30, 0.95
        for _ in range(11 if QUICK else 15):
            m = (lo + hi) / 2
            x = np.array([1.0] * NS + [m]); x[0] = r0a
            root_fell = integ(x, 3000)[0] < 0.15          # ROOT only, not whole-layer
            lo, hi = (m, hi) if root_fell else (lo, m)
        print(f"        Delta={Delta}: r0+={r0a:.2f} predicted Q_c={Qc:.4f} measured={(lo+hi)/2:.4f}")


def check_T3c1_separatrix():
    Delta = 0.55; Qc = ((A0 - (1 - Delta)) / (A0 - A1)) ** (1.0 / P)
    print(f"[T3c1 proved] exact invariant SWITCHING MANIFOLD of root: boundary=Q_c={Qc:.5f} at every eps:")

    def boundary(eps):
        lo, hi = 0.30, 0.95
        for _ in range(16 if QUICK else 22):
            m = (lo + hi) / 2
            x = np.array([1.0] * NS + [m]); x[0] = 1.0 - Delta
            root_fell = integ(x, 2500 if QUICK else 4000, eps=eps)[0] < 0.15   # ROOT only (T3c1)
            lo, hi = (m, hi) if root_fell else (lo, m)
        return (lo + hi) / 2
    for eps in (0.02, 0.005):        # both timescales in every mode: the signature is eps-independence
        b = boundary(eps)
        print(f"        eps={eps}: root boundary={b:.5f} (|b-Q_c|={abs(b-Qc):.5f}; exact, not singular-limit)")


def check_P1():
    """Proposition 1: under root-initiated ordered initial data the four state
    variables keep the prerequisite order, so their first-passage times at a common
    threshold are ordered. Second half: in the unforced system the state diagonal is
    invariant, so an exactly uniform state onset recovers synchronously and no order
    appears. Corroboration only; the proof is in the manuscript Methods, where the
    ordered cone {d_k = r_(k-1) - r_k >= 0} is shown to be forward invariant because
    the field is subtangential on each face d_k = 0 (self terms cancel there).
    The ordering asserted is non-strict, which is what the tolerances below test."""
    q = 0.5
    kappas = (0.2, 0.6) if QUICK else (0.05, 0.2, 0.6, 1.2)
    roots = (0.8, 0.99) if QUICK else (0.5, 0.8, 0.99)
    Q0s = (0.0, 0.9) if QUICK else (0.0, 0.3, 0.9)
    worst, bad, n = 0.0, 0, 0
    for kap in kappas:
        K = np.array([0.0, kap, kap, kap])
        for r0 in roots:
            for Q0 in Q0s:
                if r0 <= a_of_Q(Q0):
                    continue                      # not a root-initiated recovery
                n += 1
                sol = solve_ivp(lambda t, y: rhs(t, y, kappa=K), [0, 400],
                                np.array([r0, 0.0, 0.0, 0.0, Q0]),
                                t_eval=np.linspace(0, 400, 20001),
                                rtol=1e-10, atol=1e-12, max_step=0.5)
                if not sol.success:
                    raise RuntimeError('solver failed: ' + sol.message)
                y = sol.y
                gaps = [(y[k] - y[k + 1]).min() for k in range(NS - 1)]
                worst = min(worst, min(gaps))
                T = []
                for k in range(NS):
                    i = int(np.argmax(y[k] >= q))
                    T.append(sol.t[i] if y[k][i] >= q else np.inf)
                if min(gaps) < -1e-9 or any(T[k] > T[k + 1] + 1e-9 for k in range(NS - 1)):
                    bad += 1
    print(f"[P1 proved] root-initiated order preservation, {n} configurations "
          f"(kappa, root level, Q0): {bad} violations, worst gap "
          f"min(r_k - r_k+1) = {worst:.2e} -> {'PASS' if bad == 0 else 'FAIL'}")
    spread = 0.0
    for v in (0.3, 0.7):
        for Q0 in (0.2, 0.8):
            sol = solve_ivp(lambda t, y: rhs(t, y), [0, 400],
                            np.array([v, v, v, v, Q0]),
                            t_eval=np.linspace(0, 400, 8001),
                            rtol=1e-10, atol=1e-12, max_step=0.5)
            spread = max(spread, float(np.abs(sol.y[:NS] - sol.y[0]).max()))
    print(f"     state diagonal invariant under u = 0: max spread over the four states "
          f"= {spread:.1e} (exactly uniform onset recovers synchronously,")
    print("     so ordered recovery needs the symmetry breaking supplied by u(t))")


# ---------------------------------------------------------------- premise
def check_P2():
    """Proposition 2: within the separable update dQ = eps[alpha(z) C(r) (1 - Q)
    - rho beta(z) Q], invariance of S = {r0 = a(Q)} forces alpha(0) = beta(0) = 0.
    Corroboration only; the proof is in the manuscript Methods. On S the root drift
    vanishes, so dz/dt = -a'(Q) dQ/dt and invariance is equivalent to dQ/dt = 0 there.
    With the exclusive selectors of Axiom E this holds identically; with an overlapping
    soft pair (alpha(0) = beta(0) = 1/2) it fails, so S is not invariant. The two
    downstream configurations below isolate the two halves of the proof: C(r) = 0
    witnesses beta(0) = 0 and C(r) > 0 witnesses alpha(0) = 0."""
    def soft(z, w=0.25):
        return 1.0 / (1.0 + np.exp(-z / w))
    pairs = (('Axiom E', chi_p, chi_m),
             ('overlapping soft', lambda z: soft(z), lambda z: soft(-z)))
    rest = ((0.0, 0.0, 0.0), (0.3, 0.5, 0.7), (0.9, 0.9, 0.9))
    worst = {}
    for name, al, be in pairs:
        w = 0.0
        for Q in np.linspace(0.05, 0.95, 19):
            a = a_of_Q(Q)
            for d in rest:
                C = a * d[0] * d[1] * d[2]
                w = max(w, abs(EPS * (al(0.0) * C * (1 - Q) - RHO * be(0.0) * Q)))
        worst[name] = w
    ok = worst['Axiom E'] == 0.0 and worst['overlapping soft'] > 0.0
    print(f"[P2 proved] necessity of alpha(0) = beta(0) = 0 for invariance of S: "
          f"max |dQ/dt| on S = {worst['Axiom E']:.1e} (Axiom E) vs "
          f"{worst['overlapping soft']:.3e} (overlapping soft pair) -> "
          f"{'PASS' if ok else 'FAIL'}")


def rhs_augmented(t, x):
    """Full system with the evidence integral N(t) = int chi_+(z) C ds as a coordinate."""
    d = rhs(t, x[:NS + 1])
    r = np.clip(x[:NS], 0, 1); Q = min(max(x[NS], 0.0), 1.0)
    dN = chi_p(r[0] - a_of_Q(Q)) * float(np.prod(r))
    return np.concatenate([d, [dN]])


def check_T4():
    print("[T4 premise] closed form Q(t)=1-(1-Q0)exp(-eps N(t)) vs integration, N carried as a coordinate:")
    T = 400.0
    for Q0 in ((0.0, 0.4) if QUICK else (0.0, 0.4, 0.8)):
        x0 = np.array([1, 1, 1, 1, Q0, 0.0], float)
        s = solve_ivp(rhs_augmented, [0, T], x0, max_step=0.5, rtol=1e-9, atol=1e-11, dense_output=True)
        if not s.success:
            raise RuntimeError('solver failed: ' + s.message)
        worst = 0.0
        for t in (T / 8, T / 4, T / 2, T):
            y = s.sol(t)
            worst = max(worst, abs(y[NS] - (1 - (1 - Q0) * np.exp(-EPS * y[NS + 1]))))
        yT = s.sol(T)
        print(f"        Q0={Q0}: Q({T:.0f})={yT[NS]:.10f}  closed form={1-(1-Q0)*np.exp(-EPS*yT[NS+1]):.10f}  "
              f"max|difference|={worst:.1e}")
    xf = integ(np.array([1, 1, 1, 1, 0.4], float), 20000)[NS]
    print(f"        Q0=0.4: Q(t=20000)={xf:.4f} (Q -> 1 asymptotically; Q < 1 at every finite time)")


# ---------------------------------------------------------------- conditional
def check_T3b():
    print("[T3b] propagation: sufficient bound NOT necessary; some coupling IS:")
    M = max(r * (1 - r) * (r - A1) for r in np.linspace(A1, 1, 400)); M = max(M, 0)
    bound = M / (A1 - 0.05)
    print(f"     conservative sufficient bound kappa>{bound:.3f}; propagation vs kappa:")
    for kap in ((0.0, 0.05, 0.6) if QUICK else (0.0, 0.05, 0.1, 0.4, 0.6, 1.2)):
        x = np.array([1.0] * NS + [0.3]); x[0] = 1 - 0.55
        xf = integ(x, 3000, kappa=np.array([0.0, kap, kap, kap]))
        rmax = xf[:NS].max()
        print(f"        kappa={kap:.2f} ({'>' if kap > bound else '<'}bound): "
              f"{'COLLAPSE' if rmax < 0.15 else 'no/partial'} (max r={rmax:.3f}, Q={xf[NS]:.3f})")


def check_T3c2_absorption():
    print("[T3c2] absorption needs a RECOVERED config; not unconditional over data:")
    Delta = 0.55; Qc = ((A0 - (1 - Delta)) / (A0 - A1)) ** (1.0 / P)
    x = np.array([1.0] * NS + [Qc + 0.1]); x[0] = 1 - Delta        # recovered config, sub-threshold
    print(f"     recovered config, sub-threshold: r_min={integ(x, 3000)[:NS].min():.3f} -> absorbed")
    for kap in (0.0, 0.6):
        xf = integ(np.array([0.9, 0.0, 0.0, 0.0, 0.7]), 3000, kappa=np.array([0.0, kap, kap, kap]))
        print(f"     root-recovered but downstream=0, kappa={kap}: r={np.round(xf[:NS],2)} "
              f"({'STUCK => IC needed' if xf[:NS].min() < 0.5 else 'recovered'})")


# ---------------------------------------------------------------- Jacobian helpers (T5 weak coupling)
def _g_r(r, a):
    # d/dr [ r(1-r)(r-a) ] = -3r^2 + 2(1+a)r - a
    return -3.0 * r ** 2 + 2.0 * (1.0 + a) * r - a


def mixed_eq_eigs_analytic(xf, Kw):
    """Analytic spectrum of the mixed equilibrium on the invariant Q=1 face.
    State block (r0..r3) is lower-bidiagonal, so eigenvalues are its diagonal
    g_r(r_k,a)-kappa_k (kappa_0=0); transverse Q-eigenvalue is -eps*chi+(z)*C."""
    r = np.clip(xf[:NS], 0, 1); Q = min(max(xf[NS], 0.0), 1.0); a = a_of_Q(Q)
    lam = np.empty(NS + 1)
    lam[0] = _g_r(r[0], a)
    for k in range(1, NS):
        lam[k] = _g_r(r[k], a) - Kw[k]
    z = r[0] - a
    lam[NS] = -EPS * chi_p(z) * float(np.prod(r))
    return lam


def numeric_jacobian_inward(func, x, h=1e-6):
    """Central difference in the interior; inward one-sided difference within h of a
    [0,1] boundary, so the clipping in rhs() cannot turn a central difference into a
    biased half-step."""
    n = x.shape[0]
    J = np.zeros((n, n))
    f0 = func(x)
    for j in range(n):
        if x[j] > 1.0 - h:
            xm = x.copy(); xm[j] -= h
            J[:, j] = (f0 - func(xm)) / h
        elif x[j] < h:
            xp = x.copy(); xp[j] += h
            J[:, j] = (func(xp) - f0) / h
        else:
            xp = x.copy(); xm = x.copy(); xp[j] += h; xm[j] -= h
            J[:, j] = (func(xp) - func(xm)) / (2.0 * h)
    return J


# ---------------------------------------------------------------- global (strong coupling): T5
def check_T5():
    Kstar = max((1 - A0 + A0**2) / 3, (1 - A1 + A1**2) / 3)
    print(f"[T5] global two-outcome under strong coupling; K*=max({(1-A0+A0**2)/3:.3f},"
          f"{(1-A1+A1**2)/3:.3f})={Kstar:.3f}; illustrative kappa=0.6 {'>' if 0.6>Kstar else '<'} K*:")
    # S carries an equilibrium continuum E_S(Q)=(a(Q),...,Q): |d|=0 for all Q
    dmax = max(np.abs(rhs(0, np.array([a_of_Q(Q)]*NS + [Q]), kappa=np.array([0,0.6,0.6,0.6]))).max()
               for Q in (0.2, 0.5, 0.8))
    print(f"     S-continuum E_S(Q)=(a(Q),...,Q): max|d| over Q={dmax:.1e} (=0 => only 2 ATTRACTORS off S, not 2 equilibria)")
    # strong coupling: representative off-manifold ICs converge by sign of z(0)
    K = np.array([0, 0.6, 0.6, 0.6])
    ics = [[0.9, 0.0, 0.0, 0.0, 0.7], [0.2, 1.0, 1.0, 1.0, 0.9]] if QUICK else \
          [[0.9, 0.0, 0.0, 0.0, 0.7], [0.6, 0.2, 0.9, 0.1, 0.5], [0.2, 1.0, 1.0, 1.0, 0.9], [0.05]*4 + [0.05]]
    print("     representative off-manifold ICs (finite sample, not a proof):")
    for ic in ics:
        z0 = ic[0] - a_of_Q(ic[4]); xf = integ(np.array(ic, float), 8000, kappa=K)
        corner = 'E_rec' if xf[:NS].min() > 0.85 else ('E_path' if xf[:NS].max() < 0.15 else 'MIXED')
        ok = (z0 > 0 and corner == 'E_rec') or (z0 < 0 and corner == 'E_path')
        print(f"     z(0)={'+' if z0>0 else '-'}{abs(z0):.2f} -> {corner} {'OK' if ok else 'MISMATCH'}")
    # weak coupling: a locally asymptotically stable mixed equilibrium exists (implicit-function
    # continuation from kappa=0). kappa<a1^2/4 is the first-stage low-root threshold only.
    kw = 0.003; Kw = np.array([0, kw, kw, kw])
    xf = integ(np.array([1.0, 0.02, 0.01, 0.005, 1.0]), 20000, kappa=Kw)
    res = float(np.max(np.abs(rhs(0, xf, kappa=Kw))))
    # PRIMARY: analytic eigenvalues. On the invariant Q=1 face the 5x5 Jacobian is block
    # upper-triangular (d(dQ)/dr_i=0 at Q=1 since (1-Q)=0 and chi_-=chi_-'=0 for z>0), so
    # its spectrum is {diagonal of the lower-triangular state block} U {Q-eigenvalue}.
    lam_an = mixed_eq_eigs_analytic(xf, Kw)
    max_re_an = float(lam_an.real.max())
    # AUXILIARY: numeric Jacobian with inward one-sided differences at the [0,1] boundaries.
    J = numeric_jacobian_inward(lambda y: rhs(0, y, kappa=Kw), xf)
    max_re_num = float(np.linalg.eigvals(J).real.max())
    # the transverse Q-eigenvalue -eps*chi+(z)*C is provably negative (all factors positive on
    # the branch) and O(eps*C) tiny, because the deepest downstream node sits near 0 at weak
    # coupling; the state-block eigenvalues are strongly negative, so the equilibrium is stable.
    state_max = float(lam_an[:NS].real.max())
    stable = res < 1e-6 and max_re_an < 1e-12 and state_max < 0.0 and max_re_num < 1e-8
    print(f"     weak coupling kappa={kw} (a1^2/4={A1**2/4:.4f} is first-stage low-root threshold only):")
    print(f"        mixed eq r={np.round(xf[:NS],4)} Q={xf[NS]:.4f}; residual |f|inf={res:.1e}")
    print(f"        analytic eig: state-block max={state_max:.2e}, Q-dir={lam_an[NS]:.2e}; "
          f"numeric(inward) max={max_re_num:.2e}")
    print(f"        -> {'GENUINE STABLE mixed eq (global bistability FALSE)' if stable else 'CHECK'}")
    # Prediction 1 reads this branch. Descriptive only; nothing above is recomputed.
    print("        PREDICTION 1 reads this branch: it is NOT a mild residual state. The root is")
    print("        fully recovered while policy, motivational and fast-volatility precision are")
    print("        essentially absent at fully consolidated evidence, i.e. a severe stationary")
    print("        partial state (restored reactivity, absent initiation, absent contextual")
    print(f"        stability). State-block eigenvalues are of order {abs(state_max):.0e} while the evidence")
    print("        direction is effectively neutral, so the configuration is entered on the")
    print("        ordinary recovery timescale and then persists rather than drifting.")


# ---------------------------------------------------------------- structural dependence (Table 1)
def check_structural_dependence():
    """One ablation per row of Table 1. Illustrative: each removal loses at least the
    corresponding component. This is not a proof of logical independence or uniqueness."""
    print("[Table 1] structural dependence, one ablation per row (illustrative; not a proof")
    print("          of logical independence or uniqueness of the axioms):")

    # Row 1. Bistability (Axiom B) -> a distinct collapse basin.
    # Ablation: monostable deficit-closing field g_mono = r(1-r), as in the companion
    # recovery-order axioms. The sub-threshold excursion that collapses here recovers there.
    def rhs_mono(t, x):
        r = np.clip(x[:NS], 0, 1); Q = min(max(x[NS], 0.0), 1.0)
        z = r[0] - a_of_Q(Q)
        dr = np.zeros(NS)
        dr[0] = r[0] * (1 - r[0])
        for k in range(1, NS):
            dr[k] = r[k] * (1 - r[k]) + KAPPA[k] * (r[k - 1] - r[k])
        C = np.prod(r)
        return np.concatenate([dr, [EPS * (chi_p(z) * C * (1 - Q) - RHO * chi_m(z) * Q)]])
    x = np.array([1.0] * NS + [0.30]); x[0] = 1 - 0.55
    full_max = integ(x, 3000)[:NS].max()
    s = solve_ivp(rhs_mono, [0, 3000], x, max_step=0.5, rtol=1e-9, atol=1e-11)
    if not s.success:
        raise RuntimeError('solver failed: ' + s.message)
    mono_min = s.y[:NS, -1].min()
    print(f"     remove bistability (monostable g=r(1-r)): same excursion gives max r={full_max:.2f} "
          f"with Allee, min r={mono_min:.2f} without -> no distinct collapse basin")

    # Row 2. Concavity of a(Q) (Axiom D) -> convex evidence-to-rate gain.
    print(f"     remove concavity (p=1): a''={-1*(1-1)*(A0-A1):.1f} -> no convex evidence-to-rate gain")

    # Row 3. Evidence gating C(r) (Axiom C) -> evidence-mediated cross-level self-catalysis.
    # Ablation: dQ/dt with C replaced by 1, so evidence accrues while a prerequisite is offline
    # and downstream recovery no longer drives the evidence rate.
    xt = np.array([1.0, 1.0, 1.0, 0.0, 0.5])          # one downstream node offline
    dQ_gated = rhs(0, xt)[NS]
    z = xt[0] - a_of_Q(xt[NS])
    dQ_ungated = EPS * (chi_p(z) * (1 - xt[NS]) - RHO * chi_m(z) * xt[NS])
    print(f"     remove evidence gating (C=1): one node offline gives dQ={dQ_gated:.1e} gated vs "
          f"{dQ_ungated:.1e} ungated -> no cross-level self-catalysis")

    # Row 4. Regime exclusivity (Axiom E) -> exact finite-timescale root switching manifold.
    def rhs_soft(t, xx, eps=0.02):
        r = np.clip(xx[:NS], 0, 1); Q = min(max(xx[NS], 0), 1); a = a_of_Q(Q)
        S = 1.0 / (1.0 + np.exp(-25 * (a - r[0])))     # soft sigmoid: accumulation/attenuation overlap
        dr = np.zeros(NS); dr[0] = r[0]*(1-r[0])*(r[0]-a)
        for k in range(1, NS): dr[k] = r[k]*(1-r[k])*(r[k]-a) + KAPPA[k]*(r[k-1]-r[k])
        C = np.prod(r)
        return np.concatenate([dr, [eps*((1-S)*C*(1-Q) - RHO*S*Q)]])
    Qc = ((A0 - (1 - 0.55)) / (A0 - A1)) ** (1.0 / P)
    bnds = []
    for eps in (0.02, 0.005):
        lo, hi = 0.30, 0.95
        for _ in range(14):
            m = (lo + hi) / 2; x = np.array([1.0]*NS+[m]); x[0] = 1-0.55
            s = solve_ivp(lambda t, y: rhs_soft(t, y, eps), [0, 2500], x, max_step=0.5, rtol=1e-8, atol=1e-10)
            if not s.success:
                raise RuntimeError('solver failed: ' + s.message)
            lo, hi = (m, hi) if s.y[:NS, -1].max() < 0.15 else (lo, m)
        bnds.append((lo + hi) / 2)
    print(f"     remove regime exclusivity (soft selector): boundary eps=0.02 -> {bnds[0]:.4f}, "
          f"eps=0.005 -> {bnds[1]:.4f} (eps-dependent, NOT exact vs Q_c={Qc:.4f})")

    # Row 5. Directional coupling (Axiom A) -> whole-layer propagation.
    x = np.array([1.0] * NS + [0.3]); x[0] = 1 - 0.55
    print(f"     remove coupling (kappa=0): downstream={np.round(integ(x,3000,kappa=np.zeros(NS))[1:NS],2)} "
          f"-> no whole-layer propagation")


# ---------------------------------------------------------------- display items, in manuscript order
def _print_table(rows, widths, caption):
    line = '     +' + '+'.join('-' * (w + 2) for w in widths) + '+'
    print(line)
    for i, row in enumerate(rows):
        cells = [str(c).split('\n') for c in row]
        height = max(len(c) for c in cells)
        for h in range(height):
            parts = []
            for c, w in zip(cells, widths):
                parts.append(' ' + (c[h] if h < len(c) else '').ljust(w) + ' ')
            print('     |' + '|'.join(parts) + '|')
        print(line)
    for l in caption:
        print('     ' + l)


def _wrap(text, width):
    out, line = [], ''
    for word in text.split():
        if len(line) + len(word) + (1 if line else 0) > width:
            out.append(line); line = word
        else:
            line = (line + ' ' + word) if line else word
    if line:
        out.append(line)
    return '\n'.join(out)


TABLE1_ROWS = [('Structure removed', 'Component lost'),
               ('Bistability (Axiom B)', 'A distinct collapse basin'),
               ('Concavity of a(Q) (Axiom D)', 'Convex evidence-to-rate gain'),
               ('Evidence gating C(r) (Axiom C)', 'Evidence-mediated cross-level self-catalysis'),
               ('Regime exclusivity (Axiom E)', 'Exact finite-timescale root switching manifold'),
               ('Directional coupling (Axiom A)', 'Whole-layer propagation')]
TABLE1_CAPTION = ('Table 1. Structural dependence within the present construction: removing each '
                  'listed structure eliminates at least the corresponding component. This does not '
                  'assert logical independence or uniqueness of the axioms.')
TABLE1_WIDTHS = (32, 46)


def emit_table1():
    _print_table(TABLE1_ROWS, TABLE1_WIDTHS, _wrap(TABLE1_CAPTION, 80).split('\n'))


_FIG1_SHOWN_INLINE = False

FIG1_LEGEND = ('Figure 1. Dependency structure of the construction. Axiom A, the forward dependency chain, '
               'splits into a recovery direction, in which evidence gating (Axiom C) yields the evidence law '
               '(T4), concave basin deepening (Axiom D) then gives super-linear acceleration (T1) and a stable '
               'recovered state (T2); and a collapse direction, in which bistability and the regime-selective '
               'update (Axioms B and E), triggered by root autonomy, give the fragile window and whole-layer '
               'collapse (T3). Under strong coupling both directions resolve into the global two-outcome '
               'dichotomy (T5). The dashed edge is the self-catalytic feedback by which downstream recovery '
               'raises the evidence rate. The top row maps the paper: independently reported clinical features '
               'motivate the five axioms (A to E), which entail the theorems T1 to T5, which yield the '
               'testable predictions.')


def _draw_figure1(plt, FancyBboxPatch, FancyArrowPatch, path):
    """Draw Figure 1 and save it to path. These are the drawing calls that produced the
    figure embedded in the manuscript; changing them changes that figure."""
    plt.rcParams.update({'font.family': 'DejaVu Sans'})
    INK = '#1a1a1a'; AX = '#2f5c8f'; TH = '#7a3b2e'; GLB = '#3d6b45'; GREY = '#8a8a8a'; MAP = '#3f3f3f'

    fig, ax = plt.subplots(figsize=(6.8, 5.5), dpi=300)
    ax.set_xlim(-4, 100); ax.set_ylim(0, 120); ax.axis('off')

    def box(x, y, w, h, text, edge, fill, fs=8.4, bold=False):
        ax.add_patch(FancyBboxPatch((x - w/2, y - h/2), w, h, boxstyle='round,pad=0.6,rounding_size=2.4',
                                    linewidth=1.15, edgecolor=edge, facecolor=fill))
        ax.text(x, y, text, ha='center', va='center', fontsize=fs, color=INK,
                weight='bold' if bold else 'normal', linespacing=1.15)
        return (x, y, w, h)

    def T(n): return (n[0], n[1] + n[3]/2)
    def B(n): return (n[0], n[1] - n[3]/2)
    def L(n): return (n[0] - n[2]/2, n[1])
    def R(n): return (n[0] + n[2]/2, n[1])

    def arr(a, b, c=GREY, rad=0.0, lw=1.15, ls='-', ms=12):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle='-|>', mutation_scale=ms, lw=lw, color=c,
                                     connectionstyle=f'arc3,rad={rad}', ls=ls, shrinkA=3, shrinkB=3))

    # top banner: the map of the paper
    by, bh, bw = 112.5, 8.5, 21
    m1 = box(14, by, bw, bh, 'reported clinical\nfeatures', MAP, '#f2f2f2', 7.6)
    m2 = box(38, by, bw, bh, 'five axioms\n(A to E)', MAP, '#f2f2f2', 7.6)
    m3 = box(62, by, bw, bh, 'theorems\n(T1 to T5)', MAP, '#f2f2f2', 7.6)
    m4 = box(86, by, bw, bh, 'testable\npredictions', MAP, '#f2f2f2', 7.6)
    for a, b in ((m1, m2), (m2, m3), (m3, m4)):
        arr(R(a), L(b), MAP, lw=1.1, ms=10)
    ax.plot([-2, 100], [104, 104], color='#cccccc', lw=0.8)

    # dependency structure
    A = box(52, 93, 60, 10, 'Axiom A: forward dependency chain\n(directed acyclic graph + diffusive coupling)',
            AX, '#eef3f9', 8.3, True)
    ax.text(30, 83.2, 'recovery direction', ha='center', fontsize=7.4, color=AX, style='italic')
    ax.text(80, 83.2, 'collapse direction', ha='center', fontsize=7.4, color=TH, style='italic')
    C = box(30, 75, 38, 9, 'Axiom C   evidence gating C(r)', AX, '#eef3f9')
    T4 = box(30, 60, 38, 9, 'T4   Q = accumulated evidence', TH, '#f7efec', 8.4, True)
    D = box(30, 45, 38, 9, 'Axiom D   concave deepening a(Q)', AX, '#eef3f9')
    T12 = box(30, 29, 38, 10, 'T1  super-linear acceleration\nT2  recovered state stable', TH, '#f7efec', 8.3, True)
    BE = box(80, 67, 38, 11, 'Axiom B   bistability\nAxiom E   regime-selective update', AX, '#eef3f9', 8.1)
    T3 = box(80, 45, 36, 11, 'T3\nfragile window;\nwhole-layer collapse', TH, '#f7efec', 8.4, True)
    T5 = box(52, 10, 68, 11, 'T5   (strong coupling  \u03ba > K*)\nglobal two-outcome: full recovery or full collapse',
             GLB, '#eef6ef', 8.5, True)

    arr(B(A), T(C), AX); arr(B(C), (30, 64.5), AX); arr((30, 55.5), T(D), AX); arr((30, 40.5), T(T12), AX)
    arr(B(A), T(BE), TH)
    ax.text(64, 79, 'root autonomy', ha='center', fontsize=6.9, color=TH, style='italic')
    arr(B(BE), T(T3), TH)
    arr(B(T12), (44, 15.6), GLB, rad=-0.10)
    arr(B(T3), (60, 15.6), GLB, rad=0.10)
    arr((L(T12)[0], L(T12)[1] + 2), (L(C)[0], L(C)[1] - 2), c=GREY, rad=-0.5, lw=1.05, ls=(0, (4, 2)))
    ax.text(1.2, 52, 'self-\ncatalysis', ha='center', fontsize=6.9, color=GREY, style='italic', linespacing=1.05)

    fig.tight_layout(pad=0.4)
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def make_figure1(path=None, announce=True):
    """Regenerate Figure 1. Returns the absolute path written, or None if it could not
    be written, in which case the reason is printed."""
    path = _default_figure_path() if path is None else os.path.abspath(path)
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    except ImportError as exc:
        print(f"     [Figure 1] NOT WRITTEN: matplotlib is not available ({exc}).")
        print("     [Figure 1] install it with:  python3 -m pip install matplotlib")
        return None
    try:
        _draw_figure1(plt, FancyBboxPatch, FancyArrowPatch, path)
    except Exception as exc:
        print(f"     [Figure 1] NOT WRITTEN: {type(exc).__name__}: {exc}")
        print("     [Figure 1] re-run with --figure-only to see the full traceback")
        if FIGURE_ONLY:
            raise
        return None
    if not os.path.isfile(path):
        print(f"     [Figure 1] NOT WRITTEN: no file at {path} after saving")
        return None
    if announce:
        print(f"     [Figure 1] written to {path} ({os.path.getsize(path) / 1024:.0f} kB)")
    global _FIG1_SHOWN_INLINE
    _FIG1_SHOWN_INLINE = _show_inline(path)
    return path


def _draw_table_png(rows, widths, caption, path, fontsize=8.2):
    """Render one display table to a PNG, using the same row text and the same column
    proportions as the ASCII emitter, so the printed and the drawn table cannot drift
    apart. Booktabs-style rules: top, under the header, bottom. Nothing here computes
    anything; it only draws strings that are already fixed above."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError as exc:
        print(f"     [{caption.split('.')[0]}] NOT WRITTEN: matplotlib is not available ({exc}).")
        return None
    plt.rcParams.update({'font.family': 'DejaVu Sans'})
    INK, RULE, GREY = '#1a1a1a', '#1a1a1a', '#8a8a8a'
    ncol = len(widths)
    total = float(sum(widths))
    fig_width = 7.4 if ncol == 2 else 10.2
    cells = [[str(c).split('\n') for c in row] for row in rows]
    heights = [max(len(c) for c in cell) for cell in cells]
    line_h = fontsize * 1.62 / 72.0                     # inches per text line
    row_pad = 0.115                                    # inches of padding per row
    cap_lines = _wrap(caption, int(total * 1.12)).split('\n')
    body_h = sum(h * line_h + row_pad for h in heights)
    cap_h = len(cap_lines) * line_h + 0.10
    fig_height = body_h + cap_h + 0.42
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=300)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    xs, acc = [0.02], 0.02
    for w in widths:
        acc += 0.96 * w / total
        xs.append(acc)
    fy = lambda inches: inches / fig_height             # inches -> axes fraction
    y = 1.0 - fy(0.12)
    ax.plot([0.02, 0.98], [y, y], color=RULE, lw=1.25)  # top rule
    for i, (cell, h) in enumerate(zip(cells, heights)):
        y -= fy(row_pad * 0.5)
        for c in range(ncol):
            for ln, text in enumerate(cell[c]):
                ax.text(xs[c], y - fy(line_h * (ln + 0.5)), text, ha='left', va='center',
                        fontsize=fontsize, color=INK,
                        weight='bold' if i == 0 else 'normal')
        y -= fy(line_h * h + row_pad * 0.5)
        if i == 0:
            ax.plot([0.02, 0.98], [y, y], color=RULE, lw=1.0)
        elif i < len(cells) - 1:
            ax.plot([0.02, 0.98], [y, y], color='#d8d8d8', lw=0.5)
    ax.plot([0.02, 0.98], [y, y], color=RULE, lw=1.25)  # bottom rule
    y -= fy(0.10)
    for ln, text in enumerate(cap_lines):
        ax.text(0.02, y - fy(line_h * (ln + 0.8)), text, ha='left', va='center',
                fontsize=fontsize - 0.9, color=GREY)
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return path


def make_display_tables(fig_path=None, announce=True):
    """Draw Tables 1 to 3 as PNGs next to Figure 1. Returns {name: path or None}."""
    base = os.path.dirname(_default_figure_path() if fig_path is None else os.path.abspath(fig_path))
    os.makedirs(base or '.', exist_ok=True)
    specs = [('Table1', TABLE1_ROWS, TABLE1_WIDTHS, TABLE1_CAPTION),
             ('Table2', TABLE2_ROWS, TABLE2_WIDTHS, TABLE2_CAPTION),
             ('Table3', TABLE3_ROWS, TABLE3_WIDTHS, TABLE3_CAPTION)]
    out = {}
    for name, rows, widths, caption in specs:
        path = os.path.join(base, name.lower() + '.png')
        try:
            written = _draw_table_png(rows, widths, caption, path)
        except Exception as exc:
            print(f"     [{name}] NOT WRITTEN: {type(exc).__name__}: {exc}")
            written = None
        out[name] = written
        if announce and written:
            print(f"     [{name}] written to {written} ({os.path.getsize(written) / 1024:.0f} kB)")
        if written:
            _show_inline(written)
    return out


TABLE2_ROWS = [('Result', 'Status'),
            ('T0 positive invariance', 'Proved'),
            ('T4 evidence law Q = 1 - (1-Q0)exp(-eps N),\nmonotone in N', 'Proved, given the premise'),
            ('T1 evidence-mediated acceleration,\nconvexity, degree', 'Proved'),
            ('T2 recovered state stable', 'Proved, unconditional in rho'),
            ('T2 pathological state stable', 'Proved for rho > 0\n(non-hyperbolic at rho = 0)'),
            ('T3(a) frozen-evidence root trigger', 'Proved (exact)'),
            ('T3(b) whole-layer propagation', 'Proved under a sufficient coupling\nbound; some coupling necessary,\nbound not'),
            ('T3(c1) exact invariant switching\nmanifold of the root', 'Proved (all eps, rho)'),
            ('T3(c2) whole-layer outcome and\nabsorption', 'Proved from a recovered\nconfiguration (not over arbitrary\ndata)'),
            ('T5 two-outcome convergence off S,\nstrong coupling', 'Proved for rho > 0 and\nkappa_k > K* ~= 0.291'),
            ('Equilibrium continuum on S; stable mixed\nbranch at sufficiently weak coupling', 'Characterised; kappa < a1^2/4 is\nthe explicit first-stage low-root\ncondition'),
            ('S is the global boundary of the two\ncorner basins', 'Proved under T5 strong coupling;\nnot in the general T1 to T4 regime'),
            ('E_rec and E_path are the only equilibria', 'False (equilibrium continuum E_S\non S)'),
            ('Proposition 1 state order under\nroot-initiated recovery', 'Proved (four state nodes;\nMethods)'),
            ('Proposition 2 necessity of regime\nexclusivity for exact switching', 'Proved within the separable\nupdate class (Methods)'),
            ('Milestone order of Q relative to the\nstate nodes', 'Not proved here; the five-parameter\norder is established in the companion\ntheory under a DIFFERENT axiom class\n(refs 13, 14) and is not re-derived'),
            ('Axiom E (regime exclusivity)', 'Modelling commitment on sharpness;\nits exclusivity is necessary for the\nexact switching manifold\n(Proposition 2)'),
            ('Clinical validity of the observation model', 'Empirical; out of scope here')]
TABLE2_CAPTION = ('Table 2. Status of theoretical results. All numerical values quoted in this '
                  'table and in the text are illustrative and depend on the selector shape and '
                  'parameter choice; the theorems depend only on the class properties of Axiom E.')
TABLE2_WIDTHS = (42, 36)


def emit_table2():
    _print_table(TABLE2_ROWS, TABLE2_WIDTHS, _wrap(TABLE2_CAPTION, 80).split('\n'))


TABLE3_ROWS = [('Question', 'Companion identifiability theory\n(refs 13, 14)', 'This paper'),
            ('Why does recovery take this\norder?',
             'Derived from the zero-pattern of\nthe joint Fisher information,\nindependently of any recovery law',
             'Not re-derived. The same order is\nreproduced by a DIFFERENT\nmechanism (Proposition 1)'),
            ('Which axiom class?',
             'Availability gating; deficit-closing\nrecovery; common impaired onset',
             'Diffusive coupling; bistable field;\nordered root-initiated onset. NONE\nof the three companion axioms\nholds here, so the companion\nordering theorem does NOT apply'),
            ('What carries the order?',
             'Prerequisite gating: a capacity\ncannot recover while a prerequisite\nis offline',
             'Forward diffusive coupling, which\nstops a descendant overtaking its\nprerequisite, plus the root input u(t)'),
            ('Does slow-volatility precision\nact back on the upstream\nnodes?',
             'Not represented; the graph is\nacyclic and carries no feedback',
             'Yes. Q lowers the shared threshold\nof every node (T1 to T3), closing\nthe loop through a variable that is\nnot a node of the state graph'),
            ('Why does recovery have this\ndynamics (self-catalysis,\nstability, fragile window,\ncollapse, evidence identity)?',
             'Out of scope',
             'This paper (Theorems T1 to T5)'),
            ('Milestone order of Q against\nthe state nodes',
             'Ordered as the terminal node of\nthe chain',
             'Not proved here (Table 2)')]
TABLE3_CAPTION = ('Table 3. Division of labour with the companion theories, including what does '
                  'not transfer. The present dynamics lie OUTSIDE the companion recovery-order '
                  'axiom class: the same node names, a different axiom class.')
TABLE3_WIDTHS = (30, 34, 36)


def emit_table3():
    _print_table(TABLE3_ROWS, TABLE3_WIDTHS, _wrap(TABLE3_CAPTION, 100).split('\n'))


def main():
    mode = 'quick' if QUICK else 'full'
    print("=" * 78)
    print("A unified dynamical theory of catatonia recovery from an evidence-coupled")
    print(f"bistable cascade : reference implementation  [mode: {mode}]")
    print(f"kappa={[float(k) for k in KAPPA[1:]]} eps={EPS} rho={RHO} a0={A0} a1={A1} p={P}")
    print("=" * 78)

    # Figure 1 is written before the integrations, so that it never depends on the rest
    # of the run finishing. Its legend is printed further down, with the other display
    # items, in the order in which they appear in the manuscript.
    fig1_path, table_paths = None, {}
    if FIGURE:
        print("\n-- Display items drawn --")
        fig1_path = make_figure1()
        table_paths = make_display_tables(fig1_path)
    else:
        print("\n-- Display items skipped (--no-figure) --")
    if FIGURE_ONLY:
        return 0 if (fig1_path or not FIGURE) else 1

    t0 = time.time()
    print("\n-- Proved (unconditional) --")
    check_T0(); check_T1(); check_T2(); check_T3a(); check_T3c1_separatrix(); check_P1(); check_P2()
    print("\n-- Proved given the observation-model premise --")
    check_T4()
    print("\n-- Conditional (coupling / initial configuration) --")
    check_T3b(); check_T3c2_absorption()
    print("\n-- Global two-outcome under strong coupling (T5) --")
    check_T5()
    print("\n-- Structural dependence (Table 1) --")
    check_structural_dependence()
    print("\n" + "=" * 78)
    print("DISPLAY ITEMS, in the order in which they appear in the manuscript")
    print("=" * 78)
    print("\n-- Table 1 --")
    emit_table1()
    if table_paths.get('Table1'):
        print(f"     drawn to: {table_paths['Table1']}")
    print("\n-- Figure 1 --")
    if fig1_path:
        print(f"     file: {fig1_path}"
              + ("  (image displayed above, where it was written)" if _FIG1_SHOWN_INLINE else ""))
        for l in _wrap(FIG1_LEGEND, 80).split('\n'):
            print('     ' + l)
    elif FIGURE:
        print("     [Figure 1] not available; see the message near the top of this run")
    else:
        print("     [Figure 1] skipped (--no-figure)")
    print("\n-- Table 2 --")
    emit_table2()
    if table_paths.get('Table2'):
        print(f"     drawn to: {table_paths['Table2']}")
    print("\n-- Table 3 --")
    emit_table3()
    if table_paths.get('Table3'):
        print(f"     drawn to: {table_paths['Table3']}")
    print("\n" + "-" * 78)
    print(f"done in {time.time() - t0:.1f}s. Status mirrors Table 2; Axiom E is disclosed.")
    return 0


if __name__ == "__main__":
    _code = main()
    if not _in_notebook():          # in Jupyter or Colab, sys.exit() prints a SystemExit traceback
        sys.exit(_code)
