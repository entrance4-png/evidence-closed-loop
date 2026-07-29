#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
evidence_closed_loop.py

Reference implementation for

    Evidence-coupled bistability converts dependency structure into a
    recovery geometry

Model
-----
    dr0/dt = g(r0, Q)                      + u(t)
    drk/dt = g(rk, Q) + kappa_k (r_{k-1} - rk),      k = 1, 2, 3
    dQ/dt  = eps [ chi_+(z) C(r) (1 - Q) - rho chi_-(z) Q ]

    g(r, Q) = r (1 - r) (r - a(Q)),   C(r) = prod_j r_j,
    a(Q)    = a0 - (a0 - a1) Q**p,    z = r0 - a(Q).

The theorems are proved in the Methods for a forward acyclic chain of
n + 1 capacities, k = 1, ..., n.  This implementation is the displayed
case n = 3, the shortest chain with a root, an interior pair and a leaf.

Selectors: a four-step chain, not a single deduction
----------------------------------------------------
E1-E4 (two mutually exclusive regimes; z is the log posterior odds up to
gamma > 0; responsibility-weighted Bayesian model averaging; ambiguity-
suspended accumulation) give the finite-sharpness pair

    w_pm(z) = 1 / (1 + exp(-+ gamma z)),  so log(w_+/w_-) = gamma z
    H(z)    = -sum w log w
    lam(z)  = 1 - H(z) / log 2        (one member of the E4 class)
    chi_pm^gamma = w_pm(z) lam(z)                     Proposition 1(a)

with chi_+(0) = chi_-(0) = 0 at every finite gamma, hence S invariant, and

    chi_pm^gamma -> 1_{+- z > 0}   locally uniformly on R \ {0}
                                                      Proposition 1(b)

That limit is the discontinuous indicator pair.  The theorems are not proved
on it: premise E replaces it by a one-sided regularisation that is locally Lipschitz
on R, C^1 on the interiors of its active half-lines, and bounded below by a
positive constant on compact subsets of them, with the same support, sign and
boundary zero.  Lipschitz continuity buys uniqueness (continuity alone does
not: xdot = sqrt(|x|) is continuous and not unique), and C^1 away from the
boundary buys the Jacobians of T2.  This step is a regularisation choice made
for classical well-posedness, not a deduction; the theorems use only those
structural properties, so no functional form is assumed.  The reference
regularisation exp(-s/z**2) on the active half-line is C^infinity.

E4 likewise fixes a class, not a formula.  The identifiability weight is a
function of the posterior responsibility, lam_gamma(z) = L(w_+^gamma(z)), where
L is continuous on [0,1] with L(1/2) = 0, L(w) > 0 for w != 1/2 and
L(0) = L(1) = 1.  Uniform continuity of L on [0,1] is what carries
lam_gamma -> 1 uniformly on {|z| >= delta}, hence Proposition 1(b) for the whole
class.  Three members (entropy, entropy2, margin) are implemented and checked.

Closure (Proposition 4).  The companion generative model is implemented as
five conditionally independent Poisson channels.  The script checks that the
joint Fisher information about pi_v_slow equals the slow-channel mean, that no
other channel carries that parameter, that the slow count is bracketed between
C(r) and K C(r) (Lemma C1), that the expected number of usable per-context
estimates equals nu times the integral of C(r) (Lemma C3), and that for small
enough eps the evidence coordinate crosses a common threshold after every
capacity (the corollary).  Only the hierarchical reading (Lemma C3) supplies
Axiom C in its exact form: the first-level rate carries the prefactor
exp(pi_v_slow), which is not constant along recovery, so that reading fixes the
zero set and the C(r) dependence but not the rate.

Axiom D (Proposition 5).  A two-policy decision layer selected by expected
free energy is implemented, with Gaussian outcomes and preferences.  The
script checks the informative regime, the signs of the two partial
derivatives the implicit function theorem uses, that the derived threshold
runs from a0 to a1, that it is decreasing and concave in Q when the
coordinate scale satisfies the inequality of part (b), and that it becomes
convex when that inequality fails.

Finite sharpness (Proposition 6) and robustness (Proposition 7).  The script
also runs the finite-gamma system, checking that the margin z stays bounded
away from the switching surface, that the collapse corner is reached exactly
and that the recovered corner is displaced to the zero of
1 - Q = rho exp(-gamma(1 - a(Q))) Q; it checks a gate comparable to the
product gate, a leaky gate, and the O(eps eta) drift of the switching boundary
when the boundary zero of E4 is violated by eta.  A leak combined with finite
sharpness displaces the failed corner as well, to the solution of
Q/(1-Q) = (delta/rho) exp(-gamma a(Q)), and that too is checked.

This file reproduces the results by status and regenerates Tables 1-3,
Figures 1 and 2 and a four-panel supplementary figure.

Command line
------------
    python evidence_closed_loop.py                 # quick mode, all display items
    python evidence_closed_loop.py --full          # finer sweeps
    python evidence_closed_loop.py --fast          # coarser sweeps, ~1/4 the time
                                                   # (the switching-boundary
                                                   # measurements of Proposition 7
                                                   # keep the fine tolerance, so
                                                   # every quoted value is
                                                   # mode-independent)
    python evidence_closed_loop.py --no-figures    # checks only
    python evidence_closed_loop.py --json out.json # also dump every result

Notebook (Jupyter, Google Colab)
--------------------------------
Run the file, or import it and call run() directly:

    !python evidence_closed_loop.py                # works inside a Colab cell
    import evidence_closed_loop as m; res = m.run()

Unknown command-line arguments are ignored, so the -f kernel.json that a
notebook kernel injects into sys.argv does not abort the run, and the figures
are displayed inline when a notebook front end is detected.

Requires numpy, scipy and matplotlib, all preinstalled in Colab.

MIT licence.
"""

from __future__ import annotations

import argparse
import json
import sys
import math
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Tuple

from functools import lru_cache

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

# --------------------------------------------------------------------------
# parameters
# --------------------------------------------------------------------------

LOG2 = math.log(2.0)
FAST = False         # set by --fast: coarser sweeps, same checks
ZCUT = 1e-9          # floating-point cutoff around z = 0 (implementation only)


@dataclass(frozen=True)
class Params:
    """Illustrative parameters.  No search was performed to force any outcome."""
    a0: float = 0.60
    a1: float = 0.15
    p: float = 2.0
    kappa: Tuple[float, float, float] = (0.6, 0.6, 0.6)   # kappa_1..kappa_3
    eps: float = 0.02
    rho: float = 1.0
    gamma: float = 20.0        # finite sharpness of the regime decision
    selector: str = "sharp"    # "sharp" | "bayes" | "bma"
    sel_scale: float = 1.0     # s in chi(z) = exp(-s/z^2); any s > 0 lies in the
                               # regularisation class E, so the theorems are
                               # unaffected
    lam: str = "entropy"       # member of the E4 class: entropy | entropy2 | margin
    eta: float = 0.0           # violation of E4 at the boundary: lam -> eta + (1-eta) L
    gate: str = "product"      # evidence gate h(r): product | modulated | leaky
    gate_leak: float = 0.0     # delta in h = delta + (1-delta) prod r_j, gate = "leaky"

    def with_(self, **kw) -> "Params":
        d = {**self.__dict__, **kw}
        return Params(**d)

    # threshold and its inverse ------------------------------------------------
    def a(self, Q: float | np.ndarray):
        return self.a0 - (self.a0 - self.a1) * np.power(np.clip(Q, 0.0, 1.0), self.p)

    def a_prime(self, Q: float | np.ndarray):
        return -(self.a0 - self.a1) * self.p * np.power(np.clip(Q, 0.0, 1.0), self.p - 1.0)

    def Q_of_a(self, aval: float) -> float:
        """Q such that a(Q) = aval; the Qc of T3(a)."""
        return ((self.a0 - aval) / (self.a0 - self.a1)) ** (1.0 / self.p)

    def K_star(self) -> float:
        """K* = max_{a in [a1,a0]} (1 - a + a^2)/3; the map is convex, so the
        maximum is attained at an endpoint (Methods, T5)."""
        f = lambda a: (1.0 - a + a * a) / 3.0
        return max(f(self.a1), f(self.a0))


# --------------------------------------------------------------------------
# selectors
# --------------------------------------------------------------------------

def chi_sharp(z: float, p: Params) -> Tuple[float, float]:
    """Sharp pair: continuous, disjoint supports, chi(0) = 0."""
    if z > ZCUT:
        return math.exp(-p.sel_scale / (z * z)), 0.0
    if z < -ZCUT:
        return 0.0, math.exp(-p.sel_scale / (z * z))
    return 0.0, 0.0


def responsibilities(z: float, gamma: float) -> Tuple[float, float]:
    """Posterior responsibilities of E3.

    w_+ = 1/(1 + exp(-gamma z)), w_- = 1 - w_+, so that

        log(w_+ / w_-) = gamma z

    exactly as E2 identifies the margin with the log posterior odds.  Written
    through tanh(gamma z / 2) for numerical stability.
    """
    t = math.tanh(0.5 * gamma * z)
    return 0.5 * (1.0 + t), 0.5 * (1.0 - t)


def certainty_weight(w: float, kind: str = "entropy") -> float:
    """The function L of E4, evaluated on the posterior responsibility w.

    E4 fixes a class rather than a formula: L continuous on [0,1] with
    L(1/2) = 0, L(w) > 0 for w != 1/2 and L(0) = L(1) = 1.  Because
    lam_gamma(z) = L(w_+^gamma(z)) and w_+^gamma -> 0 or 1 uniformly on
    {|z| >= delta}, continuity of L gives lam_gamma -> 1 uniformly there, which
    is what Proposition 1(b) needs for the whole class.  Three members:
    """
    w = min(max(w, 0.0), 1.0)
    if kind == "margin":
        return abs(2.0 * w - 1.0)
    h = 0.0
    for q in (w, 1.0 - w):
        if q > 0.0:
            h -= q * math.log(q)
    lam = max(0.0, 1.0 - h / LOG2)
    return lam * lam if kind == "entropy2" else lam


def ambiguity_factor(z: float, gamma: float, kind: str = "entropy") -> float:
    """Identifiability weight lam_gamma(z) = L(w_+^gamma(z)) of E4."""
    wp, _ = responsibilities(z, gamma)
    return certainty_weight(wp, kind)


def chi_bayes(z: float, p: Params) -> Tuple[float, float]:
    """Finite-sharpness pair chi_pm = w_pm lam of Proposition 1(a).

    With p.eta > 0 the identifiability weight is relaxed to
    lam_eta = eta + (1 - eta) L(w_+), so L(1/2) = eta > 0 and the boundary-zero
    condition of Proposition 2 fails by eta.  This is the perturbation used to
    measure how far the switching surface moves when E4 is violated.
    """
    wp, wm = responsibilities(z, p.gamma)
    lam = ambiguity_factor(z, p.gamma, p.lam)
    if p.eta > 0.0:
        lam = p.eta + (1.0 - p.eta) * lam
    return wp * lam, wm * lam


def chi_bma(z: float, p: Params) -> Tuple[float, float]:
    """Plain Bayesian model averaging: responsibilities without the
    identifiability weight of E4.  Here chi_pm(0) = 1/2, so the boundary-zero
    condition of Proposition 2 fails and S is not invariant."""
    return responsibilities(z, p.gamma)


def get_selector(p: Params) -> Callable[[float, Params], Tuple[float, float]]:
    return {"sharp": chi_sharp, "bayes": chi_bayes, "bma": chi_bma}[p.selector]


# --------------------------------------------------------------------------
# vector field
# --------------------------------------------------------------------------

def gate(r: np.ndarray, p: Params) -> float:
    """Evidence gate h(r).

    "product"   h = prod_j r_j, the gate of Axiom C.
    "modulated" h = (prod_j r_j)(1 + r_3)/2, comparable to the product gate with
                constants 1/2 and 1 and with the same zero set (Proposition 7).
    "leaky"     h = delta + (1 - delta) prod_j r_j, a strictly different zero set,
                the case Proposition 7 excludes.
    """
    c = float(np.prod(np.clip(r, 0.0, 1.0)))
    if p.gate == "modulated":
        return c * 0.5 * (1.0 + float(np.clip(r[3], 0.0, 1.0)))
    if p.gate == "leaky":
        return p.gate_leak + (1.0 - p.gate_leak) * c
    return c


def g(r: float | np.ndarray, Q: float, p: Params):
    return r * (1.0 - r) * (r - p.a(Q))


def g_r(r: float, Q: float, p: Params) -> float:
    """d g / d r."""
    a = p.a(Q)
    return -3.0 * r * r + 2.0 * (1.0 + a) * r - a


def rhs(t: float, y: np.ndarray, p: Params,
        u: Callable[[float], float] | None = None) -> np.ndarray:
    r = y[:4]
    Q = float(np.clip(y[4], 0.0, 1.0))
    z = float(r[0] - p.a(Q))
    chi_p, chi_m = get_selector(p)(z, p)
    C = gate(r, p)

    dr = np.empty(4)
    dr[0] = g(r[0], Q, p) + (u(t) if u is not None else 0.0)
    for k in (1, 2, 3):
        dr[k] = g(r[k], Q, p) + p.kappa[k - 1] * (r[k - 1] - r[k])
    dQ = p.eps * (chi_p * C * (1.0 - Q) - p.rho * chi_m * Q)
    return np.concatenate([dr, [dQ]])


def integrate(y0: Sequence[float], T: float, p: Params,
              u: Callable[[float], float] | None = None,
              n_eval: int = 0, max_step: float = 0.5):
    t_eval = np.linspace(0.0, T, n_eval) if n_eval else None
    sol = solve_ivp(rhs, (0.0, T), np.asarray(y0, dtype=float), args=(p, u),
                    method="RK45", rtol=1e-9, atol=1e-11, max_step=max_step,
                    t_eval=t_eval, dense_output=bool(n_eval))
    if not sol.success:
        raise RuntimeError(f"solver failed: {sol.message}")
    return sol


def _outcome_events(p: Params):
    """Terminal events: the root has settled in one basin."""
    def hi(t, y, *a):
        return y[0] - 0.999
    def lo(t, y, *a):
        return y[0] - 0.001
    hi.terminal = True; hi.direction = 1.0
    lo.terminal = True; lo.direction = -1.0
    return [hi, lo]


def root_outcome(y0: Sequence[float], p: Params, T: float = 5.0e4,
                 max_step: float = 2.0) -> int:
    """+1 if the root settles in the recovered basin, -1 if it collapses.

    Integration stops at the first crossing, so the horizon is not a
    statement about the evidence timescale.
    """
    sol = solve_ivp(rhs, (0.0, T), np.asarray(y0, dtype=float), args=(p, None),
                    method="RK45", rtol=1e-9, atol=1e-11, max_step=max_step,
                    events=_outcome_events(p))
    if not sol.success:
        raise RuntimeError(sol.message)
    return 1 if sol.y[0, -1] > 0.5 else -1


def jacobian(y: np.ndarray, p: Params, h: float = 1e-6) -> np.ndarray:
    """Finite-difference Jacobian.

    rhs clips its arguments to the state box, so a central difference taken on
    a face of that box has one arm pinned and returns half the true derivative.
    On a face the difference is therefore taken one-sided and inward, using the
    second-order three-point formula so that the accuracy matches the central
    difference used in the interior.
    """
    y = np.asarray(y, dtype=float)
    J = np.zeros((5, 5))
    for j in range(5):
        e = np.zeros(5)
        e[j] = h
        if y[j] <= 2.0 * h:                 # lower face: second-order forward
            J[:, j] = (-3.0 * rhs(0.0, y, p) + 4.0 * rhs(0.0, y + e, p)
                       - rhs(0.0, y + 2.0 * e, p)) / (2.0 * h)
        elif y[j] >= 1.0 - 2.0 * h:         # upper face: second-order backward
            J[:, j] = (3.0 * rhs(0.0, y, p) - 4.0 * rhs(0.0, y - e, p)
                       + rhs(0.0, y - 2.0 * e, p)) / (2.0 * h)
        else:
            J[:, j] = (rhs(0.0, y + e, p) - rhs(0.0, y - e, p)) / (2.0 * h)
    return J


def corner_eigenvalues(p: Params) -> Dict[str, object]:
    """Closed-form spectra of the two corners, from the analytic Jacobian.

    At Erec the state block is lower-triangular with diagonal g_r(1,1) - kappa_k
    = -(1 - a1) - kappa_k, and since C = 1, Q = 1 and chi_- = 0 there, the
    Q-row reduces to d(dQ/dt)/dQ = -eps chi_+(1 - a1).  At Epath the diagonal is
    g_r(0,0) - kappa_k = -a0 - kappa_k, and with C = 0, Q = 0 and a'(0) = 0 for
    p > 1 the Q-eigenvalue is -eps rho chi_-(-a0).
    """
    chi_p_rec, _ = get_selector(p)(1.0 - p.a1, p)
    _, chi_m_pat = get_selector(p)(-p.a0, p)
    return {
        "Erec_state": [-(1.0 - p.a1)] + [-(1.0 - p.a1) - k for k in p.kappa],
        "Erec_Q": -p.eps * chi_p_rec,
        "Epath_state": [-p.a0] + [-p.a0 - k for k in p.kappa],
        "Epath_Q": -p.eps * p.rho * chi_m_pat,
    }


# --------------------------------------------------------------------------
# results, by status
# --------------------------------------------------------------------------

Result = Dict[str, object]


def check_positive_invariance(p: Params) -> Result:
    """T0: on the boundary of the cube the field points inward (Nagumo)."""
    ok = True
    detail = []
    for Q in (0.0, 0.25, 0.5, 0.75, 1.0):
        for k in range(4):
            for face in (0.0, 1.0):
                y = np.full(5, 0.5)
                y[4] = Q
                y[k] = face
                d = rhs(0.0, y, p)[k]
                inward = d >= -1e-12 if face == 0.0 else d <= 1e-12
                ok &= inward
                if not inward:
                    detail.append((k, face, Q, d))
        # evidence faces
        for face in (0.0, 1.0):
            y = np.full(5, 0.5)
            y[4] = face
            d = rhs(0.0, y, p)[4]
            ok &= (d >= -1e-12) if face == 0.0 else (d <= 1e-12)
    return {"name": "T0 positive invariance", "status": "proved",
            "pass": bool(ok), "violations": detail}


def check_T2(p: Params) -> Result:
    """Hyperbolicity of the two corners; rho = 0 degeneracy."""
    out = {}
    rec = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    pat = np.zeros(5)
    for rho, tag in ((p.rho, "rho>0"), (0.0, "rho=0")):
        q = p.with_(rho=rho)
        ev_rec = np.linalg.eigvals(jacobian(rec, q))
        ev_pat = np.linalg.eigvals(jacobian(pat, q))
        out[tag] = {
            "Erec_eigs": sorted(np.real(ev_rec).tolist()),
            "Epath_eigs": sorted(np.real(ev_pat).tolist()),
            "Erec_dQ": float(rhs(0.0, rec, q)[4]),
            "Epath_dQ": float(rhs(0.0, pat, q)[4]),
        }
    stable_rec = max(out["rho>0"]["Erec_eigs"]) < 0 and max(out["rho=0"]["Erec_eigs"]) < 0
    stable_pat = max(out["rho>0"]["Epath_eigs"]) < 0
    degenerate_pat = abs(max(out["rho=0"]["Epath_eigs"])) < 1e-10

    ana = corner_eigenvalues(p)
    want_rec = sorted(list(ana["Erec_state"]) + [ana["Erec_Q"]])
    want_pat = sorted(list(ana["Epath_state"]) + [ana["Epath_Q"]])
    err_rec = max(abs(a - b) for a, b in zip(out["rho>0"]["Erec_eigs"], want_rec))
    err_pat = max(abs(a - b) for a, b in zip(out["rho>0"]["Epath_eigs"], want_pat))
    out["analytic"] = ana
    out["max_abs_err_vs_analytic"] = {"Erec": err_rec, "Epath": err_pat}
    return {"name": "T2 stability of both corners", "status": "proved",
            "pass": bool(stable_rec and stable_pat and degenerate_pat
                         and err_rec < 1e-7 and err_pat < 1e-7),
            "detail": out}


BOUNDARY_TOL = 1e-7      # lowered by --fast


@lru_cache(maxsize=None)
def collapse_boundary(p: Params, Delta: float = 0.55, tol: float = 0.0) -> float:
    """Bisection in Q0 for the recovered-start outcome boundary.

    Recovered configuration r = (1,1,1,1) receives an excursion to
    r0 = 1 - Delta.  Under the sharp selector the boundary is exactly
    Qc = a^{-1}(1 - Delta) for every eps (T3(c1)); at finite gamma it drifts.
    """
    def collapses(Q0: float) -> bool:
        return root_outcome([1.0 - Delta, 1.0, 1.0, 1.0, Q0], p) < 0

    tol = tol or BOUNDARY_TOL
    lo, hi = 1e-6, 1.0 - 1e-6
    if not collapses(lo) or collapses(hi):
        return float("nan")
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if collapses(mid):
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def check_T3(p: Params, full: bool) -> Result:
    """T3(a) trigger, T3(c1) invariance of S, T3(b) propagation, T3(c2)."""
    Delta = 0.55
    Qc = p.Q_of_a(1.0 - Delta)

    # T3(c1): timescale independence of the boundary under the sharp selector
    b_fast = collapse_boundary(p.with_(eps=0.02), Delta)
    b_slow = collapse_boundary(p.with_(eps=0.005), Delta)

    # T3(b): propagation happens well below the sufficient bound, and fails at 0
    def whole_layer_collapse(kap: float) -> bool:
        q = p.with_(kappa=(kap, kap, kap))
        sol = integrate([0.05, 1.0, 1.0, 1.0, 0.3], 4000.0, q, max_step=5.0)
        return bool(np.all(sol.y[:4, -1] < 1e-2))

    prop = {f"kappa={k}": whole_layer_collapse(k) for k in
            ((0.0, 0.02, 0.05, 0.2, 0.6) if full else (0.0, 0.05, 0.6))}

    # T3(c2): absorption from a recovered configuration, failure off it
    pf = p.with_(sel_scale=0.01, eps=0.5)
    sol_abs = integrate([1.0 - Delta, 1.0, 1.0, 1.0, Qc + 1e-3], 2.0e4, pf, max_step=5.0)
    absorbed = bool(np.all(sol_abs.y[:4, -1] > 0.99) and sol_abs.y[4, -1] > 0.99)
    q_low = p.with_(kappa=(0.003, 0.003, 0.003))
    sol_off = integrate([1.0, 0.0, 0.0, 0.0, 0.9], 5000.0, q_low, max_step=5.0)
    no_uncond = bool(sol_off.y[1, -1] < 0.1)

    return {"name": "T3 fragile window and propagation", "status": "proved",
            "pass": bool(abs(b_fast - Qc) < max(1e-6, 3.0 * BOUNDARY_TOL)
                         and abs(b_slow - Qc) < max(1e-6, 3.0 * BOUNDARY_TOL)
                         and absorbed and no_uncond
                         and prop.get("kappa=0.0") is False),
            "detail": {"Qc_exact": Qc,
                       "boundary_eps_0.02": b_fast,
                       "boundary_eps_0.005": b_slow,
                       "propagation": prop,
                       "absorption_from_recovered": absorbed,
                       "absorption_not_unconditional": no_uncond}}


def check_T4(p: Params) -> Result:
    """Closed form on the recovery regime, and the general integrating-factor
    solution on a finite-gamma trajectory where both channels act."""
    # (a) recovery regime, augmented system carrying N
    def rhs_aug(t, y):
        d = rhs(t, y[:5], p)
        z = y[0] - p.a(float(np.clip(y[4], 0, 1)))
        chi_p, _ = get_selector(p)(z, p)
        C = float(np.prod(np.clip(y[:4], 0.0, 1.0)))
        return np.concatenate([d, [chi_p * C]])

    Q0 = 0.37
    y0 = np.array([0.9, 0.8, 0.7, 0.6, Q0, 0.0])
    sol = solve_ivp(rhs_aug, (0.0, 4000.0), y0, method="RK45",
                    rtol=1e-11, atol=1e-13, max_step=1.0)
    Q_num = sol.y[4, -1]
    Q_law = 1.0 - (1.0 - Q0) * math.exp(-p.eps * sol.y[5, -1])
    err_closed = abs(Q_num - Q_law)

    # (b) general linear solution with both channels active (finite gamma).
    #     The integrating factor is carried as two extra coordinates, so the
    #     comparison is limited by solver tolerance, not by quadrature.
    q = p.with_(selector="bayes", gamma=6.0)

    def AB(y):
        z = y[0] - q.a(float(np.clip(y[4], 0, 1)))
        cp, cm = chi_bayes(z, q)
        A = q.eps * cp * float(np.prod(np.clip(y[:4], 0, 1)))
        B = q.eps * q.rho * cm
        return A, B

    def rhs_int(t, y):
        d = rhs(t, y[:5], q)
        A, B = AB(y[:5])
        return np.concatenate([d, [A + B, A * math.exp(y[5])]])

    y0b = np.array([0.30, 0.5, 0.5, 0.5, 0.40, 0.0, 0.0])
    solb = solve_ivp(rhs_int, (0.0, 300.0), y0b, method="RK45",
                     rtol=1e-12, atol=1e-14, max_step=0.5)
    I, J = solb.y[5], solb.y[6]
    Q_pred = np.exp(-I) * (y0b[4] + J)
    err_general = float(np.max(np.abs(Q_pred - solb.y[4])))
    # both channels act simultaneously while the states are still positive
    minA, minB = np.inf, np.inf
    for i in range(solb.y.shape[1]):
        if float(np.prod(np.clip(solb.y[:4, i], 0, 1))) < 1e-3:
            break
        A, B = AB(solb.y[:5, i])
        minA, minB = min(minA, A), min(minB, B)

    return {"name": "T4 evidence law", "status": "proved on {z>0}; general form in Methods",
            "pass": bool(err_closed < 1e-12 and err_general < 1e-9),
            "detail": {"closed_form_abs_err": err_closed,
                       "general_solution_max_abs_err": err_general,
                       "min_accumulation_rate_while_states_online": float(minA),
                       "min_attenuation_rate_while_states_online": float(minB)}}


def check_prop1_sharp_limit(p: Params, full: bool = False) -> Result:
    """Proposition 1, in the two parts the manuscript separates.

    (a) chi_pm(0) = 0 at every finite gamma and for every member of the E4
        class, so S stays invariant and the outcome boundary stays at Qc for
        every eps;
    (b) chi_pm -> the indicator pair as gamma -> infinity, locally uniformly
        away from z = 0.  That limit is discontinuous, which is why premise E
        regularises it; the theorems are proved on the regularisation.

    Dropping E4 (plain model averaging) gives chi_pm(0) = 1/2, so the
    boundary-zero condition of Proposition 2 fails and the boundary drifts with
    the evidence timescale.
    """
    # (a) boundary zeros, for three members of the E4 class
    zeros = {}
    for lam in ("entropy", "entropy2", "margin"):
        for gam in (1.0, 5.0, 20.0, 200.0):
            zeros[f"{lam},gamma={gam:g}"] = chi_bayes(0.0, p.with_(gamma=gam, lam=lam))
    boundary_zero = all(abs(v[0]) < 1e-12 and abs(v[1]) < 1e-12 for v in zeros.values())

    # E4 class conditions on L: L(1/2) = 0, L(0) = L(1) = 1, L > 0 off 1/2
    L_class = {}
    for lam in ("entropy", "entropy2", "margin"):
        L_class[lam] = {
            "L(1/2)": certainty_weight(0.5, lam),
            "L(0)": certainty_weight(0.0, lam),
            "L(1)": certainty_weight(1.0, lam),
            "min L off 1/2": min(certainty_weight(w, lam)
                                 for w in np.linspace(0.0, 1.0, 401) if abs(w - 0.5) > 1e-3)}
    L_ok = all(abs(v["L(1/2)"]) < 1e-12 and abs(v["L(0)"] - 1) < 1e-12
               and abs(v["L(1)"] - 1) < 1e-12 and v["min L off 1/2"] > 0
               for v in L_class.values())
    # lam_gamma -> 1 uniformly on {|z| >= delta}, for every member of the class
    lam_sup = {}
    for lam in ("entropy", "entropy2", "margin"):
        lam_sup[lam] = {g_: max(abs(ambiguity_factor(float(z), g_, lam) - 1.0)
                                for z in np.concatenate([np.linspace(-0.5, -0.05, 60),
                                                         np.linspace(0.05, 0.5, 60)]))
                        for g_ in (5.0, 50.0, 500.0)}
    lam_uniform = all(v[5.0] > v[50.0] > v[500.0] and v[500.0] < 1e-3
                      for v in lam_sup.values())

    # (b) locally uniform convergence to the indicators away from the boundary
    sup_err = {}
    for gam in (2.0, 5.0, 20.0, 100.0, 500.0):
        q = p.with_(gamma=gam)
        zs = np.concatenate([np.linspace(-0.5, -0.05, 200), np.linspace(0.05, 0.5, 200)])
        e = 0.0
        for z in zs:
            cp, cm = chi_bayes(float(z), q)
            e = max(e, abs(cp - (1.0 if z > 0 else 0.0)),
                    abs(cm - (1.0 if z < 0 else 0.0)))
        sup_err[gam] = e
    decreasing_sup = all(sup_err[a] > sup_err[b] for a, b in
                         zip(sorted(sup_err)[:-1], sorted(sup_err)[1:]))
    # the limit is discontinuous at z = 0: the one-sided limit is 1, the value 0
    jump = abs(chi_bayes(1e-4, p.with_(gamma=1e7))[0] - chi_bayes(0.0, p.with_(gamma=1e7))[0])

    # invariance of S at finite gamma, for each member of the class
    Qc = p.Q_of_a(0.45)
    b_class = {lam: collapse_boundary(p.with_(selector="bayes", eps=0.02, lam=lam))
               for lam in (("entropy", "entropy2", "margin") if full else ("entropy",))}
    soft = p.with_(selector="bayes")
    b_soft = {e: collapse_boundary(soft.with_(eps=e)) for e in (0.02, 0.005)}
    avg = p.with_(selector="bma")
    b_avg = {e: collapse_boundary(avg.with_(eps=e)) for e in (0.02, 0.005, 0.00125)}

    btol = max(1e-6, 3.0 * BOUNDARY_TOL)   # the bisection cannot resolve below its own tol
    invariant_at_finite_gamma = (all(abs(v - Qc) < btol for v in b_soft.values())
                                 and all(abs(v - Qc) < btol for v in b_class.values()))
    drift = [abs(b_avg[e] - Qc) for e in (0.02, 0.005, 0.00125)]
    drifts_and_converges = drift[0] > drift[1] > drift[2] > 0.0
    return {"name": "Proposition 1 sharp-selection limit", "status": "proved (a) and (b)",
            "pass": bool(boundary_zero and L_ok and lam_uniform and decreasing_sup
                         and jump > 0.9 and invariant_at_finite_gamma
                         and drifts_and_converges),
            "detail": {"E4_class_conditions_on_L": L_class,
                       "sup_lam_minus_1_off_boundary": lam_sup,
                       "chi_at_zero_finite_gamma": {k: list(v) for k, v in zeros.items()},
                       "chi_at_zero_plain_averaging": list(chi_bma(0.0, p)),
                       "sup_err_off_boundary": sup_err,
                       "limit_jump_at_zero": jump,
                       "Qc_exact": Qc,
                       "boundary_with_E4": b_soft,
                       "boundary_by_lambda_member": b_class,
                       "boundary_without_E4_plain_averaging": b_avg}}


def check_T5(p: Params, full: bool) -> Result:
    """Two-outcome convergence off S, endpoint initial data included, plus the
    behaviour of orbits started on S."""
    K = p.K_star()
    # The theorem holds for every eps > 0 and every selector in the class of E;
    # the numerical corroboration uses a faster member (s = 0.01) and a larger
    # eps so that the evidence coordinate settles within a short horizon.
    q = p.with_(kappa=(0.6, 0.6, 0.6), sel_scale=0.01, eps=0.5)
    assert min(q.kappa) > K

    rng = np.random.default_rng(20260728)
    ics: List[Tuple[str, List[float]]] = [
        ("endpoint r0=1, z>0", [1.0, 0.0, 0.0, 0.0, 0.2]),
        ("endpoint r0=0, z<0", [0.0, 1.0, 1.0, 1.0, 0.9]),
        ("interior z>0", [0.5, 0.1, 0.05, 0.02, 0.5]),
        ("interior z<0", [0.2, 0.9, 0.9, 0.9, 0.2]),
    ]
    n = 40 if full else (4 if FAST else 12)
    for i in range(n):
        y = rng.uniform(0.0, 1.0, 5).tolist()
        ics.append((f"random {i}", y))

    rows = []
    ok = True
    for tag, y0 in ics:
        z0 = y0[0] - q.a(y0[4])
        if abs(z0) < 1e-6:
            continue
        sol = integrate(y0, 1.0e4 if FAST else 3.0e4, q, max_step=5.0)
        end = sol.y[:, -1]
        target = "Erec" if z0 > 0 else "Epath"
        reached = ("Erec" if np.all(end[:4] > 0.99) and end[4] > 0.99 else
                   "Epath" if np.all(end[:4] < 0.01) and end[4] < 0.01 else "other")
        ok &= (reached == target)
        rows.append({"ic": tag, "z0": float(z0), "target": target,
                     "reached": reached, "end": end.round(6).tolist()})

    # orbits started on S converge to the point of E_S at their frozen evidence
    Qbar = 0.4
    a_bar = float(q.a(Qbar))
    solS = integrate([a_bar, 0.9, 0.1, 0.5, Qbar], 4000.0, q, max_step=0.5)
    endS = solS.y[:, -1]
    on_S = bool(abs(endS[4] - Qbar) < 1e-12
                and np.all(np.abs(endS[1:4] - a_bar) < 1e-6)
                and abs(endS[0] - a_bar) < 1e-12)
    return {"name": "T5 two-outcome convergence", "status": "proved under kappa > K*",
            "pass": bool(ok and on_S),
            "detail": {"K_star": K, "kappa": q.kappa[0],
                       "n_trajectories": len(rows),
                       "all_as_predicted": bool(ok),
                       "S_orbit_to_ES": on_S,
                       "S_orbit_end": endS.round(9).tolist(),
                       "sample": rows[:6]}}


def mixed_branch(kappa: float, p: Params) -> Dict[str, object]:
    """Low equilibrium of the weak-coupling mixed branch at Q = 1, and its
    state-block spectrum."""
    a1 = p.a1
    disc = a1 * a1 - 4.0 * kappa
    if disc < 0:
        return {"kappa": kappa, "exists": False}
    r1 = (a1 - math.sqrt(disc)) / 2.0
    r = [1.0, r1, 0.0, 0.0]
    for k in (2, 3):
        f = lambda x, prev=r[k - 1]: x * (1 - x) * (x - a1) + kappa * (prev - x)
        r[k] = brentq(f, 0.0, max(1e-12, r[k - 1]))
    y = np.array(r + [1.0])
    q = p.with_(kappa=(kappa, kappa, kappa))
    J = jacobian(y, q)
    ev = np.sort(np.real(np.linalg.eigvals(J[:4, :4])))
    resid = float(np.max(np.abs(rhs(0.0, y, q)[:4])))
    return {"kappa": kappa, "exists": True, "equilibrium": y.round(6).tolist(),
            "state_eigenvalues": ev.round(6).tolist(),
            "hyperbolic": bool(np.all(ev < -1e-9)),
            "residual": resid,
            "double_root": bool(abs(disc) < 1e-15)}


def check_weak_coupling(p: Params) -> Result:
    a1c = p.a1 ** 2 / 4.0
    below = mixed_branch(0.003, p)
    at = mixed_branch(a1c, p)
    above = mixed_branch(0.008, p)
    return {"name": "weak-coupling mixed branch", "status": "characterised",
            "pass": bool(below["exists"] and below["hyperbolic"]
                         and at["exists"] and not above["exists"]),
            "detail": {"a1^2/4": a1c, "below": below, "saddle_node": at,
                       "above": above}}


# --------------------------------------------------------------------------
# Closure of the loop (Proposition 4): the companion observation model
#
# theta = (pi_s, beta, pi_m, pi_v_fast, pi_v_slow); five conditionally
# independent Poisson channels with means mu_j = M_j(theta) exp(theta_j),
# phi(x) = 1 - exp(-x), enactment e = 1 - exp(-a_E beta phi(pi_s)) and
#   M_s = 1, M_b = phi(pi_s), M_m = e phi(pi_s), M_f = e phi(pi_m),
#   M_slow = e phi(pi_m) phi(pi_v_fast).
# What is checked here is that the Fisher information about pi_v_slow
# accumulates at the rate M_slow, that M_slow is the product gate C(r) up
# to a bounded factor (Lemma C1), and that the per-context reading of the
# same quantity agrees with the integral of C(r) (Lemmas C2 and C3).
# --------------------------------------------------------------------------

A_E = 1.0
CHANNELS = ("s", "b", "m", "f", "slow")


def _phi(x: float) -> float:
    return 1.0 - math.exp(-x)


def companion_means(theta: Sequence[float], a_E: float = A_E) -> Dict[str, float]:
    """Poisson channel means of the companion generative model."""
    pis, beta, pim, pif, pisl = (float(v) for v in theta)
    e = 1.0 - math.exp(-a_E * beta * _phi(pis))
    M = {"s": 1.0, "b": _phi(pis), "m": e * _phi(pis),
         "f": e * _phi(pim), "slow": e * _phi(pim) * _phi(pif)}
    th = {"s": pis, "b": beta, "m": pim, "f": pif, "slow": pisl}
    return {k: M[k] * math.exp(th[k]) for k in CHANNELS}


def companion_fisher(theta: Sequence[float], h: float = 1e-6) -> np.ndarray:
    """Joint Fisher information I_kl = sum_j (1/mu_j) (d mu_j/d th_k)(d mu_j/d th_l)."""
    th = np.asarray(theta, dtype=float)
    mu0 = companion_means(th)
    J = np.zeros((len(CHANNELS), len(th)))
    for k in range(len(th)):
        tp, tm = th.copy(), th.copy()
        tp[k] += h
        tm[k] -= h
        mp, mm = companion_means(tp), companion_means(tm)
        for j, ch in enumerate(CHANNELS):
            J[j, k] = (mp[ch] - mm[ch]) / (2.0 * h)
    I = np.zeros((len(th), len(th)))
    for j, ch in enumerate(CHANNELS):
        if mu0[ch] > 0.0:
            I += np.outer(J[j], J[j]) / mu0[ch]
    return I


def check_closure(p: Params) -> Result:
    """Proposition 4: the statistic identifying pi_v_slow is the evidence variable Q.

    (i)   only the slow channel carries pi_v_slow, and I(pi_v_slow) = mu_slow;
    (ii)  Lemma C1: C(r) <= M_slow <= K C(r) with K = B/(1 - exp(-B));
    (iii) Lemma C3: the expected count of usable per-context estimates along a
          recovery trajectory is nu times the integral of C(r), which is the
          reading that supplies Axiom C exactly;
    (iv)  corollary: for small enough eps every capacity reaches a common
          threshold before Q does.
    """
    grid = np.linspace(0.2, 2.0, 4 if FAST else 6)
    B = A_E * float(grid[-1])
    K_bracket = B / (1.0 - math.exp(-B))
    max_rel_info_err = 0.0
    max_leak = 0.0
    lo_ratio, hi_ratio = np.inf, 0.0
    for pis in grid:
        for beta in grid:
            for pim in grid:
                for pif in grid:
                    theta = (pis, beta, pim, pif, 0.3)
                    mu = companion_means(theta)
                    I = companion_fisher(theta)
                    # (i) information about the top parameter equals the slow mean
                    max_rel_info_err = max(max_rel_info_err,
                                           abs(I[4, 4] - mu["slow"]) / mu["slow"])
                    # no other channel depends on pi_v_slow: rebuild the column
                    h = 1e-6
                    tp = list(theta); tp[4] += h
                    tm = list(theta); tm[4] -= h
                    mp, mm = companion_means(tp), companion_means(tm)
                    for ch in CHANNELS:
                        if ch != "slow":
                            max_leak = max(max_leak, abs(mp[ch] - mm[ch]) / (2 * h))
                    # (ii) Lemma C1 bracket
                    r = (_phi(pis), _phi(A_E * beta), _phi(pim), _phi(pif))
                    C = float(np.prod(r))
                    M_slow = mu["slow"] * math.exp(-theta[4])
                    lo_ratio = min(lo_ratio, M_slow / C)
                    hi_ratio = max(hi_ratio, M_slow / C)
    bracket_ok = bool(lo_ratio >= 1.0 - 1e-12 and hi_ratio <= K_bracket + 1e-12)

    # (iii) per-context reading: Bernoulli availability with probabilities r_k
    q = p.with_(sel_scale=0.01)
    sol = integrate([0.90, 0.50, 0.30, 0.20, 0.30], 400.0, q, n_eval=4001, max_step=0.5)
    r = np.clip(sol.y[:4], 0.0, 1.0)
    Cs = np.prod(r, axis=0)
    integral_C = float(np.trapezoid(Cs, sol.t)) if hasattr(np, "trapezoid") \
        else float(np.trapz(Cs, sol.t))
    rng = np.random.default_rng(0)
    n_ctx = 20000 if FAST else 200000
    idx = rng.integers(0, len(sol.t), size=n_ctx)             # contexts uniform in time
    draws = rng.random((4, n_ctx))
    usable = np.all(draws < r[:, idx], axis=0)
    span = float(sol.t[-1] - sol.t[0])
    mc_rate = usable.mean() * span                            # per unit context rate nu
    mc_rel_err = abs(mc_rate - integral_C) / integral_C

    # (iv) corollary: the evidence coordinate completes last for small eps
    thr = 0.8
    slow = p.with_(eps=1.0e-3, sel_scale=0.01)
    s2 = integrate([0.70, 0.50, 0.40, 0.30, 0.05], 4000.0, slow, n_eval=40001, max_step=0.5)

    def first_passage(y):
        hit = np.argmax(y >= thr)
        return float(s2.t[hit]) if y.max() >= thr else float("inf")

    T_states = [first_passage(s2.y[k]) for k in range(4)]
    T_Q = first_passage(s2.y[4])
    last_ok = bool(np.isfinite(max(T_states)) and T_Q > max(T_states))

    return {"name": "Proposition 4 closure", "status": "proved (given M1-M3 and O1-O3)",
            "pass": bool(max_rel_info_err < 1e-6 and max_leak < 1e-12
                         and bracket_ok and mc_rel_err < 0.02 and last_ok),
            "detail": {"max_rel_error_I_slow_vs_mu_slow": float(max_rel_info_err),
                       "max_leak_other_channels": float(max_leak),
                       "K_bracket": float(K_bracket),
                       "min_M_slow_over_C": float(lo_ratio),
                       "max_M_slow_over_C": float(hi_ratio),
                       "integral_C": integral_C,
                       "monte_carlo_usable_contexts": float(mc_rate),
                       "monte_carlo_relative_error": float(mc_rel_err),
                       "first_passage_states": T_states,
                       "first_passage_Q": T_Q,
                       "evidence_last": last_ok}}


# --------------------------------------------------------------------------
# Axiom D from an active-inference decision rule (Proposition 5)
#
# Two policies at each capacity, selected by expected free energy.  Under
# "enact" the predicted outcome is N(mu(r), v) with v = sigma^2 + 1/Lambda,
# preferences are N(m, 1/lam_P), and "withhold" gives a constant G0.  With
# the risk plus ambiguity decomposition,
#
#   G1(r,L) = 0.5[lam_P v + lam_P (mu(r)-m)^2 - 1 - log(lam_P v)]
#             + 0.5 log(2 pi e sigma^2).
#
# The threshold a solves G1(a,L) = G0.  Part (a) of Proposition 5 is
# a'(L) < 0; part (b) is the equivalence a''(Q) < 0 <=> Lstar a''(L) + a'(L) < 0.
# --------------------------------------------------------------------------

LAM_P = 2.0          # preference precision
SIGMA2 = 1.0         # irreducible outcome noise
LAMBDA_0 = 10.0      # baseline accumulated information
LSTAR_OK = 2.0       # coordinate scale satisfying the inequality of part (b)
LSTAR_BAD = 5.0      # coordinate scale violating it


def _W(v: float) -> float:
    return 0.5 * math.log(LAM_P * v) - 0.5 * LAM_P * v


def decision_constants(a0: float, a1: float) -> Tuple[float, float]:
    """Fix G0 (through A) and b so that a(Lambda_0) = a0 and a(inf) = a1."""
    v0 = SIGMA2 + 1.0 / LAMBDA_0
    ratio = ((1.0 - a1) / (1.0 - a0)) ** 2
    A = brentq(lambda A: (A + _W(SIGMA2)) - ratio * (A + _W(v0)), -50.0, 50.0)
    F0 = A + _W(v0)
    b2 = 2.0 * F0 / (LAM_P * (1.0 - a0) ** 2)
    return A, b2


def G1(r: float, Lam: float, A: float, b2: float) -> float:
    """Expected free energy of enacting, up to the constant absorbed in A."""
    v = SIGMA2 + 1.0 / Lam
    mu_minus_m = -math.sqrt(b2) * (1.0 - r)
    return 0.5 * (LAM_P * v + LAM_P * mu_minus_m ** 2 - 1.0
                  - math.log(LAM_P * v)) - A + 0.5


def threshold_of_Lambda(Lam: float, A: float, b2: float) -> float:
    """a(Lambda) from G1(a,Lambda) = G0, in closed form for mu(r) = m - b(1-r)."""
    F = A + _W(SIGMA2 + 1.0 / Lam)
    return 1.0 - math.sqrt(2.0 * F / (LAM_P * b2))


def check_axiom_D(p: Params) -> Result:
    """Proposition 5: the two signs of Axiom D from expected-free-energy selection."""
    A, b2 = decision_constants(p.a0, p.a1)
    n = 4001 if FAST else 20001

    # the two partial derivatives the implicit function theorem uses
    h = 1e-6
    worst_dGdr, worst_dGdL = -np.inf, -np.inf
    for Lam in np.linspace(LAMBDA_0, 400.0, 41):
        for r in np.linspace(0.05, 0.95, 19):
            dGdr = (G1(r + h, Lam, A, b2) - G1(r - h, Lam, A, b2)) / (2 * h)
            dGdL = (G1(r, Lam + h, A, b2) - G1(r, Lam - h, A, b2)) / (2 * h)
            worst_dGdr = max(worst_dGdr, dGdr)
            worst_dGdL = max(worst_dGdL, dGdL)
    informative = bool(LAM_P * (SIGMA2 + 1.0 / LAMBDA_0) > 1.0)

    def profile(Lstar: float):
        Q = np.linspace(0.0, 1.0 - 1e-6, n)
        Lam = LAMBDA_0 - Lstar * np.log(1.0 - Q)
        a = np.array([threshold_of_Lambda(float(L), A, b2) for L in Lam])
        d1 = np.gradient(a, Q)
        d2 = np.gradient(d1, Q)
        core = slice(20, -20)
        return a, d1[core], d2[core]

    a_ok, d1_ok, d2_ok = profile(LSTAR_OK)
    _, d1_bad, d2_bad = profile(LSTAR_BAD)

    # elasticity q(Lambda), analytic, and the inequality of part (b)
    Lam = np.linspace(LAMBDA_0, 2000.0, 40001)
    v = SIGMA2 + 1.0 / Lam
    Fv = A + 0.5 * np.log(LAM_P * v) - 0.5 * LAM_P * v
    F1 = (LAM_P - 1.0 / v) / (2.0 * Lam ** 2)
    F2 = -1.0 / (2.0 * Lam ** 4 * v ** 2) - (LAM_P - 1.0 / v) / Lam ** 3
    q = Lam * (F1 ** 2 / (2.0 * Fv) - F2) / F1
    margin_ok = float(np.max(q * LSTAR_OK / Lam))
    margin_bad = float(np.max(q * LSTAR_BAD / Lam))

    endpoints_ok = bool(abs(a_ok[0] - p.a0) < 1e-9
                        and abs(threshold_of_Lambda(1e9, A, b2) - p.a1) < 1e-6)
    return {"name": "Proposition 5 Axiom D", "status": "proved (given D1-D4)",
            "pass": bool(informative and endpoints_ok
                         and worst_dGdr < 0 and worst_dGdL < 0
                         and d1_ok.max() < 0 and d2_ok.max() < 0
                         and d1_bad.max() < 0 and d2_bad.max() > 0
                         and margin_ok < 1.0 < margin_bad),
            "detail": {"informative_regime": informative,
                       "max_dG_dr": float(worst_dGdr),
                       "max_dG_dLambda": float(worst_dGdL),
                       "a_at_Q0": float(a_ok[0]),
                       "a_limit": float(threshold_of_Lambda(1e9, A, b2)),
                       "max_first_derivative_Lstar_ok": float(d1_ok.max()),
                       "max_second_derivative_Lstar_ok": float(d2_ok.max()),
                       "max_second_derivative_Lstar_bad": float(d2_bad.max()),
                       "elasticity_at_Lambda0": float(q[0]),
                       "elasticity_limit": float(q[-1]),
                       "part_b_margin_ok": margin_ok,
                       "part_b_margin_bad": margin_bad}}


def check_finite_gamma(p: Params) -> Result:
    """Proposition 6: what the dichotomy becomes at finite sharpness.

    The two sides stay exactly separated (S invariant at every finite gamma),
    E_path stays an exact equilibrium, and the recovered corner is displaced to
    the zero of 1 - Q = rho exp(-gamma(1 - a(Q))) Q, within rho e^{-gamma(1-a0)}
    of E_rec.
    """
    gammas = (10.0, 20.0) if FAST else (10.0, 20.0, 40.0)
    rows = {}
    ok = True
    # the explicit margin of Lemma F, at gamma = 20
    C_L = 3.1                       # L(w) <= C_L (w-1/2)^2 for |w-1/2| <= 1/4
    A1 = p.p * (p.a0 - p.a1)        # max |a'(Q)| on [0,1]
    M_rho = max(1.0, p.rho)
    m_star = min((p.a1 / 2) * (1 - p.a1 / 2), (1 + p.a0) * (1 - p.a0) / 4)
    z_bar = min(p.a1 / 2, (1 - p.a0) / 2, 1.0 / 20.0,
                8 * m_star / (p.eps * A1 * M_rho * C_L * 20.0 ** 2))
    for gam in gammas:
        q = p.with_(selector="bayes", gamma=gam)
        sol = integrate([0.9, 0.9, 0.9, 0.9, 0.5], 60000.0, q,
                        n_eval=3001, max_step=5.0)
        r_end = sol.y[:4, -1]
        Q_end = float(sol.y[4, -1])
        z = sol.y[0] - q.a(sol.y[4])
        fixed = q.rho * math.exp(-gam * (1.0 - float(q.a(Q_end)))) * Q_end
        bound = q.rho * math.exp(-gam * (1.0 - q.a0))
        gap = 1.0 - Q_end
        rows[f"gamma={gam:g}"] = {"r_end": [float(x) for x in r_end],
                                  "one_minus_Q": gap,
                                  "fixed_point_prediction": float(fixed),
                                  "bound_rho_exp": float(bound),
                                  "min_z": float(z.min())}
        ok = ok and bool(np.all(r_end > 1 - 1e-6) and gap <= bound + 1e-12
                         and abs(gap - fixed) <= 1e-9 + 1e-3 * abs(fixed)
                         and z.min() > 0.0)
    qc = p.with_(selector="bayes", gamma=20.0)
    col = integrate([0.3, 0.3, 0.3, 0.3, 0.5], 60000.0, qc, max_step=5.0)
    collapse = [float(x) for x in col.y[:, -1]]
    ok = ok and bool(max(abs(x) for x in collapse) < 1e-9)
    return {"name": "Proposition 6 finite sharpness", "status": "proved (finite gamma)",
            "pass": bool(ok),
            "detail": {"recovery_side": rows, "collapse_end": collapse,
                       "lemma_F": {"m_star": float(m_star), "A1": float(A1),
                                   "C_L": C_L, "z_bar_at_gamma_20": float(z_bar)}}}


def check_gate_and_E4(p: Params) -> Result:
    """Proposition 7 (gate robustness) and the O(eps eta) drift of the boundary.

    A gate comparable to the product gate leaves every result unchanged; a leak
    leaves the theorems unchanged but removes the exact halting of accumulation;
    violating the boundary zero of E4 by eta moves the switching surface by an
    amount proportional to eps eta.
    """
    Qc = p.Q_of_a(0.45)
    # These boundaries are pinned to a fine bisection tolerance in every mode:
    # the drifts measured here are of order 1e-4 to 1e-3, so the coarse setting
    # used by --fast for the other sweeps would dominate them and the reported
    # ratios would not be reproducible across modes.
    btol = 1e-7
    gates = {}
    for name, kw in (("modulated", {"gate": "modulated"}),
                     ("leaky", {"gate": "leaky", "gate_leak": 0.01})):
        q = p.with_(**kw)
        collapse_boundary.cache_clear()
        b = collapse_boundary(q, 0.55, tol=btol)
        outs = [root_outcome(y0, q) for y0 in
                ([0.9, 0.9, 0.9, 0.9, 0.5], [0.3, 0.3, 0.3, 0.3, 0.5])]
        offline = float(rhs(0.0, np.array([0.9, 0.0, 0.5, 0.5, 0.5]), q)[4])
        gates[name] = {"boundary": float(b), "boundary_minus_Qc": float(b - Qc),
                       "outcomes": outs, "dQ_dt_with_one_node_offline": offline}
    drift = {}
    grid = ((0.02, 0.05), (0.02, 0.25), (0.005, 0.05), (0.005, 0.25))
    for eps, eta in grid:
        q = p.with_(selector="bayes", eta=eta, eps=eps)
        collapse_boundary.cache_clear()
        b = collapse_boundary(q, 0.55, tol=btol)
        drift[f"eps={eps:g},eta={eta:g}"] = {"boundary": float(b),
                                             "drift": float(b - Qc),
                                             "drift_over_eps_eta": float((b - Qc) / (eps * eta))}
    collapse_boundary.cache_clear()

    # a leak with finite sharpness displaces the failed corner too: on the face
    # r = 0 the equilibrium solves Q/(1-Q) = (delta/rho) exp(-gamma a(Q))
    ql = p.with_(selector="bayes", gamma=20.0, gate="leaky", gate_leak=0.01)
    sol0 = integrate([0.0, 0.0, 0.0, 0.0, 0.3], 200000.0, ql, max_step=20.0)
    Q_path = float(sol0.y[4, -1])
    pred = brentq(lambda q: ql.gate_leak * math.exp(-ql.gamma * float(ql.a(q))) * (1 - q)
                  - ql.rho * q, 0.0, 1.0)
    leak_finite = {"Q_at_failed_corner": Q_path, "prediction": float(pred),
                   "bound_delta_over_rho_exp": float((ql.gate_leak / ql.rho)
                                                     * math.exp(-ql.gamma * ql.a1))}

    ratios = [d["drift_over_eps_eta"] for d in drift.values()]
    ok = bool(abs(gates["modulated"]["boundary_minus_Qc"]) < 1e-4
              and abs(gates["leaky"]["boundary_minus_Qc"]) < 1e-4
              and gates["modulated"]["outcomes"] == [1, -1]
              and gates["leaky"]["outcomes"] == [1, -1]
              and gates["leaky"]["dQ_dt_with_one_node_offline"] > 0.0
              and all(0.85 < r < 1.05 for r in ratios)
              and abs(leak_finite["Q_at_failed_corner"] - leak_finite["prediction"])
              <= 1e-12 + 1e-6 * leak_finite["prediction"])
    return {"name": "Proposition 7 gate robustness and E4 drift",
            "status": "proved (comparable gates); drift measured",
            "pass": ok,
            "detail": {"Qc": float(Qc), "bisection_tolerance": btol,
                       "gates": gates, "boundary_drift": drift,
                       "leak_at_finite_sharpness": leak_finite,
                       "drift_ratio_range": [min(ratios), max(ratios)]}}


def check_prop3_order(p: Params) -> Result:
    """Proposition 3: state-order preservation and ordered first passage; invariance of the
    state diagonal under an exactly uniform onset."""
    q = p.with_(kappa=(0.3, 0.3, 0.3), sel_scale=0.01, eps=0.5)
    y0 = [0.90, 0.70, 0.50, 0.30, 0.60]      # ordered, root-initiated, z(0) > 0
    sol = integrate(y0, 2000.0, q, n_eval=20001, max_step=0.5)
    order_ok = bool(np.all(sol.y[0] >= sol.y[1] - 1e-9)
                    and np.all(sol.y[1] >= sol.y[2] - 1e-9)
                    and np.all(sol.y[2] >= sol.y[3] - 1e-9))
    thr = 0.8
    T = []
    for k in range(4):
        idx = np.argmax(sol.y[k] >= thr)
        T.append(float(sol.t[idx]) if sol.y[k].max() >= thr else float("inf"))
    passage_ok = all(T[i] <= T[i + 1] + 1e-9 for i in range(3))

    uni = integrate([0.5, 0.5, 0.5, 0.5, 0.6], 500.0, q, n_eval=5001, max_step=0.5)
    diag_ok = bool(np.max(np.abs(uni.y[:4] - uni.y[0])) < 1e-9)
    return {"name": "Proposition 3 order preservation", "status": "proved",
            "pass": bool(order_ok and passage_ok and diag_ok),
            "detail": {"first_passage_times": T, "order_preserved": order_ok,
                       "diagonal_invariant": diag_ok}}


def check_K_star(p: Params) -> Result:
    """K* is a uniform bound on max_x g_r, and it is sufficient, not sharp."""
    K = p.K_star()
    worst = -np.inf
    grid = 121 if FAST else 501
    for Q in np.linspace(0.0, 1.0, grid):
        for x in np.linspace(0.0, 1.0, grid):
            worst = max(worst, g_r(float(x), float(Q), p))
    # the comparison inequality used by Proposition 6(b): g(x,Q) <= K* x
    worst_ratio = -np.inf
    for Q in np.linspace(0.0, 1.0, 101):
        for x in np.linspace(1e-6, 1.0, 401):
            worst_ratio = max(worst_ratio, float(g(x, float(Q), p)) / x)
    return {"name": "K* uniform bound", "status": "proved (sufficient, not sharp)",
            "pass": bool(worst <= K + 1e-9 and worst_ratio <= K + 1e-9),
            "detail": {"K_star": K, "numerical_max_g_r": float(worst),
                       "a0_value": (1 - p.a0 + p.a0 ** 2) / 3,
                       "a1_value": (1 - p.a1 + p.a1 ** 2) / 3,
                       "max_g_over_x": float(worst_ratio)}}


def ablations(p: Params) -> Result:
    """Table 1, one row at a time."""
    out = {}
    # monostable field: a(Q) = 0 removes the collapse basin
    mono = p.with_(a0=0.0, a1=0.0)
    sol = integrate([0.02, 0.02, 0.02, 0.02, 0.1], 3000.0, mono, max_step=1.0)
    out["bistability_removed"] = {"end_r0": float(sol.y[0, -1]),
                                  "collapse_basin_lost": bool(sol.y[0, -1] > 0.9)}
    # linear a(Q): p = 1 removes the convex gain
    lin = p.with_(p=1.0)
    out["concavity_removed"] = {"a_second_derivative": 0.0,
                                "relative_curvature": 0.0,
                                "convex_gain_lost": True,
                                "p": lin.p}
    # ungated evidence: C(r) -> 1 decouples the levels
    out["evidence_gating_removed"] = {
        "note": "C(r) replaced by 1: dQ/dt no longer vanishes when a node is offline, "
                "so downstream recovery no longer controls the evidence rate"}
    # finite sharpness: supports overlap, so the closed form of T4 fails,
    # but S survives because lambda(0) = 0
    soft = p.with_(selector="bayes")
    out["sharp_selection_removed"] = {
        "Qc_exact": p.Q_of_a(0.45),
        "boundary_eps_0.02": collapse_boundary(soft.with_(eps=0.02)),
        "exclusivity_lost": True,
        "note": "both channels act on every trajectory, so the exponential closed "
                "form of T4 no longer applies and the general solution is needed; "
                "S remains invariant"}
    # dropping the ambiguity factor of E4: S is no longer invariant
    avg = p.with_(selector="bma")
    out["ambiguity_suspension_removed"] = {
        "chi_at_zero": list(chi_bma(0.0, p)),
        "boundary_eps_0.02": collapse_boundary(avg.with_(eps=0.02)),
        "boundary_eps_0.005": collapse_boundary(avg.with_(eps=0.005)),
        "boundary_eps_0.00125": collapse_boundary(avg.with_(eps=0.00125))}
    # zero coupling stops propagation
    q0 = p.with_(kappa=(0.0, 0.0, 0.0))
    sol0 = integrate([0.05, 1.0, 1.0, 1.0, 0.3], 3000.0, q0, max_step=1.0)
    out["coupling_removed"] = {"downstream_end": sol0.y[1:4, -1].round(6).tolist(),
                               "propagation_lost": bool(np.all(sol0.y[1:4, -1] > 0.9))}
    return {"name": "Table 1 ablations", "status": "illustrative", "pass": True,
            "detail": out}


# --------------------------------------------------------------------------
# display items
# --------------------------------------------------------------------------
TABLE1_ROWS = [
    ("Structure removed", "Component lost"),
    ("Bistability (Axiom B)", "A distinct collapse basin"),
    ("Concavity of a(Q) (Axiom D)", "Convex evidence-to-rate gain"),
    ("Evidence gating C(r) (Axiom C)", "Evidence-mediated cross-level self-catalysis"),
    ("Sharp regime selection (gamma -> inf limit, E)",
     "Regime exclusivity, hence the closed-form evidence law of T4 and the "
     "monotonicity of z used by T5; S itself survives at finite gamma"),
    ("Identifiability weighting at the boundary (lambda(0) = 0, E4)",
     "Invariance of S at any sharpness, which Proposition 2 shows to be exactly "
     "what invariance requires; the switching boundary "
     "becomes timescale dependent, drifting from 0.5940 at eps = 0.02 to 0.5786 "
     "at eps = 0.00125 against Qc = 0.5774"),
    ("Directional coupling (Axiom A)", "Whole-layer propagation"),
]

TABLE2_ROWS = [
    ("Result", "Status"),
    ("T0 positive invariance", "Proved"),
    ("T4 evidence law Q = 1 - (1-Q0)exp(-eps N), monotone in N",
     "Proved on the forward-invariant recovery regime {z > 0}, given the premise; "
     "the attenuation counterpart holds on {z < 0} and the general linear solution "
     "is in Methods"),
    ("T1 evidence-mediated acceleration, convexity, degree", "Proved"),
    ("T2 recovered state stable", "Proved, unconditional in rho"),
    ("T2 pathological state stable", "Proved for rho > 0 (non-hyperbolic at rho = 0)"),
    ("T3(a) frozen-evidence root trigger", "Proved (exact)"),
    ("T3(b) whole-layer propagation",
     "Proved under a sufficient coupling bound; some coupling necessary, bound not"),
    ("T3(c1) exact invariant switching manifold of the root",
     "Proved (all eps, rho; the boundary condition it uses holds at every finite gamma)"),
    ("T3(c2) whole-layer outcome and absorption",
     "Proved from a recovered configuration (not over arbitrary data)"),
    ("T5 two-outcome convergence off S, strong coupling",
     "Proved for rho > 0 and kappa_k > K* = 0.291, endpoint initial data included; "
     "K* sufficient, not sharp"),
    ("Equilibrium continuum on S; stable mixed branch at weak coupling",
     "Characterised; kappa_1 < a1^2/4 gives a hyperbolically stable low first stage, "
     "kappa_1 = a1^2/4 the saddle-node (Supplementary Methods)"),
    ("S is the global boundary of the two corner basins",
     "Proved under T5 strong coupling, the basins being identified by T5 itself; "
     "not in the general T1 to T4 regime"),
    ("Erec and Epath are the only equilibria",
     "False (equilibrium continuum ES on S, to which orbits within S converge)"),
    ("Proposition 1(a) boundary zeros and invariance of S at finite sharpness; "
     "1(b) convergence to the indicator pair",
     "Proved from E1 to E4 (Methods); the limit is discontinuous, so premise E "
     "regularises it and the theorems are proved on the regularisation"),
    ("Proposition 2 necessity of boundary-zero updating for exact switching",
     "Proved within the separable update class (Methods). Exclusivity is "
     "strictly stronger, needed for the one-sided laws of T4 and the "
     "monotonicity used in T5, not for invariance of S"),
    ("Proposition 3 state order under root-initiated recovery",
     "Proved (four state nodes; Supplementary Methods)"),
    ("Milestone order of Q relative to the state nodes",
     "Not proved here; established in the companion theory under a different "
     "axiom class and not re-derived"),
    ("Two-regime premise (E1), margin identification (E2), identifiability "
     "weighting (E4), continuous regularisation (E)",
     "Modelling commitments. E1 to E4 give the exclusive limit; E is the "
     "continuous one-sided regularisation of it, a well-posedness choice whose "
     "functional form the theorems do not use"),
    ("Clinical validity of the observation model", "Empirical; out of scope here"),
]

TABLE3_ROWS = [
    ("Study", "Input", "What it establishes", "What it does not address"),
    ("Companion clinical study (ref. 1)",
     "Consecutive inpatients, order specified a priori",
     "That the five capacities return in one order, without inversion",
     "Why the order holds; any dynamics or endpoint"),
    ("Companion computational study (ref. 2)",
     "An active-inference architecture",
     "The dependency graph, from the zero-pattern of the joint Fisher information; "
     "the chain has a unique linear extension, which is the observed order",
     "The recovery dynamics and the outcomes it can reach"),
    ("Graph-general recovery-order theory (ref. 3)",
     "Any dependency graph, with availability gating, deficit-closing recovery and a "
     "common impaired onset",
     "That the admissible orders are exactly the linear extensions of the graph",
     "Stability, consolidation, partial states and collapse"),
    ("This paper",
     "A dependency graph, carrying evidence-coupled bistable dynamics",
     "The recovery geometry: evidence-mediated acceleration (T1), two stable "
     "configurations (T2), an exact invariant switching surface (T3), and the coupling "
     "condition separating a global dichotomy from a stable partial state (T5)",
     "How the graph is obtained; the order under the companion gating axioms, none of "
     "which holds here, the order being recovered instead by forward coupling and "
     "root-initiated onset (Proposition 3)"),
]

def _wrap(text: str, width: int) -> List[str]:
    import textwrap
    out: List[str] = []
    for para in str(text).split("\n"):
        out.extend(textwrap.wrap(para, width) or [""])
    return out


def render_table_png(rows, path: str, title: str, col_frac=None,
                     fig_width: float = 11.0, fontsize: float = 9.5) -> str:
    """Draw a text table with wrapped cells and save it as a PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    ncol = len(rows[0])
    col_frac = col_frac or [1.0 / ncol] * ncol
    chars = [max(12, int(fig_width * f * 12.6)) for f in col_frac]
    wrapped = [[_wrap(c, chars[j]) for j, c in enumerate(r)] for r in rows]
    line_h = fontsize / 72.0 * 1.55
    heights = [max(len(c) for c in r) * line_h + 0.13 for r in wrapped]
    title_h = 0.42
    fig_h = sum(heights) + title_h + 0.25

    fig = plt.figure(figsize=(fig_width, fig_h))
    ax = fig.add_axes([0.02, 0.02, 0.96, 0.96])
    ax.set_xlim(0, 1); ax.set_ylim(0, fig_h); ax.axis("off")
    ax.text(0, fig_h - 0.22, title, fontsize=fontsize + 1.6, fontweight="bold",
            va="center", ha="left")

    y = fig_h - title_h
    edges = [0.0]
    for f in col_frac:
        edges.append(edges[-1] + f)
    for i, row in enumerate(wrapped):
        h = heights[i]
        for j, cell in enumerate(row):
            ax.add_patch(Rectangle((edges[j], y - h), col_frac[j], h,
                                   fc="#f4f4f4" if i == 0 else "white",
                                   ec="#555555", lw=0.8))
            for k, line in enumerate(cell):
                ax.text(edges[j] + 0.006, y - 0.085 - k * line_h, line,
                        fontsize=fontsize, va="top", ha="left",
                        fontweight="bold" if i == 0 else "normal")
        y -= h
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)
    return path


def write_tables(prefix: str = "evidence_closed_loop", draw: bool = True) -> List[str]:
    """Write Tables 1 to 3 as TSV and, if draw, also as PNG."""
    made = []
    specs = [("table1", TABLE1_ROWS, "Table 1. Structural dependence within the "
              "present construction.", [0.33, 0.67]),
             ("table2", TABLE2_ROWS, "Table 2. Status of theoretical results.",
              [0.38, 0.62]),
             ("table3", TABLE3_ROWS, "Table 3. Division of labour across the four "
              "studies, and what does not transfer.", [0.17, 0.22, 0.36, 0.25])]
    for name, rows, title, frac in specs:
        tsv = f"{prefix}_{name}.tsv"
        with open(tsv, "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write("\t".join(str(c).replace("\t", " ") for c in row) + "\n")
        made.append(tsv)
        if draw:
            made.append(render_table_png(rows, f"{prefix}_{name}.png", title, frac))
    return made


def write_figure1(path: str = "figure1.png") -> str:
    """Figure 1: the general construction, in four conceptual panels.

    a  the forward acyclic dependency chain, with the catatonia instantiation
    b  evidence accrues only while every prerequisite is online, C(r) = prod r_j
    c  accumulated evidence deforms the bistable field, a(Q) falling with Q
    d  the three outcomes the construction admits
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle

    BLUE, BLUE_E = "#eaf2fb", "#1f4e79"
    RED, RED_E = "#fbefec", "#8b2f26"
    GRN, GRN_E = "#edf6ef", "#2e6b3f"
    GREY = "#7a7a7a"

    fig = plt.figure(figsize=(12.2, 8.4))
    gs = fig.add_gridspec(2, 2, hspace=0.38, wspace=0.22,
                          left=0.075, right=0.98, top=0.92, bottom=0.08)

    # ---- a: a generic acyclic dependency graph -----------------------------
    ax = fig.add_subplot(gs[0, 0]); ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")
    ax.set_title("a   Dependency chain and the evidence variable",
                 fontsize=11.5, loc="left", fontweight="bold")
    xs = [1.0, 3.0, 5.0, 7.0]
    for k, x in enumerate(xs):
        ax.add_patch(Circle((x, 4.6), 0.5, fc=BLUE, ec=BLUE_E, lw=1.6))
        ax.text(x, 4.6, f"$r_{k}$", ha="center", va="center", fontsize=12)
        if k:
            ax.add_patch(FancyArrowPatch((xs[k-1] + 0.54, 4.6), (x - 0.54, 4.6),
                                         arrowstyle="-|>", mutation_scale=13,
                                         lw=1.6, color=BLUE_E))
            ax.text((xs[k-1] + x) / 2, 4.95, f"$\\kappa_{k}$", ha="center", fontsize=9.5)
    ax.text(1.0, 5.55, "root", ha="center", fontsize=9, color=BLUE_E)
    ax.add_patch(FancyBboxPatch((8.0, 4.05), 1.7, 1.1,
                                boxstyle="round,pad=0.08,rounding_size=0.2",
                                fc=GRN, ec=GRN_E, lw=1.6))
    ax.text(8.85, 4.6, "$Q$\nevidence", ha="center", va="center", fontsize=10)
    ax.add_patch(FancyArrowPatch((7.54, 4.6), (7.97, 4.6), arrowstyle="-|>",
                                 mutation_scale=13, lw=1.6, color=GRN_E))
    ax.add_patch(FancyArrowPatch((8.85, 4.02), (1.0, 4.08), arrowstyle="-|>",
                                 connectionstyle="arc3,rad=-0.32", ls="--",
                                 mutation_scale=13, lw=1.5, color=GREY))
    ax.text(4.9, 2.55, "$Q$ lowers the shared threshold $a(Q)$ of every capacity",
            ha="center", fontsize=9.5, style="italic", color=GREY)
    ax.text(0.1, 1.30, "catatonia instantiation", fontsize=9.5, color="#333333")
    ax.text(0.1, 0.55,
            r"$\pi_s \rightarrow \beta \rightarrow \pi_m \rightarrow \pi_{v,fast}"
            r" \rightarrow \pi_{v,slow}$", fontsize=12.5, color="#333333")

    # ---- b: evidence accrues only when every prerequisite is online --------
    ax = fig.add_subplot(gs[0, 1])
    ax.set_title("b   Evidence accrues only when the chain is complete",
                 fontsize=11.5, loc="left", fontweight="bold")
    t = np.linspace(0, 10, 1201)
    cols = ["#08519c", "#3182bd", "#6baed6", "#9ecae1"]
    C = np.ones_like(t)
    for k in range(4):
        rk = 1.0 / (1.0 + np.exp(-(t - (2.0 + 1.6 * k)) * 2.2))
        C *= rk
        ax.plot(t, rk, color=cols[k], lw=1.6, label=f"$r_{k}$")
    ax.plot(t, C, color=GRN_E, lw=2.6, label="$C(r)=\\prod_j r_j$")
    ax.fill_between(t, 0, C, color=GRN_E, alpha=0.12)
    ax.axvspan(0, 6.2, color=RED_E, alpha=0.05)
    ax.text(2.6, 0.72, "a prerequisite is still offline:\nno evidence accrues",
            ha="center", fontsize=9, color=RED_E)
    ax.set_xlabel("time"); ax.set_ylabel("capacity, evidence rate")
    ax.set_ylim(-0.03, 1.28)
    ax.legend(fontsize=8.5, loc="upper left", ncol=2, framealpha=0.95)
    ax.grid(alpha=0.25)

    # ---- c: accumulated evidence deforms the bistable field ----------------
    ax = fig.add_subplot(gs[1, 0])
    ax.set_title("c   Accumulated evidence deforms the threshold",
                 fontsize=11.5, loc="left", fontweight="bold")
    p = Params()
    x = np.linspace(0, 1, 601)
    for Qv, col in ((0.0, "#c6dbef"), (0.5, "#6baed6"), (1.0, "#08519c")):
        ax.plot(x, g(x, Qv, p), color=col, lw=2.0, label=f"$Q$ = {Qv:g}")
        ax.plot([float(p.a(Qv))], [0.0], "o", color=col, ms=6)
    ax.axhline(0, color="black", lw=0.8)
    ax.annotate("unstable threshold $a(Q)$\nfalls as evidence consolidates",
                xy=(float(p.a(0.0)), 0), xytext=(0.42, -0.055), fontsize=9,
                arrowprops=dict(arrowstyle="->", lw=0.9, color="#555555"))
    ax.set_xlabel("capacity $r$"); ax.set_ylabel("$g(r,Q)$")
    ax.set_ylim(-0.085, 0.13)
    ax.legend(fontsize=8.5, loc="upper left"); ax.grid(alpha=0.25)

    # ---- d: the three outcomes --------------------------------------------
    ax = fig.add_subplot(gs[1, 1]); ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")
    ax.set_title("d   Three outcomes admitted by the construction",
                 fontsize=11.5, loc="left", fontweight="bold")
    rows = [("full recovery", "$E_{rec}=(1,1,1,1,1)$", GRN, GRN_E,
             "$z(0)>0$, $\\kappa_k > K^{*}$ (T5)"),
            ("stable partial state", "root high, dependents $\\approx 0$", BLUE, BLUE_E,
             "$\\kappa_1 < a_1^{2}/4$: persists at $Q=1$"),
            ("whole-system collapse", "$E_{path}=(0,0,0,0,0)$", RED, RED_E,
             "$z(0)<0$, propagation (T3b)")]
    for i, (name, state, fc, ec, cond) in enumerate(rows):
        y = 5.05 - 1.62 * i
        ax.add_patch(FancyBboxPatch((0.15, y - 0.62), 5.4, 1.24,
                                    boxstyle="round,pad=0.06,rounding_size=0.2",
                                    fc=fc, ec=ec, lw=1.6))
        ax.text(2.85, y + 0.20, name, ha="center", fontsize=10.5, fontweight="bold")
        ax.text(2.85, y - 0.28, state, ha="center", fontsize=9.5)
        ax.text(5.75, y, cond, ha="left", va="center", fontsize=9, color="#333333")
    ax.text(0.15, 0.22, "$S=\\{r_0=a(Q)\\}$ is exactly invariant and separates the\n"
            "recovery regime from the collapse regime",
            fontsize=9, style="italic", color=GREY)

    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)
    return path


def write_figure2(path: str = "figure2.png") -> str:
    """Figure 2: dependency structure of the construction (axioms to theorems)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    BLUE, BLUE_E = "#eaf2fb", "#1f4e79"
    RED, RED_E = "#fbefec", "#8b2f26"
    GRN, GRN_E = "#edf6ef", "#2e6b3f"
    GREY, GREY_E = "#f2f2f2", "#3a3a3a"

    fig, ax = plt.subplots(figsize=(12.4, 10.0))
    ax.set_xlim(0, 20); ax.set_ylim(0, 17); ax.axis("off")

    def box(x0, y0, x1, y1, text, fc, ec, bold=False, size=12.5):
        ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                                    boxstyle="round,pad=0.12,rounding_size=0.28",
                                    fc=fc, ec=ec, lw=1.6))
        ax.text((x0 + x1) / 2, (y0 + y1) / 2, text, ha="center", va="center",
                fontsize=size, color="#111111",
                fontweight="bold" if bold else "normal", linespacing=1.45)

    def arrow(p1, p2, color, ls="-", rad=0.0, lw=1.6):
        ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", ls=ls, lw=lw,
                                     color=color, mutation_scale=17,
                                     connectionstyle=f"arc3,rad={rad}",
                                     shrinkA=1, shrinkB=1))

    tops = [("reported clinical\nfeatures", 1.4, 5.6),
            ("structural assumptions\n(A to D, E1 to E4)", 6.0, 10.6),
            ("theorems\n(T1 to T5)", 11.0, 15.0),
            ("testable\npredictions", 15.4, 19.4)]
    for i, (t, x0, x1) in enumerate(tops):
        box(x0, 15.3, x1, 16.6, t, GREY, GREY_E, size=12.0)
        if i:
            arrow((tops[i - 1][2] + 0.14, 15.95), (x0 - 0.14, 15.95), GREY_E, lw=1.4)
    ax.plot([0.6, 19.4], [14.75, 14.75], color="#c8c8c8", lw=1.4)

    box(4.9, 12.5, 16.6, 14.0,
        "Axiom A: forward dependency chain\n(directed acyclic graph + diffusive coupling)",
        BLUE, BLUE_E, bold=True, size=13.5)
    box(2.6, 10.2, 10.2, 11.5, "Axiom C   evidence gating C(r)", BLUE, BLUE_E)
    box(2.6, 8.1, 10.2, 9.4, "T4   Q = accumulated evidence", RED, RED_E, bold=True)
    box(2.6, 6.0, 10.2, 7.3, "Axiom D   concave deepening a(Q)", BLUE, BLUE_E)
    box(2.6, 3.6, 10.2, 5.2, "T1  super-linear acceleration\nT2  recovered state stable",
        RED, RED_E, bold=True)
    box(12.0, 9.4, 19.5, 11.4,
        "Axiom B   bistability\nE   regime-selective update,\nthe sharp limit of E1 to E4 (Prop. 1)",
        BLUE, BLUE_E, size=12.0)
    box(12.2, 5.9, 19.3, 7.7, "T3\nfragile window;\nwhole-layer collapse", RED, RED_E, bold=True)
    box(3.9, 1.0, 17.1, 2.7,
        "T5   (strong coupling  κ > K*)\nglobal two-outcome: full recovery or full collapse",
        GRN, GRN_E, bold=True, size=13.0)

    arrow((10.4, 12.5), (6.6, 11.6), BLUE_E)
    arrow((11.2, 12.5), (15.4, 11.5), RED_E)
    ax.text(3.9, 12.05, "recovery direction", fontsize=12, style="italic", color=BLUE_E)
    ax.text(15.9, 12.05, "collapse direction", fontsize=12, style="italic", color=RED_E)
    ax.text(11.35, 11.75, "root autonomy", fontsize=11.5, style="italic", color=RED_E)

    arrow((6.4, 10.2), (6.4, 9.5), BLUE_E)
    arrow((6.4, 8.1), (6.4, 7.4), BLUE_E)
    arrow((6.4, 6.0), (6.4, 5.3), BLUE_E)
    arrow((15.7, 9.4), (15.7, 7.8), RED_E)
    arrow((8.0, 3.6), (9.8, 2.8), GRN_E, rad=-0.18)
    arrow((15.0, 5.9), (12.4, 2.8), GRN_E, rad=0.18)
    arrow((2.55, 4.4), (2.55, 10.7), "#9a9a9a", ls="--", rad=-0.55, lw=1.5)
    ax.text(0.35, 7.6, "self-\ncatalysis", fontsize=11.5, style="italic",
            color="#7a7a7a", ha="center")

    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)
    return path


def write_supplementary_figure(p: Params, path: str = "supplementary_figure1.png",
                               full: bool = False) -> str:
    """Four panels: selectors, trajectories, boundary drift, mixed branch."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.0))

    # (a) selectors
    ax = axes[0, 0]
    zs = np.linspace(-0.5, 0.5, 1201)
    for gam, c in ((2.0, "#9ecae1"), (5.0, "#4292c6"), (20.0, "#08519c")):
        q = p.with_(gamma=gam)
        ax.plot(zs, [chi_bayes(float(z), q)[0] for z in zs], color=c,
                label=f"chi+ , gamma = {gam:g}")
    ax.plot(zs, [chi_sharp(float(z), p.with_(sel_scale=0.01))[0] for z in zs],
            color="#8b2f26", ls="--", label="sharp exp(-0.01/z^2)")
    ax.plot(zs, [chi_bma(float(z), p.with_(gamma=20.0))[0] for z in zs],
            color="#7a7a7a", ls=":", label="plain averaging (no E4)")
    ax.axvline(0, color="black", lw=0.7)
    ax.set_xlabel("z"); ax.set_ylabel("selector")
    ax.set_title("a  Accumulation selector: E4 forces chi(0) = 0", fontsize=10.5)
    ax.legend(fontsize=7.5, loc="upper left"); ax.grid(alpha=0.25)

    # (b) trajectories either side of S
    ax = axes[0, 1]
    q = p.with_(sel_scale=0.01, eps=0.5)
    Qc = p.Q_of_a(0.45)
    for dq, col, lab in ((+2e-3, "#1f4e79", "z(0) > 0"), (-2e-3, "#8b2f26", "z(0) < 0")):
        sol = integrate([0.45, 1.0, 1.0, 1.0, Qc + dq], 400.0, q, n_eval=2001,
                        max_step=0.5)
        ax.plot(sol.t, sol.y[0], color=col, label=f"r0, {lab}")
        ax.plot(sol.t, sol.y[4], color=col, ls="--", label=f"Q, {lab}")
    ax.set_xlabel("t"); ax.set_ylabel("state")
    ax.set_title("b  Recovered start, excursion to r0 = 0.45", fontsize=10.5)
    ax.legend(fontsize=7.5); ax.grid(alpha=0.25)

    # (c) outcome boundary against the evidence timescale
    ax = axes[1, 0]
    eps_list = ([0.02, 0.01, 0.005, 0.0025, 0.00125] if full
                else [0.02, 0.005, 0.00125])
    b_e4 = [collapse_boundary(p.with_(selector="bayes", eps=e)) for e in eps_list]
    b_no = [collapse_boundary(p.with_(selector="bma", eps=e)) for e in eps_list]
    ax.semilogx(eps_list, b_e4, "o-", color="#1f4e79", label="with E4 (chi(0) = 0)")
    ax.semilogx(eps_list, b_no, "s-", color="#7a7a7a", label="plain averaging")
    ax.axhline(Qc, color="#8b2f26", ls="--", label=f"exact Qc = {Qc:.4f}")
    ax.set_xlabel("eps"); ax.set_ylabel("collapse boundary in Q")
    ax.set_title("c  S is invariant at any sharpness only with E4", fontsize=10.5)
    ax.legend(fontsize=8); ax.grid(alpha=0.25)

    # (d) mixed branch and its saddle-node
    ax = axes[1, 1]
    kc = p.a1 ** 2 / 4.0
    ks = np.linspace(1e-5, kc, 300)
    lo = (p.a1 - np.sqrt(p.a1 ** 2 - 4 * ks)) / 2.0
    hi = (p.a1 + np.sqrt(p.a1 ** 2 - 4 * ks)) / 2.0
    ax.plot(ks, lo, color="#1f4e79", label="low root (stable)")
    ax.plot(ks, hi, color="#8b2f26", ls="--", label="upper root (unstable)")
    ax.plot([kc], [p.a1 / 2], "ko", ms=5)
    ax.annotate(f"saddle-node\nkappa1 = a1^2/4 = {kc:.6f}", (kc, p.a1 / 2),
                textcoords="offset points", xytext=(-118, 14), fontsize=8,
                arrowprops=dict(arrowstyle="-", lw=0.7, color="#555555"))
    ax.set_xlabel("kappa1"); ax.set_ylabel("r1 at Q = 1")
    ax.set_title("d  Weak-coupling mixed branch, first stage", fontsize=10.5)
    ax.legend(fontsize=8); ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

def _byname(results: List[Result], name: str) -> Dict[str, object]:
    for r in results:
        if str(r["name"]).startswith(name):
            return r.get("detail", {})  # type: ignore[return-value]
    return {}


def report_quantities(results: List[Result], tol_scale: float = 1.0) -> bool:
    """Print every quantity quoted in the manuscript next to its computed value.

    The pass/fail table above says only that each check ran and succeeded; this
    table is what a reader should compare against the text.
    """
    K = _byname(results, "K*")
    T3 = _byname(results, "T3 fragile")
    P1S = _byname(results, "Proposition 1 sharp")
    T4 = _byname(results, "T4 evidence")
    T5 = _byname(results, "T5 two-outcome")
    WK = _byname(results, "weak-coupling")
    T2 = _byname(results, "T2 stability")
    P3O = _byname(results, "Proposition 3 order")
    CL = _byname(results, "Proposition 4 closure")
    AD = _byname(results, "Proposition 5 Axiom D")
    FG = _byname(results, "Proposition 6 finite sharpness")
    GR = _byname(results, "Proposition 7 gate")

    mixed = WK.get("below", {})
    sn = WK.get("saddle_node", {})
    b_e4 = P1S.get("boundary_with_E4", {})
    b_no = P1S.get("boundary_without_E4_plain_averaging", {})

    def g(d, k, default=float("nan")):
        try:
            return d[k]
        except Exception:
            return default

    rows = [
        # label, quoted in the manuscript, computed, tolerance (None = no numeric test)
        ("K* = max (1-a+a^2)/3 on [a1,a0]", 0.291, g(K, "K_star"), 5e-4),
        ("max_x g_r over the cube (<= K*)", None, g(K, "numerical_max_g_r"), None),
        ("max g(x,Q)/x over the cube (<= K*, Proposition 6b)", None,
         g(K, "max_g_over_x"), None),
        ("Qc for Delta = 0.55 (T3a)", 0.5774, g(T3, "Qc_exact"), 5e-4),
        ("collapse boundary, sharp, eps = 0.02", 0.5774, g(T3, "boundary_eps_0.02"), 5e-4),
        ("collapse boundary, sharp, eps = 0.005", 0.5774, g(T3, "boundary_eps_0.005"), 5e-4),
        ("boundary, finite gamma with E4, eps = 0.02", 0.5774, g(b_e4, 0.02), 5e-4),
        ("boundary, finite gamma with E4, eps = 0.005", 0.5774, g(b_e4, 0.005), 5e-4),
        ("boundary, plain averaging, eps = 0.02", 0.5940, g(b_no, 0.02), 5e-4),
        ("boundary, plain averaging, eps = 0.005", 0.5820, g(b_no, 0.005), 5e-4),
        ("boundary, plain averaging, eps = 0.00125", 0.5786, g(b_no, 0.00125), 5e-4),
        ("a1^2/4, first-stage low-root threshold", 0.0056, g(WK, "a1^2/4"), 5e-5),
        ("saddle-node r1 = a1/2", 0.075, (g(sn, "equilibrium", [0, float("nan")]) or [0, 0])[1], 1e-6),
        ("I(pi_v_slow) vs mu_slow, max relative error", None,
         g(CL, "max_rel_error_I_slow_vs_mu_slow"), None),
        ("leak of pi_v_slow into the other four channels", 0.0,
         g(CL, "max_leak_other_channels"), 1e-12),
        ("Lemma C1 bracket, min M_slow / C(r) (>= 1)", None,
         g(CL, "min_M_slow_over_C"), None),
        ("Lemma C1 bracket, max M_slow / C(r) (<= K)", None,
         g(CL, "max_M_slow_over_C"), None),
        ("Lemma C1 constant K = B / (1 - exp(-B))", None, g(CL, "K_bracket"), None),
        ("per-context count vs integral of C(r), relative error", None,
         g(CL, "monte_carlo_relative_error"), None),
        ("first passage of Q vs the last capacity (corollary)", None,
         g(CL, "first_passage_Q"), None),
        ("Axiom D: max dG1/dr over the sampled range (< 0)", None,
         g(AD, "max_dG_dr"), None),
        ("Axiom D: max dG1/dLambda over the sampled range (< 0)", None,
         g(AD, "max_dG_dLambda"), None),
        ("Axiom D: max a'(Q) at Lstar = 2 (< 0)", None,
         g(AD, "max_first_derivative_Lstar_ok"), None),
        ("Axiom D: max a''(Q) at Lstar = 2 (< 0)", None,
         g(AD, "max_second_derivative_Lstar_ok"), None),
        ("Axiom D: max a''(Q) at Lstar = 5 (> 0, control)", None,
         g(AD, "max_second_derivative_Lstar_bad"), None),
        ("Axiom D: elasticity q at Lambda0", 3.91, g(AD, "elasticity_at_Lambda0"), 5e-3),
        ("Axiom D: elasticity q as Lambda -> infinity", 2.0, g(AD, "elasticity_limit"), 5e-3),
        ("Axiom D: max q Lstar / Lambda, Lstar = 2", 0.78, g(AD, "part_b_margin_ok"), 5e-3),
        ("Axiom D: max q Lstar / Lambda, Lstar = 5", 1.95, g(AD, "part_b_margin_bad"), 5e-3),
        ("finite gamma: 1 - Q at gamma = 20 (vs fixed point)", None,
         g(g(FG, "recovery_side", {}).get("gamma=20", {}), "one_minus_Q"), None),
        ("finite gamma: fixed-point prediction at gamma = 20", None,
         g(g(FG, "recovery_side", {}).get("gamma=20", {}), "fixed_point_prediction"), None),
        ("finite gamma: bound rho exp[-gamma(1-a0)] at gamma = 20", None,
         g(g(FG, "recovery_side", {}).get("gamma=20", {}), "bound_rho_exp"), None),
        ("Lemma F: explicit margin z_bar at gamma = 20", None,
         g(g(FG, "lemma_F", {}), "z_bar_at_gamma_20"), None),
        ("finite gamma: min margin z on the recovery side", None,
         g(g(FG, "recovery_side", {}).get("gamma=20", {}), "min_z"), None),
        ("comparable gate: boundary minus Qc", 0.0,
         g(g(GR, "gates", {}).get("modulated", {}), "boundary_minus_Qc"), 1e-4),
        ("leaky gate: dQ/dt with one capacity offline", None,
         g(g(GR, "gates", {}).get("leaky", {}), "dQ_dt_with_one_node_offline"), None),
        ("leak at finite sharpness: Q at the failed corner", None,
         g(g(GR, "leak_at_finite_sharpness", {}), "Q_at_failed_corner"), None),
        ("leak at finite sharpness: predicted failed-corner Q", None,
         g(g(GR, "leak_at_finite_sharpness", {}), "prediction"), None),
        ("E4 violation: drift / (eps eta), minimum", None,
         (g(GR, "drift_ratio_range", [float("nan"), float("nan")])
          or [float("nan"), float("nan")])[0], None),
        ("E4 violation: drift / (eps eta), maximum", None,
         (g(GR, "drift_ratio_range", [float("nan"), float("nan")])
          or [float("nan"), float("nan")])[1], None),
        ("T4 closed form, absolute error", None, g(T4, "closed_form_abs_err"), None),
        ("general linear solution, max error", None, g(T4, "general_solution_max_abs_err"), None),
        ("indicator limit, jump at z = 0 (why E regularises)", 1.0,
         g(P1S, "limit_jump_at_zero"), 1e-3),
        ("Erec Q-eigenvalue vs -eps chi_+(1-a1)", None,
         g(g(T2, "analytic", {}), "Erec_Q"), None),
        ("Epath Q-eigenvalue vs -eps rho chi_-(-a0)", None,
         g(g(T2, "analytic", {}), "Epath_Q"), None),
        ("corner spectra, max error against the analytic Jacobian", 0.0,
         max(g(g(T2, "max_abs_err_vs_analytic", {}), "Erec", 1.0),
             g(g(T2, "max_abs_err_vs_analytic", {}), "Epath", 1.0)), 1e-7),
        ("T5 trajectories, all as predicted", None, g(T5, "all_as_predicted"), None),
        ("orbit started on S reaches ES", None, g(T5, "S_orbit_to_ES"), None),
    ]

    print()
    print("key quantities quoted in the manuscript")
    print(f"{'quantity':<44}{'manuscript':>12}{'computed':>16}   check")
    print("-" * 88)
    ok_all = True
    for label, quoted, got, tol in rows:
        if isinstance(got, bool):
            shown, mark = str(got), "ok" if got else "MISMATCH"
            ok_all &= bool(got)
            q = "-"
        else:
            try:
                shown = f"{float(got):.6g}"
            except Exception:
                shown = str(got)
            q = "-" if quoted is None else f"{quoted:g}"
            if quoted is None or tol is None:
                mark = ""
            else:
                good = abs(float(got) - float(quoted)) <= tol * tol_scale
                ok_all &= good
                mark = "ok" if good else "MISMATCH"
        print(f"{label:<44}{q:>12}{shown:>16}   {mark}")
    print("-" * 88)

    print("mixed branch at kappa = 0.003, Q = 1")
    print(f"  equilibrium r          {g(mixed, 'equilibrium')}")
    print(f"  state eigenvalues      {g(mixed, 'state_eigenvalues')}")
    print(f"  hyperbolically stable  {g(mixed, 'hyperbolic')}")
    print(f"  saddle-node at kappa1  {g(sn, 'kappa')}  (zero eigenvalue: "
          f"{g(sn, 'state_eigenvalues')})")
    print("corner spectra")
    for tag in ("rho>0", "rho=0"):
        d = T2.get(tag, {}) if isinstance(T2, dict) else {}
        print(f"  {tag:<6} Erec {np.round(g(d, 'Erec_eigs', []), 6).tolist()}")
        print(f"  {tag:<6} Epath {np.round(g(d, 'Epath_eigs', []), 6).tolist()}")
    print(f"Proposition 3 first-passage times at q = 0.8: {g(P3O, 'first_passage_times')}")
    print(f"selectors at z = 0, finite gamma: {g(P1S, 'chi_at_zero_finite_gamma')}")
    print(f"selectors at z = 0, plain averaging: {g(P1S, 'chi_at_zero_plain_averaging')}")
    print(f"sup |chi - indicator| off the boundary: {g(P1S, 'sup_err_off_boundary')}")
    print("-" * 88)
    print("all quoted values reproduced" if ok_all else "SOME QUOTED VALUES DIFFER")
    return ok_all


def in_notebook() -> bool:
    """True inside Jupyter, Colab or any ipykernel front end."""
    try:
        from IPython import get_ipython
        ip = get_ipython()
        return ip is not None and "IPKernelApp" in getattr(ip, "config", {})
    except Exception:
        return False


def _display(paths: Sequence[str]) -> None:
    if not in_notebook():
        return
    try:
        from IPython.display import Image, display
        for path in paths:
            if str(path).endswith(".png"):
                display(Image(filename=path))
    except Exception:
        pass


def run(full: bool = False, figures: bool = True, json_path: str = "",
        strict: bool = False, fast: bool = False) -> List[Result]:
    """Run every check and, if figures, draw Tables 1 to 3 and the figures.

    Safe to call from a notebook cell:  results = run()
    """
    global FAST, BOUNDARY_TOL
    FAST = bool(fast)
    BOUNDARY_TOL = 5e-4 if FAST else 1e-7
    collapse_boundary.cache_clear()
    p = Params()
    t0 = time.time()
    results = [
        check_positive_invariance(p),
        check_K_star(p),
        check_T2(p),
        check_T4(p),
        check_prop1_sharp_limit(p, full),
        check_T3(p, full),
        check_T5(p, full),
        check_weak_coupling(p),
        check_prop3_order(p),
        check_closure(p),
        check_axiom_D(p),
        check_finite_gamma(p),
        check_gate_and_E4(p),
        ablations(p),
    ]
    elapsed = time.time() - t0

    width = max(len(str(r["name"])) for r in results) + 2
    print(f"{'result':<{width}} check   status")
    print("-" * (width + 46))
    for r in results:
        print(f"{str(r['name']):<{width}} {'pass' if r['pass'] else 'FAIL':<7} {r['status']}")
    print("-" * (width + 46))
    mode = "full" if full else ("fast" if FAST else "quick")
    print(f"mode: {mode}   elapsed: {elapsed:.1f} s")

    report_quantities(results, tol_scale=20.0 if FAST else 1.0)

    if figures:
        made = write_tables()
        made.append(write_figure1())
        made.append(write_figure2())
        made.append(write_supplementary_figure(p, full=full))
        print("wrote: " + ", ".join(made))
        _display([m for m in made if m.endswith(".png")])
    if json_path:
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=1, default=float)
        print(f"wrote {json_path}")

    failed = [r["name"] for r in results if not r["pass"]]
    if failed:
        print("FAILED: " + ", ".join(map(str, failed)))
        if strict:
            raise SystemExit(1)
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description="Reference implementation.")
    ap.add_argument("--full", action="store_true", help="finer sweeps")
    ap.add_argument("--fast", action="store_true",
                    help="coarser sweeps, same checks, for a quick reviewer run")
    ap.add_argument("--no-figures", action="store_true",
                    help="skip tables and figures")
    ap.add_argument("--json", default="", help="write all results to this JSON file")
    # parse_known_args, so that a notebook kernel's own -f argument is ignored
    args, _unknown = ap.parse_known_args([] if in_notebook() else sys.argv[1:])
    run(full=args.full, figures=not args.no_figures, json_path=args.json,
        strict=not in_notebook(), fast=args.fast)


if __name__ == "__main__":
    main()
