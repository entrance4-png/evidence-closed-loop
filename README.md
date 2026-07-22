# evidence-closed-loop

Reference implementation for the theory paper

> **A unified dynamical theory of catatonia recovery from an evidence-coupled bistable cascade**
> Hiroki Saito

This repository contains the numerical corroboration for the theorems in that paper, and it
regenerates the paper's display items. **The proofs are in the manuscript (Results and Methods);
nothing in this repository is part of a proof.** The code exists so that every quantitative
statement in the paper can be reproduced and checked independently.

## Contents

| File | Purpose |
| --- | --- |
| `evidence_closed_loop.py` | The whole implementation: model, verification by status, display items |
| `figure1.png` | Figure 1 of the paper, as produced by the script |
| `expected_output.txt` | Output of a reference run (`--quick`), for comparison |
| `requirements.txt` | Versions used for the reference run |
| `CITATION.cff` | Citation metadata |
| `.zenodo.json` | Metadata for the Zenodo archive |

## Requirements

Python 3.10 or later, with NumPy, SciPy and Matplotlib. The reference run used Python 3.12.3,
NumPy 2.4.4, SciPy 1.17.1 and Matplotlib 3.10.8.

```bash
python3 -m pip install -r requirements.txt
```

Matplotlib is needed only for Figure 1; the numerical checks run without it (`--no-figure`).

## Running

```bash
python3 evidence_closed_loop.py                 # quick mode (default)
python3 evidence_closed_loop.py --full          # wider sweeps and more initial conditions
python3 evidence_closed_loop.py --figure-only   # write Figure 1 and stop (about a second)
python3 evidence_closed_loop.py --no-figure     # verification only
python3 evidence_closed_loop.py --figure-path /somewhere/fig.png
```

A full pass takes a couple of minutes; it is integration bound. `--full` differs from `--quick`
only in sample sizes, the number of bisection steps, and the number of initial conditions and
parameter values tested. No conclusion depends on the mode.

Figure 1 is written **before** the integrations start, so it does not depend on the rest of the
run finishing, and it is written **next to this script** rather than into the current working
directory. The absolute path is printed. In Jupyter or Colab the saved PNG is also displayed in
the output cell:

```python
%run evidence_closed_loop.py --figure-only
```

## What the run produces

The output has two parts.

**1. Verification, grouped by the status of the corresponding result.** The grouping mirrors
Table 2 of the paper ("Status of theoretical results"), so a reader can see at a glance which
claims are unconditional theorems, which hold under an explicit condition, and which rest on the
observation-model premise.

- `T0` positive invariance of the state cube
- `T1` evidence-mediated acceleration, convexity, and the degree `a''/a' = (p-1)/Q`
- `T2` stability of both equilibria, including the loss of hyperbolicity at `rho = 0`
- `T3a` the frozen-evidence root trigger and the predicted threshold `Q_c`
- `T3c1` invariance of the switching manifold, checked at two values of `eps`
- `T4` the closed form `Q(t) = 1 - (1-Q0) exp(-eps N(t))`, checked against integration of the
  augmented system that carries `N` as a coordinate
- `T3b`, `T3c2` the conditional results (coupling bound; recovered initial configuration)
- `T5` the global two-outcome dichotomy under strong coupling, the equilibrium continuum on `S`,
  and the stable mixed equilibrium at weak coupling that shows the coupling condition cannot be
  dropped
- one ablation per row of Table 1 ("structural dependence")

**2. The display items, in the order in which they appear in the manuscript:**
Table 1, Figure 1, Table 2, Table 3.

The `figure1.png` produced by the run is the figure embedded in the manuscript.

## Comparing against the reference run

`expected_output.txt` is the output of `python3 evidence_closed_loop.py --quick` on the machine
described above.

```bash
python3 evidence_closed_loop.py --quick > my_output.txt
diff expected_output.txt my_output.txt
```

Four kinds of difference are expected and harmless:

- the two lines giving the path of `figure1.png`, which is absolute and therefore machine specific;
- the reported runtime on the last line;
- the last digits of quantities that sit at the floating-point noise floor, namely the
  `max|difference|` values of the T4 check, the residual of the weak-coupling mixed equilibrium,
  and its transverse eigenvalue, which is provably negative but of order `eps * C` and so tiny;
- the measured bisection thresholds, in the last digit or two.

Everything else, including every verdict (`PASS`, `OK`, `GENUINE STABLE`) and every status label,
should match.

## Model

State variables `r_k` in `[0,1]` for `k = 0..3` index sensory, policy, motivational and
fast-volatility precision. The evidence variable `Q` in `[0,1]` is identified with slow-volatility
precision. Writing `z = r_0 - a(Q)`:

```
dr_0/dt = g(r_0, Q)                                   [ + u(t) ]
dr_k/dt = g(r_k, Q) + kappa_k (r_{k-1} - r_k)          k = 1..3,  kappa_0 = 0
dQ/dt   = eps [ chi_+(z) C(r) (1 - Q) - rho chi_-(z) Q ]

g(r,Q) = r(1-r)(r - a(Q)),   C(r) = prod_j r_j,   a(Q) = a0 - (a0-a1) Q^p,   p > 1
```

Illustrative parameters: `kappa = 0.6`, `eps = 0.02`, `rho = 1`, `a0 = 0.60`, `a1 = 0.15`,
`p = 2`. No parameter search was performed to force any outcome.

## Scope

The observation-model premise, the concavity of `a(Q)` and the regime-selective update are
modelling commitments, stated as such in the paper; the theorems are conditional on them.
Clinical validity is an empirical question and is not addressed here.

## Citing

Please cite the archived release (see `CITATION.cff`) and the paper.

## License

MIT, see `LICENSE`. The manuscript itself is not included in this repository and is not covered
by that license.
