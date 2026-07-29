# evidence-closed-loop

Reference implementation for

**Evidence-coupled bistability converts dependency structure into a recovery geometry**
Hiroki Saito

This repository contains the single script that reproduces every result reported in
the manuscript and regenerates every display item: Tables 1 to 3, Figures 1 and 2,
and the supplementary figure. Every number quoted in the manuscript is printed by
the script next to its computed value, so the text and the code can be compared
line by line.

## Quick start

```bash
pip install -r requirements.txt
python evidence_closed_loop.py
```

The default (quick) mode runs all checks and writes every display item in about
100 seconds on a laptop. The run ends with `all quoted values reproduced` when the
manuscript values and the computed values agree.

### Run modes

| Command | What it does |
|---|---|
| `python evidence_closed_loop.py` | quick mode, all checks, all display items |
| `python evidence_closed_loop.py --full` | finer parameter sweeps |
| `python evidence_closed_loop.py --fast` | coarser sweeps, about a quarter of the time |
| `python evidence_closed_loop.py --no-figures` | checks only, no files written |
| `python evidence_closed_loop.py --json out.json` | also dump every result as JSON |

The switching-boundary measurements are bisected to 1e-7 in every run mode, so the
values quoted in the manuscript do not depend on the mode chosen.

### Notebooks (Jupyter, Google Colab)

```python
!python evidence_closed_loop.py           # works inside a Colab cell
import evidence_closed_loop as m; res = m.run()
```

Unknown command line arguments are ignored, so the `-f kernel.json` that a notebook
kernel injects into `sys.argv` does not abort the run, and the figures are displayed
inline when a notebook front end is detected.

## The model

```
dr0/dt = g(r0, Q)                            + u(t)
drk/dt = g(rk, Q) + kappa_k (r_{k-1} - rk),    k = 1, ..., n
dQ/dt  = eps [ chi_+(z) C(r) (1 - Q) - rho chi_-(z) Q ]

g(r, Q) = r (1 - r) (r - a(Q)),   C(r) = prod_j r_j,
a(Q)    = a0 - (a0 - a1) Q**p,    z = r0 - a(Q).
```

The theorems are proved in the Methods for a forward acyclic chain of `n + 1`
capacities. This implementation is the displayed case `n = 3`, the shortest chain
with a root, an interior pair and a leaf.

## What the script checks

| Check | Result verified |
|---|---|
| T0 | positive invariance of the state cube |
| K* | the uniform bound on `g_r`, and `g(x,Q) <= K* x` used by Proposition 6(b) |
| T2 | both corner equilibria, their spectra against the analytic Jacobian |
| T4 | the closed-form evidence law and the general linear solution |
| Proposition 1 | boundary zeros at finite sharpness, and the sharp-selection limit |
| T3 | the fragile window, propagation, and invariance of the switching surface |
| T5 | two-outcome convergence under strong coupling |
| weak coupling | the stable mixed branch and its saddle-node |
| Proposition 3 | order preservation and ordered first-passage times |
| Proposition 4 | closure of the loop: the Fisher information about the terminal node accumulates as the product gate |
| Proposition 5 | Axiom D from an active-inference decision layer |
| Proposition 6 | the dichotomy at finite decision sharpness |
| Proposition 7 | gate robustness, a leaky gate, and the O(eps eta) drift when E4 is violated |
| Table 1 | the ablations, one row at a time |

## Output files

Running with figures writes, in the working directory:

```
evidence_closed_loop_table1.tsv / .png
evidence_closed_loop_table2.tsv / .png
evidence_closed_loop_table3.tsv / .png
figure1.png
figure2.png
supplementary_figure1.png
```

## Requirements

Python 3.10 or later with numpy, scipy and matplotlib. All three are preinstalled
in Google Colab.

## Citing

If you use this code, please cite the archived release:

> Saito, H. evidence-closed-loop: reference implementation for "Evidence-coupled
> bistability converts dependency structure into a recovery geometry". Zenodo,
> https://doi.org/10.5281/zenodo.21500263 (2026).

Machine readable metadata is in `CITATION.cff`.

## Licence

MIT. See `LICENSE`.
