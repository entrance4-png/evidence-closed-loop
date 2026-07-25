# evidence-closed-loop

Reference implementation for the theory paper

> **A unified dynamical theory of catatonia recovery from an evidence-coupled bistable cascade**
> Hiroki Saito

Numerical corroboration for the theorems, and regeneration of every display item of the
manuscript. The proofs are in the manuscript (Results and Methods). **Nothing in this
repository is part of a proof**: the code checks that the analytical statements are
self-consistent and reproduces the quantitative values quoted in the paper.

## Requirements

Python 3.10 or later, with

```
numpy
scipy
matplotlib
```

```bash
python3 -m pip install -r requirements.txt
```

## Running it

```bash
python3 evidence_closed_loop.py --full        # complete verification, about 3 minutes
python3 evidence_closed_loop.py               # quick pass (default), coarser sweeps
python3 evidence_closed_loop.py --figure-only # draw the display items and exit
python3 evidence_closed_loop.py --no-figure   # verification only, no images
```

`--figure-path PATH` sets where `figure1.png` is written; the three table images go into
the same directory, which is created if it does not exist.

Runtime is integration bound. `--full` is what produced `expected_output.txt`.

## What a run produces

Four PNG files are written next to the script, **before** the integrations start, so they
do not depend on the rest of the run finishing:

| File | Display item |
|---|---|
| `table1.png` | Table 1, structural dependence |
| `figure1.png` | Figure 1, dependency structure of the construction |
| `table2.png` | Table 2, status of theoretical results |
| `table3.png` | Table 3, division of labour with the companion theories |

Each table image is drawn from the same row text and the same column proportions as the
plain-text table the script prints, so the drawn and the printed versions cannot drift
apart. In a Jupyter or Colab notebook the images are also shown inline as they are
written.

The console output is grouped by the status of each result, mirroring Table 2 of the
manuscript: proved unconditionally, proved given the observation-model premise, proved
under an explicit condition, and the two-outcome result under strong coupling.

## Checking a run against the reference

`expected_output.txt` is the full output of `--full` on the reference machine. Differences
that are harmless:

- **file paths**, which are placeholders in the reference file and will be your own paths
- **the runtime line** at the end
- **last-digit floating-point noise** in Monte-Carlo and bisection quantities

Anything else, in particular a `PASS` that has become `FAIL`, a changed boundary value or
a changed eigenvalue, is a real difference and worth reporting as an issue.

## The model

State variables `r_k` in [0,1] for k = 0..3 index sensory, policy, motivational and
fast-volatility precision. The evidence variable `Q` in [0,1] is identified with
slow-volatility precision. With `z = r_0 - a(Q)`,

```
dr_0/dt = g(r_0,Q)  [ + u(t) ]
dr_k/dt = g(r_k,Q) + kappa_k (r_{k-1} - r_k),        k = 1..3,  kappa_0 = 0
dQ/dt   = eps [ chi_+(z) C(r) (1-Q) - rho chi_-(z) Q ]

g(r,Q) = r(1-r)(r-a(Q)),   C(r) = prod_j r_j,   a(Q) = a0 - (a0-a1) Q^p,   p > 1
```

Illustrative parameters: `kappa = 0.6`, `eps = 0.02`, `rho = 1`, `a0 = 0.60`, `a1 = 0.15`,
`p = 2`. No parameter search was performed to force any outcome.

## Two things to read before quoting a number

**Axiom E is the load-bearing idealisation.** The mutual exclusivity of the selectors
(disjoint supports, both vanishing at `z = 0`) is what makes the switching manifold `S`
invariant at every timescale. The exactness of T3(c1) and its independence of `eps` are
therefore consequences of that modelling choice, not independent findings. The ablation in
Table 1 replaces the exclusive selectors by an overlapping soft pair and the boundary
becomes `eps`-dependent, approaching the exact value as the evidence timescale is made
slow: 0.5935 at `eps = 0.02`, 0.5820 at `eps = 0.005`, against `Q_c = 0.5774`.

**All printed numbers are illustrative.** Milestone and consolidation times, boundary
locations and the mixed-branch eigenvalues depend on the selector shape and the parameter
choice. Because `chi_+(z) = exp(-1/z^2)` is extremely small for small `z`, **timescales in
particular must not be read as predicted real times**. The theorems depend only on the
class properties stated in Axiom E and are unaffected by this dependence.

## Citing

Cite the archived release, not this repository. See `CITATION.cff`.

The concept DOI [10.5281/zenodo.21500263](https://doi.org/10.5281/zenodo.21500263) always
resolves to the latest archived version.

## License

MIT, see `LICENSE`. The licence covers the software in this repository. It does not cover
the manuscript.

## Changelog

**1.1.0**
- Tables 1 to 3 are now drawn as PNG files alongside Figure 1, from the same row text and
  column proportions as the plain-text tables.
- `--figure-path` now creates the target directory if it does not exist, and the three
  table images follow Figure 1 into it.
- Table 2 records the milestone-order row as established in the companion theory under a
  different axiom class rather than inherited from it, and Table 3 is expanded to state
  what does not transfer between the two axiom classes.
- The header documents Axiom E as a disclosed idealisation and flags every printed number
  as illustrative.
- Figure 1 top banner reads "reported clinical features" rather than "bedside
  observations", matching the revised manuscript.
- No change to any equation, parameter, theorem check or numerical result. Every computed
  line of `--full` is identical to the 1.0.0 output.

**1.0.0**
- First archived release.
