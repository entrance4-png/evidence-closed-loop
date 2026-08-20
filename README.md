# evidence-closed-loop

Reference implementation and verification script for

> **Evidence-coupled bistability converts dependency structure into a recovery geometry**
> Hiroki Saito

Every numerical value quoted in the manuscript and its Supplementary Information is produced by this
script. Nothing is entered by hand. The tables and figures of the paper are written by the same run.

## Model

    dr0/dt = g(r0, Q) + u(t)
    drk/dt = g(rk, Q) + kappa_k (r_{k-1} - rk),   k = 1, 2, 3
    dQ/dt  = eps [ chi_+(z) C(r) (1 - Q) - rho chi_-(z) Q ]

with

    g(r, Q) = r (1 - r) (r - a(Q)),   C(r) = prod_j r_j,
    a(Q)    = a0 - (a0 - a1) Q**p,    z = r0 - a(Q).

## Requirements

    python >= 3.10
    pip install -r requirements.txt

Only numpy, scipy and matplotlib are used. No data files are required; the script is
self-contained and deterministic, with every random seed fixed in the source.
- OS: developed and tested on macOS. The script contains no OS-specific code and
  runs on any platform with a supported Python.
- Python >= 3.10, numpy >= 1.24, scipy >= 1.10, matplotlib >= 3.7
  (as pinned in requirements.txt).
- No non-standard hardware is required; the script runs on a standard desktop or
  laptop CPU.
- Typical install time on a normal desktop computer: under one minute
  (`pip install -r requirements.txt`).


## Running

    python3 evidence_closed_loop.py            # quick mode, about 1-2 minutes
    python3 evidence_closed_loop.py --full     # finer sweeps, about 5 minutes
    python3 evidence_closed_loop.py --fast     # coarser, for a smoke test
    python3 evidence_closed_loop.py --no-figures
    python3 evidence_closed_loop.py --json results.json

A successful run ends with

    all quoted values reproduced

`expected_output.txt` in this repository is the output of `--full --no-figures` on a reference
machine; only the elapsed-time line is machine-dependent and has been replaced by a placeholder
there. Any other difference is a discrepancy worth reporting.

## What is checked

| check | status printed |
|---|---|
| T0 positive invariance | proved |
| K* uniform bound | proved (sufficient, not sharp) |
| T2 stability of both corners | proved |
| T4 evidence law | proved on {z>0}; general form in Methods |
| Proposition 1 sharp-selection limit | proved (a) and (b) |
| T3 fragile window and propagation | proved |
| T5 two-outcome convergence | proved under kappa > K* |
| weak-coupling mixed branch | characterised |
| Proposition 3 order preservation | proved |
| Proposition 4 closure | proved (given M1-M3 and O1-O3) |
| Proposition 5 Axiom D | proved (given D1-D4) |
| Proposition 6 finite sharpness | proved (finite gamma) |
| Proposition 7 gate robustness and E4 drift | proved (comparable gates); drift measured |
| Table 1 ablations | illustrative |

The status column repeats the status recorded in Table 2 of the manuscript, so that a reader can see
at a glance which lines are theorems, which hold on a stated regime, and which are illustrative.

Two checks are controls rather than confirmations. The Axiom D block reports `a''(Q)` at two values
of the evidence scale: negative at `Lstar = 2`, where the equivalence of Proposition 5(b) holds, and
positive at `Lstar = 5`, where it fails. The second is the case in which the derivation of concavity
does not go through, and it is included deliberately.

## Outputs

Running with figures writes

    evidence_closed_loop_table1.tsv / .png
    evidence_closed_loop_table2.tsv / .png
    evidence_closed_loop_table3.tsv / .png
    figure1.png
    figure2.png
    supplementary_figure1.png

Figure 1 is the conceptual figure of the paper and Figure 2 the trajectory panel; both are generated
here, so no display item in the manuscript is drawn by hand.

## Scope

The script verifies the implementation against the analytical results. It is not itself a proof.
Where a theorem holds only on a stated regime, the check is run inside that regime and the status
column says so. The premises that are commitments rather than consequences, M1 to M3 and the
bridging conditions O1 to O3 of Proposition 4, and D1 to D4 of Proposition 5, are implemented as
written; the script does not test whether they are true of any real system.

## Licence

MIT. See LICENSE.

## Citation

See CITATION.cff. The archived releases carry a concept DOI that always resolves to the latest
version: https://doi.org/10.5281/zenodo.21669949
