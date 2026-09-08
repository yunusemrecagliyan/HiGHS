# Synthetic block-structured MIPs

Same sizes as the YemYap production models, with matching block
structure (per-formula blocks + 1-3 coupling rows) and planted feasible
solutions. Used for fork-vs-stock-vs-SCIP benchmarking (see bench logs).

| file | rows | cols | nz | ints | mirrors |
|---|---|---|---|---|---|
| synth_adana.mps | 2312 | 17400 | 34559 | 2700 | Adana (2307x17422) |
| synth_salihli.mps | 4548 | 35550 | 64436 | 5085 | Adana-Salihli (4567x35540) |
| synth_onlyadana.mps | 4546 | 35550 | 64415 | 5085 | onlyadana500 (4565x35538) |
| synth_problem.mps | 4547 | 35550 | 64078 | 5085 | problem (4569x35539) |

Regenerate: `python3 generate.py` (needs numpy).
