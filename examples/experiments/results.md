# Synthetic experiment results

This fixture demonstrates the complete pipeline on recorded, synthetic answer paths.
It is not a real-model benchmark or a deployment coverage claim.

Configuration: seed 42, alpha 0.1, ten paths per source, 40 training examples,
20 calibration examples, and 40 test examples. Every method uses the same source
outputs and its own group-specific calibration thresholds.

| Method | Point accuracy | Answer rate | Selective accuracy | Mapped set coverage | Actual answer set coverage | Mean set size |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| bayesian | 45.0% | 2.5% | 100.0% | 100.0% | 97.5% | 2.60 |
| dempster_shafer | 67.5% | 15.0% | 100.0% | 100.0% | 97.5% | 2.60 |
| equal_sources | 35.0% | 0.0% | — | 100.0% | 97.5% | 2.60 |
| majority | 70.0% | 0.0% | — | 100.0% | 97.5% | 2.98 |
| parametric_only | 60.0% | 0.0% | — | 100.0% | 97.5% | 2.98 |

Most methods abstain on every test example under this conservative configuration.
High set coverage here comes with broad sets and low answer rates; it does not
establish useful selective performance or superiority of any fusion method.
Candidate recall, group results, probability metrics, calibration sample counts,
and decision reasons are available in [the report](demo/report.json).

## Reproduce

```bash
PYTHONPATH=src python3 -m knowing_where.cli.experiment demo
```

The dataset is generated with a fixed seed and the report excludes wall-clock
measurements, so identical inputs and configuration reproduce the report.

See the [experiment guide](../../docs/experiments/README.md) for the methodology,
dataset format, and custom generator interface.

## Version 0.2 verification

The demo now exports [a fitted system](demo/system.json) and
[label-free sample queries](demo/queries.jsonl). Prediction uses the same shared
routing and scoring functions as evaluation; round-trip tests reproduce all five
methods' held-out probabilities, sets, and decisions exactly.

This experiment plans and issues 4000 replay calls (100 examples × 4 sources ×
10 paths), with zero retries. The report records call telemetry, routing success,
conditional coverage among routed examples, decision-reason counts, set-size
histograms, and groups with full-frame calibration thresholds.

Training now handles constant decimal-valued columns and large finite features
more reliably. The table above was regenerated after that correction; no
calibration or test outcomes were used to tune the optimizer or selection policy.

## Version 0.3 uncertainty diagnostics

The report also exports per-question predictive uncertainty, source ambiguity and
disagreement, Bayesian source-removal sensitivity, and held-out review curves.
These descriptive additions preserve the comparison table and do not change model
fitting, calibration, generator-call counts, or final decisions. See the
[uncertainty guide](../../docs/uncertainty/README.md).
