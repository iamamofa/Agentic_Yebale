# Quantifying predictive uncertainty

The system reports uncertainty diagnostics for every scored question and method.
The same calculations are available in experiment reports, saved-system
predictions, and the Streamlit interface. They use predicted distributions and
source paths; reference answers are used only for the held-out review curve.

These are descriptive measures of the model's output distribution and source
behavior. They do not estimate a probability of error or uncertainty over model
weights. A concentrated, unanimous prediction can still be wrong.

## Predictive uncertainty

For a probability distribution `p` over the declared candidate frame:

| Diagnostic | Definition | Interpretation |
| --- | --- | --- |
| Entropy in bits | `H(p) = -sum(p(y) * log2(p(y)))` | How dispersed the probability scores are. Zero-score terms contribute zero. |
| Normalized entropy | `H(p) / log2(K)` for `K > 1`; otherwise zero | A 0–1 dispersion score relative to the complete declared frame. |
| Effective candidates | `2 ** H(p)` | The number of equally weighted outcomes with equivalent entropy. |
| Leading score | `max(p)` | Highest probability score, which may belong to `Ø`. |
| Top-two gap | Largest score minus the second-largest | How decisively one candidate leads; the UI displays percentage points. |
| Variation ratio | `1 - max(p)` | Fraction of score mass away from the leading candidate. |
| Outside score | `p(Ø)`, or zero when absent | Score assigned to truth outside the supplied candidates under the selected method. |

`K` includes every entry in the declared frame, including zero-score entries and
`Ø`. Adding zero-score candidates can change normalized entropy without changing
raw entropy. Compare raw bits and frame size as well as the percentage; avoid
interpreting normalized entropy as a correctness probability.

If routing failed, Bayesian evidence is impossible, or DS combination has total
conflict, the affected uncertainty is **unavailable**, with a reason and null
values. An outside-only abstention fallback is not displayed as a confident
posterior.

## Why sources are uncertain

Let `p_s` be the empirical answer distribution for source `s` and let `m` be their
equal-weight mixture. The source diagnostic uses the exact identity:

```text
H(m) = mean_s H(p_s) + [H(m) - mean_s H(p_s)]
       within-source     between-source Jensen–Shannon disagreement
```

The gap is nonnegative up to floating-point precision. Its normalized form divides
by `log2(min(number of sources, number of candidate labels))` when that bound is
positive. With one source or one candidate, normalized disagreement is zero.
Each source also reports its distribution, entropy, leading share, variation
ratio, and recorded path count. Controlled examples have no observed sample count.

This decomposition describes the **equal-source mixture**. It is not a
decomposition of the Bayesian posterior or DS pignistic distribution. Sources
P/W/K/M are not posterior draws of model parameters, so the UI labels the terms
within-source ambiguity and between-source disagreement, rather than identifying
them with true aleatoric and epistemic uncertainty.

The distinction matters in the controlled examples:

| Source behavior | Mixture entropy | Within-source ambiguity | Between-source disagreement |
| --- | ---: | ---: | ---: |
| All four always choose A | 0 bits | 0 bits | 0 bits |
| Every source splits equally between A and B | 1 bit | 1 bit | 0 bits |
| Two always choose A; two always choose B | 1 bit | 0 bits | 1 bit |

The last two cases have the same overall entropy but very different source
behavior. The lab's four sliders let you explore intermediate cases without
additional generation calls or changes to fitted systems.

## Source-removal sensitivity

For Bayesian fusion, the system removes each source in turn, keeps the original
candidate frame, and recomputes the product-of-ghost-masses prior for the remaining
sources. It reports:

- Total variation: half the sum of absolute differences from the original posterior.
- Whether the leading candidate changes, and the replacement candidate.
- Change in the original leading candidate's probability and the outside score.

This is a sensitivity analysis under the existing independence assumptions. It is
not a causal source attribution, reliability estimate, or guarantee that the most
influential source is correct. A failed recombination is reported as unavailable.
Sensitivity requires additional fusion calculations, but makes no generator calls.

## Uncertainty lab

Launch the app and open **Uncertainty lab**:

```bash
.venv/bin/python -m streamlit run streamlit_app.py
```

**Run results** lets you select the bundled or latest completed experiment, method,
question group, and individual question. Its fingerprint combines an entropy ring,
leading score, margin, outside score, candidate chart, source-vote heatmap,
ambiguity/disagreement decomposition, and Bayesian source-removal chart.

**Controlled examples** demonstrates agreement, within-source ambiguity, and
confident source disagreement. Four source controls update the plots immediately.
This is an equal-mixture illustration, not a trained prediction or conformal set.

**Predict** and the experiment evidence inspector also include the fingerprint.
Saved prediction decisions and conformal thresholds are unchanged by these
additional descriptive diagnostics.

## Held-out uncertainty versus error

The review curve orders valid, non-outside point predictions by normalized entropy.
At each distinct threshold, it reports the fraction of all selected test questions
retained and the canonical-answer error among those point predictions. Equal
entropy scores enter together to avoid exploiting label order within ties.
Unrouted cases, invalid distributions, and outside-leading predictions are excluded
from the review subset but remain in its coverage denominator.

The slider is exploratory: it does not change conformal answers, refit a model, or
create a calibrated deployment threshold. Because the curve uses test references,
choosing a policy from it would require independent validation afterward. Low
entropy is not assumed to imply lower observed error.

## API and exported fields

```python
from knowing_where.evidence.uncertainty import predictive_uncertainty, source_uncertainty

summary = predictive_uncertainty({"answer-a": 0.7, "answer-b": 0.2, "Ø": 0.1})
channels = source_uncertainty({"P": {"a": 1.0}, "K": {"b": 1.0}})
```

Each experiment test record and fitted-system method result includes `uncertainty`.
Audits include `source_uncertainty` and `source_sensitivity`; method comparisons
include `risk_coverage`. Aggregate metrics include available diagnostic counts,
mean normalized entropy, and mean top-two margin. These are additive report fields;
the existing artifact and calibration formats remain valid.

## References

Information-theoretic ensemble uncertainty is discussed in
[Fathullah and Gales (2022)](https://proceedings.mlr.press/v180/fathullah22a.html).
Limitations of interpreting entropy terms as aleatoric and epistemic components
are examined by [Wimmer et al.](https://arxiv.org/abs/2209.03302). This repository
uses the descriptive diagnostics above; it does not implement those papers'
training procedures or claim their empirical results.
