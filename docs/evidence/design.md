# Query-side features and evidence fusion

## Stage 1: extraction and prediction

`KAPExtractor.extract(question, QueryMetadata(...))` accepts only question-side
metadata. Each missing metadata object can be supplied by a callable probe;
provided metadata takes precedence, avoiding unnecessary calls. There are no
generation or full-retrieval clients in the extractor. Probes return:

| Feature | Input or adapter result | Numeric representation |
| --- | --- | --- |
| omega | `QuestionQuality`: scope fit, well-posedness, guardrail exposure, each in [0,1] | Three separate values; higher exposure means more interference |
| rho_P | `ParametricMetadata`: popularity decile and/or entity-name perplexity | Decile mapped to [0,1], inverse perplexity |
| rho_W | `WebMetadata`: hit count and top snippets from **one** search query | Hit count and snippet-agreement score |
| rho_K | `KnowledgeMetadata`: schema/index items matched and checked | Coverage fraction |
| rho_M | `MemoryMetadata`: similarity hits/checked and historical successes/evaluated | Hit fraction and historical success fraction |
| d | `HopEstimate`: p(d given q), optional decomposition | Full distribution retained; expected hops and P(d=0) in vector |

Example of attaching the one-query web adapter:

```python
from knowing_where.evidence.features import KAPExtractor, WebMetadata

def web_probe(question: str) -> WebMetadata:
    # One search request; return count/snippets, never fetch full documents.
    response = search_client.search(question, limit=5)  # your client
    return WebMetadata(response.total_hits, tuple(hit.snippet for hit in response.hits))

features = KAPExtractor(web_probe=web_probe).extract("What is the capital of Mauritius?")
```

The extractor invokes the web adapter once. The adapter must enforce one network
request and manage its own timeouts, cache, and credentials. Probe exceptions
propagate; failures are not silently turned into zero hits. User turns belong to
W when Stage 2 is added; do not promote repeated web hits or memory into K.

Unavailable quality estimates remain `None`; scope and guardrail judgments are
not invented by a keyword filter. The default agreement function is pairwise
token Jaccard, explicitly a **lexical overlap proxy**, which cannot detect factual
contradictions. Fewer than two usable snippets yield `None`. Replace
`agreement_scorer` with a validated semantic measure for experiments.

The default decomposer splits explicit clauses and returns uncertainty around
the estimated links. A direct lookup has d=0. This untrained heuristic is not a
validated reasoning-hop estimator; inject a query-only `decomposer` or provide
`HopEstimate` metadata. Its method name is recorded in provenance.

`LinearKAPPredictor` requires separate `SourceHeads` for all four sources, each
containing two `LogisticHead` objects. It maps the feature vector to:

- r_s = P(correct given answer in corpus, engaged, features).
- g_s = P(answer absent from corpus given engaged, features).

There are no fitted default weights. Missing values require explicit training-set
imputations, and each feature gets a `<name>__missing` indicator. Store coefficient
and imputation mappings with your experiment configuration. Fit reliability only
on engaged, corpus-present training examples and ghost heads on corpus-presence
labels. Hurdle engagement probability pi is separate from these heads; a learned
hurdle gate is not implemented in this initial scope.

Do not pass generated answers, correctness labels, gold hop counts, or test-run
outcomes as extraction metadata. Historical memory outcomes must predate the
query and exclude held-out traces. Keep answer labels in the future evaluation
harness, outside this API. The typed allowlist reduces accidental leakage but
cannot establish the provenance of caller-supplied values.

## Stage 4: fusion

Use canonical answer IDs: the caller must resolve semantic equivalence before
fusion. `SourceEvidence` supplies a source, candidate answer, `SourceEstimate`,
and optionally a normalized predictive distribution. Without that distribution,
Bayesian fusion uses a point mass on the proposed answer. Duplicate sources are
rejected to prevent accidental evidence reuse. Any nonempty subset of P/W/K/M
is supported.

### Bayesian channel rule

For every candidate and the outside option `Ø`:

```text
L_s(y) = (1-g_s) [r_s p_s(y) + (1-r_s) u(y)] + g_s u(y)
posterior(y) proportional to p0(y) product_s L_s(y)
p0(Ø) = gamma
```

The default noise `u` is uniform across the entire frame. The remaining prior
is uniform over candidate answers or specified with `answer_prior`. Custom
`noise` must cover the entire frame; distributions must sum to one.

`joint_ghost_mass=gamma` is required. It must satisfy the Frechet bounds
`max(0, sum(g_s)-(n-1)) <= gamma <= min(g_s)` for the consulted sources.
The product of ghost masses is appropriate only with an explicit independence
assumption, as in the toy example. For fusion over fewer than four sources,
gamma describes absence from those consulted corpora.

Gamma sets the **prior**, so posterior probability on `Ø` can differ. This initial
model equates the probability of unwritten truth with the outside-option prior;
it does not separately model a written correct answer missing from the candidate
set. Candidate coverage must be assessed before interpreting its posterior.

The result includes posterior, prior, channel distributions, log evidence, and
`log_evidence_ratio = log Z_joint - sum_s log Z_s`. Negative ratios flag tension
relative to the chosen model/prior; they are not calibrated conflict probabilities.
Fusion uses log-space normalization; zero-probability evidence raises `ValueError`.

### Dempster–Shafer rule

Each source places r_s on its answer singleton and 1-r_s on the whole frame
Theta. Theta includes `Ø`, preserving ignorance even with a single candidate.
This follows the requested comparator exactly: **g_s does not discount r_s**
in this rule. It therefore does not provide a ghost-aware posterior.

The result reports normalized focal masses, singleton belief and plausibility,
and a pignistic probability transform (uniform allocation within focal sets).
Belief is not a Bayesian posterior. It also returns pairwise kappa and aggregate
conjunctive conflict **before normalization**. With total conflict, `status` is
`total_conflict` and all probability/belief summaries are empty. A downstream
decision rule must handle that status explicitly.

Both fusion comparators assume independent source evidence conditional on the
answer. Correlated web pages, memory copied from a source, or repeated evidence
can produce unjustified certainty; dependence modeling is future work. Joint
corpus-absence dependence (gamma) and channel-evidence dependence are distinct.

## Integrated experiment pipeline

The earlier milestone list is now implemented as a connected offline experiment
pipeline. It includes validated dataset acquisition, seeded group-isolated
40/20/40 splits, train-only KAP fitting, ten-path generation adapters, empirical
source confidence, fusion, baseline comparisons, held-out metrics, and Mondrian
prediction sets with explicit abstention decisions.

See the [experiment guide](../experiments/README.md) for commands, the dataset
contract, methodology, and a milestone-by-milestone status table. Run:

```bash
PYTHONPATH=src python3 -m knowing_where.cli.experiment demo
```

The included data and model profile are synthetic. A real benchmark still requires
a dataset-specific conversion, independently annotated corpus-presence labels,
and recorded or live-generated answer paths. A generator interface is available;
no live provider client is bundled. No real benchmark performance or deployment
coverage is claimed.

The original reference was described as *Knowing Where Not to Look: query- and
source-resolved confidence for agents* (November 11, 2026). Its full text is not
included in this repository; the new experiment guide records the choices made
for the previously unspecified stages.
