# Knowing Where Not to Look

**Context preparation, model routing, and source-resolved evidence fusion for AI agents.**

Knowing Where Not to Look is a Python research prototype for deciding what context
to retain, which model can process it within a budget, and how to combine evidence
from different sources. It makes those decisions inspectable through source-linked
excerpts, explicit conflict records, routing explanations, and retention reports.

The core library supports Python 3.9+ with no third-party runtime dependencies.
The optional Streamlit workbench adds an interactive interface.
It now opens with a beginner walkthrough, a source-first routing simulation,
an algorithm explorer, and a research notebook separating reported measurements
from synthetic examples. See the [app guide](docs/app/README.md) for the tour.
The **Live models** workspace also connects to Hugging Face Inference Providers
for actual hosted model responses. Configure `HF_TOKEN` or enter a session token
in the app; the research examples remain available without credentials.

**Start and share:** double-click `start_app.command` on this Mac, or run
`.venv/bin/python start_app.py`. It opens the app with a public HTTPS link and
shows the link in the sidebar. Keep the launcher running; a new link is created
on each restart. See the [public sharing guide](docs/app/README.md#start-with-a-public-link).
The complete experiment fits prediction heads and calibrates decisions using
separate data partitions. Included examples use synthetic data and model profiles;
the standalone fusion demo retains illustrative, untrained coefficients. These
examples demonstrate behavior rather than validated gains on real benchmarks.

Version 0.3 adds predictive uncertainty, source-disagreement decomposition,
source-removal sensitivity, and an interactive uncertainty lab. Reusable fitted
systems, call budgets, and frozen conformal decisions remain part of the pipeline.

## Capabilities

| Component | Purpose |
| --- | --- |
| Context summaries | Select query-relevant excerpts while preserving instructions, required documents, declared prerequisites, and explicit claim conflicts. |
| Model summaries | Describe each model's declared capabilities, context capacity, output reserve, estimated cost, and latency. |
| Budget-aware routing | Build a context packet for each eligible model and select a model using capability, capacity, cost, latency, and coverage constraints. |
| Query-side features | Extract question-side KAP features and apply configurable reliability and corpus-absence prediction heads. |
| Evidence fusion | Combine source estimates using Bayesian channel fusion or Dempster–Shafer combination, with conflict diagnostics. |
| Uncertainty | Quantify predictive entropy, margins, outside scores, source ambiguity, and source sensitivity; explore held-out uncertainty versus error. |
| Saved systems | Export validated fitted artifacts and run new queries without reference answers or retraining. |
| Experiment pipeline | Run group-isolated train/calibration/test splits, fit KAP heads, generate ten paths per source, compare five methods, and produce Mondrian sets and abstention decisions. |
| Offline evaluation | Measure provenance and evidence retention, predictive accuracy, probability quality, selective risk, and prediction-set coverage. |

## How it works

### Context preparation and routing

1. Supply a question, instructions, documents, model profiles, and routing policy.
2. Check each model against required capabilities and estimated cost and latency limits.
3. Calculate its input budget: `context_capacity - output_reserve`.
4. Preserve mandatory content, then select optional excerpts using lexical
   relevance, new query terms, source diversity, and utility per added budget unit.
5. Measure the complete serialized packet, including metadata, against the budget.
6. Rank eligible models and return the selected model, its packet, and an audit
   of retained and omitted material.

Context summaries are **extractive**: retained passages remain verbatim and carry
source references, character offsets, and content hashes. The compiler suppresses
optional exact duplicates and records disagreements between explicitly supplied
structured claims without choosing a winner.

Routing supports two policies:

- **Quality:** prioritize declared capability suitability, then lexical query-term
  coverage, lower cost, and lower latency.
- **Economy:** prioritize lower cost after all eligibility checks pass, then
  suitability, coverage, and latency.

If no model meets the policy, the result includes rejection reasons and no
selected model. Mandatory material is never silently discarded to make a packet fit.

### Source identity and evidence fusion

Evidence retains its source identity independently of the model selected to
process it.

| Source | Meaning |
| --- | --- |
| **P** | Parametric knowledge associated with model parameters |
| **W** | Web material and user turns |
| **K** | Knowledge designated as verified by the caller |
| **M** | Interaction memory |

The evidence subsystem implements query-side feature extraction and configurable
prediction heads, plus fusion over caller-supplied canonical answer IDs and source
estimates. The experiment runner connects these components through ten-path source
generation, fitted source estimates, fusion, and calibrated prediction sets.

## Streamlit app

Install the optional UI and launch the evidence workbench from the project root:

```bash
python -m pip install -e ".[ui]"
python -m streamlit run streamlit_app.py
```

Use **Overview**, **Context studio**, **Experiments**, **Uncertainty lab**, and **Predict** to compile
context, compare fusion methods, inspect evidence and conflicts, export fitted
systems, and score label-free queries. Sample inputs are included, and custom data
can be uploaded or pasted. The uncertainty lab adds an entropy ring, source-vote
heatmap, interactive source scenarios, and a review-coverage explorer. See the
[app guide](docs/app/README.md) and [uncertainty methodology](docs/uncertainty/README.md).

## Quick start

Run the complete synthetic experiment, including training, conformal calibration,
five-method comparisons, and final decisions:

```bash
PYTHONPATH=src python3 -m knowing_where.cli.experiment demo
```

This creates a dataset, model profile, fitted system, sample queries, and report in
`examples/experiments/demo/`. See the [experiment guide](docs/experiments/README.md)
for your own datasets, checksum-verified downloads, and generator integration.

### Predict with a saved system

```bash
PYTHONPATH=src python3 -m knowing_where.cli.experiment predict \
  examples/experiments/demo/system.json \
  examples/experiments/demo/queries.jsonl \
  --max-calls 120 \
  --output results/predictions.json
```

Prediction inputs contain questions, context, metadata, and recorded paths for
replay, with no gold answers or corpus-presence labels. The fitted system retains
training coefficients and calibrated group thresholds. Its outputs include each
method's candidate probabilities, prediction set, decision, and audit.

### Context preparation

Run the examples from the project root without installing the package:

```bash
PYTHONPATH=src python3 -m knowing_where.cli.context \
  examples/context/request.json
```

To save a routing and context report:

```bash
PYTHONPATH=src python3 -m knowing_where.cli.context \
  examples/context/request.json \
  --output examples/context/report.json
```

The supplied request selects `demo-medium`, produces packets for the eligible
models, and rejects `demo-large` because its estimated cost exceeds the limit.
It also preserves the disagreement between the earlier 3000 ms latency target
and the current 1500 ms target.

To compare evidence retention across input budgets:

```bash
PYTHONPATH=src python3 -m knowing_where.cli.context \
  examples/context/request.json \
  --sweep-budgets 1800 2200 2500 3200 5000 \
  --output examples/context/report.json
```

Run the separate evidence-fusion demonstration:

```bash
PYTHONPATH=src python3 -m knowing_where
```

### Install for development

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

After installation, the commands above can be run without `PYTHONPATH=src`.
The original `python -m knowing_where.context_cli` entry point remains available.

## Working with reports

The context CLI accepts a JSON request containing `question`, `documents`, and
`models`, with optional `instructions`, `required_document_ids`, `policy`, and
`evaluation_spans`. Use the [example request](examples/context/request.json) as a
starting point.

The resulting report includes:

- Model summaries, the selected model ID, the ranking rule, and rejection reasons.
- A separate context packet for each eligible model.
- Budget usage, selected and omitted document IDs, and excerpt omission reasons.
- Provenance checks, lexical coverage, and evidence-retention measurements when
  evaluation spans are supplied.
- Budget-sweep results when requested.

For downstream model integration, send the chosen candidate's **`payload`** string.
That is the exact serialized content measured against the budget. The report's
`context` field is an inspection view; the complete report is not the model input.

## Project structure

```text
.
├── src/knowing_where/
│   ├── context/         # Context compiler and retention evaluation
│   ├── routing/         # Model profiles and selection policy
│   ├── evidence/        # KAP features, prediction, and fusion
│   ├── experiments/     # Datasets, training, generation, calibration, metrics
│   ├── ui/              # Streamlit views and validated app actions
│   ├── cli/             # Context and experiment command-line tools
│   ├── types.py         # Shared source identifiers and validation
│   ├── __main__.py      # Evidence-fusion demonstration
│   └── context_cli.py   # Original CLI compatibility entry point
├── docs/
│   ├── context/         # Compression and budget contract
│   ├── routing/         # Model cards and routing behavior
│   ├── evidence/        # Feature extraction and fusion design
│   └── experiments/     # Pipeline methodology and dataset contract
├── examples/
│   ├── context/         # Context request, report, and retention results
│   └── experiments/     # Synthetic dataset, model profile, report, and results
├── tests/
│   ├── context/         # Context, routing, and retention tests
│   ├── evidence/        # Feature, prediction, and fusion tests
│   └── experiments/     # Leakage, split, training, calibration, and pipeline tests
├── streamlit_app.py     # Streamlit entry point
├── .streamlit/          # Theme and upload settings
├── requirements.txt     # Optional app installation
└── pyproject.toml       # Package metadata and build configuration
```

## Validation

Run the test suite from the project root:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

Tests cover exact packet budgets, mandatory content, dependencies, conflict
retention, provenance, routing constraints, feature validation, and mathematical
reference cases for evidence fusion. Experiment tests also verify train-only
fitting, group isolation, label-free generation requests, finite-sample conformal
thresholds, conservative sparse-group handling, artifact round trips, generation
budgets, retry accounting, numerical stability, and reproducibility. See the
[context results](examples/context/results.md) for compression measurements and
the [experiment results](examples/experiments/results.md) for synthetic pipeline
comparisons.

## Current scope

- **Budget units:** the default compiler and CLI count UTF-8 bytes. Actual model
  integration requires a matching tokenizer and reserves for output and provider
  wrappers outside the packet.
- **Selection quality:** context selection uses deterministic, untrained
  heuristics. Lexical coverage and retained-text fractions do not establish
  evidence sufficiency or answer correctness.
- **Conflict detection:** the compiler compares explicit structured claim values.
  General natural-language contradiction detection is not implemented.
- **Model estimates:** capability, cost, and latency values come from the caller.
  Routing does not measure model accuracy or enforce live spending.
- **Capacity management:** context budgets are supported. Live server congestion,
  request queues, and provider load balancing are outside the current scope.
- **Integration:** bundled experiments replay recorded paths. The downloader can
  fetch prepared datasets over HTTPS, and the Python API accepts custom generators.
  No live provider adapter is bundled; real benchmark data and source-presence
  annotations must be supplied independently.
- **Calibration:** source heads are now fitted on the training partition, and
  Mondrian thresholds use only calibration examples. Coverage depends on
  within-group exchangeability and a frozen pipeline. Missing reference answers
  map to `Ø`; mapped coverage and actual answer coverage are reported separately.

## Documentation

- [Predictive uncertainty and methodology](docs/uncertainty/README.md)
- [Streamlit app and launch guide](docs/app/README.md)
- [Context summaries and capacity management](docs/context/design.md)
- [Model summaries and routing](docs/routing/README.md)
- [End-to-end experiments and milestone status](docs/experiments/README.md)
- [Query-side features and evidence fusion](docs/evidence/design.md)
- [Context example results](examples/context/results.md)
- [Synthetic experiment results](examples/experiments/results.md)
