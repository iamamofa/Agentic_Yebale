# Model summaries and routing

Implementation: [`router.py`](../../src/knowing_where/routing/router.py).

Each `ModelProfile` declares capabilities, context capacity, output reserve,
estimated request cost and latency, limitations, and the origin of those values.
Capability scores express declared suitability, not measured answer accuracy.

`route_context` rejects profiles that fail required capabilities, cost or latency
limits, mandatory context capacity, suitability floors, or lexical coverage
floors. It compiles a separate context packet for each eligible model.

- `quality` ranks suitability, query-term coverage, cost, latency, then model ID.
- `economy` ranks cost first, then suitability, coverage, latency, and model ID.

Unknown cost or latency fails a corresponding hard limit. If no model qualifies,
the result contains no selected model and explains each rejection. These checks
use caller-supplied estimates; they do not observe live provider congestion.

See the [full budget and routing contract](../context/design.md#budgets-and-routing)
and the [runnable request](../../examples/context/request.json).
