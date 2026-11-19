# Context compression demo results

These are offline results on a synthetic request with four manually marked evidence spans. Model capabilities, cost, and latency are synthetic. Evidence recall measures retention, not answer accuracy.

## Model packets

| Model | Used / available bytes | Evidence spans retained | Query-term coverage |
| --- | ---: | ---: | ---: |
| demo-medium | 3203 / 5000 | 4 / 4 | 44.4% |
| demo-small | 2488 / 2500 | 3 / 4 | 44.4% |

Selected demo-medium using quality ranking after capability, budget, cost, latency, and coverage gates.

The large profile exceeds the cost limit. Both compiled packets preserve required instructions, the required architecture document, and the structured latency conflict. The small packet omits at least one annotated evidence span; the conflict ledger remains available separately.

## Input budget sweep

| Input budget (bytes) | Used bytes | Annotated evidence recall |
| ---: | ---: | ---: |
| 1800 | 1727 | 25% |
| 2200 | 2195 | 75% |
| 2500 | 2490 | 75% |
| 3200 | 3110 | 100% |
| 5000 | 3204 | 100% |

The sweep uses the fixed model label `budget-sweep`, so its serialized sizes differ slightly from the named model packets. These measurements do not establish performance on unseen tasks.

## Reproduce

```bash
PYTHONPATH=src python3 -m knowing_where.cli.context examples/context/request.json --sweep-budgets 1800 2200 2500 3200 5000 --output examples/context/report.json
PYTHONPATH=src python3 -m unittest discover -s tests -v
```
