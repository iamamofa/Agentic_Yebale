"""Train-only imputation, scaling, and regularized logistic KAP fitting."""

from dataclasses import asdict
from math import exp, fsum, isfinite, log, sqrt

from ..evidence.prediction import LinearKAPPredictor, LogisticHead, SourceHeads
from ..types import Source


def fit_head(vectors, labels, *, epochs=300, learning_rate=0.15, l2=0.01):
    """Full-batch logistic descent; standardize using only these training rows.

    Return raw-feature coefficients compatible with the existing predictor.
    A Laplace-smoothed intercept handles single-class training subsets explicitly.
    """
    if not vectors or len(vectors) != len(labels):
        raise ValueError('head fitting requires aligned, nonempty training rows')
    if type(epochs) is not int or epochs < 1 or not isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError('invalid optimizer configuration')
    if not isfinite(l2) or l2 < 0 or any(type(y) is not bool for y in labels):
        raise ValueError('invalid regularization or binary labels')
    names = sorted(vectors[0])
    if any(set(row) != set(names) or any(not isfinite(v) for v in row.values()) for row in vectors):
        raise ValueError('training feature schema must be consistent and finite')
    n = len(vectors)
    # Scale before centering to avoid overflow for finite, very large features.
    peaks = {k: max(abs(row[k]) for row in vectors) or 1.0 for k in names}
    centers = {k: fsum(row[k] / peaks[k] / n for row in vectors) for k in names}
    scales = {k: sqrt(fsum((row[k] / peaks[k] - centers[k]) ** 2 / n for row in vectors)) or 1.0
              for k in names}
    matrix = [[(row[k] / peaks[k] - centers[k]) / scales[k] for k in names] for row in vectors]
    prevalence = (sum(labels) + 1) / (n + 2)
    intercept = log(prevalence / (1 - prevalence))
    weights = [0.0] * len(names)
    epochs_run = 0
    if len(set(labels)) > 1:
        for _ in range(epochs):
            epochs_run += 1
            errors = [LogisticHead({}, intercept + sum(w * x for w, x in zip(weights, row))).predict({}) - y
                      for row, y in zip(matrix, labels)]
            intercept -= learning_rate * sum(errors) / n
            weights = [w - learning_rate * (sum(e * row[j] for e, row in zip(errors, matrix)) / n + l2 * w)
                       for j, w in enumerate(weights)]
    raw = {k: w / scales[k] / peaks[k] for k, w in zip(names, weights)}
    head = LogisticHead(raw, intercept - fsum(w * centers[k] / scales[k] for k, w in zip(names, weights)))
    loss = 0.0
    for row, y in zip(matrix, labels):
        z = intercept + fsum(w * x for w, x in zip(weights, row))
        loss += (max(z, 0) - z * y + log(1 + exp(-abs(z)))) / n
    return head, {'rows': n, 'positives': sum(labels), 'single_class': len(set(labels)) == 1,
                  'epochs_run': epochs_run, 'training_log_loss': loss}


def train_predictor(rows, features, confidence):
    """Reliability uses corpus-present training rows; absence uses all engaged rows."""
    if not rows:
        raise ValueError('predictor fitting requires training examples')
    raw = {row.id: features[row.id].vector() for row in rows}
    names = sorted(next(iter(raw.values())))
    imputations = {}
    for name in names:
        observed = [v[name] for v in raw.values() if v[name] is not None]
        imputations[name] = fsum(value / len(observed) for value in observed) if observed else 0.0
    vectors = {id_: {**{k: imputations[k] if v is None else v for k, v in values.items()},
                      **{k + '__missing': float(v is None) for k, v in values.items()}}
               for id_, values in raw.items()}
    heads, audit = {}, {}
    for source in Source:
        present = [row for row in rows if row.corpus_present[source]]
        if not present:
            raise ValueError(f'no corpus-present training examples for source {source.value}')
        reliability, reliability_audit = fit_head(
            [vectors[row.id] for row in present],
            [confidence[row.id][source].answer == row.gold for row in present])
        ghost, ghost_audit = fit_head([vectors[row.id] for row in rows],
                                     [not row.corpus_present[source] for row in rows])
        heads[source] = SourceHeads(reliability, ghost)
        audit[source.value] = {'reliability': reliability_audit, 'ghost_mass': ghost_audit}
    predictor = LinearKAPPredictor(heads, imputations)
    return predictor, {'heads': {s.value: asdict(h) for s, h in heads.items()},
                       'missing_values': imputations, 'fit_audit': audit,
                       'training_ids': [row.id for row in rows],
                       'optimizer': {'epochs': 300, 'learning_rate': 0.15, 'l2': 0.01}}
