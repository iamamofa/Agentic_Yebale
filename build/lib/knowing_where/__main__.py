"""Offline smoke example. All data and coefficients are synthetic."""

from dataclasses import asdict
import json
from math import prod

from .evidence.features import (
    KAPExtractor, KnowledgeMetadata, MemoryMetadata, ParametricMetadata,
    QueryMetadata, QuestionQuality, WebMetadata,
)
from .evidence.fusion import SourceEvidence, bayesian_fusion, dempster_shafer_fusion
from .evidence.prediction import LinearKAPPredictor, LogisticHead, SourceHeads
from .types import Source


def main() -> None:
    features = KAPExtractor().extract("What is the capital of Mauritius?", QueryMetadata(
        quality=QuestionQuality(1.0, 0.95, 0.0),
        parametric=ParametricMetadata(7, 2.0),
        web=WebMetadata(2, ("Mauritius capital Port Louis", "Port Louis capital Mauritius")),
        knowledge=KnowledgeMetadata(1, 1),
        memory=MemoryMetadata(3, 5, 2, 3),
    ))
    # Demonstration coefficients only: these have never been fitted/calibrated.
    heads = {source: SourceHeads(
        LogisticHead({"popularity": 1.0, "distance_mean": -0.5}, intercept=1.0),
        LogisticHead({"knowledge_coverage": -1.0}, intercept=-1.0),
    ) for source in Source}
    predictor = LinearKAPPredictor(heads, missing_values={})
    estimates = predictor.predict(features)
    evidence = [SourceEvidence(source, "Port Louis" if source != Source.M else "Curepipe", estimates[source]) for source in Source]
    # Explicit toy assumption: corpus absence is independent across sources.
    gamma = prod(item.estimate.ghost_mass for item in evidence)
    bayes = bayesian_fusion(evidence, joint_ghost_mass=gamma)
    ds = dempster_shafer_fusion(evidence)
    print(json.dumps({
        "warning": "Synthetic inputs and untrained coefficients; no calibration claims.",
        "features": features.vector(),
        "provenance": features.provenance,
        "estimates": {source.value: asdict(estimate) for source, estimate in estimates.items()},
        "bayesian": {"posterior": bayes.posterior, "joint_ghost_prior": gamma,
                     "log_evidence": bayes.log_evidence, "log_evidence_ratio": bayes.log_evidence_ratio},
        "dempster_shafer": {"belief": ds.belief, "plausibility": ds.plausibility,
                            "pignistic": ds.pignistic, "conflict": ds.conflict, "status": ds.status},
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
