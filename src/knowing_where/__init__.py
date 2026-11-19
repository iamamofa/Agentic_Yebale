"""Query-side features, evidence fusion, context packets, and model routing."""

from .context import ContextCompiler, ContextDocument, ContextRequest, ContextPacket, BudgetExceeded
from .context.evaluation import EvidenceSpan, evaluate_packet
from .evidence.features import KAPExtractor, KAPFeatures, QueryMetadata
from .evidence.fusion import bayesian_fusion, dempster_shafer_fusion
from .evidence.prediction import LinearKAPPredictor, SourceEstimate
from .evidence.uncertainty import predictive_uncertainty, source_uncertainty
from .routing import ModelProfile, RoutingPolicy, route_context
from .types import OUTSIDE, Source

__all__ = [
    "KAPExtractor", "KAPFeatures", "QueryMetadata", "LinearKAPPredictor",
    "SourceEstimate", "Source", "OUTSIDE", "bayesian_fusion",
    "dempster_shafer_fusion",
    "ContextCompiler", "ContextDocument", "ContextRequest", "ContextPacket",
    "BudgetExceeded", "ModelProfile", "RoutingPolicy", "route_context",
    "EvidenceSpan", "evaluate_packet", "predictive_uncertainty", "source_uncertainty",
]
