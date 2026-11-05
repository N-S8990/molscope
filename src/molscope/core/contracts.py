"""Explicit contracts for feature applicability and provenance.

Feature values must never use zero to mean a feature was unavailable.  This
module models that distinction independently of any particular analyzer.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any
import json
import numpy as np


class ApplicabilityStatus(str, Enum):
    """Whether an analyzer can produce a scientifically meaningful result."""

    APPLICABLE = "applicable"
    NOT_APPLICABLE = "not_applicable"
    MISSING_INPUT = "missing_input"
    UNSUPPORTED = "unsupported"
    INSUFFICIENT_DATA = "insufficient_data"
    FAILED = "failed"


@dataclass(frozen=True)
class FeatureProvenance:
    """Information needed to reproduce and interpret a feature calculation."""

    analyzer_id: str
    analyzer_version: str
    parameters: dict[str, Any] = field(default_factory=dict)
    source_topology: str | None = None
    source_trajectory: str | None = None
    elapsed_seconds: float | None = None

    def to_dict(self) -> dict[str, Any]:
        def normalize(value):
            if isinstance(value, np.generic):
                return value.item()
            if isinstance(value, np.ndarray):
                return value.tolist()
            raise TypeError(f"Unsupported provenance value: {type(value).__name__}")
        return json.loads(json.dumps(asdict(self), default=normalize))


@dataclass(frozen=True)
class FeatureAssessment:
    """A feature's applicability decision before values are computed."""

    feature_id: str
    family: str
    status: ApplicabilityStatus
    reason: str
    granularity: str
    units: str | None
    missing_capabilities: tuple[str, ...] = ()
    missing_inputs: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["status"] = self.status.value
        return payload
