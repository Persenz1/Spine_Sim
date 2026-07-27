"""Independent physical, numerical, model and run-state dimensions."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any, Mapping

from .errors import ConfigurationError


class PhysicalState(StrEnum):
    FREE = "free"
    CONTACT = "contact"
    STICK = "stick"
    SLIDE = "slide"
    HARD_STOP = "hard_stop"
    FAILED = "failed"


class NumericalState(StrEnum):
    NOT_RUN = "not_run"
    CONVERGED = "converged"
    NONCONVERGED = "nonconverged"
    INVALID_RESIDUAL = "invalid_residual"


class ModelState(StrEnum):
    COVERED = "covered"
    PARAMETER_UNCLOSED = "parameter_unclosed"
    OUT_OF_SCOPE = "out_of_scope"


class RunState(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETE = "complete"
    CANCELLED = "cancelled"
    EXECUTION_ERROR = "execution_error"


class EventType(StrEnum):
    CONTACT = "contact"
    DETACH = "detach"
    RECONTACT = "recontact"
    SLIP_START = "slip_start"
    HARD_STOP = "hard_stop"
    TERRAIN_BOUNDS = "terrain_bounds"
    NUMERICAL_RETRY = "numerical_retry"
    NUMERICAL_FAILURE = "numerical_failure"
    CANCELLED = "cancelled"
    RESUMED = "resumed"


@dataclass(frozen=True)
class StateBundle:
    physical_state: PhysicalState
    numerical_state: NumericalState
    model_state: ModelState
    run_state: RunState

    def as_dict(self) -> dict[str, str]:
        return {key: value.value for key, value in asdict(self).items()}

    @classmethod
    def from_mapping(cls, value: Mapping[str, str]) -> "StateBundle":
        required = {
            "physical_state",
            "numerical_state",
            "model_state",
            "run_state",
        }
        if set(value) != required:
            raise ConfigurationError(
                f"state bundle requires exactly {sorted(required)}"
            )
        try:
            return cls(
                PhysicalState(value["physical_state"]),
                NumericalState(value["numerical_state"]),
                ModelState(value["model_state"]),
                RunState(value["run_state"]),
            )
        except ValueError as exc:
            raise ConfigurationError(f"invalid state value: {exc}") from exc


@dataclass(frozen=True)
class Event:
    event_type: EventType
    sequence: int
    case_id: str
    path_position_m: float | None = None
    timestamp_utc: str | None = None
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.sequence < 0:
            raise ConfigurationError("event sequence must be non-negative")
        if not self.case_id:
            raise ConfigurationError("event case_id cannot be empty")

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["event_type"] = self.event_type.value
        return value
