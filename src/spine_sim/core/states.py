"""Canonical physical states and independent diagnostic state dimensions."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any, Mapping

from .errors import ConfigurationError


class PhysicalState(StrEnum):
    SEARCH = "search"
    CONTACT = "contact"
    STICK = "stick"
    SLIP = "slip"
    DETACH = "detach"
    REBOUND = "rebound"
    HARDSTOP = "hardstop"
    FAILED = "failed"


class NumericalState(StrEnum):
    NOT_RUN = "not_run"
    CONVERGED = "converged"
    NONCONVERGED = "nonconverged"
    INVALID_RESIDUAL = "invalid_residual"


class ModelState(StrEnum):
    CLOSED = "closed"
    PARAMETER_UNCLOSED = "parameter_unclosed"
    OUT_OF_SCOPE = "out_of_scope"


class RunState(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETE = "complete"
    CANCELLED = "cancelled"
    EXECUTION_ERROR = "execution_error"


class SpringBranch(StrEnum):
    RIGID = "rigid"
    LOWER_STOP = "lower_stop"
    INTERIOR = "interior"
    HARDSTOP = "hardstop"


class ContinuationAction(StrEnum):
    PERMANENT_REMOVE = "permanent_remove"
    STOP_MODEL_LIMIT = "stop_model_limit"


class EventType(StrEnum):
    """Physical events only; numerical and run events have separate enums."""

    CONTACT = "contact"
    CONTACT_REJECT = "contact_reject"
    STICK_START = "stick_start"
    SLIP_START = "slip_start"
    RESTICK = "restick"
    DETACH = "detach"
    REBOUND_START = "rebound_start"
    REBOUND_COMPLETE = "rebound_complete"
    REENGAGE = "reengage"
    HARDSTOP = "hardstop"
    HARDSTOP_RELEASE = "hardstop_release"
    MATERIAL_FAILURE = "material_failure"


class NumericalEventType(StrEnum):
    STEP_REDUCTION = "step_reduction"
    NEWTON_RETRY = "newton_retry"
    ACTIVE_SET_RECOMPUTE = "active_set_recompute"
    NONCONVERGENCE = "nonconvergence"
    INVALID_RESIDUAL = "invalid_residual"


class ModelEventType(StrEnum):
    PARAMETER_UNCLOSED = "parameter_unclosed"
    MODEL_LIMIT = "model_limit"
    OUT_OF_SCOPE = "out_of_scope"


class RunEventType(StrEnum):
    CANCELLED = "cancelled"
    RESUMED = "resumed"
    EXECUTION_ERROR = "execution_error"


_CONTACT_BEARING_STATES = (
    PhysicalState.CONTACT,
    PhysicalState.STICK,
    PhysicalState.SLIP,
    PhysicalState.HARDSTOP,
)

ALLOWED_PHYSICAL_TRANSITIONS: Mapping[
    EventType, frozenset[tuple[PhysicalState, PhysicalState | None]]
] = {
    EventType.CONTACT: frozenset(
        {(PhysicalState.SEARCH, PhysicalState.CONTACT)}
    ),
    EventType.CONTACT_REJECT: frozenset(
        {(PhysicalState.CONTACT, PhysicalState.SEARCH)}
    ),
    EventType.REENGAGE: frozenset(
        {(PhysicalState.SEARCH, PhysicalState.CONTACT)}
    ),
    EventType.STICK_START: frozenset(
        {(PhysicalState.CONTACT, PhysicalState.STICK)}
    ),
    EventType.SLIP_START: frozenset(
        {
            (PhysicalState.CONTACT, PhysicalState.SLIP),
            (PhysicalState.STICK, PhysicalState.SLIP),
        }
    ),
    EventType.RESTICK: frozenset(
        {(PhysicalState.SLIP, PhysicalState.STICK)}
    ),
    EventType.DETACH: frozenset(
        {(state, PhysicalState.DETACH) for state in _CONTACT_BEARING_STATES}
    ),
    EventType.REBOUND_START: frozenset(
        {(PhysicalState.DETACH, PhysicalState.REBOUND)}
    ),
    EventType.REBOUND_COMPLETE: frozenset(
        {(PhysicalState.REBOUND, PhysicalState.SEARCH)}
    ),
    EventType.HARDSTOP: frozenset(
        {
            (PhysicalState.CONTACT, PhysicalState.HARDSTOP),
            (PhysicalState.STICK, PhysicalState.HARDSTOP),
            (PhysicalState.SLIP, PhysicalState.HARDSTOP),
        }
    ),
    EventType.HARDSTOP_RELEASE: frozenset(
        {(PhysicalState.HARDSTOP, PhysicalState.CONTACT)}
    ),
    EventType.MATERIAL_FAILURE: frozenset(
        {
            *((state, PhysicalState.FAILED) for state in _CONTACT_BEARING_STATES),
            *((state, None) for state in _CONTACT_BEARING_STATES),
        }
    ),
}


def validate_physical_transition(
    event_type: EventType,
    from_state: PhysicalState,
    to_state: PhysicalState | None,
) -> None:
    """Reject any physical transition outside the frozen state graph."""

    if (from_state, to_state) not in ALLOWED_PHYSICAL_TRANSITIONS[event_type]:
        target = "MODEL_LIMIT" if to_state is None else to_state.value
        raise ConfigurationError(
            f"illegal physical transition for {event_type.value}: "
            f"{from_state.value}->{target}"
        )


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
    """One validated physical event."""

    event_type: EventType
    sequence: int
    from_state: PhysicalState
    to_state: PhysicalState | None
    case_id: str | None = None
    spine_id: str | None = None
    load_parameter: float | None = None
    path_position_m: float | None = None
    timestamp_utc: str | None = None
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.sequence < 0:
            raise ConfigurationError("event sequence must be non-negative")
        if not self.case_id and not self.spine_id:
            raise ConfigurationError("event requires a case_id or spine_id")
        validate_physical_transition(
            self.event_type, self.from_state, self.to_state
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "event_type": self.event_type.value,
            "sequence": self.sequence,
            "from_state": self.from_state.value,
            "to_state": self.to_state.value if self.to_state is not None else None,
            "case_id": self.case_id,
            "spine_id": self.spine_id,
            "load_parameter": self.load_parameter,
            "path_position_m": self.path_position_m,
            "timestamp_utc": self.timestamp_utc,
            "details": dict(self.details),
        }


@dataclass(frozen=True)
class NumericalEvent:
    event_type: NumericalEventType
    sequence: int
    case_id: str | None = None
    spine_id: str | None = None
    details: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelEvent:
    event_type: ModelEventType
    sequence: int
    case_id: str | None = None
    spine_id: str | None = None
    details: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RunEvent:
    event_type: RunEventType
    sequence: int
    case_id: str
    details: Mapping[str, Any] = field(default_factory=dict)
