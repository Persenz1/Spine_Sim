"""仿真的标准状态维度和物理事件图。

物理状态、数值收敛状态、模型闭合状态和任务运行状态彼此独立；例如“数值收敛”
不能替代“参数已闭合”，物理不可行也不应被包装成执行异常。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Mapping

from .errors import ConfigurationError


class PhysicalState(StrEnum):
    """单刺在搜索、承载、脱离和失效过程中的离散物理状态。"""

    SEARCH = "search"
    CONTACT = "contact"
    STICK = "stick"
    SLIP = "slip"
    DETACH = "detach"
    REBOUND = "rebound"
    HARDSTOP = "hardstop"
    FAILED = "failed"


class NumericalState(StrEnum):
    """非线性/平衡求解器的数值结论。"""

    NOT_RUN = "not_run"
    CONVERGED = "converged"
    NONCONVERGED = "nonconverged"
    INVALID_RESIDUAL = "invalid_residual"


class ModelState(StrEnum):
    """当前结论是否被已有参数和模型适用范围支撑。"""

    CLOSED = "closed"
    PARAMETER_UNCLOSED = "parameter_unclosed"
    OUT_OF_SCOPE = "out_of_scope"


class RunState(StrEnum):
    """campaign case 的调度与执行状态。"""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETE = "complete"
    CANCELLED = "cancelled"
    EXECUTION_ERROR = "execution_error"


class SpringBranch(StrEnum):
    """单边轴向弹簧的互补分支。"""

    RIGID = "rigid"
    LOWER_STOP = "lower_stop"
    INTERIOR = "interior"
    HARDSTOP = "hardstop"


class ContinuationAction(StrEnum):
    """发生材料失效后，阵列层应采取的延续策略。"""

    PERMANENT_REMOVE = "permanent_remove"
    STOP_MODEL_LIMIT = "stop_model_limit"


class EventType(StrEnum):
    """求解器可发出的物理事件类型。"""

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


_CONTACT_BEARING_STATES = (
    PhysicalState.CONTACT,
    PhysicalState.STICK,
    PhysicalState.SLIP,
    PhysicalState.HARDSTOP,
)

ALLOWED_PHYSICAL_TRANSITIONS: Mapping[
    EventType, frozenset[tuple[PhysicalState, PhysicalState | None]]
] = {
    # 这是冻结的状态图；CONTACT 是 trial 内的瞬时状态，必须继续进入承载或返回搜索。
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
    """拒绝冻结状态图之外的物理跃迁。"""

    if (from_state, to_state) not in ALLOWED_PHYSICAL_TRANSITIONS[event_type]:
        target = "MODEL_LIMIT" if to_state is None else to_state.value
        raise ConfigurationError(
            f"illegal physical transition for {event_type.value}: "
            f"{from_state.value}->{target}"
        )


@dataclass(frozen=True)
class Event:
    """一条已经过状态图校验、可持久化的物理事件。"""

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
        """检查序号、归属对象和前后状态是否合法。"""

        if self.sequence < 0:
            raise ConfigurationError("event sequence must be non-negative")
        if not self.case_id and not self.spine_id:
            raise ConfigurationError("event requires a case_id or spine_id")
        validate_physical_transition(
            self.event_type, self.from_state, self.to_state
        )

    def as_dict(self) -> dict[str, Any]:
        """转换为 JSON 友好的事件记录。"""

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
