from __future__ import annotations

import abc
import enum
import logging

from miles.utils.test_utils.fault_injector import FailureMode
from miles.utils.workers.worker_provider.base import CellInfo

logger = logging.getLogger(__name__)

TERMINATE_INCARNATION_TIMEOUT_SECONDS = 120.0
CELL_TERMINATION_NOT_CONFIRMED = "not_confirmed"
INCARNATION_CHECK_TIMEOUT_SECONDS = 30.0


class FaultInjectionOutcome(enum.Enum):
    INJECTED = "injected"
    ACCEPTED = "accepted"
    STALE = "stale"
    UNKNOWN = "unknown"


class CellTerminationOutcome(enum.Enum):
    TERMINATED = "terminated"
    ALREADY_GONE = "already_gone"
    STALE = "stale"


class CellTerminationNotConfirmedError(Exception):
    pass


class BaseCellOperations(abc.ABC):
    @abc.abstractmethod
    async def cell_infos(self, *, pool_ids: list[str]) -> dict[str, CellInfo]: ...

    @abc.abstractmethod
    async def suspend(self, *, cell_id: str) -> None: ...

    @abc.abstractmethod
    async def resume(self, *, cell_id: str) -> None: ...

    @abc.abstractmethod
    async def terminate_incarnation(
        self,
        *,
        cell_id: str,
        expected_workers_hash: str,
        timeout: float = TERMINATE_INCARNATION_TIMEOUT_SECONDS,
    ) -> CellTerminationOutcome: ...

    @abc.abstractmethod
    async def inject_fault(
        self,
        *,
        cell_id: str,
        mode: FailureMode,
        sub_index: int | None = None,
        expected_workers_hash: str | None = None,
    ) -> FaultInjectionOutcome: ...

    @abc.abstractmethod
    async def incarnation_is_current(self, *, cell_id: str, expected_workers_hash: str) -> bool: ...
