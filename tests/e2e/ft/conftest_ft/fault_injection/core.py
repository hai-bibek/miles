# NOTE: You MUST read tests/e2e/ft/README.md as source-of-truth and documentations

import logging
import random
import threading
import time
from collections.abc import Callable
from datetime import datetime, timezone

import requests
from tests.e2e.ft.conftest_ft.fault_injection.fault_forms import BaseFaultForm, CellFaultForms
from tests.e2e.ft.conftest_ft.fault_injection.state import Event, EventLog, cell_type_of
from tests.e2e.ft.conftest_ft.fault_injection.views import (
    PendingInjection,
    compute_pending_injections,
    compute_successful_form_names,
)

logger = logging.getLogger(__name__)

POLL_INTERVAL_SECONDS: float = 2.0
UNKNOWN_INJECTION_RESOLUTION_TIMEOUT_SECONDS: float = 900.0
LIST_TIMEOUT_SECONDS: float = 5.0


class UnresolvedInjectionError(RuntimeError):
    pass


def _compute_next_injection_time(rng: random.Random, mean_interval_seconds: float) -> float:
    return time.monotonic() + rng.expovariate(1.0 / mean_interval_seconds)


def run_fault_injection_loop(
    *,
    base_url: str,
    seed: int,
    mean_interval_seconds_of_cell_type: dict[str, float],
    stop_event: threading.Event,
    event_log: EventLog,
    cell_fault_forms: CellFaultForms,
    get_virtual_cells: Callable[[], list[dict]] | None = None,
    injection_enabled: Callable[[], bool] | None = None,
    collect_hook_fires: Callable[[], None] | None = None,
    poll_interval_seconds: float = POLL_INTERVAL_SECONDS,
    unknown_injection_timeout_seconds: float = UNKNOWN_INJECTION_RESOLUTION_TIMEOUT_SECONDS,
) -> None:
    rng = random.Random(seed)
    next_injection_time_of_cell_type: dict[str, float] = {
        cell_type: _compute_next_injection_time(rng, mean_interval_seconds)
        for cell_type, mean_interval_seconds in sorted(mean_interval_seconds_of_cell_type.items())
    }

    while not stop_event.is_set():
        if stop_event.wait(timeout=poll_interval_seconds):
            break

        cells = list_cells(base_url=base_url, cell_types=set(mean_interval_seconds_of_cell_type))
        if cells is None:
            continue
        if get_virtual_cells is not None:
            cells.extend(get_virtual_cells())

        # Record every poll so the post-run witnesses see the same stream the injector saw.
        event_log.observe(cells)
        if collect_hook_fires is not None:
            collect_hook_fires()
        _raise_if_an_unknown_injection_is_stuck(
            compute_pending_injections(event_log.events), timeout_seconds=unknown_injection_timeout_seconds
        )

        if stop_event.is_set():
            break

        cells_of_type: dict[str, list[dict]] = {cell_type: [] for cell_type in next_injection_time_of_cell_type}
        for cell in cells:
            cells_of_type[cell_type_of(cell)].append(cell)
        now: float = time.monotonic()
        due_types = sorted(kind for kind, due_at in next_injection_time_of_cell_type.items() if now >= due_at)
        if not due_types:
            continue

        ready_types = [kind for kind in due_types if len(cells_of_type[kind]) > 1]
        if not ready_types:
            continue

        cell_type = rng.choice(ready_types)
        target = rng.choice(cells_of_type[cell_type])
        cell_name = target["metadata"]["name"]
        form = _draw_form(
            cell_fault_forms[cell_type], events=event_log.events, cell=target, cell_type=cell_type, rng=rng
        )
        if form is None:
            logger.info("Deferring injection: no fault form of %s can be applied to %s yet", cell_type, cell_name)
            continue
        if injection_enabled is not None and not injection_enabled():
            continue
        next_injection_time_of_cell_type[cell_type] = _compute_next_injection_time(
            rng, mean_interval_seconds_of_cell_type[cell_type]
        )
        try:
            form.inject(target, rng)
        except Exception:
            if not form.records_own_attempt:
                event_log.note_injection_attempt(
                    cell_name=cell_name, form_name=form.name, succeeded=False, harmed=form.harms_cell
                )
            logger.info("Failed to inject fault %s into %s", form.name, cell_name, exc_info=True)
            continue

        if not form.records_own_attempt:
            event_log.note_injection_attempt(
                cell_name=cell_name, form_name=form.name, succeeded=True, harmed=form.harms_cell
            )
        logger.info("Injected fault %s into %s", form.name, cell_name)


def _raise_if_an_unknown_injection_is_stuck(pending: list[PendingInjection], *, timeout_seconds: float) -> None:
    now = datetime.now(timezone.utc)
    for entry in pending:
        if not entry.outcome_known and (now - entry.injected_at).total_seconds() >= timeout_seconds:
            raise UnresolvedInjectionError(
                f"Hook request {entry.request_id} from {entry.cell_name} ({entry.workers_hash}) has no "
                f"trusted fire or pre-arm refusal after {timeout_seconds:.0f}s"
            )


def _draw_form(
    forms: list[BaseFaultForm], *, events: list[Event], cell: dict, cell_type: str, rng: random.Random
) -> BaseFaultForm | None:
    available = [form for form in forms if form.is_available(cell)]
    if not available:
        return None

    worked = compute_successful_form_names(events, cell_type=cell_type)
    unproven = [form for form in available if form.name not in worked]
    return rng.choice(unproven or available)


def list_cells(*, base_url: str, cell_types: set[str]) -> list[dict] | None:
    items = _list_items(base_url=base_url, path="/api/v1/cells", description="cells")
    if items is None:
        return None
    return [c for c in items if cell_type_of(c) in cell_types]


def list_fault_hook_sources(*, base_url: str) -> list[dict] | None:
    return _list_items(base_url=base_url, path="/api/v1/fault-hook-sources", description="fault hook sources")


def _list_items(*, base_url: str, path: str, description: str) -> list[dict] | None:
    try:
        resp = requests.get(f"{base_url}{path}", timeout=LIST_TIMEOUT_SECONDS)
        resp.raise_for_status()
        return resp.json()["items"]
    except Exception:
        logger.info("Failed to list %s from api server", description, exc_info=True)
        return None
