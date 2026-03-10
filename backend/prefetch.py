"""Shared async prefetch helpers for CPU/GPU pipelining."""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from queue import Full, Queue
from typing import TypeVar

_PREFETCH_SENTINEL = object()
_PREFETCH_POLL_INTERVAL = 0.05
_PrefetchItem = TypeVar("_PrefetchItem")


@dataclass
class _PrefetchFailure:
    error: BaseException


def iter_prefetched_items(
    load_next_item: Callable[[], _PrefetchItem | None],
    *,
    prefetch_count: int = 2,
) -> Iterator[_PrefetchItem]:
    """Load future items on a background thread while the caller consumes the current one."""
    item_queue: Queue[object] = Queue(maxsize=max(1, prefetch_count))
    stop_event = threading.Event()

    def _put(item: object) -> None:
        while not stop_event.is_set():
            try:
                item_queue.put(item, timeout=_PREFETCH_POLL_INTERVAL)
                return
            except Full:
                continue

    def _producer() -> None:
        try:
            while not stop_event.is_set():
                item = load_next_item()
                if item is None:
                    _put(_PREFETCH_SENTINEL)
                    return
                _put(item)
            _put(_PREFETCH_SENTINEL)
        except BaseException as exc:
            _put(_PrefetchFailure(exc))

    producer = threading.Thread(target=_producer, name="prefetch-worker", daemon=True)
    producer.start()

    try:
        while True:
            item = item_queue.get()
            if item is _PREFETCH_SENTINEL:
                return
            if isinstance(item, _PrefetchFailure):
                raise item.error
            yield item
    finally:
        stop_event.set()
        producer.join(timeout=1.0)
