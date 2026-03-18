"""
Lightweight internal event bus for Msanii.
Allows existing endpoints to emit events (e.g., 'royalty_calculated')
that integrations can subscribe to for push notifications and sync.
"""

import asyncio
import logging
from typing import Callable, Dict, List, Any

logger = logging.getLogger(__name__)

# Type for event handlers: async functions that take event_name and payload
EventHandler = Callable[[str, Dict[str, Any]], Any]

# Registry of handlers per event type
_handlers: Dict[str, List[EventHandler]] = {}


def on(event_name: str, handler: EventHandler):
    """Register a handler for an event type."""
    if event_name not in _handlers:
        _handlers[event_name] = []
    _handlers[event_name].append(handler)
    logger.info(f"Registered handler {handler.__name__} for event '{event_name}'")


def off(event_name: str, handler: EventHandler):
    """Unregister a handler for an event type."""
    if event_name in _handlers:
        _handlers[event_name] = [h for h in _handlers[event_name] if h != handler]


async def emit(event_name: str, payload: Dict[str, Any]):
    """
    Emit an event to all registered handlers.
    Handlers run concurrently. Failures in one handler don't affect others.
    """
    handlers = _handlers.get(event_name, [])
    if not handlers:
        return

    logger.info(f"Emitting event '{event_name}' to {len(handlers)} handler(s)")

    tasks = []
    for handler in handlers:
        tasks.append(_safe_call(handler, event_name, payload))

    await asyncio.gather(*tasks)


async def _safe_call(handler: EventHandler, event_name: str, payload: Dict[str, Any]):
    """Call a handler, catching and logging any exceptions."""
    try:
        result = handler(event_name, payload)
        if asyncio.iscoroutine(result):
            await result
    except Exception as e:
        logger.error(
            f"Error in event handler {handler.__name__} for '{event_name}': {e}",
            exc_info=True,
        )


# Standard event names used across the app
CONTRACT_UPLOADED = "contract_uploaded"
CONTRACT_DELETED = "contract_deleted"
ROYALTY_CALCULATED = "royalty_calculated"
ARTIST_CREATED = "artist_created"
TASK_CREATED = "task_created"
TASK_UPDATED = "task_updated"
TASK_COMPLETED = "task_completed"
