"""
Core Event Bus System
Implements a lightweight Pub/Sub mechanism for decoupling system components.
"""

import logging
from typing import Dict, List, Callable, Any, Type
from enum import Enum
import threading
from collections import defaultdict

# Configure logging
logger = logging.getLogger("EventBus")

class Priority(Enum):
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3

class Event:
    """Base Event class"""
    def __init__(self, name: str, data: Any = None, priority: Priority = Priority.NORMAL):
        self.name = name
        self.data = data
        self.priority = priority

class EventBus:
    """
    Singleton Event Bus.
    Thread-safe implementation for managing events and subscribers.
    """
    _instance = None
    _lock = threading.RLock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(EventBus, cls).__new__(cls)
                cls._instance._subscribers: Dict[str, List[Callable]] = defaultdict(list)
                cls._instance._wildcard_subscribers: List[Callable] = []
            return cls._instance

    def subscribe(self, event_name: str, callback: Callable[[Event], None]):
        """Subscribe to a specific event"""
        with self._lock:
            if event_name == "*":
                self._wildcard_subscribers.append(callback)
            else:
                self._subscribers[event_name].append(callback)
            logger.debug(f"[EventBus] Subscribed to {event_name}")

    def unsubscribe(self, event_name: str, callback: Callable[[Event], None]):
        """Unsubscribe from an event"""
        with self._lock:
            if event_name == "*":
                if callback in self._wildcard_subscribers:
                    self._wildcard_subscribers.remove(callback)
            elif event_name in self._subscribers:
                if callback in self._subscribers[event_name]:
                    try:
                        self._subscribers[event_name].remove(callback)
                    except ValueError:
                        pass # Already removed

    def publish(self, event: Event):
        """Publish an event to all subscribers"""
        # We process in a separate thread/task eventually, but for now synchronous for simplicity
        # or we could make it async if we were fully async.
        # Given this is mixed Sync/Async, let's keep it safe.
        
        # Notify specific subscribers
        listeners = []
        with self._lock:
            # Use list() to create a copy and avoid modification during iteration
            listeners.extend(list(self._subscribers.get(event.name, [])))
            listeners.extend(list(self._wildcard_subscribers))
        
        if not listeners:
            return

        # Simple dispatch
        for listener in listeners:
            try:
                listener(event)
            except Exception as e:
                logger.error(f"[EventBus] Error in listener for {event.name}: {e}")

# Global instance
bus = EventBus()
