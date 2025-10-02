"""
Component Event Bus for Dashboard Feature Coordination

This implements an event-driven system where dashboard components can:
1. Work independently without dependencies
2. Optionally share data when beneficial
3. Coordinate through events without tight coupling

Usage:
    from .component_event_bus import ComponentEventBus, DashboardEvent

    # Subscribe to events
    bus = ComponentEventBus()
    bus.subscribe('sensor_selected', my_callback)

    # Emit events
    bus.emit('sensor_selected', {'sensor_id': 'SMAP-PWR-001', 'timestamp': datetime.now()})
"""

import logging
import threading
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Standard dashboard event types"""

    # Data events
    SENSOR_SELECTED = "sensor_selected"
    SENSOR_DATA_UPDATED = "sensor_data_updated"
    TIME_RANGE_CHANGED = "time_range_changed"

    # Analysis events
    ANOMALY_DETECTED = "anomaly_detected"
    FORECAST_GENERATED = "forecast_generated"
    ALERT_TRIGGERED = "alert_triggered"

    # UI events
    TAB_CHANGED = "tab_changed"
    FILTER_APPLIED = "filter_applied"
    VIEW_MODE_CHANGED = "view_mode_changed"

    # System events
    MODEL_TRAINED = "model_trained"
    SERVICE_STATUS_CHANGED = "service_status_changed"
    PERFORMANCE_UPDATED = "performance_updated"


@dataclass
class DashboardEvent:
    """Standard dashboard event structure"""

    event_type: EventType
    source_component: str
    timestamp: datetime
    data: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None


class ComponentEventBus:
    """
    Event bus for coordinating between dashboard components

    Features:
    - Thread-safe event handling
    - Component isolation (no tight coupling)
    - Optional data sharing
    - Event filtering and routing
    """

    def __init__(self):
        """Initialize the component event bus"""
        self._subscribers: Dict[EventType, List[Callable]] = defaultdict(list)
        self._event_history: List[DashboardEvent] = []
        self._max_history = 1000  # Keep last 1000 events
        self._lock = threading.RLock()

        # Component registration
        self._registered_components: Dict[str, Dict[str, Any]] = {}

        logger.info("Component Event Bus initialized")

    def register_component(self, component_id: str, component_info: Dict[str, Any]):
        """
        Register a component with the event bus

        Args:
            component_id: Unique component identifier
            component_info: Component metadata (type, capabilities, etc.)
        """
        with self._lock:
            self._registered_components[component_id] = {
                "info": component_info,
                "registered_at": datetime.now(),
                "last_activity": datetime.now(),
                "events_emitted": 0,
                "events_received": 0,
            }

        logger.info(f"Component registered: {component_id}")

    def unregister_component(self, component_id: str):
        """Unregister a component"""
        with self._lock:
            if component_id in self._registered_components:
                del self._registered_components[component_id]
                logger.info(f"Component unregistered: {component_id}")

    def subscribe(
        self,
        event_type: EventType,
        callback: Callable[[DashboardEvent], None],
        component_id: str = "unknown",
    ):
        """
        Subscribe to an event type

        Args:
            event_type: Type of event to listen for
            callback: Function to call when event occurs
            component_id: ID of subscribing component
        """
        with self._lock:
            # Wrap callback to track component activity
            def wrapped_callback(event: DashboardEvent):
                try:
                    if component_id in self._registered_components:
                        self._registered_components[component_id]["events_received"] += 1
                        self._registered_components[component_id]["last_activity"] = datetime.now()
                    callback(event)
                except Exception as e:
                    logger.error(f"Error in event callback for {component_id}: {e}")

            self._subscribers[event_type].append(wrapped_callback)

        logger.debug(f"Component {component_id} subscribed to {event_type.value}")

    def unsubscribe(self, event_type: EventType, callback: Callable):
        """Unsubscribe from an event type"""
        with self._lock:
            if event_type in self._subscribers:
                try:
                    self._subscribers[event_type].remove(callback)
                    logger.debug(f"Unsubscribed from {event_type.value}")
                except ValueError:
                    pass

    def emit(
        self,
        event_type: EventType,
        source_component: str,
        data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Emit an event to all subscribers

        Args:
            event_type: Type of event
            source_component: Component emitting the event
            data: Event data payload
            metadata: Optional metadata
        """
        event = DashboardEvent(
            event_type=event_type,
            source_component=source_component,
            timestamp=datetime.now(),
            data=data,
            metadata=metadata or {},
        )

        with self._lock:
            # Update component activity
            if source_component in self._registered_components:
                self._registered_components[source_component]["events_emitted"] += 1
                self._registered_components[source_component]["last_activity"] = datetime.now()

            # Store in history
            self._event_history.append(event)
            if len(self._event_history) > self._max_history:
                self._event_history.pop(0)

            # Notify subscribers
            subscribers = self._subscribers[event_type].copy()

        # Call subscribers outside of lock to prevent deadlock
        for callback in subscribers:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Error notifying subscriber of {event_type.value}: {e}")

        logger.debug(f"Event emitted: {event_type.value} from {source_component}")

    def get_recent_events(self, event_type: Optional[EventType] = None, limit: int = 50) -> List[DashboardEvent]:
        """
        Get recent events, optionally filtered by type

        Args:
            event_type: Optional event type filter
            limit: Maximum number of events to return

        Returns:
            List of recent events
        """
        with self._lock:
            events = self._event_history.copy()

        if event_type:
            events = [e for e in events if e.event_type == event_type]

        # Return most recent events
        return list(reversed(events))[-limit:]

    def get_component_stats(self) -> Dict[str, Any]:
        """Get statistics about registered components"""
        with self._lock:
            stats = {
                "total_components": len(self._registered_components),
                "total_events": len(self._event_history),
                "components": {},
            }

            for comp_id, comp_data in self._registered_components.items():
                stats["components"][comp_id] = {
                    "registered_at": comp_data["registered_at"],
                    "last_activity": comp_data["last_activity"],
                    "events_emitted": comp_data["events_emitted"],
                    "events_received": comp_data["events_received"],
                    "info": comp_data["info"],
                }

        return stats

    def clear_history(self):
        """Clear event history"""
        with self._lock:
            self._event_history.clear()
        logger.info("Event history cleared")


# Global singleton instance for dashboard
_global_event_bus: Optional[ComponentEventBus] = None


def get_event_bus() -> ComponentEventBus:
    """Get the global dashboard event bus singleton"""
    global _global_event_bus
    if _global_event_bus is None:
        _global_event_bus = ComponentEventBus()
    return _global_event_bus


class EventDrivenComponent:
    """
    Base class for event-driven dashboard components

    Provides common functionality for components that use the event bus
    """

    def __init__(self, component_id: str, component_type: str):
        """
        Initialize event-driven component

        Args:
            component_id: Unique component identifier
            component_type: Type of component (e.g., 'sensor_monitor', 'forecaster')
        """
        self.component_id = component_id
        self.component_type = component_type
        self.event_bus = get_event_bus()

        # Register with event bus
        self.event_bus.register_component(component_id, {"type": component_type, "created_at": datetime.now()})

        logger.info(f"Event-driven component initialized: {component_id} ({component_type})")

    def emit_event(
        self,
        event_type: EventType,
        data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Emit an event from this component"""
        self.event_bus.emit(event_type, self.component_id, data, metadata)

    def subscribe_to_event(self, event_type: EventType, callback: Callable[[DashboardEvent], None]):
        """Subscribe to an event type"""
        self.event_bus.subscribe(event_type, callback, self.component_id)

    def cleanup(self):
        """Clean up component resources"""
        self.event_bus.unregister_component(self.component_id)
        logger.info(f"Component cleaned up: {self.component_id}")


# Utility functions for common event patterns
def emit_sensor_selection(component_id: str, sensor_id: str, additional_data: Dict[str, Any] = None):
    """Utility to emit sensor selection event"""
    bus = get_event_bus()
    data = {"sensor_id": sensor_id}
    if additional_data:
        data.update(additional_data)
    bus.emit(EventType.SENSOR_SELECTED, component_id, data)


def emit_anomaly_alert(
    component_id: str,
    sensor_id: str,
    severity: str,
    score: float,
    additional_data: Dict[str, Any] = None,
):
    """Utility to emit anomaly detection alert"""
    bus = get_event_bus()
    data = {"sensor_id": sensor_id, "severity": severity, "score": score}
    if additional_data:
        data.update(additional_data)
    bus.emit(EventType.ANOMALY_DETECTED, component_id, data)


def emit_forecast_update(
    component_id: str,
    sensor_id: str,
    forecast_horizon: int,
    accuracy: float,
    additional_data: Dict[str, Any] = None,
):
    """Utility to emit forecast generation event"""
    bus = get_event_bus()
    data = {
        "sensor_id": sensor_id,
        "forecast_horizon": forecast_horizon,
        "accuracy": accuracy,
    }
    if additional_data:
        data.update(additional_data)
    bus.emit(EventType.FORECAST_GENERATED, component_id, data)
