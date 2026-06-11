"""
Smart Advisor Service.
Implements the 'Smart Advisor' logic from the Developer Guide.
Listens to training events and provides intelligent suggestions.
"""

import logging
import random
from core.event_bus import bus, Event, Priority
from core.events import SystemEvents

logger = logging.getLogger(__name__)

class SmartAdvisorService:
    def __init__(self):
        self.enabled = True
        logger.info("[SmartAdvisor] Initializing...")
        self._register_listeners()

    def _register_listeners(self):
        """Subscribe to relevant events"""
        bus.subscribe(SystemEvents.TRAINING_COMPLETED, self._on_training_completed)
        bus.subscribe(SystemEvents.VRAM_WARNING, self._on_vram_warning)

    def _on_training_completed(self, event: Event):
        """Analyze completed training session and suggest improvements"""
        if not self.enabled: return
        
        data = event.data or {}
        duration = data.get("duration_seconds", 0)
        gpu_name = data.get("gpu_name", "Unknown GPU")
        engine = data.get("engine", "lulynx")

        logger.info(f"[SmartAdvisor] Analyzing training session: {duration}s on {gpu_name} using {engine}")

        suggestions = []

        # Batch size suggestion (if we had vram data)
        vram_peak = data.get("vram_peak", 0)
        vram_total = data.get("vram_total", 24)
        if vram_peak < vram_total * 0.5:
             suggestions.append({
                "type": "resource_underutilized",
                "title": "⚡ 显存未充分利用",
                "message": f"当前显存占用仅 {int(vram_peak)}GB。您可以尝试将 Batch Size 翻倍以加快训练速度。",
                "action": "increase_batch_size",
                "confidence": 0.8
            })

        # Push suggestions back to the bus
        for suggestion in suggestions:
            bus.publish(Event(
                name=SystemEvents.ADVISOR_SUGGESTION,
                data=suggestion,
                priority=Priority.HIGH
            ))
            logger.info(f"[SmartAdvisor] Generated suggestion: {suggestion['title']}")

    def _on_vram_warning(self, event: Event):
        """Handle real-time VRAM warnings"""
        logger.warning("[SmartAdvisor] Detected VRAM pressure. Preparing optimization plan...")
        # In a real implementation, this could trigger dynamic parameter adjustment
