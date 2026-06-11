"""
Standard System Events Definition.
Centralized registry of all event types to maintain strict contract.
"""

class SystemEvents:
    # System Lifecycle
    STARTUP = "system:startup"
    SHUTDOWN = "system:shutdown"
    
    # Training Lifecycle
    TRAINING_STARTED = "training:started"
    TRAINING_PAUSED = "training:paused"
    TRAINING_RESUMED = "training:resumed"
    TRAINING_STOPPED = "training:stopped"
    TRAINING_COMPLETED = "training:completed"
    TRAINING_FAILED = "training:failed"
    
    # Progress & Metrics
    TRAINING_STEP_UPDATE = "training:step_update"
    TRAINING_EPOCH_COMPLETED = "training:epoch_completed"
    
    # Hardware & Resources
    VRAM_WARNING = "resource:vram_warning"
    VRAM_CRITICAL = "resource:vram_critical"
    GPU_THERMAL_WARNING = "resource:thermal_warning"
    
    # Advisor & Notifications
    ADVISOR_SUGGESTION = "advisor:suggestion"
    USER_NOTIFICATION = "ui:notification"
