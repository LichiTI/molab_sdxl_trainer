# Re-export shim — canonical source: training_components.ema
from core.training_components.ema import (  # noqa: F401
    EMAModel,
    EMACallback,
    EMAStateTracker,
    create_ema,
    create_ema_callback,
)
