"""

Lulynx Trainer Configuration (Facade)

This file now aliases the UnifiedTrainingConfig from core.configs.
"""

from ..configs import (
    UnifiedTrainingConfig as LulynxConfig,
    ModelArch,
    NetworkType,
    OptimizerType,
    SchedulerType,
    LyCORISAlgo, # Adding for completeness if needed
)
