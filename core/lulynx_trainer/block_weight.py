# Re-export shim — canonical source: training_components.block_weight
from core.training_components.block_weight import (  # noqa: F401
    BlockWeightManager,
    BlockWeightConfig,
    BlockWeightPreset,
    get_preset_list,
    create_block_weight_manager,
    create_block_weight_manager_from_settings,
)
