# Re-export shim — canonical source: training_components.coreset
from core.training_components.coreset import (  # noqa: F401
    CoresetManager,
    WeightedBatchSampler,
    SampleStats,
    create_coreset_manager,
    create_weighted_sampler,
)
