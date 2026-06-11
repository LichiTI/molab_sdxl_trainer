# Re-export shim — canonical source: training_components.smart_rank
from core.training_components.smart_rank import (  # noqa: F401
    RankAdvice,
    SmartRankController,
    advise_rank,
    advise_rank_from_weight,
    infer_rank_from_svd,
)