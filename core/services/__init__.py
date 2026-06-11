try:
    from core.services.tagger_service import TaggerService
    from core.services.training_service import TrainingService
    __all__ = ['TaggerService', 'TrainingService']
except ImportError:
    __all__ = []