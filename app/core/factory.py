from .interfaces import BaseLarvaDetector
from .strategies.dl_strategy import DLStrategy
from .strategies.trad_strategy import TraditionalDetectorStrategy

class DetectorFactory:
    @staticmethod
    def get_detector(strategy_type: str) -> BaseLarvaDetector:
        if strategy_type == 'deep_learning':
            return DLStrategy()
        elif strategy_type == 'traditional':
            return TraditionalDetectorStrategy()
        else:
            raise ValueError(f"Unknown strategy type: {strategy_type}")