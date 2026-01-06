from abc import ABC, abstractmethod
from typing import List, Tuple
import numpy as np
from pydantic import BaseModel

# --- Data Models ---
class DetectionResult(BaseModel):
    """Standardized output for a single object in a frame"""
    id: int                # Unique ID (from Registry/Tracker)
    box: Tuple[int, int, int, int] # (x, y, w, h)
    confidence: float      # 0.0 to 1.0 (1.0 for Traditional)
    class_id: int = 0      # 0 for Larva

class FrameAnalysis(BaseModel):
    """Output for a single processed frame"""
    frame_index: int
    larvae_count: int      # Active larvae in this frame
    detections: List[DetectionResult]

# --- The Strategy Interface ---
class BaseLarvaDetector(ABC):
    """
    The Strategy Interface. 
    All algorithms (Traditional or DL) must inherit from this.
    """
    
    @abstractmethod
    def detect(self, frame: np.ndarray, frame_index: int) -> FrameAnalysis:
        """
        Process a single frame and return standardized results.
        """
        pass

    @abstractmethod
    def get_total_unique_count(self) -> int:
        """
        Return the final count from the internal Registry.
        """
        pass