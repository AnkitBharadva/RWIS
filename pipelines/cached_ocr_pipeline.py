"""Cached OCR Pipeline for improved performance.

This module wraps the OCR pipeline with intelligent caching to avoid
redundant OCR processing on similar images.
"""

import hashlib
import numpy as np
from collections import OrderedDict
from typing import Optional
import logging

from utils.data_models import OCRResult


logger = logging.getLogger(__name__)


class CachedOCRPipeline:
    """OCR pipeline with LRU caching for performance."""
    
    def __init__(self, ocr_pipeline, cache_size: int = 100):
        """Initialize cached OCR pipeline.
        
        Args:
            ocr_pipeline: Underlying OCR pipeline instance
            cache_size: Maximum number of cached results
        """
        self.ocr_pipeline = ocr_pipeline
        self.cache_size = cache_size
        self.cache: OrderedDict[str, OCRResult] = OrderedDict()
        self.hits = 0
        self.misses = 0
    
    def _compute_hash(self, roi: np.ndarray) -> str:
        """Compute hash of ROI for caching.
        
        Uses a downsampled version to be robust to minor variations.
        
        Args:
            roi: Input ROI image
            
        Returns:
            Hash string
        """
        # Downsample to 32x32 for fast hashing
        import cv2
        small = cv2.resize(roi, (32, 32))
        # Convert to grayscale if needed
        if len(small.shape) == 3:
            small = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        # Compute hash
        return hashlib.md5(small.tobytes()).hexdigest()
    
    def extract_text(self, roi: np.ndarray) -> OCRResult:
        """Extract text with caching.
        
        Args:
            roi: Input ROI image
            
        Returns:
            OCR result (from cache or fresh)
        """
        # Compute hash
        roi_hash = self._compute_hash(roi)
        
        # Check cache
        if roi_hash in self.cache:
            self.hits += 1
            # Move to end (most recently used)
            self.cache.move_to_end(roi_hash)
            result = self.cache[roi_hash]
            logger.debug(f"OCR cache hit (hit rate: {self.get_hit_rate():.1%})")
            return result
        
        # Cache miss - run OCR
        self.misses += 1
        result = self.ocr_pipeline.extract_text(roi)
        
        # Add to cache
        self.cache[roi_hash] = result
        
        # Evict oldest if cache is full
        if len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        
        return result
    
    def get_hit_rate(self) -> float:
        """Get cache hit rate.
        
        Returns:
            Hit rate as fraction (0.0 to 1.0)
        """
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
    
    def clear_cache(self):
        """Clear the cache."""
        self.cache.clear()
        self.hits = 0
        self.misses = 0
    
    def get_stats(self) -> dict:
        """Get cache statistics.
        
        Returns:
            Dictionary with cache stats
        """
        return {
            'cache_size': len(self.cache),
            'max_size': self.cache_size,
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': self.get_hit_rate()
        }
