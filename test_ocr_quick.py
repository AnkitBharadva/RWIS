"""Quick test for PaddleOCR integration."""
import numpy as np
import os
os.environ['DISABLE_MODEL_SOURCE_CHECK'] = 'True'
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Disable GPU for testing

from pipelines.ocr_pipeline import OCRPipeline

# Create pipeline (GPU disabled via env var)
pipeline = OCRPipeline(gpu_enabled=False, language='en')

# Create a test image with white background
test_roi = np.ones((100, 300, 3), dtype=np.uint8) * 255

# Test extraction
result = pipeline.extract_text(test_roi)
print(f'OCR Result: text="{result.text}", confidence={result.confidence}')
print(f'Pipeline initialized: {pipeline.is_initialized()}')
print(f'Pipeline available: {pipeline.is_available()}')

if pipeline.is_available():
    print('PaddleOCR is working correctly!')
else:
    print(f'PaddleOCR error: {pipeline.get_initialization_error()}')
