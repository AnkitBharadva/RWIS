"""Debug script to check ByteTrack output format."""
import sys
sys.path.insert(0, '.')

from ultralytics.trackers.byte_tracker import BYTETracker
from types import SimpleNamespace
import numpy as np

args = SimpleNamespace(track_thresh=0.5, track_buffer=30, match_thresh=0.8, mot20=False)
tracker = BYTETracker(args, frame_rate=30)

det = np.array([[100, 100, 200, 200, 0.9, 0]], dtype=np.float32)
result = tracker.update(det, (480, 640), (480, 640))

print('Result type:', type(result))
print('Length:', len(result))
if len(result) > 0:
    item = result[0]
    print('First item type:', type(item))
    print('Is ndarray:', isinstance(item, np.ndarray))
    print('Has track_id:', hasattr(item, 'track_id'))
    print('Has tlwh:', hasattr(item, 'tlwh'))
    print('Has score:', hasattr(item, 'score'))
    print('Has conf:', hasattr(item, 'conf'))
    if not isinstance(item, np.ndarray):
        print('Dir:', [x for x in dir(item) if not x.startswith('_')])
    else:
        print('Array shape:', item.shape)
        print('Array values:', item)
