# Lint as: python3
# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Object tracking loader.

Based on filename, the loader will instantiate an inference engine.
"""

from tracker_package.object_tracking.base_object_detection import BaseObjectDetectionInference
from tracker_package.object_tracking.camshift_object_tracker import CamshiftObjectTracker
from tracker_package.object_tracking.config import ObjectTrackingConfig
from tracker_package.types import Format
from tracker_package.types import NormalizedBoundingBox
from tracker_package.types import ObjectTrackingAnnotation
from tracker_package.types import Size
from tracker_package.types import Tracker
from tracker_package.utils import format_from_filename
from tracker_package.object_tracking.sort import *

def load(frozen_graph_path,
         label_map_path,
         config,
         file_format=Format.UNDEFINED):
  
  # type: (str, str, ObjectTrackingConfig, Format) -> BaseObjectDetectionInference
  
  """Instantiates an inference engine based on the file format.

  Args:
    frozen_graph_path: Path to the model frozen graph to be used.
    label_map_path: Path to the labelmap .pbtxt file.
    config: An ObjectTrackingConfig instance.
    file_format: Specifies which format the graph is in. If undefined, will make
      assumptions based on filename.

  Returns:
    An instantiated inference engine.
  """
  if file_format == Format.UNDEFINED:
    file_format = format_from_filename(frozen_graph_path)

  print('Loading: {} <{}> {}'.format(frozen_graph_path, file_format,
                                     label_map_path))

  engine = None

  # Some modules may never even be loaded. Only hotloads what is necessary.
  
  if file_format == Format.TFLITE:
    from tracker_package.object_tracking.tflite_object_detection import TFLiteObjectDetectionInference
    engine = TFLiteObjectDetectionInference(frozen_graph_path, label_map_path,
                                            config)
  elif file_format == Format.TENSORFLOW:
    from tracker_package.object_tracking.tf_object_detection import TFObjectDetectionInference
    engine = TFObjectDetectionInference(frozen_graph_path, label_map_path,
                                        config)
  else:
    engine = BaseObjectDetectionInference(None, None, None)

  if config.tracker == Tracker.FAST_INACCURATE:
    return CamshiftObjectTracker(engine, config)
  elif config.tracker == Tracker.BASIC:
    from tracker_package.object_tracking.mediapipe_object_tracker import MediaPipeObjectTracker  
    return MediaPipeObjectTracker(engine, config)
  elif config.tracker == Tracker.SORT:
    return SortObjectTracker(engine, config)
  elif not config.tracker or config.tracker == Tracker.NONE:
    return engine
  else:
    raise NotImplementedError('Invalid or unimplemented tracker type.')
  
