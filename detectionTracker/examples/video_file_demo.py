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
r"""Runs object detection on video files.

Can output to either a video file or to the UI (default).

Requires cv2 from `sudo apt-get install python3-opencv`

Note: this specific example uses the EdgeTPU .tflite model. 

For Coral Devices:
  python3 examples/video_file_demo.py \
    --input_video data/traffic_frames.mp4
    --output_video data/traffic_frames_annotated.mp4
    --model data/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite \
    --label data/coco_labels.txt

To output to UI instead of file, do not include the "--output_video" argument.

python3 examples/video_file_demo.py \
  --input_video data/traffic_frames.mp4
  --model data/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite \
  --label data/coco_labels.txt

Press Q key to exit.
"""
import argparse
from tracker_package import object_tracking as vot
import utils

try:
  import cv2  
except:  
  print("Couldn't load cv2. Try running: sudo apt install python3-opencv.")


def main():
  default_video = 'data/traffic_frames.mp4'
  default_model = 'data/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite'
  default_labels = 'data/coco_labels.txt'
  parser = argparse.ArgumentParser()
  parser.add_argument('--model', help='model path', default=default_model)
  parser.add_argument(
      '--labels', help='label file path', default=default_labels)
  parser.add_argument(
      '--input_video', help='input video file path', default=default_video)
  parser.add_argument(
      '--output_video', help='output video file path', default='')
  parser.add_argument(
      '--threshold', type=float, default=0.2, help='class score threshold')
  parser.add_argument(
      '--use_tracker', type=str, default=None, help='use an object tracker', choices=[None,'mediapipe','sort','camshift'])
  args = parser.parse_args()
  if args.use_tracker == 'mediapipe': #For MediaPipe Object Tracker
    trackerToBeUsed = vot.Tracker.BASIC
  elif args.use_tracker == 'sort': #For Sort Object Tracker
    trackerToBeUsed = vot.Tracker.SORT
  elif args.use_tracker == 'camshift': #For CamShift object Tracker
    trackerToBeUsed = vot.Tracker.FAST_INACCURATE
  else:
    trackerToBeUsed = vot.Tracker.NONE

  print('Loading %s with %s labels.' % (args.model, args.labels))
  
  config = vot.ObjectTrackingConfig(
      score_threshold=args.threshold,
      tracker=trackerToBeUsed
)
  engine = vot.load(args.model, args.labels, config)
  input_size = engine.input_size()
  fps_calculator = utils.FpsCalculator()

  cap = cv2.VideoCapture(args.input_video)

  writer = None
  if cap.isOpened() and args.output_video:
    writer = cv2.VideoWriter(args.output_video, cv2.VideoWriter_fourcc(*'mp4v'),
                             cap.get(cv2.CAP_PROP_FPS),
                             (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                              int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

  timestamp = 0
  while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
      break

    # Resizes frame.
    resized_frame = cv2.resize(frame, (input_size.width, input_size.height))

    # Calculates current microsecond for timestamp.
    timestamp = int(timestamp + (1/cap.get(cv2.CAP_PROP_FPS)) * 1000 * 1000)

    # Run inference engine to populate annotations array.
    annotations = []
    if engine.run(timestamp, resized_frame, annotations):
      frame = utils.render_bbox(frame, annotations)
    # Calculate FPS, then visualize it.
    fps, latency = fps_calculator.measure()
    print ('fps =',fps)
    print ('latency =',latency)
    frame = cv2.putText(frame, '{} fps'.format(fps), (0, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    frame = cv2.putText(frame, '{} ms'.format(latency), (0, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    if writer:
      writer.write(frame)
    else:
      cv2.imshow('frame', frame)
      if cv2.waitKey(1) & 0xFF == ord('q'):
        break

  if writer:
    writer.release()
  else:
    cv2.destroyAllWindows()
  cap.release()


if __name__ == '__main__':
  main()
