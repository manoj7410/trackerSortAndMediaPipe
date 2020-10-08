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

"""This class provides the support for Tracker Object.

"""

from sort import *

class ObjectTracker(object):
    def __init__(self, trackerObjectName):
        if trackerObjectName == 'mediapipe':
            self.trackerObject = MediaPipe()
        elif trackerObjectName == 'sort':
            self.trackerObject = SortTracker()
        elif trackerObjectName == 'deepsort':
            self.trackerObject = DeepSortTracker()
        else:
            print("Invalid Tracker Name")
            self.trackerObject = None
class MediaPipe(ObjectTracker):
    def __init__(self):
        raise NotImplementedError

class SortTracker(ObjectTracker):
    def __init__(self):       
        self.mot_tracker = Sort()    

class DeepSortTracker(ObjectTracker):
    def __init__(self):       
        raise NotImplementedError

