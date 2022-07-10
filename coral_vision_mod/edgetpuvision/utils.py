# Copyright 2019 Google LLC
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

import collections
import os
import re
import time

from pycoral.adapters import common
from pycoral.utils import edgetpu
from pycoral.utils import dataset


def load_labels(path):
    return dataset.read_label_file(path)

def input_image_size(interpreter):
    return common.input_size(interpreter)

def same_input_image_sizes(interpreters):
    return len({input_image_size(interpreter) for interpreter in interpreters}) == 1

def avg_fps_counter(window_size):
    window = collections.deque(maxlen=window_size)
    prev = time.monotonic()
    yield 0.0  # First fps value.

    while True:
        curr = time.monotonic()
        window.append(curr - prev)
        prev = curr
        yield len(window) / sum(window)

def make_interpreters(models):
    interpreters, titles = [], {}
    for model in models.split(','):
        if '@' in model:
            model_path, title = model.split('@')
        else:
            model_path, title = model, os.path.basename(os.path.normpath(model))
        interpreter = edgetpu.make_interpreter(model_path)
        interpreter.allocate_tensors()
        interpreters.append(interpreter)
        titles[interpreter] = title
    return interpreters, titles
