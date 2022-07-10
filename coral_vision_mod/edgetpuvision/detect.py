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

"""A demo which runs object detection on camera frames.

export TEST_DATA=/usr/lib/python3/dist-packages/edgetpu/test_data

Run face detection model:
python3 -m edgetpuvision.detect \
  --model ${TEST_DATA}/mobilenet_ssd_v2_face_quant_postprocess_edgetpu.tflite

Run coco model:
python3 -m edgetpuvision.detect \
  --model ${TEST_DATA}/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite \
  --labels ${TEST_DATA}/coco_labels.txt
"""

import argparse
import colorsys
import itertools
import time

from pycoral.adapters import detect
from pycoral.utils import edgetpu

from . import svg
from . import utils
from .apps import run_app

CSS_STYLES = str(svg.CssStyle({'.back': svg.Style(fill='black',
                                                  stroke='black',
                                                  stroke_width='0.5em'),
                               '.bbox': svg.Style(fill_opacity=0.0,
                                                  stroke_width='0.1em')}))

def size_em(length):
    return '%sem' % str(0.6 * (length + 1))

def color(i, total):
    return tuple(int(255.0 * c) for c in colorsys.hsv_to_rgb(i / total, 1.0, 1.0))

def make_palette(keys):
    return {key : svg.rgb(color(i, len(keys))) for i, key in enumerate(keys)}

def make_get_color(color, labels):
    if color:
        return lambda obj_id: color

    if labels:
        palette = make_palette(labels.keys())
        return lambda obj_id: palette[obj_id]

    return lambda obj_id: 'white'

def overlay(title, objs, get_color, labels, inference_time, inference_rate, layout):
    x0, y0, width, height = layout.window
    font_size = 0.03 * height

    defs = svg.Defs()
    defs += CSS_STYLES

    doc = svg.Svg(width=width, height=height,
                  viewBox='%s %s %s %s' % layout.window,
                  font_size=font_size, font_family='monospace', font_weight=500)
    doc += defs

    for obj in objs:
        percent = int(100 * obj.score)
        if labels:
            caption = '%d%% %s' % (percent, labels[obj.id])
        else:
            caption = '%d%%' % percent

        color = get_color(obj.id)
        inference_width, inference_height = layout.inference_size
        bbox = obj.bbox.scale(1.0 / inference_width, 1.0 / inference_height).scale(*layout.size)
        x, y, w, h = bbox.xmin, bbox.ymin, bbox.width, bbox.height
        doc += svg.Rect(x=x, y=y, width=w, height=h,
                        style='stroke:%s' % color, _class='bbox')
        doc += svg.Rect(x=x, y=y+h ,
                        width=size_em(len(caption)), height='1.2em', fill=color)
        t = svg.Text(x=x, y=y+h, fill='black')
        t += svg.TSpan(caption, dy='1em')
        doc += t

    ox = x0 + 20
    oy1, oy2 = y0 + 20 + font_size, y0 + height - 20

    # Title
    if title:
        doc += svg.Rect(x=0, y=0, width=size_em(len(title)), height='1em',
                        transform='translate(%s, %s) scale(1,-1)' % (ox, oy1), _class='back')
        doc += svg.Text(title, x=ox, y=oy1, fill='white')

    # Info
    lines = [
        'Objects: %d' % len(objs),
        'Inference time: %.2f ms (%.2f fps)' % (inference_time * 1000, 1.0 / inference_time)
    ]

    for i, line in enumerate(reversed(lines)):
        y = oy2 - i * 1.7 * font_size
        doc += svg.Rect(x=0, y=0, width=size_em(len(line)), height='1em',
                       transform='translate(%s, %s) scale(1,-1)' % (ox, y), _class='back')
        doc += svg.Text(line, x=ox, y=y, fill='white')

    return str(doc)

def print_results(inference_rate, objs):
    print('\nInference (rate=%.2f fps):' % inference_rate)
    for i, obj in enumerate(objs):
        print('    %d: %s, area=%.2f' % (i, obj, obj.bbox.area))

def render_gen(args):
    fps_counter  = utils.avg_fps_counter(30)

    interpreters, titles = utils.make_interpreters(args.model)
    assert utils.same_input_image_sizes(interpreters)
    interpreters = itertools.cycle(interpreters)
    interpreter = next(interpreters)

    labels = utils.load_labels(args.labels) if args.labels else None
    filtered_labels = set(l.strip() for l in args.filter.split(',')) if args.filter else None
    get_color = make_get_color(args.color, labels)

    draw_overlay = True

    width, height = utils.input_image_size(interpreter)
    yield width, height

    output = None
    while True:
        tensor, layout, command = (yield output)

        inference_rate = next(fps_counter)
        if draw_overlay:
            start = time.monotonic()
            edgetpu.run_inference(interpreter, tensor)
            inference_time = time.monotonic() - start

            objs = detect.get_objects(interpreter, args.threshold)[:args.top_k]
            if labels and filtered_labels:
                objs = [obj for obj in objs if labels[obj.id] in filtered_labels]

            objs = [obj for obj in objs \
                    if args.min_area <= obj.bbox.scale(1.0 / width, 1.0 / height).area <= args.max_area]

            if args.print:
                print_results(inference_rate, objs)

            title = titles[interpreter]
            output = overlay(title, objs, get_color, labels, inference_time, inference_rate, layout)
        else:
            output = None

        if command == 'o':
            draw_overlay = not draw_overlay
        elif command == 'n':
            interpreter = next(interpreters)

def add_render_gen_args(parser):
    parser.add_argument('--model',
                        help='.tflite model path', required=True)
    parser.add_argument('--labels',
                        help='labels file path')
    parser.add_argument('--top_k', type=int, default=50,
                        help='Max number of objects to detect')
    parser.add_argument('--threshold', type=float, default=0.1,
                        help='Detection threshold')
    parser.add_argument('--min_area', type=float, default=0.0,
                        help='Min bounding box area')
    parser.add_argument('--max_area', type=float, default=1.0,
                        help='Max bounding box area')
    parser.add_argument('--filter', default=None,
                        help='Comma-separated list of allowed labels')
    parser.add_argument('--color', default=None,
                        help='Bounding box display color'),
    parser.add_argument('--print', default=False, action='store_true',
                        help='Print inference results')

def main():
    run_app(add_render_gen_args, render_gen)

if __name__ == '__main__':
    main()
