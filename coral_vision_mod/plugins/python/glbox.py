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

import numpy
import os
import time

import gi
gi.require_version('GLib', '2.0')
gi.require_version('GObject', '2.0')
gi.require_version('Gst', '1.0')
gi.require_version('GstGL', '1.0')
gi.require_version('GstVideo', '1.0')
from gi.repository import GLib, GObject, Gst, GstGL, GstVideo

from OpenGL.arrays.arraydatatype import ArrayDatatype
from OpenGL.GLES3 import (
    glActiveTexture, glBindBuffer, glBindTexture, glBindVertexArray, glBufferData, glDeleteBuffers,
    glClear, glClearColor, glDeleteVertexArrays, glDrawElements, glEnableVertexAttribArray,
    glGenBuffers, glGenVertexArrays, glVertexAttribPointer)
from OpenGL.GLES3 import (
    GL_ARRAY_BUFFER, GL_COLOR_BUFFER_BIT,GL_ELEMENT_ARRAY_BUFFER, GL_FALSE, GL_FLOAT,
    GL_STATIC_DRAW, GL_TEXTURE0, GL_TEXTURE_2D, GL_TRIANGLES, GL_UNSIGNED_SHORT, GL_VERTEX_SHADER)


SINK_CAPS = 'video/x-raw(memory:GLMemory),format=RGBA,width=[1,{max_int}],height=[1,{max_int}],texture-target=2D'
SINK_CAPS = Gst.Caps.from_string(SINK_CAPS.format(max_int=GLib.MAXINT))

SRC_CAPS = 'video/x-raw(memory:GLMemory),format=RGBA,width=[1,{max_int}],height=[1,{max_int}],texture-target=2D'
SRC_CAPS = Gst.Caps.from_string(SRC_CAPS.format(max_int=GLib.MAXINT))

VERTEX_SHADER = '''
attribute vec4 a_position;
attribute vec2 a_texcoord;
varying vec2 v_texcoord;
uniform float u_scale_x;
uniform float u_scale_y;
void main()
{
  v_texcoord = a_texcoord;
  gl_Position = vec4(a_position.x * u_scale_x, a_position.y * u_scale_y, a_position.zw);
}
'''

POSITIONS = numpy.array([
        -1.0, -1.0,
         1.0, -1.0,
         1.0,  1.0,
        -1.0,  1.0,
    ], dtype=numpy.float32)

TEXCOORDS = numpy.array([
         0.0, 0.0,
         1.0, 0.0,
         1.0, 1.0,
         0.0, 1.0,
    ], dtype=numpy.float32)

INDICES = numpy.array([
         0, 1, 2, 0, 2, 3
    ], dtype=numpy.uint16)

class GlBox(GstGL.GLFilter):
    __gstmetadata__ = ('GlBox',
                       'Filter/Converter/Video',
                       'Scale video preserving aspect ratio',
                       'Coral <coral-support@google.com>')
    __gsttemplates__ = (Gst.PadTemplate.new('sink',
                        Gst.PadDirection.SINK,
                        Gst.PadPresence.ALWAYS,
                        SINK_CAPS),
                        Gst.PadTemplate.new('src',
                        Gst.PadDirection.SRC,
                        Gst.PadPresence.ALWAYS,
                        SRC_CAPS))
    __gproperties__ = {
        'x': (int,
                'Frame x coordinate',
                'Frame x coordinate',
                0,
                GLib.MAXINT,
                0,
                GObject.ParamFlags.READABLE),
        'y': (int,
                'Frame y coordinate',
                'Frame y coordinate',
                0,
                GLib.MAXINT,
                0,
                GObject.ParamFlags.READABLE),
        'width': (int,
                'Frame width',
                'Frame width',
                0,
                GLib.MAXINT,
                0,
                GObject.ParamFlags.READABLE),
        'height': (int,
                'Frame height',
                'Frame height',
                0,
                GLib.MAXINT,
                0,
                GObject.ParamFlags.READABLE),
        'scale-x': (float,
                'Frame scaling factor x',
                'Frame scaling factor x',
                0,
                GLib.MAXFLOAT,
                0,
                GObject.ParamFlags.READABLE),
        'scale-y': (float,
                'Frame scaling factor y',
                'Frame scaling factor y',
                0,
                GLib.MAXFLOAT,
                0,
                GObject.ParamFlags.READABLE),
    }

    def __init__(self):
        GstGL.GLFilter.__init__(self)
        self.x, self.y, self.w, self.h = 0, 0, 0, 0
        self.scale_x, self.scale_y = 1.0, 1.0

        self.shader = None
        self.vao_id = 0
        self.positions_buffer = 0
        self.texcoords_buffer = 0
        self.vbo_indices_buffer = 0
        self.print_fps = int(os.environ.get('PRINT_FPS', '0'))
        self.fps_start = 0
        self.frames = 0

    def do_get_property(self, prop):
        if prop.name == 'x':
            return self.x
        elif prop.name == 'y':
            return self.y
        elif prop.name == 'width':
            return self.w
        elif prop.name == 'height':
            return self.h
        elif prop.name == 'scale-x':
            return self.scale_x
        elif prop.name == 'scale-y':
            return self.scale_y
        else:
            raise AttributeError('Unknown property %s' % prop.name)

    def do_transform_internal_caps(self, direction, caps, filter_caps):
        res = SINK_CAPS if direction == Gst.PadDirection.SRC else SRC_CAPS

        if filter_caps:
            res = res.intersect(filter_caps)
        return res


    def do_gl_start(self):
        frag_stage = GstGL.GLSLStage.new_default_fragment(self.context)
        vert_stage = GstGL.GLSLStage.new_with_string(self.context,
            GL_VERTEX_SHADER,
            GstGL.GLSLVersion.NONE,
            GstGL.GLSLProfile.COMPATIBILITY | GstGL.GLSLProfile.ES,
            VERTEX_SHADER)

        self.shader = GstGL.GLShader.new(self.context)
        self.shader.compile_attach_stage(vert_stage)
        self.shader.compile_attach_stage(frag_stage)
        self.shader.link()

        a_position = self.shader.get_attribute_location('a_position')
        a_texcoord = self.shader.get_attribute_location('a_texcoord')

        self.vao_id = glGenVertexArrays(1)
        glBindVertexArray(self.vao_id)

        self.positions_buffer = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.positions_buffer)
        glBufferData(GL_ARRAY_BUFFER, ArrayDatatype.arrayByteCount(POSITIONS), POSITIONS, GL_STATIC_DRAW)

        self.texcoords_buffer = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.texcoords_buffer)
        glBufferData(GL_ARRAY_BUFFER, ArrayDatatype.arrayByteCount(TEXCOORDS), TEXCOORDS, GL_STATIC_DRAW)

        self.vbo_indices_buffer = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_indices_buffer)
        glBufferData(GL_ARRAY_BUFFER, ArrayDatatype.arrayByteCount(INDICES), INDICES, GL_STATIC_DRAW)

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.vbo_indices_buffer);
        glBindBuffer(GL_ARRAY_BUFFER, self.positions_buffer);
        glVertexAttribPointer.wrappedOperation(a_position, 2, GL_FLOAT, GL_FALSE, 0, None)
        glBindBuffer(GL_ARRAY_BUFFER, self.texcoords_buffer);
        glVertexAttribPointer.wrappedOperation(a_texcoord, 2, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(a_position)
        glEnableVertexAttribArray(a_texcoord)

        glBindVertexArray(0)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

        return True

    def do_gl_stop(self):
        self.shader = None
        glDeleteVertexArrays(1, [self.vao_id])
        self.vao_id = None
        glDeleteBuffers(1, [self.positions_buffer])
        self.positions_buffer = None
        glDeleteBuffers(1, [self.texcoords_buffer])
        self.texcoords_buffer = None
        glDeleteBuffers(1, [self.vbo_indices_buffer])
        self.vbo_indices_buffer = None

    def do_gst_gl_filter_set_caps(self, in_caps, out_caps):
        in_info = GstVideo.VideoInfo()
        in_info.from_caps(in_caps)

        out_info = GstVideo.VideoInfo()
        out_info.from_caps(out_caps)

        in_ratio = in_info.width / in_info.height
        out_ratio = out_info.width / out_info.height

        if in_ratio > out_ratio:
            w = out_info.width
            h = out_info.width / in_ratio
            x = 0
            y = (out_info.height - h) / 2
        elif in_ratio < out_ratio:
            w = out_info.height * in_ratio
            h = out_info.height
            x = (out_info.width - w) / 2
            y = 0
        else:
            w = out_info.width
            h = out_info.height
            x = 0
            y = 0

        self.x = int(x)
        self.y = int(y)
        self.w = int(w)
        self.h = int(h)
        self.scale_x = self.w / out_info.width
        self.scale_y = self.h / out_info.height
        return True

    def do_filter_texture(self, in_tex, out_tex):
        self.render_to_target(in_tex, out_tex, self.do_render)
        self.frames += 1
        if not self.fps_start:
            self.fps_start = time.monotonic()

        elapsed = time.monotonic() - self.fps_start
        if self.print_fps and elapsed > self.print_fps:
            print('glbox: out {} ({:.2f} fps)'.format(
                self.frames, self.frames / elapsed))
            self.fps_start = time.monotonic()
            self.frames = 0
        return True

    def do_render(self, filter, in_tex):
        # Black borders.
        glClearColor(0.0, 0.0, 0.0, 0.0)
        glClear(GL_COLOR_BUFFER_BIT)

        glBindVertexArray(self.vao_id)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, in_tex.tex_id)
        self.shader.use()
        self.shader.set_uniform_1f('u_scale_x', self.scale_x)
        self.shader.set_uniform_1f('u_scale_y', self.scale_y)
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_SHORT, None)

__gstelementfactory__ = ("glbox", Gst.Rank.NONE, GlBox)
