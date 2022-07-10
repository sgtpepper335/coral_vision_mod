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

import cairo
import collections
import contextlib
import ctypes
import numpy
import os
import threading
import time

import gi
gi.require_version('Gdk', '3.0')
gi.require_version('GObject', '2.0')
gi.require_version('GLib', '2.0')
gi.require_version('Gst', '1.0')
gi.require_version('GstAllocators', '1.0')
gi.require_version('GstBase', '1.0')
gi.require_version('GstGL', '1.0')
gi.require_version('GstVideo', '1.0')
from gi.repository import Gdk, GObject, GLib, Gst, GstAllocators, GstBase, GstGL, GstVideo

from OpenGL.arrays.arraydatatype import ArrayDatatype
from OpenGL.GLES3 import (
    glActiveTexture, glBindBuffer, glBindTexture, glBindVertexArray, glBlendEquation, glBlendFunc,
    glBufferData, glDeleteBuffers, glDeleteVertexArrays, glDisable, glDrawElements, glEnable,
    glEnableVertexAttribArray, glGenBuffers, glGenVertexArrays, glGetUniformLocation,
    glUniformMatrix4fv, glVertexAttribPointer, glViewport)
from OpenGL.GLES3 import (
    GL_ARRAY_BUFFER, GL_BLEND, GL_ELEMENT_ARRAY_BUFFER, GL_FALSE, GL_FLOAT, GL_FUNC_ADD,
    GL_ONE_MINUS_SRC_ALPHA, GL_SRC_ALPHA, GL_STATIC_DRAW, GL_TEXTURE0, GL_TEXTURE_2D,
    GL_TRIANGLES, GL_UNSIGNED_SHORT, GL_VERTEX_SHADER)

# Gst.Buffer.map(Gst.MapFlags.WRITE) is broken, this is a workaround. See
# http://lifestyletransfer.com/how-to-make-gstreamer-buffer-writable-in-python/
# https://gitlab.gnome.org/GNOME/gobject-introspection/issues/69
class GstMapInfo(ctypes.Structure):
    _fields_ = [('memory', ctypes.c_void_p),                # GstMemory *memory
                ('flags', ctypes.c_int),                    # GstMapFlags flags
                ('data', ctypes.POINTER(ctypes.c_byte)),    # guint8 *data
                ('size', ctypes.c_size_t),                  # gsize size
                ('maxsize', ctypes.c_size_t),               # gsize maxsize
                ('user_data', ctypes.c_void_p * 4),         # gpointer user_data[4]
                ('_gst_reserved', ctypes.c_void_p * 4)]     # GST_PADDING

# ctypes imports for missing or broken introspection APIs.
libgst = ctypes.CDLL('libgstreamer-1.0.so.0')
GST_MAP_INFO_POINTER = ctypes.POINTER(GstMapInfo)
libgst.gst_buffer_map.argtypes = [ctypes.c_void_p, GST_MAP_INFO_POINTER, ctypes.c_int]
libgst.gst_buffer_map.restype = ctypes.c_int
libgst.gst_buffer_unmap.argtypes = [ctypes.c_void_p, GST_MAP_INFO_POINTER]
libgst.gst_buffer_unmap.restype = None
libgst.gst_mini_object_is_writable.argtypes = [ctypes.c_void_p]
libgst.gst_mini_object_is_writable.restype = ctypes.c_int
libgst.gst_context_writable_structure.restype = ctypes.c_void_p
libgst.gst_context_writable_structure.argtypes = [ctypes.c_void_p]
libgst.gst_structure_set.restype = ctypes.c_void_p
libgst.gst_structure_set.argtypes = [ctypes.c_void_p, ctypes.c_char_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]

libgstgl = ctypes.CDLL('libgstgl-1.0.so.0')
libgstgl.gst_gl_memory_get_texture_id.argtypes = [ctypes.c_void_p]
libgstgl.gst_gl_memory_get_texture_id.restype = ctypes.c_uint
libgstgl.gst_gl_sync_meta_wait_cpu.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
libgstgl.gst_gl_sync_meta_wait_cpu.restype = None

libcairo = ctypes.CDLL('libcairo.so.2')
libcairo.cairo_image_surface_create_for_data.restype = ctypes.c_void_p
libcairo.cairo_image_surface_create_for_data.argtypes = [ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
libcairo.cairo_surface_flush.restype = None
libcairo.cairo_surface_flush.argtypes = [ctypes.c_void_p]
libcairo.cairo_surface_destroy.restype = None
libcairo.cairo_surface_destroy.argtypes = [ctypes.c_void_p]
libcairo.cairo_format_stride_for_width.restype = ctypes.c_int
libcairo.cairo_format_stride_for_width.argtypes = [ctypes.c_int, ctypes.c_int]
libcairo.cairo_create.restype = ctypes.c_void_p
libcairo.cairo_create.argtypes = [ctypes.c_void_p]
libcairo.cairo_destroy.restype = None
libcairo.cairo_destroy.argtypes = [ctypes.c_void_p]

librsvg = ctypes.CDLL('librsvg-2.so.2')
librsvg.rsvg_handle_new_from_data.restype = ctypes.c_void_p
librsvg.rsvg_handle_new_from_data.argtypes = [ctypes.c_char_p, ctypes.c_size_t, ctypes.c_void_p]
librsvg.rsvg_handle_render_cairo.restype = ctypes.c_bool
librsvg.rsvg_handle_render_cairo.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
librsvg.rsvg_handle_close.restype = ctypes.c_bool
librsvg.rsvg_handle_close.argtypes = [ctypes.c_void_p, ctypes.c_void_p]

libgobject = ctypes.CDLL('libgobject-2.0.so.0')
libgobject.g_object_unref.restype = None
libgobject.g_object_unref.argtypes = [ctypes.c_void_p]

libgdk = ctypes.CDLL('libgdk-3.so.0')
libgdk.gdk_wayland_window_get_wl_surface.restype = ctypes.c_void_p
libgdk.gdk_wayland_window_get_wl_surface.argtypes = [ctypes.c_void_p]
libgdk.gdk_wayland_display_get_wl_display.restype = ctypes.c_void_p
libgdk.gdk_wayland_display_get_wl_display.argtypes = [ctypes.c_void_p]

@contextlib.contextmanager
def _gst_buffer_map(buffer, flags):
    ptr = hash(buffer)
    if flags & Gst.MapFlags.WRITE and libgst.gst_mini_object_is_writable(ptr) == 0:
        raise ValueError('Buffer not writable')

    mapping = GstMapInfo()
    success = libgst.gst_buffer_map(ptr, mapping, flags)
    if not success:
        raise RuntimeError('gst_buffer_map failed')
    try:
        yield ctypes.cast(mapping.data, ctypes.POINTER(ctypes.c_byte * mapping.size)).contents
    finally:
        libgst.gst_buffer_unmap(ptr, mapping)

def _get_gl_texture_id(buf):
    if not buf:
        return 0
    memory = buf.peek_memory(0)
    assert GstGL.is_gl_memory(memory)
    return libgstgl.gst_gl_memory_get_texture_id(hash(memory))

VERTEX_SHADER_SRC = '''
    uniform mat4 u_transformation;
    attribute vec4 a_position;
    attribute vec2 a_texcoord;
    varying vec2 v_texcoord;
    void main()
    {
       gl_Position = u_transformation * a_position;
       v_texcoord = a_texcoord;
    }
'''

POSITIONS = numpy.array([
         1.0,  1.0,
        -1.0,  1.0,
        -1.0, -1.0,
         1.0, -1.0,
    ], dtype=numpy.float32)

TEXCOORDS = numpy.array([
         1.0, 0.0,
         0.0, 0.0,
         0.0, 1.0,
         1.0, 1.0,
    ], dtype=numpy.float32)

INDICES = numpy.array([
         0, 1, 2, 0, 2, 3
    ], dtype=numpy.uint16)

IDENTITY_MATRIX = numpy.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=numpy.float16)

HFLIP_MATRIX = numpy.array([
        [-1.0, 0.0, 0.0, 0.0],
        [ 0.0, 1.0, 0.0, 0.0],
        [ 0.0, 0.0, 1.0, 0.0],
        [ 0.0, 0.0, 0.0, 1.0],
    ], dtype=numpy.float16)

VFLIP_MATRIX = numpy.array([
      [1.0,  0.0, 0.0, 0.0],
      [0.0, -1.0, 0.0, 0.0],
      [0.0,  0.0, 1.0, 0.0],
      [0.0,  0.0, 0.0, 1.0],
    ], dtype=numpy.float16)

NUM_BUFFERS = 2

class DmaOverlayBuffer():
    def __init__(self, pool, glupload):
        self.glcontext = glupload.context
        res, self.dmabuf = pool.acquire_buffer()
        assert res == Gst.FlowReturn.OK
        assert GstAllocators.is_dmabuf_memory(self.dmabuf.peek_memory(0))
        with _gst_buffer_map(self.dmabuf, Gst.MapFlags.WRITE) as mapped:
            self.ptr = ctypes.addressof(mapped)
            self.len = ctypes.sizeof(mapped)
            self.clear()
        meta = GstVideo.buffer_get_video_meta(self.dmabuf)
        assert meta
        self.surface = libcairo.cairo_image_surface_create_for_data(
                self.ptr,
                int(cairo.FORMAT_ARGB32),
                meta.width,
                meta.height,
                meta.stride[0])
        self.cairo = libcairo.cairo_create(self.surface)
        res, self.gl_buffer = glupload.perform_with_buffer(self.dmabuf)
        assert res == GstGL.GLUploadReturn.DONE
        memory = self.gl_buffer.peek_memory(0)
        assert GstGL.is_gl_memory(memory)
        self.texture_id = libgstgl.gst_gl_memory_get_texture_id(hash(memory))
        self.sync = GstGL.buffer_add_gl_sync_meta(self.glcontext, self.gl_buffer)

    def __del__(self):
        if self.surface:
            libcairo.cairo_surface_destroy(self.surface)
        if self.cairo:
            libcairo.cairo_destroy(self.cairo)

    def render_svg(self, svg):
        self.wait_sync()
        self.clear()
        if svg:
            data = svg.encode()
            handle = librsvg.rsvg_handle_new_from_data(data, len(data), 0)
            if handle:
                librsvg.rsvg_handle_render_cairo(handle, self.cairo)
                librsvg.rsvg_handle_close(handle, 0)
                libgobject.g_object_unref(handle)
            libcairo.cairo_surface_flush(self.surface)

    def clear(self):
        ctypes.memset(self.ptr, 0, self.len)

    def set_sync_point(self):
        self.sync.set_sync_point(self.glcontext)

    def wait_sync(self):
        libgstgl.gst_gl_sync_meta_wait_cpu(hash(self.sync), hash(self.glcontext))


class GlSvgOverlaySink(Gst.Bin, GstVideo.VideoOverlay):
    __gstmetadata__ = ('GlSvgOverlaySink',
                       'Sink/Video',
                       'glimagesink with SVG overlays',
                       'Coral <coral-support@google.com>')
    __gsttemplates__ = (Gst.PadTemplate.new('sink',
                        Gst.PadDirection.SINK,
                        Gst.PadPresence.ALWAYS,
                        GstGL.GLUpload.get_input_template_caps()
                        ))
    __gproperties__ = {
        'svg': (str,
            'SVG data',
            'SVG overlay data',
            '',
            GObject.ParamFlags.WRITABLE
            ),
        'rotate-method': (str,
            'Rotate method',
            'Rotate method according to glimagesink',
            'none',
            GObject.ParamFlags.WRITABLE
            ),
        'qos': (bool,
            'Generate Quality-of-Service events upstream',
            'Generate Quality-of-Service events upstream',
            True,
            GObject.ParamFlags.READWRITE
            ),
        'sync': (bool,
            'Sync on the clock',
            'Sync on the clock',
            True,
            GObject.ParamFlags.READWRITE
            ),
        'max-lateness': (int,
            'Maximum frame lateness (ns)',
            'Maximum number of nanoseconds that a buffer can be late before it is dropped (-1 unlimited)',
            -1,
            GLib.MAXINT,
            20000000,
            GObject.ParamFlags.READWRITE
            ),
        'fullscreen': (bool,
            'Fullscreen',
            'Requests that internally created windows are fullscreen',
            False,
            GObject.ParamFlags.READWRITE
            ),
        }
    __gsignals__ = {
        'drawn': (GObject.SignalFlags.RUN_LAST, None, ())
    }

    def __init__(self):
        Gst.Bin.__init__(self)
        self.shader = None
        self.vao = 0
        self.positions_buffer = 0
        self.texcoords_buffer = 0
        self.vbo_indices = 0
        self.u_transformation = 0
        self.glcontext = None
        self.glimagesink = Gst.ElementFactory.make('glimagesink')
        self.add(self.glimagesink)
        self.add_pad(Gst.GhostPad('sink', self.glimagesink.get_static_pad('sink')))
        self.glimagesink.connect('client-draw', self.on_draw)
        self.glimagesink.connect('client-reshape', self.on_reshape)
        self.glimagesink.get_static_pad('sink').add_probe(
            Gst.PadProbeType.EVENT_UPSTREAM, self.on_glimagesink_event)
        self.get_static_pad('sink').add_probe(
            Gst.PadProbeType.QUERY_DOWNSTREAM, self.on_downstream_query)
        self.render_thread = None
        self.cond = threading.Condition()
        self.rendering = False
        self.svg = None
        self.buffers = [None] * NUM_BUFFERS
        self.index = 0
        self.matrix = IDENTITY_MATRIX

        self.print_fps = int(os.environ.get('PRINT_FPS', '0'))
        self.incoming_frames = 0
        self.incoming_overlays = 0
        self.rendered_overlays = 0
        self.draws = 0
        self.fps_start = 0
        if self.print_fps:
            self.glimagesink.get_static_pad('sink').add_probe(
                Gst.PadProbeType.BUFFER, self.on_incoming_frame)

    def do_expose(self):
        self.glimagesink.expose()

    def do_handle_events(self, handle_events):
        self.glimagesink.handle_events(handle_events)

    def do_set_render_rectangle(self, x, y, width, height):
        return self.glimagesink.set_render_rectangle(x, y, width, height)

    def do_set_window_handle(self, handle):
        self.glimagesink.set_window_handle(handle)

    def on_glimagesink_event(self, pad, info):
        event = info.get_event()
        if event.type == Gst.EventType.RECONFIGURE:
            return Gst.PadProbeReturn.DROP
        return Gst.PadProbeReturn.OK

    def on_downstream_query(self, pad, info):
        query = info.get_query()
        if query.type == Gst.QueryType.ALLOCATION:
            # Ask glimagesink, but remove the metas we don't support.
            # Need to fiddle with refcount as Python bindings are buggy.
            # refcount is really 1, but Python took another ref making
            # it 2 and hence query is 'not writable'.
            assert query.mini_object.refcount == 2
            try:
                query.mini_object.refcount = 1
                if self.glimagesink.get_static_pad('sink').query(query):
                    for i in reversed(range(0, query.get_n_allocation_metas())):
                        gtype, params = query.parse_nth_allocation_meta(i)
                        if (gtype.name == 'GstVideoAffineTransformationAPI' or
                            gtype.name == 'GstVideoOverlayCompositionMetaAPI'):
                            query.remove_nth_allocation_meta(i)
                    return Gst.PadProbeReturn.HANDLED
            finally:
                query.mini_object.refcount = 2
        return Gst.PadProbeReturn.OK

    def on_glimagesink_event(self, pad, info):
        event = info.get_event()
        if event.type == Gst.EventType.RECONFIGURE:
            return Gst.PadProbeReturn.DROP
        return Gst.PadProbeReturn.OK

    def on_incoming_frame(self, pad, info):
        self.incoming_frames += 1
        return Gst.PadProbeReturn.OK

    def post_error(self, string, debug=''):
        Gst.error(string)
        gerror = GLib.Error.new_literal(Gst.ResourceError.quark(), string, Gst.CoreError.FAILED)
        message = Gst.Message.new_error(self, gerror, debug)
        return self.post_message(message)

    def do_change_state(self, transition):
        if transition == Gst.StateChange.READY_TO_NULL:
            self.glcontext.thread_add(self.deinit_gl)
            self.glcontext = None
        elif transition == Gst.StateChange.PAUSED_TO_READY:
            with self.cond:
                self.rendering = False
                self.cond.notify_all()
            self.render_thread.join()
            self.glcontext.thread_add(self.free_buffers)

        result = Gst.Bin.do_change_state(self, transition)

        if transition == Gst.StateChange.NULL_TO_READY:
            self.glcontext = self.glimagesink.get_property('context')
            if self.glcontext:
                self.glcontext.thread_add(self.init_gl)
            else:
                self.post_error('failed to get gl context')
                result = Gst.StateChangeReturn.FAILURE
        elif transition == Gst.StateChange.READY_TO_PAUSED:
            self.try_create_buffers()
            self.rendering = True
            self.render_thread = threading.Thread(target=self.render_loop)
            self.render_thread.start()
        elif transition == Gst.StateChange.PAUSED_TO_PLAYING:
            self.try_create_buffers()

        return result

    def do_set_property(self, prop, value):
        if prop.name == 'svg':
            if not self.get_back_buffer():
                Gst.warning('Not ready to draw overlays, dropping data')
                return
            with self.cond:
                self.incoming_overlays += 1
                self.svg = value or ''
                self.cond.notify_all()
        elif prop.name == 'rotate-method':
            value = int(value) if value.isnumeric() else value
            self.glimagesink.set_property(prop.name, value)
            value = int(self.glimagesink.get_property(prop.name))
            if value == 0:
                self.matrix = IDENTITY_MATRIX
            elif value == 4:
                self.matrix = HFLIP_MATRIX
            elif value == 5:
                self.matrix = VFLIP_MATRIX
            else:
                Gst.warning('Unsupported rotate-method')
                self.matrix = IDENTITY_MATRIX
        else:
            self.glimagesink.set_property(prop.name, value)

    def do_get_property(self, prop):
        return self.glimagesink.get_property(prop.name)

    def init_gl(self, glcontext):
        assert not self.shader
        assert glcontext == self.glcontext

        frag_stage = GstGL.GLSLStage.new_default_fragment(self.glcontext)
        vert_stage = GstGL.GLSLStage.new_with_string(self.glcontext,
            GL_VERTEX_SHADER,
            GstGL.GLSLVersion.NONE,
            GstGL.GLSLProfile.COMPATIBILITY | GstGL.GLSLProfile.ES,
            VERTEX_SHADER_SRC)
        self.shader = GstGL.GLShader.new(self.glcontext)
        self.shader.compile_attach_stage(vert_stage)
        self.shader.compile_attach_stage(frag_stage)
        self.shader.link()

        self.u_transformation = glGetUniformLocation.wrappedOperation(
            self.shader.get_program_handle(), 'u_transformation')

        a_position = self.shader.get_attribute_location('a_position')
        a_texcoord = self.shader.get_attribute_location('a_texcoord')

        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)

        self.positions_buffer = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.positions_buffer)
        glBufferData(GL_ARRAY_BUFFER, ArrayDatatype.arrayByteCount(POSITIONS), POSITIONS, GL_STATIC_DRAW)

        self.texcoords_buffer = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.texcoords_buffer)
        glBufferData(GL_ARRAY_BUFFER, ArrayDatatype.arrayByteCount(TEXCOORDS), TEXCOORDS, GL_STATIC_DRAW)

        self.vbo_indices = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_indices)
        glBufferData(GL_ARRAY_BUFFER, ArrayDatatype.arrayByteCount(INDICES), INDICES, GL_STATIC_DRAW)

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.vbo_indices);
        glBindBuffer(GL_ARRAY_BUFFER, self.positions_buffer);
        glVertexAttribPointer.wrappedOperation(a_position, 2, GL_FLOAT, GL_FALSE, 0, None)
        glBindBuffer(GL_ARRAY_BUFFER, self.texcoords_buffer);
        glVertexAttribPointer.wrappedOperation(a_texcoord, 2, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(a_position)
        glEnableVertexAttribArray(a_texcoord)

        glBindVertexArray(0)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

    def deinit_gl(self, glcontext):
        assert glcontext == self.glcontext
        self.overlay_shader = None
        self.shader = None
        glDeleteVertexArrays(1, [self.vao])
        self.vao = None
        glDeleteBuffers(1, [self.positions_buffer])
        self.positions_buffer = None
        glDeleteBuffers(1, [self.texcoords_buffer])
        self.texcoords_buffer = None
        glDeleteBuffers(1, [self.vbo_indices])
        self.vbo_indices = None

    def on_reshape(self, sink, context, width, height):
        sinkelement = self.glimagesink.get_by_interface(GstVideo.VideoOverlay)
        if sinkelement.width and sinkelement.height:
            src_ratio = sinkelement.width / sinkelement.height
            dst_ratio = width / height
            if src_ratio > dst_ratio:
              w = width
              h = width / src_ratio
              x = 0
              y = (height - h) / 2
            elif src_ratio < dst_ratio:
              w = height * src_ratio
              h = height
              x = (width - w) / 2
              y = 0
            else:
              w = width
              h = height;
              x = 0
              y = 0
        else:
            w = width
            h = height
            x = 0
            y = 0

        glViewport(int(x), int(y), int(w), int(h))
        return True

    # TODO: affine gltransformation support
    def on_draw(self, sink, context, sample):
        assert context == self.glcontext
        self.draws += 1

        assert context == self.glcontext
        frame_texture = _get_gl_texture_id(sample.get_buffer())
        overlay_buffer = self.get_front_buffer()
        overlay_texture = overlay_buffer.texture_id if overlay_buffer else 0

        glDisable(GL_BLEND)

        glBindVertexArray(self.vao)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, frame_texture)

        self.shader.use()
        self.shader.set_uniform_1i('frame', 0)
        glUniformMatrix4fv(self.u_transformation, 1, GL_FALSE, self.matrix)

        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_SHORT, None)

        if overlay_texture:
            glBindTexture(GL_TEXTURE_2D, overlay_texture)
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glBlendEquation(GL_FUNC_ADD)
            glUniformMatrix4fv(self.u_transformation, 1, GL_FALSE, IDENTITY_MATRIX)
            glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_SHORT, None)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, 0)
        glDisable(GL_BLEND)
        context.clear_shader()
        glBindVertexArray(0)

        if overlay_buffer:
            overlay_buffer.set_sync_point()

        self.emit('drawn')

        if not self.fps_start:
            self.fps_start = time.monotonic()
        elapsed = time.monotonic() - self.fps_start
        if self.print_fps and elapsed > self.print_fps:
            incoming_fps = self.incoming_frames / elapsed
            draw_fps = self.draws / elapsed
            incoming_overlay_fps = self.incoming_overlays / elapsed
            render_fps = self.rendered_overlays / elapsed
            print('glsvgoverlaysink: in frames {} ({:.2f} fps) svg {} ({:.2f} fps), rendered {} ({:.2f} fps), draw {} ({:.2f} fps)'.format(
                    self.incoming_frames, self.incoming_frames / elapsed,
                    self.incoming_overlays, self.incoming_overlays / elapsed,
                    self.rendered_overlays, self.rendered_overlays / elapsed,
                    self.draws, self.draws / elapsed))
            self.incoming_frames = 0
            self.incoming_overlays = 0
            self.rendered_overlays = 0
            self.draws = 0
            self.fps_start = time.monotonic()

        return True

    def render_loop(self):
        while True:
            with self.cond:
                rendering = self.rendering
                svg = self.svg
                self.svg = None

            if not rendering:
                break

            if svg is None:
                with self.cond:
                    self.cond.wait()
                continue

            buf = self.get_back_buffer()
            buf.render_svg(svg)
            self.rendered_overlays += 1
            self.swap_buffers()

    def try_create_buffers(self):
        if self.buffers[0]:
            return

        sink = self.glimagesink.get_by_interface(GstVideo.VideoOverlay)
        width = sink.width
        height = sink.height
        if not width or not height:
            return

        render_caps = 'video/x-raw, width={}, height={}, format=BGRA'
        gl_caps = 'video/x-raw(memory:GLMemory), width={}, height={}, format=RGBA, texture-target=2D'
        min_stride = libcairo.cairo_format_stride_for_width(int(cairo.FORMAT_ARGB32), width)
        render_caps = Gst.Caps.from_string(render_caps.format(width, height))
        gl_caps = Gst.Caps.from_string(gl_caps.format(width, height))
        glupload = GstGL.GLUpload.new(self.glcontext)
        glupload.set_caps(render_caps, gl_caps)
        query = Gst.Query.new_allocation(render_caps, True)
        glupload.propose_allocation(None, query)
        assert query.get_n_allocation_pools()
        pool, size, min_bufs, max_bufs = query.parse_nth_allocation_pool(0)
        assert pool.set_active(True)

        for i in range(0, NUM_BUFFERS):
            self.buffers[i] = DmaOverlayBuffer(pool, glupload)
        assert pool.set_active(False)

    def free_buffers(self, glcontext):
        self.buffers = [None] * NUM_BUFFERS

    def get_front_buffer(self):
        return self.buffers[(self.index - 1) % len(self.buffers)]

    def get_back_buffer(self):
        return self.buffers[self.index]

    def swap_buffers(self):
        self.index = (self.index + 1) % len(self.buffers)

    def get_default_wayland_display_context(self):
        wl_display = libgdk.gdk_wayland_display_get_wl_display(hash(Gdk.Display.get_default()))
        context = Gst.Context.new('GstWaylandDisplayHandleContextType', True)
        structure = libgst.gst_context_writable_structure(hash(context))
        libgst.gst_structure_set(structure, ctypes.c_char_p('display'.encode()),
            hash(GObject.TYPE_POINTER), wl_display, 0)
        return context

    def get_gl_display_context(self):
        if not self.glcontext:
            return None
        context = Gst.Context.new(GstGL.GL_DISPLAY_CONTEXT_TYPE, True)
        GstGL.context_set_gl_display(context, self.glcontext.get_display())
        return context

    def get_wayland_window_handle(self, widget):
        return libgdk.gdk_wayland_window_get_wl_surface(hash(widget.get_window()))

    def get_sharable_local_context(self):
        if not self.glcontext:
            return None
        _, new_glcontext = self.glcontext.get_display().create_context(self.glcontext)
        gst_context = Gst.Context.new('gst.gl.local_context', True)
        structure = libgst.gst_context_writable_structure(hash(gst_context))
        libgst.gst_structure_set(structure, ctypes.c_char_p('context'.encode()),
                hash(GObject.GType.from_name('GstGLContext')), hash(new_glcontext), 0)
        return gst_context


__gstelementfactory__ = ('glsvgoverlaysink', Gst.Rank.NONE, GlSvgOverlaySink)
