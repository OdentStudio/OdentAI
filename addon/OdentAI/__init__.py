# SPDX-License-Identifier: GPL-2.0-or-later
#
# This add-on script uses the Blender Python API and is licensed under the
# GPL, as Blender's add-on policy requires. The segmentation engine it launches
# (bin/) is a separate executable and is not part of this script.
#
# Blender add-on: CBCT -> nine anatomical surfaces.
#
# The work is done by a separate process, bin/dental9/dental9, not by Blender
# itself. Two reasons. First, onnxruntime and vtk cannot be installed into
# Blender's Python without breaking its dependencies, and a graphics card is
# involved on top of that. Second, running out of video memory kills the
# process — better ours than the user's session with unsaved work.
bl_info = {
    "name": "OdentAI Segment",
    "author": "Dr. Ilya Fomenko DMD, Dr. Essaid Issam Dakir DMD, Dr. Krasouski Dmitry DMD",
    "version": (1, 1, 0),
    "blender": (3, 3, 0),
    "location": "3D View > Sidebar (N) > OdentAI",
    "description": "See more. Plan better. CBCT (DICOM) to teeth, jaws, canals, "
                   "sinuses and pharynx — AI segmentation for Blender",
    "category": "Import-Export",
}

# Brand. The name goes everywhere the user can see it; the internal package
# and executable keep the name dental9 — nobody sees those.
BRAND = "OdentAI Segment"
BRAND_SHORT = "OdentAI"
VERSION_LABEL = "1.1.0"
TAGLINE = "See more. Plan better."
AUTHORS = ["Dr. Ilya Fomenko DMD", "Dr. Essaid Issam Dakir DMD", "Dr. Krasouski Dmitry DMD"]
# One universe with ODent: the same community, the same link.
TELEGRAM_LINK = "https://t.me/odent_blender"

import atexit
import json
import math
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time

import bpy
from bpy.props import (BoolProperty, EnumProperty, FloatProperty, IntProperty,
                       PointerProperty, StringProperty)
from bpy.types import AddonPreferences, Operator, Panel, PropertyGroup

# Order and ids must match the model. Change them only together with retraining.
#
# Colours are given as sRGB hex because that is how they were designed and
# checked on rendered meshes. Blender wants linear values, so they are converted
# on the way in — feeding sRGB straight into Base Color washes the colour out.
#
# The palette reads as three families, so a glance separates them:
#   bone      warm ivory, low saturation, the neutral background of the scene
#   teeth     brighter and cooler than bone, enamel rather than bone
#   airways   one blue family graded by depth: sinus azure, nasal cavity a
#             lighter cyan, pharynx a deeper blue
#   canal     the only saturated red in the scene, so it cannot be mistaken
#   palate    muted salmon, soft tissue, related to the canal but far calmer
#
# Roughness follows the same grouping: enamel is glossy, bone is matte.
CLASSES = [
    # key                label               hex        rough  alpha  on
    ("mandible",         "Mandible",         "#EADBC0", 0.52,  0.80,  True),
    ("upper_skull",      "Upper Skull",      "#CDBB9C", 0.55,  0.80,  True),
    ("upper_teeth",      "Upper Teeth",      "#FFFDF5", 0.22,  1.00,  True),
    ("lower_teeth",      "Lower Teeth",      "#F2EFE4", 0.22,  1.00,  True),
    ("mandibular_canal", "Mandibular Canal", "#F03E2F", 0.35,  1.00,  True),
    ("maxillary_sinus",  "Maxillary Sinus",  "#2E8FD8", 0.32,  1.00,  True),
    ("nasal_cavity",     "Nasal Cavity",     "#45C8E8", 0.32,  1.00,  True),
    ("pharynx",          "Pharynx",          "#1F5FA8", 0.35,  1.00,  True),
    ("soft_palate",      "Soft Palate",      "#F2A08C", 0.42,  1.00,  True),
]
# Second pass — teeth as separate objects. Natural teeth keep the enamel look
# of the arch they belong to; the prosthetic parts get their own materials so
# an implant is never mistaken for a tooth in the viewport: titanium grey for
# the screw, a cooler porcelain white for the crown on it, bridges likewise.
TEETH_MATERIALS = {
    "upper":         ("#FFFDF5", 0.22),
    "lower":         ("#F2EFE4", 0.22),
    "implant":       ("#8C9096", 0.30),
    "implant_crown": ("#EEF3F8", 0.18),
    "bridge":        ("#E8EEF5", 0.20),
}
# Both jaws carry a little transparency. Not for looks: the canal and the tooth
# roots sit inside the bone, and at 1.0 they are simply not there to be seen.
# 0.90 is invisible, 0.65 turns the bone into a ghost — 0.80 shows what is
# inside while the jaw still reads as a solid body.

# Sharper than this is a real edge of the anatomy, not a facet of the marching
# cubes grid, and smoothing across it would round off a genuine ridge.
SMOOTH_ANGLE_DEG = 40.0

ADDON_DIR = os.path.dirname(os.path.abspath(__file__))

# Custom icons live in a preview collection for the life of the add-on.
# Blender frees them on unregister; forgetting that leaks and warns on reload.
_ICONS = None


def _icon(name: str) -> int:
    """icon_value for layout calls; 0 when the icon is missing, which Blender
    renders as no icon rather than an error."""
    if _ICONS is None:
        return 0
    it = _ICONS.get(name)
    return it.icon_id if it else 0


# Fixed size in pixels (16 px per unit of scale, times the UI scale). The tile
# does not follow the panel width: a narrower panel clips it, a wider one
# leaves margins — it never stretches.
LOGO_SCALE = 10.0
# Height of the Segment button in row units; the icon beside it is drawn at
# 20 * SEGMENT_BUTTON_SCALE - 8 px.
SEGMENT_BUTTON_SCALE = 1.6



def _read_png_rgba(path: str):
    """Decode an 8-bit RGBA PNG without PIL or bpy.data.

    Why hand-rolled: previews.load() squeezes every image into a 256 px
    buffer, which is what made the logo mushy, and bpy.data.images.load()
    must not be called while the add-on registers. The decoder covers exactly
    what our own files are — 8-bit RGBA, non-interlaced — and nothing else.
    Returns (width, height, pixels) with pixels as floats, bottom row first,
    the way Blender's previews want them.
    """
    import struct
    import zlib
    with open(path, "rb") as fh:
        data = fh.read()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("not a PNG")
    pos, idat, w = 8, [], 0
    while pos < len(data):
        n, = struct.unpack(">I", data[pos:pos + 4])
        kind = data[pos + 4:pos + 8]
        body = data[pos + 8:pos + 8 + n]
        if kind == b"IHDR":
            w, h, depth, ctype, _c, _f, interlace = struct.unpack(">IIBBBBB", body)
            if depth != 8 or ctype != 6 or interlace != 0:
                raise ValueError("only 8-bit RGBA non-interlaced PNG is supported")
        elif kind == b"IDAT":
            idat.append(body)
        elif kind == b"IEND":
            break
        pos += 12 + n
    raw = zlib.decompress(b"".join(idat))
    stride = w * 4
    prev = bytearray(stride)
    rows = []
    for y in range(h):
        off = y * (stride + 1)
        ftype = raw[off]
        cur = bytearray(raw[off + 1:off + 1 + stride])
        for i in range(stride):
            a = cur[i - 4] if i >= 4 else 0
            b = prev[i]
            c = prev[i - 4] if i >= 4 else 0
            if ftype == 1:
                cur[i] = (cur[i] + a) & 255
            elif ftype == 2:
                cur[i] = (cur[i] + b) & 255
            elif ftype == 3:
                cur[i] = (cur[i] + ((a + b) >> 1)) & 255
            elif ftype == 4:
                pp = a + b - c
                pa, pb, pc = abs(pp - a), abs(pp - b), abs(pp - c)
                cur[i] = (cur[i] + (a if pa <= pb and pa <= pc else b if pb <= pc else c)) & 255
        rows.append(cur)
        prev = cur
    flat = []
    for row in reversed(rows):                 # PNG is top-down, Blender bottom-up
        flat.extend(v / 255.0 for v in row)
    return w, h, flat


def _resample_rgba(w: int, h: int, px, size: int):
    """Shrink an RGBA float image to size x size with a Lanczos-3 filter.

    Blender draws a preview with plain bilinear sampling and no mipmaps, so a
    512 px buffer squeezed into ~190 px on screen shimmers and blurs thin
    strokes — the lettering in the logo most of all. Handing it a buffer of
    exactly the drawn size makes the copy 1:1 and the text crisp.

    The result is left premultiplied: Blender composites icon previews with
    premultiplied blending, and straight alpha shows up as a light fringe on
    every rounded corner.
    """
    import numpy as np

    def weights(n_out: int, n_in: int):
        scale = n_in / n_out
        x = (np.arange(n_out) + 0.5) * scale - 0.5
        i = np.arange(n_in)
        t = (i[None, :] - x[:, None]) / scale
        a = 3.0
        with np.errstate(divide="ignore", invalid="ignore"):
            k = np.where(np.abs(t) < 1e-9, 1.0,
                         a * np.sin(np.pi * t) * np.sin(np.pi * t / a) / (np.pi * t) ** 2)
        k[np.abs(t) >= a] = 0.0
        return k / k.sum(axis=1, keepdims=True)

    img = np.asarray(px, dtype=np.float32).reshape(h, w, 4)
    img[..., :3] *= img[..., 3:4]
    wy, wx = weights(size, h), weights(size, w)
    # two separable passes as plain matrix products; einsum on the full
    # three-index expression falls back to a naive loop and takes a minute
    tmp = (wy @ img.reshape(h, w * 4)).reshape(size, w, 4)      # rows
    out = (tmp.transpose(0, 2, 1) @ wx.T).transpose(0, 2, 1)     # columns
    out = np.clip(out, 0.0, 1.0)
    out[..., :3] = np.minimum(out[..., :3], out[..., 3:4])   # keep it premultiplied
    return out.ravel().tolist()


def _ui_scale() -> float:
    ui = 0.0
    try:
        ui = bpy.context.preferences.system.ui_scale        # 0 without a window
        if ui <= 0.0:
            ui = bpy.context.preferences.view.ui_scale      # the user's setting
    except Exception:                                       # noqa: BLE001
        pass
    return ui if ui > 0.0 else 1.0


def _button_icon_pixels() -> int:
    """The icon beside the Segment button: a square of the button's height
    minus the preview padding."""
    return max(8, int(round(20 * SEGMENT_BUTTON_SCALE * _ui_scale())) - 8)


def _logo_pixels() -> int:
    """How many screen pixels template_icon(scale=LOGO_SCALE) actually covers.

    The button is 20 px per unit of scale (UI_UNIT_X), times the interface
    scale, and the preview is drawn inside it with a 4 px pad on each side.
    Measured on a screenshot: scale 10 at ui_scale 1.0 gives 192 px."""
    return max(32, int(round(20 * LOGO_SCALE * _ui_scale())) - 8)


def _load_icons() -> None:
    global _ICONS
    import bpy.utils.previews
    _ICONS = bpy.utils.previews.new()
    d = os.path.join(ADDON_DIR, "icons")
    for name in ("mark", "logo_dark", "logo_horizontal", "icon_orange", "icon_dark",
                 "icon_light", "telegram"):
        f = os.path.join(d, f"{name}.png")
        if os.path.isfile(f):
            _ICONS.load(name, f, "IMAGE")
    # The big logo tile goes in by hand, shrunk to exactly the size it is
    # drawn at (see _resample_rgba); previews.load() would cap it at 256 and
    # leave the downscale to the GPU. Changing the interface scale in the
    # preferences needs the add-on re-enabled to pick up the new size.
    _load_exact("logo_tile", os.path.join(d, "logo_tile.png"), _logo_pixels())
    _load_exact("rocket", os.path.join(d, "rocket.png"), _button_icon_pixels())


def _load_exact(name: str, path: str, size: int) -> None:
    """Load a PNG into a preview of exactly size x size pixels, resampling
    ourselves so the GPU copies it 1:1 on screen (see _resample_rgba)."""
    if not os.path.isfile(path):
        return
    try:
        w, h, px = _read_png_rgba(path)
        if (w, h) != (size, size):
            px, w, h = _resample_rgba(w, h, px, size), size, size
        else:
            import numpy as np
            arr = np.asarray(px, dtype=np.float32).reshape(-1, 4)
            arr[:, :3] *= arr[:, 3:4]                       # premultiplied, as above
            px = arr.ravel().tolist()
        p = _ICONS.new(name)
        p.image_size = (w, h)
        p.image_pixels_float = px
    except Exception:                                       # noqa: BLE001
        pass


def _free_icons() -> None:
    global _ICONS
    if _ICONS is not None:
        import bpy.utils.previews
        bpy.utils.previews.remove(_ICONS)
        _ICONS = None


def _draw_brand_header(layout, scale: float = LOGO_SCALE) -> None:
    """The brand tile at a fixed size, the same in the panel and the
    preferences. Falls back to the bare mark and text if the tile is missing."""
    if _icon("logo_tile"):
        r = layout.row()
        r.alignment = "CENTER"
        r.template_icon(icon_value=_icon("logo_tile"), scale=scale)
        r = layout.row()
        r.alignment = "CENTER"
        r.scale_y = 0.7
        r.label(text=VERSION_LABEL)
        return
    if _icon("mark"):
        r = layout.row()
        r.alignment = "CENTER"
        r.template_icon(icon_value=_icon("mark"), scale=scale * 0.6)
    r = layout.row()
    r.alignment = "CENTER"
    r.label(text=f"{BRAND} · {VERSION_LABEL}")


def _draw_telegram(layout) -> None:
    r = layout.row()
    r.scale_y = 1.3
    r.operator("wm.url_open", text="Join the Telegram community",
               icon_value=_icon("telegram")).url = TELEGRAM_LINK

# Result of the last hardware check. A module-level variable rather than a
# scene property: it describes the machine, not the file, and has no business
# surviving a .blend save.
_DIAG = {}


def _exe() -> str:
    name = "dental9.exe" if sys.platform == "win32" else "dental9"
    return os.path.join(ADDON_DIR, "bin", "dental9", name)


def _ensure_executable(path: str) -> None:
    """Blender's installer drops the executable bit, so the binary arrives
    unrunnable. Zip does carry the permission, but the installer ignores it."""
    if sys.platform != "win32" and os.path.isfile(path) and not os.access(path, os.X_OK):
        try:
            os.chmod(path, 0o755)
        except OSError:
            pass


# What the last segmentation in this session ran on, for the preferences.
# Module-level for the same reason as _DIAG.
_LAST_RUN = {}


def _model() -> str:
    """The weights live next to the binary as a separate file, not inside it:
    they weigh over a quarter of a gigabyte and must be replaceable without a
    rebuild — by dropping new files over these. There is no path in the
    preferences any more (2026-10-02): the zip always carries the weights, and
    a path field only offered a way to point at the wrong file."""
    return os.path.join(ADDON_DIR, "bin", "dental9", "dental9.onnx")


def _teeth_model() -> str:
    """The tooth-numbering weights for "Separate teeth", shipped next to the
    main model."""
    return os.path.join(ADDON_DIR, "bin", "dental9", "teeth_fdi.onnx")


def _model_info(model: str) -> dict:
    """Preprocessing parameters live next to the weights, not in the code:
    they change together with the model, and forgetting one of the two is
    easy."""
    for c in (os.path.splitext(model)[0] + ".json",
              os.path.join(os.path.dirname(model), "dental9.json")):
        if os.path.isfile(c):
            try:
                with open(c, encoding="utf-8") as f:
                    return json.load(f)
            except Exception:                                   # noqa: BLE001
                pass
    return {}


def _prefs(context):
    return context.preferences.addons[__name__].preferences


# Each build carries one GPU provider: DirectML on Windows, CUDA on Linux.
# Offering both made "CUDA" a dead end on Windows (2026-10-02: "this
# onnxruntime build has no such provider"). The numbers are fixed so that a
# value saved on the other platform is not misread as a different device.
_GPU_ITEM = (("dml", "DirectML (GPU)", "Any graphics card: NVIDIA, AMD or Intel", 1)
             if sys.platform == "win32" else
             ("cuda", "CUDA (GPU)", "NVIDIA graphics card", 2))
_DEVICE_ITEMS = [("auto", "Automatic", "Use the graphics card if it works, "
                  "otherwise the CPU", 0),
                 _GPU_ITEM,
                 ("cpu", "CPU", "Many times slower, fallback only", 3)]


def _device(prefs) -> str:
    """The device for the worker. A value this build cannot use (one saved
    before the list became per platform) comes back empty from Blender —
    fall back to automatic rather than fail the run."""
    d = prefs.device
    return d if d in {i[0] for i in _DEVICE_ITEMS} else "auto"


class Dental9Prefs(AddonPreferences):
    bl_idname = __name__

    device: EnumProperty(
        name="Compute on", default="auto", items=_DEVICE_ITEMS)

    def draw(self, context):
        _draw_brand_header(self.layout)
        r = self.layout.row()
        r.alignment = "CENTER"
        r.scale_y = 0.6
        r.label(text=TAGLINE)
        _draw_telegram(self.layout)
        self.layout.separator()
        col = self.layout.column()
        col.prop(self, "device")
        col.separator()

        # What is installed: read-only, the zip brings all of it.
        box = col.box()
        box.label(text="Installed")
        if os.path.isfile(_exe()):
            box.label(text="segmentation engine", icon="CHECKMARK")
        else:
            box.label(text="segmentation engine missing — reinstall the add-on",
                      icon="ERROR")
        m = _model()
        if os.path.isfile(m):
            info = _model_info(m)
            box.label(text=f"model: {os.path.getsize(m)/1e6:.0f} MB, "
                           f"trained on {info.get('dataset', '?')}, "
                           f"{info.get('spacing', '?')} mm, epoch {info.get('epoch', '?')}",
                      icon="CHECKMARK")
        else:
            box.label(text="model missing — reinstall the add-on", icon="ERROR")
        tm = _teeth_model()
        if os.path.isfile(tm):
            info = _model_info(tm)
            box.label(text=f"teeth model: {os.path.getsize(tm)/1e6:.0f} MB, "
                           f"epoch {info.get('epoch', '?')}", icon="CHECKMARK")
        else:
            box.label(text="teeth model missing — \"Separate teeth\" will not work",
                      icon="INFO")

        # What the last run actually used: the setting above is a wish, this
        # is what happened (a GPU can fail and leave the CPU, a tile can shrink).
        box = col.box()
        box.label(text="Last run")
        if not _LAST_RUN:
            box.label(text="no segmentation in this session yet", icon="BLANK1")
        else:
            dev = _LAST_RUN.get("device", "?")
            box.label(text=f"computed on: {dev}",
                      icon="CHECKMARK" if "GPU" in dev else "ERROR")
            tile, want = _LAST_RUN.get("tile"), _LAST_RUN.get("requested_tile")
            if tile:
                t = f"tile {tile[0]}x{tile[1]}x{tile[2]}"
                if want and list(want) != list(tile):
                    t += " (reduced: not enough video memory)"
                box.label(text=t, icon="DOT")
            if _LAST_RUN.get("seconds"):
                box.label(text=f"{_LAST_RUN['seconds']} s in total", icon="TIME")
        col.separator()
        col.operator("dental9.diagnose", icon="SYSTEM")
        for a in _DIAG.get("advice", []):
            col.label(text=a["text"],
                      icon={"ok": "CHECKMARK", "warn": "INFO",
                            "bad": "ERROR"}[a["level"]])


class Dental9Props(PropertyGroup):
    # A plain string on purpose, without the FILE_PATH/DIR_PATH subtype: each
    # of those draws its own browse icon that can return only a file OR only a
    # folder, and a scan arrives as either. The Browse button next to the field
    # handles both, so a built-in icon would be a duplicate that does less.
    input_path: StringProperty(
        name="Scan",
        description="Folder with a DICOM series (nested exports are searched "
                    "too), a single volume file, or a .zip of either")
    out_dir: StringProperty(
        name="Output", subtype="DIR_PATH", default="",
        description="Where to keep the STL files and the report; empty means "
                    "an OdentAI_out folder next to the scan")
    smooth: IntProperty(name="Smoothing", default=20, min=0, max=100,
                        description="Taubin iterations on the mesh. 0 leaves it raw")
    decimate: FloatProperty(name="Lighten mesh", default=0.0, min=0.0, max=0.95,
                            description="Fraction of triangles to drop. 0.5 halves "
                                        "the mesh and is hard to see")
    keep_files: BoolProperty(
        name="Keep STL files on disk", default=False,
        description="Off: the meshes only end up in the scene, and the working "
                    "folder is removed afterwards")
    separate_teeth: BoolProperty(
        name="Separate teeth (FDI numbers)", default=False,
        description="Second pass after the main segmentation: every tooth as its "
                    "own object named by its FDI number, plus implants, implant "
                    "crowns and bridges. Adds about 20 seconds on a GPU")
    use_crop: BoolProperty(
        name="Only inside the box", default=False,
        description="Segment only what is inside the crop box. A large field of "
                    "view needs gigabytes of memory and many times longer; the "
                    "jaws rarely fill more than a third of it")
    crop_box: PointerProperty(
        name="Crop box", type=bpy.types.Object,
        description="Any object: the region is its bounding box. \"Bone preview\" "
                    "creates one around the bone")
    preview_obj: PointerProperty(
        name="Preview", type=bpy.types.Object,
        description="The quick bone surface the box is drawn on; the box is "
                    "read in its coordinates")


# The nine checkboxes are generated from the class list rather than written
# out by hand. Blender properties must be declared through __annotations__;
# a plain setattr is not picked up.
Dental9Props.__annotations__.update(
    {k: BoolProperty(name=l, default=o) for k, l, _h, _r, _a, o in CLASSES})


def _srgb_to_linear(v: int) -> float:
    """Blender stores Base Color linearly; hex is sRGB."""
    c = v / 255.0
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def _material(name, hexcol, rough, alpha=1.0):
    m = bpy.data.materials.get(name)
    if m is None:
        m = bpy.data.materials.new(name)
        m.use_nodes = True
        h = hexcol.lstrip("#")
        rgb = tuple(_srgb_to_linear(int(h[i:i + 2], 16)) for i in (0, 2, 4))
        bsdf = m.node_tree.nodes.get("Principled BSDF")
        if bsdf:
            bsdf.inputs["Base Color"].default_value = (*rgb, 1.0)
            if "Roughness" in bsdf.inputs:
                bsdf.inputs["Roughness"].default_value = rough
            # Anatomy is not a mirror: the default specular reads as wet plastic
            # on bone.
            if "Specular IOR Level" in bsdf.inputs:
                bsdf.inputs["Specular IOR Level"].default_value = 0.4
            if alpha < 1.0:
                bsdf.inputs["Alpha"].default_value = alpha
                # The property was replaced in 4.2 when EEVEE Next landed, so
                # set whichever one this Blender has.
                if hasattr(m, "surface_render_method"):
                    m.surface_render_method = "BLENDED"
                if hasattr(m, "blend_method"):
                    m.blend_method = "BLEND"
                # Without this the inner wall of the bone shell shows through
                # and the jaw reads as a hollow cast.
                m.show_transparent_back = False
        # Solid view does not read the shader nodes at all — it reads this one
        # property, alpha included. Without it everything is the default grey
        # and the colours only appear once the viewport is switched to Material
        # preview, which is not where this work is done.
        #
        # The viewport itself is left alone on purpose: Solid colour is already
        # set to Material by default, and overriding a user's shading settings
        # to show our colours would be taking over their window.
        m.diffuse_color = (*rgb, alpha)
        m.roughness = rough
    return m


def _shade_auto_smooth(obj) -> None:
    """Smooth across gentle curvature, keep genuine edges sharp.

    The operator was renamed in 4.1 and the old mesh flag is gone there, so
    both paths are needed to cover the Blender versions in use.
    """
    me = obj.data
    if hasattr(me, "shade_auto_smooth"):                 # Blender 4.1+
        me.shade_auto_smooth(angle=math.radians(SMOOTH_ANGLE_DEG))
        return
    for poly in me.polygons:
        poly.use_smooth = True
    if hasattr(me, "use_auto_smooth"):                   # Blender 3.x / 4.0
        me.use_auto_smooth = True
        me.auto_smooth_angle = math.radians(SMOOTH_ANGLE_DEG)


def _import_stl(path: str):
    """The operator name changed between Blender versions, so try both."""
    before = set(bpy.data.objects)
    if hasattr(bpy.ops.wm, "stl_import"):
        bpy.ops.wm.stl_import(filepath=path)
    else:
        bpy.ops.import_mesh.stl(filepath=path)
    return [o for o in bpy.data.objects if o not in before]


class DENTAL9_OT_pick_path(Operator):
    """Browse for a scan: a folder, a single file or a zip"""

    bl_idname = "dental9.pick_path"
    bl_label = "Browse"

    filepath: StringProperty(subtype="FILE_PATH")
    directory: StringProperty(subtype="DIR_PATH")
    # Everything the reader accepts. Two cases this filter cannot express, and
    # that is why the field also takes a typed path: DICOM exports often have
    # no extension at all (IM000001), and a folder is what gets picked most of
    # the time anyway.
    filter_glob: StringProperty(
        default="*.dcm;*.zip;*.mha;*.mhd;*.nrrd;*.nhdr;*.nii;*.nii.gz;*.img;"
                "*.hdr;*.vtk;*.gipl;*.mnc;*.mrc;*.rec;*.spr;*.h5;*.tif;*.tiff",
        options={"HIDDEN"})

    def execute(self, context):
        # filepath holds the directory itself when the user opened a folder and
        # pressed Accept without clicking any file inside it
        chosen = self.filepath if os.path.isfile(self.filepath) else self.directory
        context.scene.dental9.input_path = chosen
        return {"FINISHED"}

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}


class DENTAL9_OT_diagnose(Operator):
    bl_idname = "dental9.diagnose"
    bl_label = "Check system"
    bl_description = ("See whether the graphics card is actually used, and how "
                      "long one scan will take")

    def execute(self, context):
        global _DIAG
        exe = _exe()
        _ensure_executable(exe)
        if not os.path.isfile(exe):
            self.report({"ERROR"}, f"executable missing: {exe}")
            return {"CANCELLED"}
        model = _model()
        flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        win = context.window          # there is no window in background mode
        if win:
            win.cursor_set("WAIT")
        try:
            # The check opens a session and runs a probe tile: a few seconds
            # on the CPU, so wait quietly but with an hourglass cursor.
            r = subprocess.run([exe, "--diagnose", "--json", "-m", model],
                               capture_output=True, text=True, timeout=300,
                               creationflags=flags, encoding="utf-8",
                               errors="replace")
            _DIAG = json.loads(r.stdout)
        except Exception as e:                                  # noqa: BLE001
            self.report({"ERROR"}, f"system check failed: {e}")
            return {"CANCELLED"}
        finally:
            if win:
                win.cursor_set("DEFAULT")

# Print the whole report to the console: it carries versions and the
        # provider list, which is exactly what has to be sent if something is
        # wrong.
        print("[OdentAI] " + "=" * 46)
        for k in ("platform", "onnxruntime", "gpus", "model", "gpu", "cpu"):
            if k in _DIAG:
                print(f"[OdentAI] {k}: {_DIAG[k]}")
        for a in _DIAG.get("advice", []):
            print(f"[OdentAI] {a['level'].upper()}: {a['text']}")
        print("[OdentAI] " + "=" * 46)

        bad = [a for a in _DIAG.get("advice", []) if a["level"] == "bad"]
        self.report({"WARNING"} if bad else {"INFO"},
                    bad[0]["text"] if bad else "hardware is fine")
        return {"FINISHED"}


def _console(line: str) -> None:
    """Echo a line of the worker's output into Blender's console.

    Blender's stdout on Windows may sit on the locale code page, and a scan
    path with a letter outside it must not take the whole run down with a
    UnicodeEncodeError — the line is only an echo.
    """
    msg = "[OdentAI] " + line            # one string: print() would emit the
    try:                                  # prefix before failing on the rest
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode("ascii", "replace").decode("ascii"))


def _log_path() -> str:
    """Where the full output of the last run is kept.

    A file, not only the console: on someone else's machine the system
    console is closed, and by the time a run fails its output is gone. The
    file survives, and it is what gets sent to the developer. Overwritten by
    each run — it is the last run that matters.
    """
    d = os.path.join(tempfile.gettempdir(), "OdentAI")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, "last_run.log")


# The worker that is running now, if any: one at a time (two runs would share
# the GPU, the log file and the scene), and stopped on quit or reload.
_RUNNING = None


def _kill_running() -> None:
    """Stop the running worker. Blender calls an operator's cancel() when it
    quits or loads another file, but not on every exit path (a crash, a
    disabled add-on); this is the safety net for those. Before 2026-10-02 the
    worker went on computing a 200 mm scan after Blender had closed."""
    global _RUNNING
    proc, _RUNNING = _RUNNING, None
    if proc is not None and proc.poll() is None:
        try:
            proc.kill()
            proc.wait(timeout=10)
        except Exception:                                       # noqa: BLE001
            pass


atexit.register(_kill_running)

# What the panel shows while a run goes: which run, the stage, 0..1, when it
# started. Read from the worker's own output lines, so any worker build
# drives it. Module-level like _DIAG: it describes this session, not the file.
_PROGRESS = {}
# Set by the Stop button; the running operator sees it on its next tick.
_STOP_REQUESTED = False

# Shares of the bar per stage, measured on 20 clinic scans (2026-10-02): the
# network takes most of a run, meshes and teeth the rest.
_P_READ, _P_NET, _P_MESH, _P_TEETH = 0.05, 0.70, 0.90, 0.99


def _progress_from_line(line: str, n_classes: int, teeth: bool) -> None:
    """Advance _PROGRESS by one line of the worker's output."""
    import re
    s = line.strip()
    p = _PROGRESS
    stage = p.get("phase", "")
    mesh_end = _P_MESH if teeth else 0.98
    m = re.match(r"tile (\d+)/(\d+)", s)
    if s.startswith("reading:"):
        p.update(phase="read", label="Reading the scan", factor=0.01)
    elif s.startswith("resampled to"):
        p.update(factor=_P_READ)
    elif s.startswith("computing on:"):
        dev = s.split(":", 1)[1]
        p.update(phase="net", label="Segmenting on " + ("GPU" if "GPU" in dev else "CPU"),
                 factor=_P_READ)
    elif m and stage == "net":
        f = int(m.group(1)) / max(1, int(m.group(2)))
        p["factor"] = _P_READ + (_P_NET - _P_READ) * f
    elif s.startswith("canal looks broken"):
        p.update(label="Second pass for the canal")
    elif s.startswith("surfaces prepared"):
        p.update(phase="mesh", label="Building surfaces", factor=_P_NET, meshes=0)
    elif stage == "mesh" and re.match(r"[A-Z][\w ]+: ", s):
        p["meshes"] = p.get("meshes", 0) + 1
        p["factor"] = _P_NET + (mesh_end - _P_NET) * min(1.0, p["meshes"] / max(1, n_classes))
    elif s.startswith("separating teeth"):
        p.update(phase="teeth", label="Numbering the teeth", factor=_P_MESH, arch=0)
    elif stage == "teeth" and re.match(r"(upper|lower) arch", s):
        p["arch"] = p.get("arch", 0) + 1
        p["factor"] = _P_MESH + (_P_TEETH - _P_MESH) * p["arch"] / 2
    elif m and stage == "teeth":
        f = int(m.group(1)) / max(1, int(m.group(2)))
        p["factor"] = _P_MESH + (_P_TEETH - _P_MESH) * (p.get("arch", 0) + f) / 2
    elif s.startswith("done in"):
        p.update(label="Loading into the scene", factor=1.0)


def _draw_progress(layout) -> bool:
    """The progress box under the Segment button; False when nothing runs."""
    if _RUNNING is None or _RUNNING.poll() is not None or not _PROGRESS:
        return False
    box = layout.box()
    secs = int(time.time() - _PROGRESS.get("start", time.time()))
    f = max(0.0, min(1.0, _PROGRESS.get("factor", 0.0)))
    text = f"{int(f * 100)}%  {secs // 60}:{secs % 60:02d}  {_PROGRESS.get('label', 'Starting')}"
    if hasattr(box, "progress"):                                # Blender 4.0+
        box.progress(factor=f, type="BAR", text=text)
    else:
        box.label(text=text, icon="SORTTIME")
    box.operator("dental9.stop", icon="CANCEL")
    return True


class DENTAL9_OT_stop(Operator):
    bl_idname = "dental9.stop"
    bl_label = "Stop"
    bl_description = "Stop the running segmentation or preview; nothing is loaded"

    @classmethod
    def poll(cls, context):
        return _RUNNING is not None and _RUNNING.poll() is None

    def execute(self, context):
        global _STOP_REQUESTED
        _STOP_REQUESTED = True
        return {"FINISHED"}


def _redraw_panels(context) -> None:
    """The sidebar redraws only on events; a timer tick is not one for it."""
    for win in context.window_manager.windows:
        for area in win.screen.areas:
            if area.type == "VIEW_3D":
                area.tag_redraw()


class _WorkerRun:
    """Runs the worker executable without freezing Blender: the output is
    read on a thread, echoed to the console, written to the log file and
    shown in the status bar; the subclass finishes the job in _on_done.
    Esc stops it; so does quitting Blender or opening another file."""

    _timer = None
    _proc = None
    _q = None
    _log = None
    _error = ""

    @classmethod
    def poll(cls, context):
        if _RUNNING is not None and _RUNNING.poll() is None:
            if hasattr(cls, "poll_message_set"):                # Blender 3.0+
                cls.poll_message_set("A run is in progress: wait for it, or "
                                     "press Stop under the Segment button")
            return False
        return True

    def _launch(self, context, cmd) -> set:
        global _RUNNING
        self._error = ""
        try:
            self._log = open(_log_path(), "w", encoding="utf-8", errors="replace")
            self._log.write(f"{BRAND} {VERSION_LABEL}, Blender {bpy.app.version_string}, "
                            f"{time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            self._log.write("command: " + subprocess.list2cmdline(cmd) + "\n\n")
        except OSError:
            self._log = None                # no log is no reason to not run
        # CREATE_NO_WINDOW: without it a black console window pops up over
        # Blender on every run under Windows.
        flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        try:
            self._proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                universal_newlines=True, bufsize=1, encoding="utf-8",
                errors="replace", creationflags=flags)
        except OSError as e:
            # An antivirus quarantining the exe, a blocked download: say so
            # instead of a traceback.
            if self._log:
                self._log.write(f"could not start: {e}\n")
                self._log.close()
                self._log = None
            self.report({"ERROR"}, f"could not start the segmentation engine: {e}. "
                                   "Reinstall the add-on; check that the antivirus "
                                   "did not block it")
            self._on_stopped()
            return {"CANCELLED"}
        _RUNNING = self._proc
        global _STOP_REQUESTED
        _STOP_REQUESTED = False
        _PROGRESS.clear()
        _PROGRESS.update(start=time.time(), factor=0.0, phase="",
                         label=self._progress_label)
        _redraw_panels(context)
        self._q = queue.Queue()
        threading.Thread(target=self._reader, daemon=True).start()
        wm = context.window_manager
        self._timer = wm.event_timer_add(0.2, window=context.window)
        wm.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def _reader(self):
        for line in self._proc.stdout:
            self._q.put(line.rstrip())
        self._q.put(None)

    def _stop(self, context, why: str) -> None:
        """Kill the worker and tidy up after it: timer, status bar, log,
        and whatever the subclass made (its temporary folder)."""
        global _RUNNING
        if self._proc is not None and self._proc.poll() is None:
            try:
                self._proc.kill()
                self._proc.wait(timeout=10)
            except Exception:                                   # noqa: BLE001
                pass
        if _RUNNING is self._proc:
            _RUNNING = None
        _PROGRESS.clear()
        try:
            _redraw_panels(context)
        except Exception:                                       # noqa: BLE001
            pass
        if self._timer:
            try:
                context.window_manager.event_timer_remove(self._timer)
            except Exception:                                   # noqa: BLE001
                pass
            self._timer = None
        try:
            context.workspace.status_text_set(None)
        except Exception:                                       # noqa: BLE001
            pass
        if self._log:
            self._log.write(f"\n{why}\n")
            self._log.close()
            self._log = None
        self._on_stopped()

    def cancel(self, context):
        """Blender calls this when it quits or loads another file while the
        run is going."""
        self._stop(context, "stopped: Blender closed or another file was opened")

    def _on_stopped(self) -> None:
        """Cleanup of a run that did not finish; the subclass removes its
        temporary folder here."""

    _progress_label = "Starting"

    def modal(self, context, event):
        stop = event.type == "ESC" and event.value == "PRESS"
        if event.type == "TIMER" and _STOP_REQUESTED:
            stop = True                     # the Stop button in the panel
        if stop:
            self._stop(context, "stopped by the user")
            self.report({"WARNING"}, "stopped")
            return {"CANCELLED"}
        if event.type != "TIMER":
            return {"PASS_THROUGH"}
        done = False
        while True:
            try:
                line = self._q.get_nowait()
            except queue.Empty:
                break
            if line is None:
                done = True
                break
            if not line:
                continue
            _console(line)
            if self._log:
                self._log.write(line + "\n")
            _progress_from_line(line, len(getattr(self, "_selected", []) or []),
                                bool(getattr(self, "_teeth", False)))
            # The details after the error are for the log, not the status bar.
            if line.startswith("ERROR:") and not self._error:
                self._error = line[6:].strip()
            if not self._error:
                context.workspace.status_text_set(f"{line[:110]}   ·   Esc: stop")
        _redraw_panels(context)
        if not done:
            return {"RUNNING_MODAL"}

        global _RUNNING
        if self._timer:
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None
        context.workspace.status_text_set(None)
        rc = self._proc.wait()
        if _RUNNING is self._proc:
            _RUNNING = None
        _PROGRESS.clear()
        _redraw_panels(context)
        if self._log:
            self._log.write(f"\nexit code {rc}\n")
            self._log.close()
            self._log = None
        if rc != 0:
            msg = self._error or "the worker stopped with no message"
            print(f"[OdentAI] full log: {_log_path()}")
            self.report({"ERROR"}, f"{msg[:300]} — log: {_log_path()}")
        return self._on_done(context, rc)

    def _on_done(self, context, rc: int) -> set:
        raise NotImplementedError


def _crop_args(p) -> list:
    """The crop box as --crop arguments, in the scan's own frame.

    The frame is the preview's: the preview mesh arrives in patient
    coordinates, so its local space IS the scan frame, and reading the box
    there keeps working if the user moved the preview and the box together
    (the two are not parented, so moving one alone does shift the crop).
    Without a preview the world frame is used — the segmented meshes land in
    it unmoved, so a box drawn around an earlier result works too. A rotated
    box is taken by its enclosing axis-aligned box.
    """
    box = p.crop_box
    if not (p.use_crop and box):
        return []
    from mathutils import Matrix, Vector
    # matrix_world lags behind a location or size just typed into the panel
    # until the depsgraph runs; without this the previous box would be sent.
    bpy.context.view_layer.update()
    to_scan = (p.preview_obj.matrix_world.inverted() if p.preview_obj
               else Matrix.Identity(4)) @ box.matrix_world
    pts = [to_scan @ Vector(c) for c in box.bound_box]
    lo = [min(v[i] for v in pts) for i in range(3)]
    hi = [max(v[i] for v in pts) for i in range(3)]
    return ["--crop", *(f"{v:.2f}" for v in lo + hi)]


def _remove_object(obj) -> None:
    if obj is None:
        return
    data = obj.data
    bpy.data.objects.remove(obj, do_unlink=True)
    if data is not None and data.users == 0 and isinstance(data, bpy.types.Mesh):
        bpy.data.meshes.remove(data)


def _in_scene(scene, coll) -> bool:
    if hasattr(scene.collection, "children_recursive"):   # Blender 3.2+
        return coll in scene.collection.children_recursive
    stack = list(scene.collection.children)
    while stack:
        c = stack.pop()
        if c == coll:
            return True
        stack.extend(c.children)
    return False


def _brand_collection(scene):
    """The scene's OdentAI collection, made and linked when missing. ODent5
    keeps every segmentation under a collection of this name, so ours is
    looked up by it rather than made anew."""
    coll = bpy.data.collections.get(BRAND_SHORT)
    if coll is None:
        coll = bpy.data.collections.new(BRAND_SHORT)
    if not _in_scene(scene, coll):
        scene.collection.children.link(coll)
    return coll


REGION_COLLECTION = "Region"
# Ours is found by this tag, not by the name: the OdentAI collection is shared
# with ODent5, and a "Region" the user made there is not ours to delete.
REGION_TAG = "odentai_region"


def _our_region(parent):
    for c in parent.children:
        if c.get(REGION_TAG):
            return c
    return None


def _region_collection(scene):
    """OdentAI > Region: the bone preview and the crop box (2026-10-02, they
    used to land in whatever collection was active, among the user's data)."""
    parent = _brand_collection(scene)
    coll = _our_region(parent)
    if coll is None:
        coll = bpy.data.collections.new(REGION_COLLECTION)
        coll[REGION_TAG] = True
        parent.children.link(coll)
    return coll


RUN_TAG = "odentai_run"
# Blender's limit for an ID name is 63 bytes (UTF-8: a Cyrillic letter is 2).
_ID_NAME_BYTES = 63


def _scan_label(src: str) -> str:
    """A short name for the scan: the file or folder name, with the folder
    above it when the name alone says nothing ("CT", "dicom", "Data")."""
    src = os.path.normpath(src) if src else ""
    base = os.path.basename(src)
    for ext in (".nii.gz", ".zip", ".mha", ".mhd", ".nrrd", ".nii", ".dcm"):
        if base.lower().endswith(ext):
            base = base[:-len(ext)]
            break
    if base.lower() in {"ct", "dicom", "data", "dcm", "cbct", "kt", "vol_1",
                        "vol_2", "vol_3", "scan", "images"}:
        parent = os.path.basename(os.path.dirname(src))
        if parent:
            base = f"{parent}/{base}"
    return base or "scan"


def _run_collection(scene, src: str):
    """OdentAI > "<n> · <scan>": one sub-collection per run (2026-10-02).
    Runs on different CTs in one project stay apart and a run is removed
    with one click; ODent5 merges a second top-level OdentAI.001 into
    OdentAI, which used to mix the objects of two runs."""
    parent = _brand_collection(scene)
    # The next number after the highest, not a count: with run 1 deleted a
    # count would give the next run the name of the one still there.
    n = 1 + max([int(c.get(RUN_TAG, 0)) for c in parent.children] + [0])
    name = f"{n} · {_scan_label(src)}"
    while len(name.encode("utf-8")) > _ID_NAME_BYTES:
        name = name[:-1]
    coll = bpy.data.collections.new(name)
    coll[RUN_TAG] = n
    coll["odentai_source"] = src
    parent.children.link(coll)
    return coll


def _move_to(obj, coll) -> None:
    for c in list(obj.users_collection):
        c.objects.unlink(obj)
    coll.objects.link(obj)


def _drop_region_if_empty(scene) -> None:
    parent = bpy.data.collections.get(BRAND_SHORT)
    coll = _our_region(parent) if parent else None
    if coll is not None and not coll.all_objects:
        bpy.data.collections.remove(coll)


def _make_box(name: str, lo, hi):
    """The crop box: a unit cube's eight corners spanning lo..hi, never rendered.

    Corners only, no edges or faces, so Blender itself draws nothing: the box
    is drawn by the add-on's overlay and edited through its handles (see
    DENTAL9_GGT_crop_box). A plain wireframe object was hard to grab — the
    click lands on the bone behind it — and S/G along an axis moves both walls
    at once. The object still holds the box, so the numeric fields in the
    panel keep working. Scaled and moved as an object, so it stays a true box.
    """
    verts = [(x, y, z) for x in (-0.5, 0.5) for y in (-0.5, 0.5) for z in (-0.5, 0.5)]
    me = bpy.data.meshes.new(name)
    me.from_pydata(verts, [], [])
    obj = bpy.data.objects.new(name, me)
    obj.location = [(a + b) / 2 for a, b in zip(lo, hi)]
    obj.scale = [max(b - a, 1.0) for a, b in zip(lo, hi)]
    obj.hide_render = True
    obj.hide_select = True
    # Deliberately not parented to the preview: a child draws Blender's dashed
    # relationship line to its parent's origin, right across the box.
    return obj


class DENTAL9_OT_segment(_WorkerRun, Operator):
    bl_idname = "dental9.segment"
    bl_label = "Segment"
    bl_description = "Run the segmentation and load the surfaces into the scene"
    bl_options = {"REGISTER"}

    _outdir = ""
    _tmp_out = False
    _selected = []
    _teeth = False
    _src = ""

    def _start(self, context):
        """Checks and the command line; None when something is missing."""
        p = context.scene.dental9
        pr = _prefs(context)
        exe = _exe()
        _ensure_executable(exe)
        if not os.path.isfile(exe):
            self.report({"ERROR"}, f"executable missing: {exe}")
            return None
        src = bpy.path.abspath(p.input_path)
        if not src or not os.path.exists(src):
            self.report({"ERROR"}, "no scan selected")
            return None
        self._src = src
        model = _model()
        if not os.path.isfile(model):
            self.report({"ERROR"}, "model file missing, reinstall the add-on")
            return None

        self._selected = [k for k, _l, _h, _r, _a, _o in CLASSES if getattr(p, k)]
        if not self._selected:
            self.report({"ERROR"}, "no classes selected")
            return None

        # Without "keep" the files are only a way to get the meshes into the
        # scene: they go to a temporary folder that is removed after loading.
        if p.keep_files:
            self._outdir = bpy.path.abspath(p.out_dir) or os.path.join(
                os.path.dirname(src), f"{BRAND_SHORT}_out")
            try:
                os.makedirs(self._outdir, exist_ok=True)
            except OSError as e:
                # A scan on a CD or a read-only network share.
                self._outdir = ""
                self.report({"ERROR"}, f"cannot write the files to {e.filename}: "
                                       "choose another output folder")
                return None
            self._tmp_out = False
        else:
            self._outdir = tempfile.mkdtemp(prefix="odentai_")
            self._tmp_out = True

        # The second canal pass is always on and has no switch: it fires only
        # when the canal comes out broken, and on a healthy scan it costs
        # nothing. A switch would only offer a way to make things worse.
        cmd = [exe, src, "-o", self._outdir, "-m", model,
               "--device", _device(pr), "--taubin", str(p.smooth),
               "-c", *self._selected]
        if p.decimate > 0:
            cmd += ["--decimate", f"{p.decimate:.2f}"]
        self._teeth = bool(p.separate_teeth)
        if self._teeth:
            tm = _teeth_model()
            if not os.path.isfile(tm):
                self.report({"ERROR"}, "teeth model missing, reinstall the add-on "
                                       "or untick \"Separate teeth\"")
                return None
            cmd += ["--separate-teeth", "--teeth-model", tm]
        if p.use_crop:
            if not p.crop_box:
                self.report({"ERROR"}, "\"Only inside the box\" is on, but there is "
                                       "no box: press \"Bone preview\" first")
                return None
            cmd += _crop_args(p)
        return cmd

    def _on_done(self, context, rc):
        if rc != 0:
            self._drop_tmp()
            return {"CANCELLED"}
        try:
            self._load(context)
        except Exception as e:                                  # noqa: BLE001
            # A run that computed fine but did not load must still end in a
            # message, not a traceback from a timer.
            import traceback
            traceback.print_exc()
            self.report({"ERROR"}, f"the result could not be loaded: {e}")
            return {"CANCELLED"}
        finally:
            self._drop_tmp()
        # The preview and the box have done their job; they would only sit in
        # the result's way. The eye button in the panel brings them back.
        p = context.scene.dental9
        for o in (p.preview_obj, p.crop_box):
            if o is not None:
                o.hide_set(True)
        return {"FINISHED"}

    def _on_stopped(self):
        self._drop_tmp()

    def _drop_tmp(self):
        """Remove the working folder when the user did not ask to keep files."""
        if self._tmp_out and self._outdir:
            import shutil
            shutil.rmtree(self._outdir, ignore_errors=True)
            self._outdir = ""

    def _load(self, context):
        report = {}
        rp = os.path.join(self._outdir, "report.json")
        if os.path.isfile(rp):
            with open(rp, encoding="utf-8") as f:
                report = json.load(f)

        coll = _run_collection(context.scene, getattr(self, "_src", ""))
        n = 0
        for i, (key, label, hexcol, rough, alpha, _o) in enumerate(CLASSES, start=1):
            if key not in self._selected:
                continue
            path = os.path.join(self._outdir, f"{i:02d}_{key}.stl")
            if not os.path.isfile(path) or os.path.getsize(path) < 100:
                continue
            for obj in _import_stl(path):
                obj.name = label
                obj.data.materials.clear()
                obj.data.materials.append(_material(f"{BRAND_SHORT}_{key}", hexcol, rough, alpha))
                _shade_auto_smooth(obj)
                for c in list(obj.users_collection):
                    c.objects.unlink(obj)
                coll.objects.link(obj)
                n += 1
        n_teeth = 0
        if self._teeth:
            n_teeth = self._load_teeth(context, coll, report.get("teeth", {}))

        # Print the summary separately from the output stream: the console
        # then shows what computed it, at which tile and how long it took —
        # without that it is a mystery why one machine takes fifteen seconds
        # and another three minutes.
        tile = report.get("tile")
        want = report.get("requested_tile")
        secs = report.get("seconds")
        _LAST_RUN.clear()
        _LAST_RUN.update(device=report.get("device", "?"), tile=tile,
                         requested_tile=want, seconds=secs)
        print("[OdentAI] " + "-" * 46)
        print(f"[OdentAI] device:     {report.get('device', '?')}")
        if tile:
            line = f"[OdentAI] tile:       {tile[0]}x{tile[1]}x{tile[2]}"
            if want and list(want) != list(tile):
                line += (f"  (asked for {want[0]}x{want[1]}x{want[2]}, reduced — "
                         "not enough video memory)")
            print(line)
        if report.get("predict_seconds"):
            print(f"[OdentAI] inference:  {report['predict_seconds']} s")
        if secs:
            print(f"[OdentAI] total:      {secs} s (reading, meshing and loading "
                  "included)")
        print(f"[OdentAI] objects:    {n}")
        if self._teeth:
            t = report.get("teeth", {})
            arches = t.get("arches", {})
            line = f"[OdentAI] teeth:      {n_teeth} separated"
            for a in ("upper", "lower"):
                if arches.get(a, {}).get("skipped"):
                    line += f", {a} arch skipped"
            if t.get("prosthesis"):
                line += ", " + ", ".join(f"{k}: {v['components']}" for k, v in t["prosthesis"].items())
            if t.get("seconds"):
                line += f" ({t['seconds']} s)"
            print(line)
        print("[OdentAI] " + "-" * 46)

        # If it ran on the CPU the user must learn that at once, rather than
        # wonder why they waited six minutes.
        if "GPU" not in report.get("device", ""):
            print("[OdentAI] this ran on the CPU. Press \"Check system\" — it "
                  "explains why the graphics card was not used")
            self.report({"WARNING"}, "ran on the CPU — press \"Check system\"")

        msg = f"objects loaded: {n}"
        if tile:
            msg += f", tile {tile[0]}x{tile[1]}x{tile[2]}"
        if secs:
            msg += f", {secs} s on {report.get('device', '')}"
        self.report({"INFO"}, msg)

    def _load_teeth(self, context, parent, treport: dict) -> int:
        """Teeth from the second pass into their own sub-collection. Each object
        is named by its FDI number so the outliner reads like a dental chart."""
        tdir = os.path.join(self._outdir, "teeth")
        if not os.path.isdir(tdir):
            return 0
        # Numbered after the run: collection names are unique per file, and
        # "Teeth.001" says less than "Teeth 2" next to run 2.
        run = parent.get(RUN_TAG)
        coll = bpy.data.collections.new(f"Teeth {run}" if run else "Teeth")
        parent.children.link(coll)
        n = 0
        items = []
        for key, e in treport.get("teeth", {}).items():
            fdi = int(e.get("fdi", key))
            items.append((e.get("stl"), f"Tooth {fdi}", "upper" if fdi < 30 else "lower",
                          e.get("name", f"Tooth {fdi}")))
        for key, e in treport.get("prosthesis", {}).items():
            items.append((e.get("stl"), key.replace("_", " ").title(), key, key))
        for rel, name, mat_key, _full in items:
            if not rel:
                continue
            path = os.path.join(self._outdir, rel)
            if not os.path.isfile(path) or os.path.getsize(path) < 100:
                continue
            hexcol, rough = TEETH_MATERIALS.get(mat_key, TEETH_MATERIALS["upper"])
            for obj in _import_stl(path):
                obj.name = name
                obj.data.materials.clear()
                obj.data.materials.append(_material(f"{BRAND_SHORT}_teeth_{mat_key}", hexcol, rough))
                _shade_auto_smooth(obj)
                for c in list(obj.users_collection):
                    c.objects.unlink(obj)
                coll.objects.link(obj)
                n += 1
        return n

    def execute(self, context):
        cmd = self._start(context)
        if not cmd:
            self._drop_tmp()
            return {"CANCELLED"}
        return self._launch(context, cmd)


PREVIEW_NAME = f"{BRAND_SHORT} Bone Preview"
BOX_NAME = f"{BRAND_SHORT} Crop Box"
# Air around the bone when the box is first drawn: enough not to clip a
# crown or the chin, small next to any field of view.
BOX_MARGIN_MM = 3.0


class DENTAL9_OT_preview(_WorkerRun, Operator):
    """A quick bone surface by threshold, no network, and a crop box around it"""

    bl_idname = "dental9.preview"
    bl_label = "Bone preview"
    _progress_label = "Bone preview (no network, seconds)"
    bl_description = ("Quick bone surface without the network (seconds), with a box "
                      "around it. Shrink the box to the area of interest and only "
                      "that is segmented")
    bl_options = {"REGISTER"}

    _outdir = ""
    _src = ""

    def execute(self, context):
        p = context.scene.dental9
        exe = _exe()
        _ensure_executable(exe)
        if not os.path.isfile(exe):
            self.report({"ERROR"}, f"executable missing: {exe}")
            return {"CANCELLED"}
        self._src = bpy.path.abspath(p.input_path)
        if not self._src or not os.path.exists(self._src):
            self.report({"ERROR"}, "no scan selected")
            return {"CANCELLED"}
        self._outdir = tempfile.mkdtemp(prefix="odentai_preview_")
        return self._launch(context, [exe, self._src, "-o", self._outdir, "--preview"])

    def _on_done(self, context, rc):
        import shutil
        try:
            if rc != 0:
                return {"CANCELLED"}
            return self._load(context)
        except Exception as e:                                  # noqa: BLE001
            import traceback
            traceback.print_exc()
            self.report({"ERROR"}, f"the preview could not be loaded: {e}")
            return {"CANCELLED"}
        finally:
            shutil.rmtree(self._outdir, ignore_errors=True)

    def _on_stopped(self):
        import shutil
        if self._outdir:
            shutil.rmtree(self._outdir, ignore_errors=True)

    def _load(self, context):
        p = context.scene.dental9
        stl = os.path.join(self._outdir, "preview.stl")
        if not os.path.isfile(stl) or os.path.getsize(stl) < 100:
            self.report({"ERROR"}, "the preview came out empty — no bone found")
            return {"CANCELLED"}
        # One preview and one box per scene: a new preview replaces both, the
        # old box belongs to the old scan.
        _remove_object(p.crop_box)
        _remove_object(p.preview_obj)
        for name in (BOX_NAME, PREVIEW_NAME):
            _remove_object(bpy.data.objects.get(name))

        objs = _import_stl(stl)
        if not objs:
            self.report({"ERROR"}, "could not import the preview")
            return {"CANCELLED"}
        prev = objs[0]
        prev.name = PREVIEW_NAME
        prev.data.materials.clear()
        prev.data.materials.append(_material(f"{BRAND_SHORT}_preview", "#B8B2A6", 0.6, 0.55))
        prev.hide_select = True          # clicks go to the handles, not the bone
        prev["odentai_source"] = self._src
        _move_to(prev, _region_collection(context.scene))

        p.preview_obj, p.use_crop = prev, True
        box = _fit_box(p)
        _frame_box(context, box)
        self.report({"INFO"}, "bone preview ready: drag the arrows to shrink the "
                              "box, the ring in the middle moves it; then Segment")
        return {"FINISHED"}


def _fit_box(p):
    """A new crop box around the preview bone, replacing any old one."""
    prev = p.preview_obj
    _remove_object(p.crop_box)
    _remove_object(bpy.data.objects.get(BOX_NAME))
    lo = [min(v.co[i] for v in prev.data.vertices) - BOX_MARGIN_MM for i in range(3)]
    hi = [max(v.co[i] for v in prev.data.vertices) + BOX_MARGIN_MM for i in range(3)]
    box = _make_box(BOX_NAME, lo, hi)
    for c in prev.users_collection:
        c.objects.link(box)
    # lo/hi are in the preview's own space; place the box wherever the preview
    # is now. Setting the world matrix also makes it current before the next
    # depsgraph update (the overlay and the handles read it at once).
    box.matrix_world = prev.matrix_world @ box.matrix_basis
    p.crop_box = box
    return box


def _frame_box(context, box) -> None:
    """Bring the box into view; purely a courtesy, so never an error."""
    try:
        from mathutils import Vector
        m = box.matrix_world
        centre = m @ Vector((0.0, 0.0, 0.0))
        size = max(m.to_scale())
        for area in context.screen.areas:
            if area.type == "VIEW_3D":
                rv3d = area.spaces.active.region_3d
                rv3d.view_location = centre
                rv3d.view_distance = size * 1.8
                break
    except Exception:                                       # noqa: BLE001
        pass


class DENTAL9_OT_remove_preview(Operator):
    bl_idname = "dental9.remove_preview"
    bl_label = "Remove preview and box"
    bl_description = "Delete the bone preview and the crop box; segment the whole scan"

    def execute(self, context):
        p = context.scene.dental9
        _remove_object(p.crop_box)
        _remove_object(p.preview_obj)
        p.use_crop = False
        _drop_region_if_empty(context.scene)
        return {"FINISHED"}


class DENTAL9_OT_fit_box(Operator):
    bl_idname = "dental9.fit_box"
    bl_label = "Fit box to bone"
    bl_description = "Put the box back around the whole preview bone"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return context.scene.dental9.preview_obj is not None

    def execute(self, context):
        _fit_box(context.scene.dental9)
        return {"FINISHED"}


class DENTAL9_OT_toggle_region(Operator):
    bl_idname = "dental9.toggle_region"
    bl_label = "Show or hide the preview and the box"
    bl_description = ("Show or hide the bone preview and the crop box. Hidden, the "
                      "box still applies when \"Only inside the box\" is on")

    def execute(self, context):
        p = context.scene.dental9
        show = not _region_visible(p)
        for o in (p.preview_obj, p.crop_box):
            if o is not None:
                o.hide_set(not show)
        return {"FINISHED"}


def _region_visible(p) -> bool:
    box = p.crop_box
    try:
        return box is not None and box.visible_get()
    except RuntimeError:                  # not in the current view layer
        return False


# ---------------------------------------------------------------------------
# Crop box handles and overlay.
#
# Handles are Blender gizmos, the same machinery as the arrows on a light or
# a camera: they highlight under the cursor, keep their size on screen at any
# zoom, and drag with snapping (Ctrl) and precision (Shift) for free. Six
# arrows, one per wall, each moving only its own wall — the opposite one stays
# put, which is what "cut here" means. A ring in the middle moves the whole
# box. The box itself and its sizes are drawn with the gpu module.
# ---------------------------------------------------------------------------

# Axis colours as in Blender's own navigation gizmo, so X/Y/Z read at once.
AXIS_RGB = ((0.96, 0.30, 0.33), (0.47, 0.80, 0.18), (0.24, 0.55, 0.96))
BOX_RGB = (1.0, 0.58, 0.16)             # the brand orange, for faces and the ring
MIN_BOX_MM = 5.0


def _box_frame(box):
    """The box as (world matrix, local min corner, local max corner). Read from
    the mesh, not assumed to be the unit cube: it may have been edited."""
    from mathutils import Vector
    bb = [Vector(c) for c in box.bound_box]
    lo = Vector([min(c[i] for c in bb) for i in range(3)])
    hi = Vector([max(c[i] for c in bb) for i in range(3)])
    return box.matrix_world.copy(), lo, hi


def _wall_anchor(box, axis: int, sign: int):
    """For the wall on side `sign` of `axis`: the centre of the OPPOSITE wall
    in world space, the outward direction, and the box's extent along it.
    The arrow hangs from the opposite wall at a distance equal to the extent,
    so the extent is the arrow's value and the opposite wall stays fixed."""
    m, lo, hi = _box_frame(box)
    c = (lo + hi) / 2
    c[axis] = lo[axis] if sign > 0 else hi[axis]
    rot = m.to_3x3()
    col = rot.col[axis]
    direction = col.normalized() * sign
    extent = col.length * (hi[axis] - lo[axis])
    return m @ c, direction, extent


def _set_wall(box, axis: int, sign: int, extent: float) -> None:
    """Move one wall so the box is `extent` mm long on that axis."""
    from mathutils import Matrix
    m, lo, hi = _box_frame(box)
    local_len = hi[axis] - lo[axis]
    if local_len <= 1e-9:
        return
    anchor_local = (lo + hi) / 2
    anchor_local[axis] = lo[axis] if sign > 0 else hi[axis]
    anchor_world = m @ anchor_local
    loc, rot, scl = m.decompose()
    scl[axis] = max(extent, MIN_BOX_MM) / local_len
    new = Matrix.LocRotScale(loc, rot, scl)
    # Scaling pivots on the object origin; shift back so the opposite wall
    # is exactly where it was.
    loc = loc + (anchor_world - new @ anchor_local)
    box.matrix_world = Matrix.LocRotScale(loc, rot, scl)


def _box_centre(box):
    m, lo, hi = _box_frame(box)
    return m @ ((lo + hi) / 2)


def _move_box_to(box, centre) -> None:
    from mathutils import Vector
    m = box.matrix_world.copy()
    m.translation += Vector(centre) - _box_centre(box)
    box.matrix_world = m


def _active_box(context):
    """The box the handles and the overlay work on: present, visible, and in
    Object Mode (in Edit Mode the handles would fight Blender's own)."""
    p = getattr(context.scene, "dental9", None)
    if p is None or context.mode != "OBJECT" or not _region_visible(p):
        return None
    return p.crop_box


class DENTAL9_GGT_crop_box(bpy.types.GizmoGroup):
    bl_idname = "DENTAL9_GGT_crop_box"
    bl_label = "OdentAI crop box"
    bl_space_type = "VIEW_3D"
    bl_region_type = "WINDOW"
    # SHOW_MODAL_ALL keeps every handle visible while one is dragged, so the
    # box never looks half-finished mid-drag.
    bl_options = {"3D", "PERSISTENT", "SHOW_MODAL_ALL"}

    @classmethod
    def poll(cls, context):
        return _active_box(context) is not None

    def setup(self, context):
        self.walls = []
        for axis in range(3):
            for sign in (1, -1):
                gz = self.gizmos.new("GIZMO_GT_arrow_3d")
                gz.target_set_handler("offset", get=self._getter(axis, sign),
                                      set=self._setter(axis, sign))
                gz.color = AXIS_RGB[axis]
                gz.alpha = 0.85
                gz.color_highlight = tuple(min(1.0, v + 0.25) for v in AXIS_RGB[axis])
                gz.alpha_highlight = 1.0
                gz.line_width = 3.0
                gz.scale_basis = 1.6
                self.walls.append((gz, axis, sign))

        mv = self.gizmos.new("GIZMO_GT_move_3d")
        mv.target_set_handler("offset", get=self._get_centre, set=self._set_centre)
        mv.draw_options = {"ALIGN_VIEW", "FILL_SELECT"}
        mv.color = BOX_RGB
        mv.alpha = 0.8
        mv.color_highlight = (1.0, 0.8, 0.5)
        mv.alpha_highlight = 1.0
        mv.line_width = 3.0
        mv.scale_basis = 0.3
        self.mover = mv

    # Handlers look the box up on every call: the gizmo group outlives any one
    # box (a new preview replaces it), and a stale reference would crash.
    @staticmethod
    def _getter(axis, sign):
        def get():
            box = _active_box(bpy.context)
            return _wall_anchor(box, axis, sign)[2] if box else 0.0
        return get

    @staticmethod
    def _setter(axis, sign):
        def set_(value):
            box = _active_box(bpy.context)
            if box:
                _set_wall(box, axis, sign, value)
        return set_

    @staticmethod
    def _get_centre():
        box = _active_box(bpy.context)
        return tuple(_box_centre(box)) if box else (0.0, 0.0, 0.0)

    @staticmethod
    def _set_centre(value):
        box = _active_box(bpy.context)
        if box:
            _move_box_to(box, value)

    def draw_prepare(self, context):
        """Put every handle where the box is now — also mid-drag."""
        from mathutils import Matrix
        box = _active_box(context)
        if box is None:
            return
        for gz, axis, sign in self.walls:
            anchor, direction, _extent = _wall_anchor(box, axis, sign)
            rot = direction.to_track_quat("Z", "Y").to_matrix().to_4x4()
            gz.matrix_basis = Matrix.Translation(anchor) @ rot
        # The mover's value IS the centre, in world space, so its basis stays
        # at the origin.
        self.mover.matrix_basis = Matrix.Identity(4)


# Corner order of Object.bound_box: 0 (-,-,-) 1 (-,-,+) 2 (-,+,+) 3 (-,+,-)
# 4 (+,-,-) 5 (+,-,+) 6 (+,+,+) 7 (+,+,-). Edges grouped by the axis they run
# along, so each is drawn in its axis colour.
_BOX_EDGES = (((0, 4), (1, 5), (2, 6), (3, 7)),        # along X
              ((0, 3), (1, 2), (4, 7), (5, 6)),        # along Y
              ((0, 1), (3, 2), (4, 5), (7, 6)))        # along Z
_BOX_TRIS = ((0, 1, 2), (0, 2, 3), (4, 7, 6), (4, 6, 5), (0, 4, 5), (0, 5, 1),
             (3, 2, 6), (3, 6, 7), (0, 3, 7), (0, 7, 4), (1, 5, 6), (1, 6, 2))
_DRAW_HANDLERS = []


def _box_corners(box):
    m = box.matrix_world
    from mathutils import Vector
    return [m @ Vector(c) for c in box.bound_box]


def _draw_box_3d():
    """The box: faint orange walls, edges in their axis colours, seen through
    the bone so the walls are never lost behind it."""
    box = _active_box(bpy.context)
    if box is None:
        return
    import gpu
    from gpu_extras.batch import batch_for_shader
    pts = [tuple(v) for v in _box_corners(box)]
    gpu.state.blend_set("ALPHA")
    gpu.state.depth_test_set("NONE")
    try:
        sh = gpu.shader.from_builtin("UNIFORM_COLOR")
        batch = batch_for_shader(sh, "TRIS", {"pos": pts}, indices=_BOX_TRIS)
        sh.uniform_float("color", (*BOX_RGB, 0.03))
        batch.draw(sh)

        sh = gpu.shader.from_builtin("POLYLINE_UNIFORM_COLOR")
        region = bpy.context.region
        sh.uniform_float("viewportSize", (region.width, region.height))
        sh.uniform_float("lineWidth", 2.0 * _ui_scale())
        for axis, edges in enumerate(_BOX_EDGES):
            seg = [pts[i] for e in edges for i in e]
            batch = batch_for_shader(sh, "LINES", {"pos": seg})
            sh.uniform_float("color", (*AXIS_RGB[axis], 0.9))
            batch.draw(sh)
    finally:
        gpu.state.blend_set("NONE")
        gpu.state.depth_test_set("LESS_EQUAL")


def _draw_box_labels():
    """The size along each axis, in millimetres, beside the middle of an edge."""
    box = _active_box(bpy.context)
    if box is None:
        return
    import blf
    from bpy_extras.view3d_utils import location_3d_to_region_2d
    region, rv3d = bpy.context.region, bpy.context.region_data
    if rv3d is None:
        return
    pts = _box_corners(box)
    font = 0
    s = _ui_scale()
    blf.size(font, 13 * s)
    blf.enable(font, blf.SHADOW)
    blf.shadow(font, 3, 0.0, 0.0, 0.0, 0.8)
    for axis, edges in enumerate(_BOX_EDGES):
        # Label the edge whose middle is lowest on screen: it is the one in
        # front more often than not and does not sit on the arrows.
        best = None
        for i, j in edges:
            mid = location_3d_to_region_2d(region, rv3d, (pts[i] + pts[j]) / 2)
            if mid is not None and (best is None or mid.y < best.y):
                best = mid
        if best is None:
            continue
        size = (pts[edges[0][1]] - pts[edges[0][0]]).length
        blf.color(font, *AXIS_RGB[axis], 1.0)
        blf.position(font, best.x + 6 * s, best.y + 6 * s, 0)
        blf.draw(font, f"{'XYZ'[axis]} {size:.1f} mm")
    blf.disable(font, blf.SHADOW)


def _add_draw_handlers() -> None:
    sv = bpy.types.SpaceView3D
    _DRAW_HANDLERS.append(sv.draw_handler_add(_draw_box_3d, (), "WINDOW", "POST_VIEW"))
    _DRAW_HANDLERS.append(sv.draw_handler_add(_draw_box_labels, (), "WINDOW", "POST_PIXEL"))


def _remove_draw_handlers() -> None:
    for h in _DRAW_HANDLERS:
        try:
            bpy.types.SpaceView3D.draw_handler_remove(h, "WINDOW")
        except (ValueError, RuntimeError):
            pass
    _DRAW_HANDLERS.clear()


class DENTAL9_OT_open_log(Operator):
    bl_idname = "dental9.open_log"
    bl_label = "Open last log"
    bl_description = ("The full output of the last run: system, memory, the error "
                      "and its details. Send this file when something fails")

    def execute(self, context):
        path = _log_path()
        if not os.path.isfile(path):
            self.report({"WARNING"}, "no run yet")
            return {"CANCELLED"}
        bpy.ops.wm.path_open(filepath=path)
        return {"FINISHED"}


class DENTAL9_OT_select_all(Operator):
    bl_idname = "dental9.select_all"
    bl_label = "Select"
    state: BoolProperty(default=True)

    def execute(self, context):
        for key, _l, _h, _r, _a, _o in CLASSES:
            setattr(context.scene.dental9, key, self.state)
        return {"FINISHED"}


def _draw_region(layout, context, p) -> None:
    """Bone preview and the crop box: what part of the scan gets computed."""
    box = layout.box()
    row = box.row(align=True)
    row.label(text="Region")
    if p.preview_obj or p.crop_box:
        row.operator("dental9.toggle_region", text="",
                     icon="HIDE_OFF" if _region_visible(p) else "HIDE_ON")
        row.operator("dental9.fit_box", text="", icon="SHADING_BBOX")
        row.operator("dental9.remove_preview", text="", icon="TRASH")
    box.operator("dental9.preview", icon="MESH_CUBE")
    if not p.crop_box:
        box.label(text="whole scan", icon="INFO")
        return
    box.prop(p, "use_crop")
    if _region_visible(p) and context.mode == "OBJECT":
        box.label(text="drag the arrows, the ring moves", icon="INFO")
    col = box.column(align=True)
    col.enabled = p.use_crop
    col.prop(p.crop_box, "location", text="Centre")
    # Dimensions rather than scale: millimetres, whatever the box's history.
    col.prop(p.crop_box, "dimensions", text="Size (mm)")
    src = p.preview_obj.get("odentai_source") if p.preview_obj else None
    if src and os.path.normcase(src) != os.path.normcase(bpy.path.abspath(p.input_path)):
        box.label(text="the preview is of another scan", icon="ERROR")


class DENTAL9_PT_panel(Panel):
    bl_label = BRAND
    bl_idname = "DENTAL9_PT_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = BRAND_SHORT

    def draw(self, context):
        p = context.scene.dental9
        _draw_brand_header(self.layout)
        # Also at the top: the panel is taller than a laptop screen, and the
        # Segment button with the bar under it ends up below the edge.
        _draw_progress(self.layout)
        col = self.layout.column()
        row = col.row(align=True)
        row.prop(p, "input_path")
        row.operator("dental9.pick_path", text="", icon="FILEBROWSER")

        _draw_region(self.layout, context, p)

        box = self.layout.box()
        box.label(text="Classes")
        grid = box.column(align=True)
        for key, _label, _h, _r, _a, _o in CLASSES:
            grid.prop(p, key)
        row = box.row(align=True)
        row.operator("dental9.select_all", text="All").state = True
        row.operator("dental9.select_all", text="None").state = False
        box.separator()
        box.prop(p, "separate_teeth")

        box = self.layout.box()
        box.label(text="Mesh")
        box.prop(p, "smooth")
        box.prop(p, "decimate")
        box.prop(p, "keep_files")
        if p.keep_files:
            box.prop(p, "out_dir", text="")

        # A button cannot draw its icon larger than 16 px, so the mark sits
        # beside the button as its own element. The icon is a square of the
        # button's height (scale_y applies to the button only), and the
        # visible image is that square minus the 4 px preview padding.
        r = self.layout.row(align=True)
        if _icon("rocket"):
            r.template_icon(icon_value=_icon("rocket"), scale=SEGMENT_BUTTON_SCALE)
        c = r.column(align=True)
        c.scale_y = SEGMENT_BUTTON_SCALE
        c.operator("dental9.segment")
        _draw_progress(self.layout)

        box = self.layout.box()
        row = box.row(align=True)
        row.label(text="Hardware")
        row.operator("dental9.open_log", text="", icon="TEXT")
        row.operator("dental9.diagnose", text="", icon="FILE_REFRESH")
        if not _DIAG:
            box.label(text="not checked yet", icon="QUESTION")
            box.operator("dental9.diagnose", icon="SYSTEM")
        else:
            g = _DIAG.get("gpu", {})
            box.label(text=f"computing on: {g.get('device_used', '?')}",
                      icon="CHECKMARK" if "GPU" in g.get("device_used", "")
                      else "ERROR")
            gpus = _DIAG.get("gpus") or [{"name": n} for n in _DIAG.get("gpu_names", [])]
            for gpu in gpus[:2]:
                vram = f", {gpu['vram_gb']:g} GB" if gpu.get("vram_gb") else ""
                box.label(text=(gpu["name"][:34 - len(vram)] + vram), icon="DOT")
            if g.get("full_tile_seconds"):
                box.label(text=f"working tile: {g['full_tile_seconds']} s", icon="DOT")
            if g.get("case_seconds_estimate"):
                box.label(text=f"~{g['case_seconds_estimate']} s per scan",
                          icon="TIME")
            ICON = {"ok": "CHECKMARK", "warn": "INFO", "bad": "ERROR"}
            for a in _DIAG.get("advice", []):
                if a["level"] == "ok":
                    continue
                col = box.column(align=True)
                # Long text does not fit the panel on one line and Blender
                # cannot wrap it, so wrap by words here.
                line, words = "", a["text"].split()
                first = True
                for w in words:
                    if len(line) + len(w) > 34:
                        col.label(text=line,
                                  icon=ICON[a["level"]] if first else "BLANK1")
                        line, first = w, False
                    else:
                        line = f"{line} {w}".strip()
                if line:
                    col.label(text=line, icon=ICON[a["level"]] if first else "BLANK1")



class DENTAL9_PT_about(Panel):
    """Collapsed by default: whoever needs the authors and the licences opens
    it; everyone else gets a shorter panel."""
    bl_label = "About"
    bl_idname = "DENTAL9_PT_about"
    bl_parent_id = "DENTAL9_PT_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        box = self.layout
        c = box.column(align=True)
        c.scale_y = 0.8
        c.label(text=TAGLINE)
        for a in AUTHORS:
            c.label(text=a, icon="USER")
        _draw_telegram(box)
        # The training data and the models behind the labels come with
        # attribution licences (ToothFairy2 CC BY-SA, DentVoxel CC BY,
        # DentalSegmentator CC BY, TotalSegmentator and nnU-Net Apache-2.0).
        # Acknowledging them is an obligation, not a courtesy.
        c = box.column(align=True)
        c.scale_y = 0.8
        c.label(text="Built on open data and methods:")
        c.label(text="ToothFairy2, DentVoxel, nnU-Net,")
        c.label(text="DentalSegmentator, TotalSegmentator")
        box.operator("wm.path_open", text="Third-party notices",
                     icon="TEXT").filepath = os.path.join(ADDON_DIR, "THIRD_PARTY_NOTICES.md")


# The About sub-panel must come after its parent: Blender resolves
# bl_parent_id at registration time.
CLASSES_RNA = (Dental9Prefs, Dental9Props, DENTAL9_OT_segment, DENTAL9_OT_stop,
               DENTAL9_OT_preview, DENTAL9_OT_remove_preview, DENTAL9_OT_open_log,
               DENTAL9_OT_fit_box, DENTAL9_OT_toggle_region, DENTAL9_GGT_crop_box,
               DENTAL9_OT_diagnose, DENTAL9_OT_pick_path, DENTAL9_OT_select_all,
               DENTAL9_PT_panel, DENTAL9_PT_about)


# The module was called dental9_addon up to 1.0.1. Both register the same
# operators, panels and Scene.dental9, so with the old one still enabled each
# would quietly replace the other's classes and break on unregister. Refuse
# with a message the user can act on instead.
LEGACY_MODULE = "dental9_addon"


def register():
    if LEGACY_MODULE in bpy.context.preferences.addons:
        raise RuntimeError(
            f"An older {BRAND} is still enabled (folder \"{LEGACY_MODULE}\"). "
            "Remove it in Preferences > Add-ons, restart Blender, then enable "
            "this one.")
    _load_icons()
    for c in CLASSES_RNA:
        bpy.utils.register_class(c)
    bpy.types.Scene.dental9 = bpy.props.PointerProperty(type=Dental9Props)
    _add_draw_handlers()


def unregister():
    _kill_running()
    _remove_draw_handlers()
    del bpy.types.Scene.dental9
    for c in reversed(CLASSES_RNA):
        bpy.utils.unregister_class(c)
    _free_icons()
