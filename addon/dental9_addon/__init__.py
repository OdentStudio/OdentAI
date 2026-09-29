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
    "version": (1, 0, 0),
    "blender": (3, 0, 0),
    "location": "3D View > Sidebar (N) > OdentAI",
    "description": "See more. Plan better. CBCT (DICOM) to teeth, jaws, canals, "
                   "sinuses and pharynx — AI segmentation for Blender",
    "warning": "Beta",
    "category": "Import-Export",
}

# Brand. The name goes everywhere the user can see it; the internal package
# and executable keep the name dental9 — nobody sees those.
BRAND = "OdentAI Segment"
BRAND_SHORT = "OdentAI"
VERSION_LABEL = "1.0 Beta"
TAGLINE = "See more. Plan better."
AUTHORS = ["Dr. Ilya Fomenko DMD", "Dr. Essaid Issam Dakir DMD", "Dr. Krasouski Dmitry DMD"]
# One universe with ODent: the same community, the same link.
TELEGRAM_LINK = "https://t.me/odent_blender"

import json
import math
import os
import queue
import subprocess
import sys
import threading

import bpy
from bpy.props import (BoolProperty, EnumProperty, FloatProperty, IntProperty,
                       StringProperty)
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


def _default_model() -> str:
    """The weights live next to the binary as a separate file, not inside it:
    they weigh over a quarter of a gigabyte and must be replaceable without a
    rebuild."""
    p = os.path.join(ADDON_DIR, "bin", "dental9", "dental9.onnx")
    return p if os.path.isfile(p) else ""


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


def _teeth_model(prefs) -> str:
    """Explicit path from the preferences, otherwise teeth_fdi.onnx next to the
    main model — the two ship together in bin/dental9/."""
    p = bpy.path.abspath(prefs.teeth_model_path) if prefs.teeth_model_path else ""
    if p:
        return p
    m = bpy.path.abspath(prefs.model_path)
    return os.path.join(os.path.dirname(m), "teeth_fdi.onnx") if m else ""


class Dental9Prefs(AddonPreferences):
    bl_idname = __name__

    model_path: StringProperty(
        name="Model file (.onnx)", subtype="FILE_PATH", default=_default_model(),
        description="Network weights. To update the model, replace this file and "
                    "the dental9.json next to it — both change together")
    teeth_model_path: StringProperty(
        name="Teeth model (.onnx)", subtype="FILE_PATH", default="",
        description="Weights of the tooth-numbering model for \"Separate teeth\". "
                    "Empty = teeth_fdi.onnx next to the main model")
    device: EnumProperty(
        name="Compute on", default="auto",
        items=[("auto", "Automatic", "Use the GPU if onnxruntime can see one"),
               ("dml", "DirectML (GPU)", "Windows, any graphics card"),
               ("cuda", "CUDA (GPU)", "NVIDIA only"),
               ("cpu", "CPU", "Many times slower, fallback only")])

    def draw(self, context):
        _draw_brand_header(self.layout)
        r = self.layout.row()
        r.alignment = "CENTER"
        r.scale_y = 0.6
        r.label(text=TAGLINE)
        _draw_telegram(self.layout)
        self.layout.separator()
        col = self.layout.column()
        col.prop(self, "model_path")
        col.prop(self, "device")
        exe = _exe()
        if not os.path.isfile(exe):
            col.label(text=f"executable missing: {exe}", icon="ERROR")
        m = bpy.path.abspath(self.model_path)
        if m and os.path.isfile(m):
            info = _model_info(m)
            col.label(text=f"model: {os.path.getsize(m)/1e6:.0f} MB, "
                           f"trained on {info.get('dataset', '?')}, "
                           f"{info.get('spacing', '?')} mm, epoch {info.get('epoch', '?')}",
                      icon="CHECKMARK")
        else:
            col.label(text="model file not found", icon="ERROR")
        col.prop(self, "teeth_model_path")
        tm = _teeth_model(self)
        if tm and os.path.isfile(tm):
            info = _model_info(tm)
            col.label(text=f"teeth model: {os.path.getsize(tm)/1e6:.0f} MB, "
                           f"epoch {info.get('epoch', '?')}", icon="CHECKMARK")
        else:
            col.label(text="teeth model not found — \"Separate teeth\" will not work",
                      icon="INFO")
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
                    "a dental9_out folder next to the scan")
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
        model = bpy.path.abspath(_prefs(context).model_path)
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
        for k in ("platform", "onnxruntime", "gpu_names", "model", "gpu", "cpu"):
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


class DENTAL9_OT_segment(Operator):
    bl_idname = "dental9.segment"
    bl_label = "Segment"
    bl_description = "Run the segmentation and load the surfaces into the scene"
    bl_options = {"REGISTER"}

    _timer = None
    _proc = None
    _q = None
    _outdir = ""
    _tmp_out = False
    _selected = []
    _teeth = False

    def _start(self, context):
        p = context.scene.dental9
        pr = _prefs(context)
        exe = _exe()
        _ensure_executable(exe)
        if not os.path.isfile(exe):
            self.report({"ERROR"}, f"executable missing: {exe}")
            return False
        src = bpy.path.abspath(p.input_path)
        if not src or not os.path.exists(src):
            self.report({"ERROR"}, "no scan selected")
            return False
        model = bpy.path.abspath(pr.model_path)
        if not model or not os.path.isfile(model):
            self.report({"ERROR"}, "model file not found, set it in the add-on "
                                   "preferences")
            return False

        self._selected = [k for k, _l, _h, _r, _a, _o in CLASSES if getattr(p, k)]
        if not self._selected:
            self.report({"ERROR"}, "no classes selected")
            return False

        # Without "keep" the files are only a way to get the meshes into the
        # scene: they go to a temporary folder that is removed after loading.
        if p.keep_files:
            self._outdir = bpy.path.abspath(p.out_dir) or os.path.join(
                os.path.dirname(src), "dental9_out")
            os.makedirs(self._outdir, exist_ok=True)
            self._tmp_out = False
        else:
            import tempfile
            self._outdir = tempfile.mkdtemp(prefix="odentai_")
            self._tmp_out = True

        # The second canal pass is always on and has no switch: it fires only
        # when the canal comes out broken, and on a healthy scan it costs
        # nothing. A switch would only offer a way to make things worse.
        cmd = [exe, src, "-o", self._outdir, "-m", model,
               "--device", pr.device, "--taubin", str(p.smooth),
               "-c", *self._selected]
        if p.decimate > 0:
            cmd += ["--decimate", f"{p.decimate:.2f}"]
        self._teeth = bool(p.separate_teeth)
        if self._teeth:
            tm = _teeth_model(pr)
            if not tm or not os.path.isfile(tm):
                self.report({"ERROR"}, "teeth model (teeth_fdi.onnx) not found, set "
                                       "it in the add-on preferences")
                return False
            cmd += ["--separate-teeth", "--teeth-model", tm]

        # CREATE_NO_WINDOW: without it a black console window pops up over
        # Blender on every run under Windows.
        flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        self._proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            universal_newlines=True, bufsize=1, encoding="utf-8",
            errors="replace", creationflags=flags)
        self._q = queue.Queue()
        threading.Thread(target=self._reader, daemon=True).start()
        return True

    def _reader(self):
        for line in self._proc.stdout:
            self._q.put(line.rstrip())
        self._q.put(None)

    def modal(self, context, event):
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
            if line:
                _console(line)
                context.workspace.status_text_set(line[:120])
        if not done:
            return {"RUNNING_MODAL"}

        self._finish(context)
        rc = self._proc.wait()
        if rc != 0:
            self.report({"ERROR"}, "segmentation failed, see the console")
            self._drop_tmp()
            return {"CANCELLED"}
        try:
            self._load(context)
        finally:
            self._drop_tmp()
        return {"FINISHED"}

    def _drop_tmp(self):
        """Remove the working folder when the user did not ask to keep files."""
        if self._tmp_out and self._outdir:
            import shutil
            shutil.rmtree(self._outdir, ignore_errors=True)
            self._outdir = ""

    def _finish(self, context):
        if self._timer:
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None
        context.workspace.status_text_set(None)

    def _load(self, context):
        report = {}
        rp = os.path.join(self._outdir, "report.json")
        if os.path.isfile(rp):
            with open(rp, encoding="utf-8") as f:
                report = json.load(f)

        coll = bpy.data.collections.new(BRAND_SHORT)
        context.scene.collection.children.link(coll)
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
        coll = bpy.data.collections.new("Teeth")
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
        if not self._start(context):
            return {"CANCELLED"}
        wm = context.window_manager
        self._timer = wm.event_timer_add(0.2, window=context.window)
        wm.modal_handler_add(self)
        return {"RUNNING_MODAL"}


class DENTAL9_OT_select_all(Operator):
    bl_idname = "dental9.select_all"
    bl_label = "Select"
    state: BoolProperty(default=True)

    def execute(self, context):
        for key, _l, _h, _r, _a, _o in CLASSES:
            setattr(context.scene.dental9, key, self.state)
        return {"FINISHED"}


class DENTAL9_PT_panel(Panel):
    bl_label = BRAND
    bl_idname = "DENTAL9_PT_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = BRAND_SHORT

    def draw(self, context):
        p = context.scene.dental9
        _draw_brand_header(self.layout)
        col = self.layout.column()
        row = col.row(align=True)
        row.prop(p, "input_path")
        row.operator("dental9.pick_path", text="", icon="FILEBROWSER")

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

        box = self.layout.box()
        row = box.row(align=True)
        row.label(text="Hardware")
        row.operator("dental9.diagnose", text="", icon="FILE_REFRESH")
        if not _DIAG:
            box.label(text="not checked yet", icon="QUESTION")
            box.operator("dental9.diagnose", icon="SYSTEM")
        else:
            g = _DIAG.get("gpu", {})
            box.label(text=f"computing on: {g.get('device_used', '?')}",
                      icon="CHECKMARK" if "GPU" in g.get("device_used", "")
                      else "ERROR")
            names = _DIAG.get("gpu_names")
            if names:
                box.label(text=names[0][:34], icon="DOT")
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
CLASSES_RNA = (Dental9Prefs, Dental9Props, DENTAL9_OT_segment,
               DENTAL9_OT_diagnose, DENTAL9_OT_pick_path, DENTAL9_OT_select_all,
               DENTAL9_PT_panel, DENTAL9_PT_about)


def register():
    _load_icons()
    for c in CLASSES_RNA:
        bpy.utils.register_class(c)
    bpy.types.Scene.dental9 = bpy.props.PointerProperty(type=Dental9Props)


def unregister():
    del bpy.types.Scene.dental9
    for c in reversed(CLASSES_RNA):
        bpy.utils.unregister_class(c)
    _free_icons()
