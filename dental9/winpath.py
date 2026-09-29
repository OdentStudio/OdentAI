"""Non-ASCII paths on Windows: keep the native libraries out of trouble.

Python itself is fine with any path. The libraries underneath are not: ITK,
GDCM and VTK open files with a narrow `fopen`, which goes through the system
code page, and a folder called "КТ Иванов" may simply fail to open on a
machine whose code page is not UTF-8. Linux has no such problem — verified
with Cyrillic, spaces and dashes in the input folder, the zip name and the
output path.

Two lines of defence, in order:

1. The 8.3 short name (GetShortPathNameW). It is pure ASCII and points at the
   same file, so nothing is copied. Available on the system drive by default;
   other volumes may have short names disabled.
2. Staging through an ASCII folder. Input is copied there before reading;
   output is written there and moved into the real folder at the end. Costs a
   copy of the scan, which is seconds on an SSD.

The short name only helps for the top of the tree: the files and sub-folders
inside keep their long names, and a series exported as "王伟_0001.dcm" fails
just the same. So the whole tree is checked, and when anything inside is
non-ASCII the scan is staged with every name replaced (f000001.dcm, d001):
a DICOM series does not depend on its file names. Archives are unpacked the
same way, which also covers names stored in a local code page (an archive
from a Chinese Explorer keeps GBK names without the UTF-8 flag).

The staging folder is itself chosen to be ASCII: %TEMP% lives under the user
profile, and a user called Иван has a non-ASCII %TEMP%. Hence the fallbacks.
"""
import os
import shutil
import sys
import tempfile
from typing import Callable, Tuple


def _is_ascii(p: str) -> bool:
    try:
        p.encode("ascii")
        return True
    except UnicodeEncodeError:
        return False


def _short_name(p: str) -> str:
    """The 8.3 form, or "" when Windows cannot give one."""
    if sys.platform != "win32":
        return ""
    try:
        import ctypes
        buf = ctypes.create_unicode_buffer(4096)
        n = ctypes.windll.kernel32.GetShortPathNameW(p, buf, 4096)
        return buf.value if 0 < n < 4096 else ""
    except Exception:                                   # noqa: BLE001
        return ""


def _ascii_temp_root() -> str:
    """A folder to stage through that is ASCII on any Windows machine."""
    cands = []
    t = tempfile.gettempdir()
    cands.append(t if _is_ascii(t) else _short_name(t))
    pub = os.environ.get("PUBLIC", r"C:\Users\Public")
    cands.append(os.path.join(pub, "OdentAI", "tmp"))
    cands.append(os.path.join(os.environ.get("SystemDrive", "C:") + os.sep, "OdentAI_tmp"))
    for c in cands:
        if not c or not _is_ascii(c):
            continue
        try:
            os.makedirs(c, exist_ok=True)
            probe = tempfile.mkdtemp(prefix="odentai_", dir=c)
            return probe
        except OSError:
            continue
    raise RuntimeError("could not find an ASCII-only folder for temporary files")


def _tree_is_ascii(path: str) -> bool:
    for root, dirs, files in os.walk(path):
        for n in dirs + files:
            if not _is_ascii(n):
                return False
    return True


def _ascii_ext(name: str) -> str:
    e = _ext(name)
    return e if _is_ascii(e) else ""


def copy_tree_ascii(src: str, dst: str) -> int:
    """Copy a folder giving every file and sub-folder an ASCII name.

    Structure is kept (sub-folders become d001, d002, ...), files become
    f000001 with their extension when it is ASCII. Returns the file count.
    """
    n = 0
    dirmap = {src: dst}
    for root, dirs, files in os.walk(src):
        here = dirmap[root]
        os.makedirs(here, exist_ok=True)
        dirs.sort()
        for i, d in enumerate(dirs, 1):
            dirmap[os.path.join(root, d)] = os.path.join(here, f"d{i:03d}")
        for f in sorted(files):
            n += 1
            shutil.copy2(os.path.join(root, f), os.path.join(here, f"f{n:06d}{_ascii_ext(f)}"))
    return n


def extract_zip_ascii(zip_path: str, dst: str) -> int:
    """Unpack an archive with every entry renamed the way copy_tree_ascii does.

    Entry names are never trusted: besides being non-ASCII they may be
    mis-decoded (a local code page without the UTF-8 flag) or try to climb
    out of the folder.
    """
    import zipfile
    n = 0
    dirmap = {"": dst}
    with zipfile.ZipFile(zip_path) as z:
        for info in sorted(z.infolist(), key=lambda i: i.filename):
            if info.is_dir():
                continue
            parts = [p for p in info.filename.replace("\\", "/").split("/")[:-1]
                     if p not in ("", ".", "..")]
            key = ""
            for p in parts:
                child = key + "/" + p
                if child not in dirmap:
                    parent = dirmap[key]
                    k = sum(1 for d in dirmap if d.rsplit("/", 1)[0] == key and d != key) + 1
                    dirmap[child] = os.path.join(parent, f"d{k:03d}")
                key = child
            here = dirmap[key]
            os.makedirs(here, exist_ok=True)
            n += 1
            with z.open(info) as fin, open(
                    os.path.join(here, f"f{n:06d}{_ascii_ext(info.filename)}"), "wb") as fout:
                shutil.copyfileobj(fin, fout)
    return n


def temp_dir() -> str:
    """A fresh temporary folder that native libraries can open: ASCII on
    Windows, the ordinary one elsewhere. The caller removes it."""
    if sys.platform == "win32":
        return _ascii_temp_root()
    return tempfile.mkdtemp(prefix="dental9_")


def readable(path: str) -> Tuple[str, Callable[[], None]]:
    """A path the native readers can open, plus a cleanup to call afterwards.

    On Linux, and for ASCII paths anywhere, this is the path itself.
    """
    if sys.platform != "win32":
        return path, lambda: None
    is_dir = os.path.isdir(path)
    if _is_ascii(path) and (not is_dir or _tree_is_ascii(path)):
        return path, lambda: None
    if not is_dir or _tree_is_ascii(path):
        # only the way to the scan is non-ASCII: the short name costs nothing
        short = _short_name(path)
        if short and _is_ascii(short):
            return short, lambda: None
    stage = _ascii_temp_root()
    dst = os.path.join(stage, "input")
    if is_dir:
        copy_tree_ascii(path, dst)
    else:
        os.makedirs(dst, exist_ok=True)
        # keep the extension: the reader tells a zip from a volume by it
        dst = os.path.join(dst, "scan" + _ascii_ext(path))
        shutil.copy2(path, dst)
    return dst, lambda: shutil.rmtree(stage, ignore_errors=True)


def writable(outdir: str) -> Tuple[str, Callable[[], None]]:
    """A folder the native writers can write into, plus a finisher that moves
    the result into the real folder. Same rules as `readable`."""
    os.makedirs(outdir, exist_ok=True)
    if sys.platform != "win32" or _is_ascii(outdir):
        return outdir, lambda: None
    short = _short_name(outdir)
    if short and _is_ascii(short):
        return short, lambda: None
    stage = _ascii_temp_root()

    def finish():
        for root, dirs, files in os.walk(stage):
            rel = os.path.relpath(root, stage)
            target = outdir if rel == "." else os.path.join(outdir, rel)
            os.makedirs(target, exist_ok=True)
            for f in files:
                shutil.move(os.path.join(root, f), os.path.join(target, f))
        shutil.rmtree(stage, ignore_errors=True)

    return stage, finish


def _ext(path: str) -> str:
    low = path.lower()
    for e in (".nii.gz", ".img.gz", ".gipl.gz"):
        if low.endswith(e):
            return e
    return os.path.splitext(path)[1]
