"""What the machine has and what the run is using: RAM, video memory, peaks.

Exists for one reason: when a run fails on someone else's machine, the log
has to say what it ran into — video memory, system memory or the driver —
without a remote session. Every function here is best effort: a missing
answer is None, never an exception, because diagnostics must not be the
thing that breaks a run.
"""
import os
import subprocess
import sys
from typing import List, Optional, Tuple

_NO_WINDOW = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0


def ram_gb() -> Tuple[Optional[float], Optional[float]]:
    """(total, available) system memory in GB."""
    try:
        if sys.platform == "win32":
            import ctypes

            class MS(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong),
                            ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong),
                            ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong),
                            ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong),
                            ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            m = MS()
            m.dwLength = ctypes.sizeof(MS)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
            return m.ullTotalPhys / 2 ** 30, m.ullAvailPhys / 2 ** 30
        total = avail = None
        with open("/proc/meminfo", encoding="ascii") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    total = int(line.split()[1]) / 2 ** 20
                elif line.startswith("MemAvailable:"):
                    avail = int(line.split()[1]) / 2 ** 20
        return total, avail
    except Exception:                                   # noqa: BLE001
        return None, None


def process_peak_gb(committed: bool = False) -> Optional[float]:
    """The most memory this process has held at once. On a failed run this is
    the number that tells whether it ran out of RAM.

    Resident (working set) by default. `committed` is what the process
    reserved, page file included — much larger under DirectML, whose driver
    commits system memory for its own heaps without touching it.
    """
    try:
        if sys.platform == "win32":
            import ctypes
            from ctypes import wintypes

            class PMC(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD),
                            ("PageFaultCount", wintypes.DWORD),
                            ("PeakWorkingSetSize", ctypes.c_size_t),
                            ("WorkingSetSize", ctypes.c_size_t),
                            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                            ("PagefileUsage", ctypes.c_size_t),
                            ("PeakPagefileUsage", ctypes.c_size_t)]
            c = PMC()
            c.cb = ctypes.sizeof(PMC)
            k32 = ctypes.windll.kernel32
            k32.GetCurrentProcess.restype = wintypes.HANDLE
            # argtypes are required: the pseudo-handle is -1 as a pointer, and
            # the default int conversion overflows on it.
            gpmi = ctypes.windll.psapi.GetProcessMemoryInfo
            gpmi.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
            gpmi.restype = wintypes.BOOL
            ok = gpmi(k32.GetCurrentProcess(), ctypes.byref(c), c.cb)
            if not ok:
                return None
            return (c.PeakPagefileUsage if committed else c.PeakWorkingSetSize) / 2 ** 30
        if committed:
            return None
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20
    except Exception:                                   # noqa: BLE001
        return None


def _registry_gpus() -> List[dict]:
    """Adapters and their dedicated memory from the display class key.

    Not Win32_VideoController.AdapterRAM: that field is 32-bit and reports
    every card with more than 4 GB as 4 GB — exactly the cards in question.
    """
    import winreg
    out = []
    root = (r"SYSTEM\CurrentControlSet\Control\Class"
            r"\{4d36e968-e325-11ce-bfc1-08002be10318}")
    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, root) as cls:
        i = 0
        while True:
            try:
                sub = winreg.EnumKey(cls, i)
            except OSError:
                break
            i += 1
            if not sub.isdigit():
                continue
            try:
                with winreg.OpenKey(cls, sub) as k:
                    name = winreg.QueryValueEx(k, "DriverDesc")[0]
                    mem = None
                    for v in ("HardwareInformation.qwMemorySize",
                              "HardwareInformation.MemorySize"):
                        try:
                            raw = winreg.QueryValueEx(k, v)[0]
                            mem = (int.from_bytes(raw, "little") if isinstance(raw, bytes)
                                   else int(raw))
                            break
                        except OSError:
                            continue
                    try:
                        drv = winreg.QueryValueEx(k, "DriverVersion")[0]
                    except OSError:
                        drv = None
                out.append({"name": name,
                            "vram_gb": round(mem / 2 ** 30, 1) if mem else None,
                            "driver": drv})
            except OSError:
                continue
    return out


def _nvidia_smi() -> List[dict]:
    """Live numbers for NVIDIA cards: the driver ships nvidia-smi on Windows too."""
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,driver_version",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, creationflags=_NO_WINDOW)
        if r.returncode != 0:
            return []
        out = []
        for line in r.stdout.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 4:
                out.append({"name": parts[0], "vram_gb": round(int(parts[1]) / 1024, 1),
                            "used_gb": round(int(parts[2]) / 1024, 1),
                            "driver": parts[3]})
        return out
    except Exception:                                   # noqa: BLE001
        return []


def gpus() -> List[dict]:
    """Graphics adapters with their video memory, as much as can be learned."""
    smi = _nvidia_smi()
    try:
        reg = _registry_gpus() if sys.platform == "win32" else []
    except Exception:                                   # noqa: BLE001
        reg = []
    # nvidia-smi knows the live usage, the registry knows the other vendors.
    names = {g["name"] for g in smi}
    return smi + [g for g in reg if g["name"] not in names]


def vram_used() -> Optional[str]:
    """Video memory in use right now, if the driver can tell (NVIDIA only)."""
    smi = _nvidia_smi()
    if not smi:
        return None
    return ", ".join(f"{g['used_gb']}/{g['vram_gb']} GB" for g in smi)


def describe_gpu(g: dict) -> str:
    s = g["name"]
    if g.get("vram_gb"):
        s += f", {g['vram_gb']} GB"
    if g.get("driver"):
        s += f", driver {g['driver']}"
    return s


def snapshot() -> str:
    """One line for the log: free RAM, the process peak and live VRAM."""
    total, avail = ram_gb()
    parts = []
    if avail is not None and total:
        parts.append(f"RAM free {avail:.1f}/{total:.1f} GB")
    peak = process_peak_gb()
    if peak is not None:
        com = process_peak_gb(committed=True)
        parts.append(f"process peak {peak:.1f} GB"
                     + (f" (committed {com:.1f})" if com is not None else ""))
    v = vram_used()
    if v:
        parts.append(f"VRAM used {v}")
    return ", ".join(parts) or "memory figures unavailable"


def cpu_name() -> str:
    try:
        if sys.platform == "win32":
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as k:
                return winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
        with open("/proc/cpuinfo", encoding="utf-8", errors="replace") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except Exception:                                   # noqa: BLE001
        pass
    return f"{os.cpu_count()} cores"
