"""The model's nine classes: names, colours, speck thresholds.

Order and ids must match dataset.json of Dataset003_Dental9. They cannot be
changed here — only together with retraining.
"""
from typing import NamedTuple, Tuple


class Klass(NamedTuple):
    id: int
    key: str            # file name and add-on property name
    name: str           # what the user sees
    color: Tuple[float, float, float]
    min_cm3: float      # smaller than this is a speck and is dropped


# Thresholds chosen by anatomy, not by how the picture looks: the whole
# mandibular canal is about 1 cm3, so its threshold is an order of magnitude
# below the teeth, and the soft palate is lower still. Too high a threshold
# silently eats the class.
CLASSES = (
    Klass(1, "mandible",        "Mandible",         (0.90, 0.86, 0.78), 1.0),
    Klass(2, "upper_skull",     "Upper Skull",      (0.86, 0.83, 0.76), 1.0),
    Klass(3, "upper_teeth",     "Upper Teeth",      (1.00, 0.99, 0.94), 0.02),
    Klass(4, "lower_teeth",     "Lower Teeth",      (0.98, 0.97, 0.92), 0.02),
    Klass(5, "mandibular_canal","Mandibular Canal", (0.90, 0.30, 0.30), 0.02),
    Klass(6, "maxillary_sinus", "Maxillary Sinus",  (0.40, 0.70, 0.95), 0.20),
    Klass(7, "nasal_cavity",    "Nasal Cavity",     (0.55, 0.80, 0.90), 0.20),
    Klass(8, "pharynx",         "Pharynx",          (0.60, 0.55, 0.85), 0.50),
    Klass(9, "soft_palate",     "Soft Palate",      (0.95, 0.60, 0.60), 0.10),
)

BY_KEY = {k.key: k for k in CLASSES}
BY_ID = {k.id: k for k in CLASSES}


def resolve(names) -> list:
    """User strings -> classes. Understands a key, a number and 'all'."""
    if not names:
        return list(CLASSES)
    out = []
    for n in names:
        n = str(n).strip().lower()
        if n == "all":
            return list(CLASSES)
        if n.isdigit() and int(n) in BY_ID:
            out.append(BY_ID[int(n)])
        elif n in BY_KEY:
            out.append(BY_KEY[n])
        else:
            raise SystemExit(f"unknown class: {n}. Available: "
                             + ", ".join(k.key for k in CLASSES))
    return out
