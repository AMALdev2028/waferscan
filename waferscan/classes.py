"""The 9 WM-811K failure-pattern classes, in one fixed order used everywhere
(model outputs, metrics, API). Aliases map the spellings found in the wild
(the original LSWMD.pkl uses 'none'/'Loc', the Roboflow export uses 'Local')."""

CLASSES = ["Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc", "Near-full", "Random", "Scratch", "None"]
NUM_CLASSES = len(CLASSES)
INDEX = {c: i for i, c in enumerate(CLASSES)}

_ALIASES = {
    "none": "None", "normal": "None", "local": "Loc", "loc": "Loc",
    "near-full": "Near-full", "near_full": "Near-full", "nearfull": "Near-full",
    "edge-loc": "Edge-Loc", "edgeloc": "Edge-Loc", "edge-ring": "Edge-Ring", "edgering": "Edge-Ring",
    "center": "Center", "donut": "Donut", "random": "Random", "scratch": "Scratch",
}


def canonical(name: str) -> str:
    """'Local' -> 'Loc', 'none' -> 'None'. Raises KeyError for unknown labels."""
    key = name.strip().lower().replace(" ", "-")
    if key in _ALIASES:
        return _ALIASES[key]
    raise KeyError(f"unknown wafer class label: {name!r}")
