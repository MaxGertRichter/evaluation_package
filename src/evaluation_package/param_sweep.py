from evaluation_package.filetools import load_yaml, save_yaml
import os
from itertools import product
from typing import Any, Dict, List, Optional, Iterable, Tuple
from pathlib import Path

def set_by_dotted_path(d: Dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    cur = d
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value

def get_by_dotted_path(d: Dict[str, Any], dotted: str, default: Any=None) -> Any:
    cur = d
    for p in dotted.split("."):
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur

def format_value_for_filename(val: Any) -> str:
    if isinstance(val, float):
        if abs(val) >= 1e6 or (0 < abs(val) < 1e-3):
            return f"{val:.3e}".replace("+", "")
        s = f"{val:.6f}".rstrip("0").rstrip(".")
        return s if s else "0"
    if isinstance(val, int):
        return str(val)
    if isinstance(val, str):
        s = val.strip().replace(" ", "")
        s = s.replace("/", "-per-").replace("\\", "-").replace(":", "")
        return s
    return str(val)

def make_filename(base_stem: str,
                  sweep_items: Dict[str, Any],
                  index: Optional[int] = None,
                  pad_width: Optional[int] = None) -> str:
    """Filename like: 01__CASR_sensitivity__mixing_frequency-5.000e08__N-16.yaml"""
    parts = []
    for k, v in sweep_items.items():
        short = k.split(".")[-1]
        parts.append(f"{short}-{format_value_for_filename(v)}")
    middle = "__".join(parts)
    core = f"{base_stem}__{middle}.yaml" if middle else f"{base_stem}.yaml"
    if index is not None:
        if pad_width is None:
            pad_width = max(2, len(str(index)))  # default at least 2 digits
        prefix = f"{str(index).zfill(pad_width)}__"
        return prefix + core
    return core

def _cartesian_dicts(sweeps: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    keys = list(sweeps.keys())
    combos = []
    for combo in product(*[sweeps[k] for k in keys]):
        combos.append(dict(zip(keys, combo)))
    return combos

def _zip_dicts(sweeps: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    keys = list(sweeps.keys())
    lengths = [len(sweeps[k]) for k in keys]
    if len(set(lengths)) != 1:
        raise ValueError(f"Zip mode requires equal list lengths, got {lengths}")
    out = []
    for i in range(lengths[0]):
        out.append({k: sweeps[k][i] for k in keys})
    return out

def _sort_combos(combos: List[Dict[str, Any]], order_by: Optional[List[str]]) -> List[Dict[str, Any]]:
    if not order_by:
        return combos
    def keyf(c):
        return tuple(c.get(k) for k in order_by)
    return sorted(combos, key=keyf)

def generate_sweeps(
    base_yaml: Path,
    out_dir: Path,
    sweeps: Dict[str, List[Any]],
    mode: str = "cartesian",             # "cartesian" or "zip"
    prefix_index: bool = True,           # add 01__, 02__, ...
    start_index: int = 1,                # first index value
    pad_width: Optional[int] = None,     # None -> auto from total count (>=2)
    order_by: Optional[List[str]] = None # optional dotted keys to sort by before numbering
):
    out_dir.mkdir(parents=True, exist_ok=True)
    base_stem = base_yaml.stem

    # 1) Materialize combinations in the generation order
    if mode == "cartesian":
        combos = _cartesian_dicts(sweeps)
    elif mode == "zip":
        combos = _zip_dicts(sweeps)
    else:
        raise ValueError("mode must be 'cartesian' or 'zip'")

    # 2) Optional sorting by one or more dotted keys
    combos = _sort_combos(combos, order_by)

    # 3) Figure out padding once (total count)
    total = len(combos)
    if pad_width is None:
        pad_width = max(2, len(str(start_index + total - 1)))

    # 4) Generate files in that fixed order, adding index prefix if requested
    results = []
    for i, values in enumerate(combos, start=start_index):
        cfg = load_yaml(base_yaml)
        for dotted_key, v in values.items():
            set_by_dotted_path(cfg, dotted_key, v)
        fname = make_filename(
            base_stem,
            values,
            index=i if prefix_index else None,
            pad_width=pad_width
        )
        out_path = out_dir / fname
        save_yaml(cfg, out_path)
        results.append(out_path)

    return results