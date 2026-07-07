#!/usr/bin/env python3
"""
swmm_to_epanet_v5.py

Convert a basic EPA SWMM .inp network into an EPANET .inp skeleton.

This version is designed for real SWMM exports that contain object IDs with
spaces or IDs that overflow SWMM's pretty fixed-width table columns. It uses
section-specific semantic parsing instead of naive whitespace splitting.

Mapped:
- [JUNCTIONS]  -> [JUNCTIONS]
- [OUTFALLS]   -> [RESERVOIRS], when elevation/head can be parsed
- [STORAGE]    -> [TANKS], approximate where possible
- [CONDUITS] + [XSECTIONS] circular/force_main -> [PIPES]
- [PUMPS] + PUMP3 [CURVES] -> EPANET HEAD pump curves where possible
- [COORDINATES] and [VERTICES]
- Basic [OPTIONS] flow units

Big caveat, because reality insists: SWMM and EPANET are different hydraulic
engines. This produces a reviewable EPANET skeleton, not a validated water
model.

python swmm_to_epanet_v5.py input_swmm.inp output_epanet.inp --warnings warnings.txt

"""

from __future__ import annotations

import argparse
import math
import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

SECTION_RE = re.compile(r"^\s*\[([^\]]+)\]\s*$")
SHAPES = {
    "CIRCULAR", "FORCE_MAIN", "FILLED_CIRCULAR", "RECT_CLOSED", "RECT_OPEN",
    "TRAPEZOIDAL", "TRIANGULAR", "HORIZ_ELLIPSE", "VERT_ELLIPSE", "ARCH",
    "PARABOLIC", "POWER", "RECT_TRIANGULAR", "RECT_ROUND", "MODBASKETHANDLE",
    "EGG", "HORSESHOE", "GOTHIC", "CATENARY", "SEMIELLIPTICAL", "BASKETHANDLE",
    "SEMICIRCULAR", "IRREGULAR", "CUSTOM"
}
OUTFALL_TYPES = {"FREE", "NORMAL", "FIXED", "TIDAL", "TIMESERIES"}
CURVE_TYPES = {
    "STORAGE", "SHAPE", "DIVERSION", "TIDAL", "RATING", "CONTROL",
    "PUMP1", "PUMP2", "PUMP3", "PUMP4", "PUMP5", "WEIR"
}
PUMP_CURVE_TYPES = {"PUMP1", "PUMP2", "PUMP3", "PUMP4", "PUMP5"}


@dataclass
class RawSection:
    name: str
    lines: List[str] = field(default_factory=list)
    header: List[str] = field(default_factory=list)


@dataclass
class SwmmModel:
    sections: Dict[str, RawSection]


class IdMapper:
    def __init__(self) -> None:
        self.original_to_safe: Dict[str, str] = {}
        self.safe_used: Dict[str, str] = {}
        self.warnings: List[str] = []

    def safe(self, original: Any) -> str:
        text = str(original).strip()
        if not text:
            text = "Unnamed"
        if text in self.original_to_safe:
            return self.original_to_safe[text]

        safe = re.sub(r"\s+", "_", text)
        safe = re.sub(r"[^A-Za-z0-9_.:-]", "_", safe)
        safe = safe.strip("_") or "Unnamed"
        base = safe[:28] if len(safe) > 31 else safe
        safe = base
        i = 2
        while safe in self.safe_used and self.safe_used[safe] != text:
            suffix = f"_{i}"
            safe = f"{base[:31-len(suffix)]}{suffix}"
            i += 1

        self.original_to_safe[text] = safe
        self.safe_used[safe] = text
        if safe != text:
            self.warnings.append(f"Renamed ID '{text}' to '{safe}' for EPANET compatibility")
        return safe


def read_swmm(path: Path) -> SwmmModel:
    sections: Dict[str, RawSection] = {}
    current: Optional[RawSection] = None
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        raw = raw.rstrip("\r\n")
        m = SECTION_RE.match(raw.strip())
        if m:
            name = m.group(1).upper()
            current = RawSection(name=name)
            sections[name] = current
            continue
        if current is None:
            continue
        if not raw.strip():
            continue
        if raw.lstrip().startswith(";;"):
            current.header.append(raw.lstrip()[2:].rstrip())
            continue
        if raw.lstrip().startswith(";"):
            continue
        current.lines.append(raw)
    return SwmmModel(sections=sections)


def remove_comment(line: str) -> str:
    # Good enough for SWMM IDs. Avoid making semicolon-in-quote a whole lifestyle.
    return line.split(";", 1)[0].strip()


def tokens(line: str) -> List[str]:
    body = remove_comment(line)
    if not body:
        return []
    try:
        return shlex.split(body)
    except ValueError:
        return body.split()


def to_float(s: Any) -> Optional[float]:
    try:
        return float(str(s))
    except Exception:
        return None


def is_float_token(s: str) -> bool:
    return to_float(s) is not None


def join_name(parts: Sequence[str]) -> str:
    return " ".join(p for p in parts if p).strip()


def find_first_numeric_index(parts: Sequence[str]) -> Optional[int]:
    for i, p in enumerate(parts):
        if is_float_token(p):
            return i
    return None


def parse_numeric_suffix_row(line: str, min_numbers: int = 1) -> Optional[Tuple[str, List[float]]]:
    """
    Parse rows like: multi word name 12.3 4.5 0.0 ...

    Important: SWMM IDs can themselves be numeric, so do not assume the first
    numeric token starts the data fields. Instead, find the earliest split that
    leaves at least one name token and an all-numeric suffix.
    """
    parts = tokens(line)
    for i in range(1, len(parts)):
        suffix = parts[i:]
        if len(suffix) < min_numbers:
            continue
        nums = [to_float(p) for p in suffix]
        if all(v is not None for v in nums):
            return join_name(parts[:i]), [v for v in nums if v is not None]
    return None


def get_option(model: SwmmModel, key: str, default: str = "") -> str:
    sec = model.sections.get("OPTIONS")
    if not sec:
        return default
    for line in sec.lines:
        p = tokens(line)
        if len(p) >= 2 and p[0].upper() == key.upper():
            return p[1]
    return default


def is_metric_flow_units(flow_units: str) -> bool:
    return flow_units.upper() in {"CMS", "LPS", "MLD", "CMH", "CMD"}


def epanet_units_from_swmm(flow_units: str, warnings: List[str]) -> str:
    units = flow_units.upper()
    if units == "CMS":
        warnings.append("SWMM FLOW_UNITS is CMS. EPANET does not support CMS, so output UNITS is set to LPS.")
        return "LPS"
    if units in {"CFS", "GPM", "MGD", "LPS", "MLD"}:
        return units
    if units in {"CMH", "CMD"}:
        warnings.append(f"SWMM FLOW_UNITS is {units}. EPANET output UNITS is set to LPS and pump curve flows are converted.")
        return "LPS"
    warnings.append(f"Unknown/unsupported SWMM FLOW_UNITS '{flow_units}'. Output UNITS set to LPS.")
    return "LPS"


def flow_to_cms(q: float, units: str) -> float:
    u = units.upper()
    factors = {
        "CMS": 1.0,
        "LPS": 0.001,
        "MLD": 1000.0 / 86400.0,
        "CMH": 1.0 / 3600.0,
        "CMD": 1.0 / 86400.0,
        "CFS": 0.028316846592,
        "GPM": 0.0000630901964,
        "MGD": 0.0438126364,
    }
    if u not in factors:
        raise ValueError(f"Unsupported flow units: {units}")
    return q * factors[u]


def cms_to_flow(q_cms: float, units: str) -> float:
    u = units.upper()
    factors = {
        "CMS": 1.0,
        "LPS": 0.001,
        "MLD": 1000.0 / 86400.0,
        "CMH": 1.0 / 3600.0,
        "CMD": 1.0 / 86400.0,
        "CFS": 0.028316846592,
        "GPM": 0.0000630901964,
        "MGD": 0.0438126364,
    }
    if u not in factors:
        raise ValueError(f"Unsupported flow units: {units}")
    return q_cms / factors[u]


def convert_flow_units(q: float, from_units: str, to_units: str) -> float:
    return cms_to_flow(flow_to_cms(q, from_units), to_units)


def swmm_diam_to_epanet(geom1: float, metric: bool) -> float:
    return geom1 * 1000.0 if metric else geom1 * 12.0


def manning_to_dw_roughness(n: float, metric: bool) -> float:
    if n <= 0:
        return 0.001 if metric else 0.00328
    ks_m = (26.0 * n) ** 6
    return ks_m * 1000.0 if metric else ks_m * 3.280839895 * 1000.0


def parse_junctions(sec: Optional[RawSection], warnings: List[str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if not sec:
        return out
    for line in sec.lines:
        parsed = parse_numeric_suffix_row(line, min_numbers=1)
        if not parsed:
            warnings.append(f"[JUNCTIONS] skipped row: {line.strip()}")
            continue
        name, nums = parsed
        out[name] = nums[0]
    return out


def parse_outfalls(sec: Optional[RawSection], warnings: List[str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if not sec:
        return out
    for line in sec.lines:
        p = tokens(line)
        if not p:
            continue

        # Valid simple form is: Name Elevation Type ...
        # Names may contain spaces, so find an outfall type whose immediately
        # preceding token is numeric. That token is the elevation.
        candidates = []
        for i, part in enumerate(p):
            if part.upper() in OUTFALL_TYPES and i >= 1 and to_float(p[i - 1]) is not None:
                candidates.append(i)
        if not candidates:
            warnings.append(f"[OUTFALLS] skipped row because elevation/type pattern is not valid: {line.strip()}")
            continue

        type_idx = candidates[0]
        name_parts = p[: type_idx - 1]
        # If the supposed name still contains an outfall type token, the export
        # has already shifted columns. Skipping is safer than inventing a bad ID.
        if any(x.upper() in OUTFALL_TYPES for x in name_parts):
            warnings.append(f"[OUTFALLS] skipped ambiguous shifted row: {line.strip()}")
            continue

        elev = to_float(p[type_idx - 1])
        if elev is None or not name_parts:
            warnings.append(f"[OUTFALLS] skipped row because elevation is not numeric: {line.strip()}")
            continue
        name = join_name(name_parts)
        out_type = p[type_idx].upper()
        head = elev
        if out_type == "FIXED" and type_idx + 1 < len(p):
            fixed = to_float(p[type_idx + 1])
            if fixed is not None:
                head = fixed
        out[name] = head
    return out


def parse_storage(sec: Optional[RawSection], warnings: List[str]) -> Dict[str, Tuple[float, float, float, float]]:
    out: Dict[str, Tuple[float, float, float, float]] = {}
    if not sec:
        return out
    for line in sec.lines:
        parsed = parse_numeric_suffix_row(line, min_numbers=3)
        if not parsed:
            warnings.append(f"[STORAGE] skipped row: {line.strip()}")
            continue
        name, nums = parsed
        elev, max_depth, init_depth = nums[0], nums[1], nums[2]
        diameter = 1.0
        if len(nums) >= 4 and nums[3] > 0:
            diameter = math.sqrt(4.0 * nums[3] / math.pi)
        out[name] = (elev, init_depth, max_depth, diameter)
    return out


def parse_xsections(sec: Optional[RawSection], warnings: List[str]) -> Dict[str, Tuple[str, float]]:
    out: Dict[str, Tuple[str, float]] = {}
    if not sec:
        return out
    for line in sec.lines:
        p = tokens(line)
        shape_idx = None
        for i, part in enumerate(p):
            if part.upper() in SHAPES:
                shape_idx = i
                break
        if shape_idx is None or shape_idx == 0 or shape_idx + 1 >= len(p):
            warnings.append(f"[XSECTIONS] skipped row: {line.strip()}")
            continue
        name = join_name(p[:shape_idx])
        geom1 = to_float(p[shape_idx + 1])
        if geom1 is None:
            warnings.append(f"[XSECTIONS] skipped row, invalid Geom1: {line.strip()}")
            continue
        out[name] = (p[shape_idx].upper(), geom1)
    return out


def parse_curves(sec: Optional[RawSection], warnings: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    Parse SWMM [CURVES].

    SWMM export normally writes one X,Y pair per row:
      Name Type X Y
      Name      X Y

    Numeric curve names are valid, so this parser reads the last two tokens as
    X and Y and treats anything before them as Name[/Type].
    """
    curves: Dict[str, Dict[str, Any]] = {}
    if not sec:
        return curves

    for line in sec.lines:
        p = tokens(line)
        if len(p) < 3:
            continue
        x = to_float(p[-2])
        y = to_float(p[-1])
        if x is None or y is None:
            warnings.append(f"[CURVES] skipped row, invalid X/Y: {line.strip()}")
            continue

        name_type = p[:-2]
        if not name_type:
            warnings.append(f"[CURVES] skipped row, missing curve name: {line.strip()}")
            continue

        curve_type: Optional[str] = None
        if name_type[-1].upper() in CURVE_TYPES:
            curve_type = name_type[-1].upper()
            name_parts = name_type[:-1]
        else:
            name_parts = name_type

        name = join_name(name_parts)
        if not name:
            warnings.append(f"[CURVES] skipped row, missing curve name: {line.strip()}")
            continue

        rec = curves.setdefault(name, {"type": None, "points": []})
        if curve_type:
            if rec["type"] and rec["type"] != curve_type:
                warnings.append(f"[CURVES] curve {name} changes type from {rec['type']} to {curve_type}; using {rec['type']}")
            else:
                rec["type"] = curve_type
        rec["points"].append((x, y))

    return curves


def split_known_node_pair(parts: List[str], known_nodes: set[str]) -> Optional[Tuple[str, str, List[str]]]:
    """
    Given prefix after link name and before numeric fields, split into from_node + to_node.
    Prefer longest from-node match, then longest to-node match.
    """
    n = len(parts)
    for i in range(n, 0, -1):
        from_node = join_name(parts[:i])
        if from_node not in known_nodes:
            continue
        remaining = parts[i:]
        for j in range(len(remaining), 0, -1):
            to_node = join_name(remaining[:j])
            if to_node in known_nodes:
                extra = remaining[j:]
                return from_node, to_node, extra
    return None


def parse_link_row(line: str, link_names: set[str], known_nodes: set[str], section: str, warnings: List[str]) -> Optional[Tuple[str, str, str, List[float]]]:
    """
    Parse [CONDUITS] rows using the conduit length/roughness signature.

    This is deliberately not "first numeric token" parsing because real SWMM
    exports often have numeric IDs and occasional over-width names. For a
    conduit, the reliable pair is usually:
        Length Roughness
    where roughness is Manning n, normally a small positive number.
    """
    p = tokens(line)
    if len(p) < 5:
        warnings.append(f"[{section}] skipped row: {line.strip()}")
        return None

    def split_prefix(prefix: List[str]) -> Optional[Tuple[str, str, str]]:
        # Prefer a link-name prefix found in [XSECTIONS].
        for link_end in range(len(prefix) - 2, 0, -1):
            link = join_name(prefix[:link_end])
            if link not in link_names:
                continue
            rest = prefix[link_end:]
            pair = split_known_node_pair(rest, known_nodes)
            if pair:
                n1, n2, _ = pair
                return link, n1, n2
            # Fallback: from-node known, to-node is whatever remains. This is
            # needed when the to-node is a malformed/over-width outfall ID.
            for from_end in range(len(rest), 0, -1):
                from_node = join_name(rest[:from_end])
                if from_node in known_nodes and from_end < len(rest):
                    to_node = join_name(rest[from_end:])
                    return link, from_node, to_node
        return None

    # Find plausible Length/Roughness starts. Manning n is normally around
    # 0.009-0.03, but keep the upper bound generous for dirty sewer models.
    candidates: List[int] = []
    for i in range(1, len(p) - 1):
        length = to_float(p[i])
        roughness = to_float(p[i + 1])
        if length is None or roughness is None:
            continue
        if length > 0 and 0 < roughness <= 0.2:
            candidates.append(i)

    for length_idx in candidates:
        prefix = p[:length_idx]
        split = split_prefix(prefix)
        if not split:
            continue
        link, n1, n2 = split
        nums: List[float] = []
        for x in p[length_idx:]:
            v = to_float(x)
            if v is None:
                break
            nums.append(v)
        if len(nums) >= 2:
            return link, n1, n2, nums

    # Last fallback for unusual roughness values: original numeric-suffix method.
    numeric_idx = None
    for i, part in enumerate(p):
        if i == 0:  # numeric link IDs are common; not a data field
            continue
        if is_float_token(part):
            numeric_idx = i
            break
    if numeric_idx is not None and numeric_idx >= 3:
        prefix = p[:numeric_idx]
        nums = []
        for x in p[numeric_idx:]:
            v = to_float(x)
            if v is None:
                break
            nums.append(v)
        split = split_prefix(prefix)
        if split:
            link, n1, n2 = split
            return link, n1, n2, nums
        return p[0], p[1], join_name(p[2:numeric_idx]), nums

    warnings.append(f"[{section}] skipped row: {line.strip()}")
    return None


def parse_pump_row(line: str, known_nodes: set[str], curve_names: set[str], warnings: List[str]) -> Optional[Tuple[str, str, str, Optional[str], str]]:
    """Parse a SWMM [PUMPS] row into pump, from_node, to_node, curve, status."""
    p = tokens(line)
    if len(p) < 4:
        warnings.append(f"[PUMPS] skipped row: {line.strip()}")
        return None

    status = "OPEN"
    status_idx = len(p)
    for i, part in enumerate(p):
        if part.upper() in {"ON", "OFF"}:
            status = "OPEN" if part.upper() == "ON" else "CLOSED"
            status_idx = i
            break
    prefix = p[:status_idx]

    # Preferred: find a known curve name at the end of the prefix, then split the
    # remaining prefix into pump name + known from/to node pair.
    for curve_start in range(len(prefix) - 1, 0, -1):
        curve_name = join_name(prefix[curve_start:])
        if curve_name not in curve_names:
            continue
        before_curve = prefix[:curve_start]
        for pump_end in range(1, len(before_curve) - 1):
            pump_name = join_name(before_curve[:pump_end])
            node_parts = before_curve[pump_end:]
            pair = split_known_node_pair(node_parts, known_nodes)
            if pair:
                n1, n2, _ = pair
                return pump_name, n1, n2, curve_name, status

    # Fallback: normal SWMM export without spaces in the three first IDs.
    if len(prefix) >= 4:
        pump_name, n1, n2, curve_name = prefix[0], prefix[1], prefix[2], join_name(prefix[3:])
        return pump_name, n1, n2, curve_name if curve_name else None, status

    warnings.append(f"[PUMPS] skipped row: {line.strip()}")
    return None


def parse_coordinates(sec: Optional[RawSection], known_ids: set[str], warnings: List[str], section: str) -> List[Tuple[str, float, float]]:
    out: List[Tuple[str, float, float]] = []
    if not sec:
        return out
    for line in sec.lines:
        p = tokens(line)
        if len(p) < 3:
            continue
        # Last two numeric tokens are coordinates; everything before is name.
        x = to_float(p[-2])
        y = to_float(p[-1])
        if x is None or y is None:
            warnings.append(f"[{section}] skipped row: {line.strip()}")
            continue
        name = join_name(p[:-2])
        # If fixed export split a name oddly but the joined name exists, this is fine.
        if name:
            out.append((name, x, y))
    return out


def fmt(x: Any) -> str:
    if isinstance(x, float):
        if abs(x) >= 1000:
            return f"{x:.3f}".rstrip("0").rstrip(".")
        return f"{x:.6g}"
    return str(x)


def section_rows(rows: Iterable[Iterable[Any]], widths: Tuple[int, ...]) -> List[str]:
    out = []
    for row in rows:
        cells = [fmt(v) for v in row]
        out.append(" ".join(cells[i].ljust(widths[i] if i < len(widths) else 12) for i in range(len(cells))).rstrip())
    return out


def convert(model: SwmmModel, *, include_pumps: bool = True, include_non_circular: bool = False) -> Tuple[str, List[str]]:
    warnings: List[str] = []
    ids = IdMapper()

    flow_units = get_option(model, "FLOW_UNITS", "LPS")
    epanet_units = epanet_units_from_swmm(flow_units, warnings)
    metric = is_metric_flow_units(flow_units)

    junction_elev = parse_junctions(model.sections.get("JUNCTIONS"), warnings)
    reservoirs_head = parse_outfalls(model.sections.get("OUTFALLS"), warnings)
    storage = parse_storage(model.sections.get("STORAGE"), warnings)
    xsections = parse_xsections(model.sections.get("XSECTIONS"), warnings)
    curves = parse_curves(model.sections.get("CURVES"), warnings)

    junctions: Dict[str, List[Any]] = {name: [ids.safe(name), elev, 0] for name, elev in junction_elev.items()}
    reservoirs: Dict[str, List[Any]] = {name: [ids.safe(name), head] for name, head in reservoirs_head.items()}
    tanks: Dict[str, List[Any]] = {
        name: [ids.safe(name), elev, init, 0, max_depth, diameter, 0]
        for name, (elev, init, max_depth, diameter) in storage.items()
    }
    known_nodes = set(junctions) | set(reservoirs) | set(tanks)

    pipes: List[List[Any]] = []
    skipped_links = 0
    conduit_sec = model.sections.get("CONDUITS")
    if conduit_sec:
        link_names = set(xsections)
        for line in conduit_sec.lines:
            parsed = parse_link_row(line, link_names, known_nodes, "CONDUITS", warnings)
            if not parsed:
                skipped_links += 1
                continue
            name, n1, n2, nums = parsed
            if len(nums) < 2:
                warnings.append(f"[CONDUITS] skipped {name}: could not parse length and roughness from {line.strip()}")
                skipped_links += 1
                continue
            length, manning_n = nums[0], nums[1]
            x = xsections.get(name)
            if not x:
                warnings.append(f"[CONDUITS] skipped {name}: no matching [XSECTIONS] row")
                skipped_links += 1
                continue
            shape, geom1 = x
            if geom1 <= 0:
                warnings.append(f"[CONDUITS] skipped {name}: invalid diameter/Geom1")
                skipped_links += 1
                continue
            if shape not in {"CIRCULAR", "FORCE_MAIN"} and not include_non_circular:
                warnings.append(f"[CONDUITS] skipped {name}: non-circular shape {shape}")
                skipped_links += 1
                continue
            if n1 not in known_nodes:
                junctions[n1] = [ids.safe(n1), 0, 0]
                known_nodes.add(n1)
                warnings.append(f"Created missing node {n1} at elevation 0")
            if n2 not in known_nodes:
                junctions[n2] = [ids.safe(n2), 0, 0]
                known_nodes.add(n2)
                warnings.append(f"Created missing node {n2} at elevation 0")
            pipes.append([
                ids.safe(name), ids.safe(n1), ids.safe(n2), length,
                swmm_diam_to_epanet(geom1, metric), manning_to_dw_roughness(manning_n, metric), 0, "Open"
            ])

    pumps: List[List[Any]] = []
    epanet_pump_curves: Dict[str, List[Tuple[float, float]]] = {}
    if include_pumps:
        pump_sec = model.sections.get("PUMPS")
        curve_names = set(curves)
        if pump_sec:
            for line in pump_sec.lines:
                parsed_pump = parse_pump_row(line, known_nodes, curve_names, warnings)
                if not parsed_pump:
                    continue
                name, n1, n2, curve_name, status = parsed_pump
                if n1 not in known_nodes:
                    junctions[n1] = [ids.safe(n1), 0, 0]
                    known_nodes.add(n1)
                    warnings.append(f"Created missing pump node {n1} at elevation 0")
                if n2 not in known_nodes:
                    junctions[n2] = [ids.safe(n2), 0, 0]
                    known_nodes.add(n2)
                    warnings.append(f"Created missing pump node {n2} at elevation 0")

                if curve_name and curve_name in curves:
                    curve_rec = curves[curve_name]
                    curve_type = (curve_rec.get("type") or "").upper()
                    points = curve_rec.get("points", [])
                    if curve_type == "PUMP3" and points:
                        # SWMM PUMP3 curves are head difference (X) vs flow (Y).
                        # EPANET HEAD pump curves are flow (X) vs head gain (Y).
                        epanet_curve_id = ids.safe(f"PC_{curve_name}")
                        converted_points: List[Tuple[float, float]] = []
                        for head, flow in points:
                            try:
                                q = convert_flow_units(flow, flow_units, epanet_units)
                            except ValueError:
                                q = flow
                            converted_points.append((q, head))
                        converted_points.sort(key=lambda xy: xy[0])

                        # Remove duplicate/non-increasing flow points because EPANET rejects them.
                        cleaned: List[Tuple[float, float]] = []
                        last_q: Optional[float] = None
                        for q, h in converted_points:
                            if last_q is not None and q <= last_q:
                                warnings.append(f"Pump curve {curve_name}: dropped duplicate/non-increasing flow point Q={q}, H={h}")
                                continue
                            cleaned.append((q, h))
                            last_q = q
                        if len(cleaned) >= 1:
                            epanet_pump_curves[epanet_curve_id] = cleaned
                            pumps.append([ids.safe(name), ids.safe(n1), ids.safe(n2), "HEAD", epanet_curve_id])
                            if status == "CLOSED":
                                warnings.append(f"Pump {name} is OFF in SWMM. EPANET [PUMPS] does not carry initial status here; add a [STATUS] row if required.")
                        else:
                            pumps.append([ids.safe(name), ids.safe(n1), ids.safe(n2), "POWER", 1.0])
                            warnings.append(f"Pump {name}: curve {curve_name} had no usable points; used placeholder POWER 1.0")
                    elif curve_type in PUMP_CURVE_TYPES:
                        pumps.append([ids.safe(name), ids.safe(n1), ids.safe(n2), "POWER", 1.0])
                        warnings.append(f"Pump {name}: SWMM {curve_type} curve {curve_name} is not directly convertible to an EPANET head-flow curve; used placeholder POWER 1.0")
                    else:
                        pumps.append([ids.safe(name), ids.safe(n1), ids.safe(n2), "POWER", 1.0])
                        warnings.append(f"Pump {name}: referenced curve {curve_name} is type {curve_type or 'UNKNOWN'}, not PUMP3; used placeholder POWER 1.0")
                else:
                    pumps.append([ids.safe(name), ids.safe(n1), ids.safe(n2), "POWER", 1.0])
                    warnings.append(f"Pump {name}: no matching pump curve found; used placeholder POWER 1.0")

    coords = parse_coordinates(model.sections.get("COORDINATES"), known_nodes, warnings, "COORDINATES")
    vertices = parse_coordinates(model.sections.get("VERTICES"), set(xsections), warnings, "VERTICES")

    warnings = warnings + ids.warnings
    warnings.insert(0, f"Converted junctions={len(junctions)}, reservoirs={len(reservoirs)}, tanks={len(tanks)}, pipes={len(pipes)}, pumps={len(pumps)}, pump_curves={len(epanet_pump_curves)}, skipped_links={skipped_links}")

    lines: List[str] = []
    lines.append("[TITLE]")
    lines.append("; Converted from EPA SWMM to EPANET skeleton")
    lines.append("; Review hydraulics carefully before use.")
    lines.append("")

    lines.append("[OPTIONS]")
    lines.extend(section_rows([
        ["UNITS", epanet_units],
        ["HEADLOSS", "D-W"],
        ["HYDRAULICS", "NONE"],
        ["QUALITY", "NONE"],
        ["TRIALS", 40],
        ["ACCURACY", 0.001],
    ], widths=(14, 14)))
    lines.append("")

    lines.append("[JUNCTIONS]")
    lines.append(";ID               Elevation      Demand")
    lines.extend(section_rows(junctions.values(), widths=(18, 14, 14)))
    lines.append("")

    if reservoirs:
        lines.append("[RESERVOIRS]")
        lines.append(";ID               Head")
        lines.extend(section_rows(reservoirs.values(), widths=(18, 14)))
        lines.append("")

    if tanks:
        lines.append("[TANKS]")
        lines.append(";ID               Elevation      InitLevel      MinLevel       MaxLevel       Diameter      MinVol")
        lines.extend(section_rows(tanks.values(), widths=(18, 14, 14, 14, 14, 14, 14)))
        lines.append("")

    lines.append("[PIPES]")
    lines.append(";ID               Node1          Node2          Length        Diameter      Roughness     MinorLoss     Status")
    lines.extend(section_rows(pipes, widths=(18, 14, 14, 13, 13, 13, 13, 10)))
    lines.append("")

    if pumps:
        lines.append("[PUMPS]")
        lines.append(";ID               Node1          Node2          Parameters")
        lines.extend(section_rows(pumps, widths=(18, 14, 14, 10, 10)))
        lines.append("")

    if epanet_pump_curves:
        lines.append("[CURVES]")
        lines.append(";ID               Flow          Head")
        curve_rows: List[List[Any]] = []
        for curve_id, pts in epanet_pump_curves.items():
            for q, h in pts:
                curve_rows.append([curve_id, q, h])
        lines.extend(section_rows(curve_rows, widths=(18, 14, 14)))
        lines.append("")

    if coords:
        lines.append("[COORDINATES]")
        lines.append(";Node             X-Coord        Y-Coord")
        lines.extend(section_rows(([ids.safe(n), x, y] for n, x, y in coords), widths=(18, 14, 14)))
        lines.append("")

    if vertices:
        lines.append("[VERTICES]")
        lines.append(";Link             X-Coord        Y-Coord")
        lines.extend(section_rows(([ids.safe(n), x, y] for n, x, y in vertices), widths=(18, 14, 14)))
        lines.append("")

    lines.append("[END]")
    lines.append("")
    return "\n".join(lines), warnings


def convert_file(input_path: Path, output_path: Path, warning_path: Optional[Path] = None, *, include_non_circular: bool = False, include_pumps: bool = True) -> List[str]:
    model = read_swmm(input_path)
    text, warnings = convert(model, include_pumps=include_pumps, include_non_circular=include_non_circular)
    output_path.write_text(text, encoding="utf-8")
    if warning_path:
        warning_path.write_text("\n".join(warnings) + "\n", encoding="utf-8")
    return warnings


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a basic EPA SWMM .inp file to an EPANET .inp skeleton.")
    parser.add_argument("input", type=Path, help="Input EPA SWMM .inp file")
    parser.add_argument("output", type=Path, nargs="?", help="Output EPANET .inp file")
    parser.add_argument("--warnings", type=Path, help="Optional warning report path")
    parser.add_argument("--include-non-circular", action="store_true", help="Use Geom1 as equivalent diameter for non-circular conduits")
    parser.add_argument("--no-pumps", action="store_true", help="Do not include pumps")
    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(f"Input file not found: {args.input}")
    output = args.output or args.input.with_name(args.input.stem + "_epanet.inp")
    warnings = convert_file(args.input, output, args.warnings, include_non_circular=args.include_non_circular, include_pumps=not args.no_pumps)

    print(f"Wrote EPANET file: {output}")
    if args.warnings:
        print(f"Wrote warnings:    {args.warnings}")
    print("\nSummary")
    print("-------")
    print(warnings[0] if warnings else "No warnings")
    if len(warnings) > 1:
        print(f"Warnings: {len(warnings)}")
        print("\nFirst warnings")
        print("--------------")
        for w in warnings[1:21]:
            print(f"- {w}")
        if len(warnings) > 21:
            print(f"... {len(warnings) - 21} more")


if __name__ == "__main__":
    main()
