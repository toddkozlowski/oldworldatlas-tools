"""One-off utility to bake SVG transforms into element geometry.

This script reads a source SVG, composes all ancestor/element transforms,
applies the final affine matrix directly to element coordinates/path data,
removes all `transform` attributes, and writes a sibling output SVG with
`_baked` added to the filename.
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from xml.etree import ElementTree as ET


# SVG affine matrix tuple:
# x' = a*x + c*y + e
# y' = b*x + d*y + f
TransformMatrix = Tuple[float, float, float, float, float, float]
IDENTITY_MATRIX: TransformMatrix = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)

DEFAULT_INPUT = Path(
    r"C:\Users\toddc\dev\personal\old-world-atlas\oldworldatlas-maps\OLD_WORLD_ATLAS.svg"
)


@dataclass
class BakeStats:
    """Tracks a compact summary of baking activity."""

    transforms_removed: int = 0
    elements_rewritten: int = 0
    path_elements_rewritten: int = 0


def format_float(value: float) -> str:
    """Format floats compactly while keeping enough precision for SVG output."""
    if abs(value) < 1e-12:
        value = 0.0
    text = f"{value:.6f}".rstrip("0").rstrip(".")
    return text if text else "0"


def local_tag(tag: str) -> str:
    """Return a namespace-stripped XML tag name."""
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def is_command_token(token: str) -> bool:
    """Return True when token is a single SVG path command letter."""
    return len(token) == 1 and token.isalpha()


def compose_transforms(outer: TransformMatrix, inner: TransformMatrix) -> TransformMatrix:
    """Compose two affine transforms: apply `inner` first, then `outer`."""
    a1, b1, c1, d1, e1, f1 = outer
    a2, b2, c2, d2, e2, f2 = inner
    return (
        a1 * a2 + c1 * b2,
        b1 * a2 + d1 * b2,
        a1 * c2 + c1 * d2,
        b1 * c2 + d1 * d2,
        a1 * e2 + c1 * f2 + e1,
        b1 * e2 + d1 * f2 + f1,
    )


def apply_point(x: float, y: float, matrix: TransformMatrix) -> Tuple[float, float]:
    """Apply affine transform to a point."""
    a, b, c, d, e, f = matrix
    return (a * x + c * y + e, b * x + d * y + f)


def apply_vector(x: float, y: float, matrix: TransformMatrix) -> Tuple[float, float]:
    """Apply only matrix linear terms to a vector (translation removed)."""
    a, b, c, d, _, _ = matrix
    return (a * x + c * y, b * x + d * y)


def parse_float_list(raw: str) -> List[float]:
    """Parse a coordinate list from an SVG attribute string."""
    return [
        float(v)
        for v in re.findall(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", raw)
    ]


def parse_transform_to_matrix(transform_str: str) -> TransformMatrix:
    """Parse an SVG transform attribute into a single affine matrix."""
    if not transform_str or not transform_str.strip():
        return IDENTITY_MATRIX

    result: TransformMatrix = IDENTITY_MATRIX
    for match in re.finditer(r"([a-zA-Z]+)\s*\(([^)]+)\)", transform_str):
        func = match.group(1)
        args = [
            float(v)
            for v in re.split(r"[,\s]+", match.group(2).strip())
            if v.strip()
        ]

        if func == "translate":
            tx = args[0] if len(args) >= 1 else 0.0
            ty = args[1] if len(args) >= 2 else 0.0
            part: TransformMatrix = (1.0, 0.0, 0.0, 1.0, tx, ty)
        elif func == "scale":
            sx = args[0] if len(args) >= 1 else 1.0
            sy = args[1] if len(args) >= 2 else sx
            part = (sx, 0.0, 0.0, sy, 0.0, 0.0)
        elif func == "matrix" and len(args) >= 6:
            part = (args[0], args[1], args[2], args[3], args[4], args[5])
        elif func == "rotate":
            angle = math.radians(args[0] if args else 0.0)
            cos_a = math.cos(angle)
            sin_a = math.sin(angle)
            if len(args) >= 3:
                cx, cy = args[1], args[2]
                part = (
                    cos_a,
                    sin_a,
                    -sin_a,
                    cos_a,
                    cx - cx * cos_a + cy * sin_a,
                    cy - cx * sin_a - cy * cos_a,
                )
            else:
                part = (cos_a, sin_a, -sin_a, cos_a, 0.0, 0.0)
        elif func == "skewX" and len(args) >= 1:
            t = math.tan(math.radians(args[0]))
            part = (1.0, 0.0, t, 1.0, 0.0, 0.0)
        elif func == "skewY" and len(args) >= 1:
            t = math.tan(math.radians(args[0]))
            part = (1.0, t, 0.0, 1.0, 0.0, 0.0)
        else:
            continue

        result = compose_transforms(outer=result, inner=part)

    return result


def matrix_is_axis_aligned(matrix: TransformMatrix, eps: float = 1e-12) -> bool:
    """Return True when transform has only scale/translate (no shear/rotation)."""
    _, b, c, _, _, _ = matrix
    return abs(b) < eps and abs(c) < eps


def transform_pair_attributes(elem: ET.Element, matrix: TransformMatrix) -> bool:
    """Transform common coordinate attribute pairs if they exist on the element."""
    changed = False
    for ax, ay in (("x1", "y1"), ("x2", "y2"), ("cx", "cy"), ("fx", "fy")):
        if ax in elem.attrib and ay in elem.attrib:
            x = float(elem.get(ax, "0"))
            y = float(elem.get(ay, "0"))
            tx, ty = apply_point(x, y, matrix)
            elem.set(ax, format_float(tx))
            elem.set(ay, format_float(ty))
            changed = True
    return changed


def transform_text_like_position_attributes(elem: ET.Element, matrix: TransformMatrix) -> bool:
    """Transform text/tspan x/y/dx/dy attribute value lists."""
    changed = False

    if "x" in elem.attrib and "y" in elem.attrib:
        x_vals = parse_float_list(elem.get("x", ""))
        y_vals = parse_float_list(elem.get("y", ""))
        if x_vals and y_vals:
            pair_count = min(len(x_vals), len(y_vals))
            tx_vals: List[str] = []
            ty_vals: List[str] = []
            for idx in range(pair_count):
                tx, ty = apply_point(x_vals[idx], y_vals[idx], matrix)
                tx_vals.append(format_float(tx))
                ty_vals.append(format_float(ty))
            elem.set("x", " ".join(tx_vals))
            elem.set("y", " ".join(ty_vals))
            changed = True
    elif "x" in elem.attrib:
        x_vals = parse_float_list(elem.get("x", ""))
        if x_vals:
            tx_vals: List[str] = []
            for x in x_vals:
                tx, _ = apply_point(x, 0.0, matrix)
                tx_vals.append(format_float(tx))
            elem.set("x", " ".join(tx_vals))
            changed = True
    elif "y" in elem.attrib:
        y_vals = parse_float_list(elem.get("y", ""))
        if y_vals:
            ty_vals: List[str] = []
            for y in y_vals:
                _, ty = apply_point(0.0, y, matrix)
                ty_vals.append(format_float(ty))
            elem.set("y", " ".join(ty_vals))
            changed = True

    if "dx" in elem.attrib and "dy" in elem.attrib:
        dx_vals = parse_float_list(elem.get("dx", ""))
        dy_vals = parse_float_list(elem.get("dy", ""))
        if dx_vals and dy_vals:
            pair_count = min(len(dx_vals), len(dy_vals))
            tdx_vals: List[str] = []
            tdy_vals: List[str] = []
            for idx in range(pair_count):
                tdx, tdy = apply_vector(dx_vals[idx], dy_vals[idx], matrix)
                tdx_vals.append(format_float(tdx))
                tdy_vals.append(format_float(tdy))
            elem.set("dx", " ".join(tdx_vals))
            elem.set("dy", " ".join(tdy_vals))
            changed = True

    return changed


def transform_points_attribute(elem: ET.Element, matrix: TransformMatrix) -> bool:
    """Transform polygon/polyline points attribute values."""
    points_raw = elem.get("points", "")
    coords = parse_float_list(points_raw)
    if len(coords) < 2:
        return False

    output_points: List[str] = []
    for idx in range(0, len(coords) - 1, 2):
        tx, ty = apply_point(coords[idx], coords[idx + 1], matrix)
        output_points.append(f"{format_float(tx)},{format_float(ty)}")

    elem.set("points", " ".join(output_points))
    return True


def transform_arc_parameters(
    rx: float,
    ry: float,
    x_axis_rotation: float,
    sweep_flag: int,
    matrix: TransformMatrix,
) -> Tuple[float, float, float, int]:
    """Approximate transformed SVG arc parameters for affine transforms."""
    a, b, c, d, _, _ = matrix
    phi = math.radians(x_axis_rotation)

    ux, uy = (rx * math.cos(phi), rx * math.sin(phi))
    vx, vy = (-ry * math.sin(phi), ry * math.cos(phi))

    tux, tuy = apply_vector(ux, uy, matrix)
    tvx, tvy = apply_vector(vx, vy, matrix)

    new_rx = max(math.hypot(tux, tuy), 1e-9)
    new_ry = max(math.hypot(tvx, tvy), 1e-9)
    new_rot = math.degrees(math.atan2(tuy, tux))

    det = a * d - b * c
    new_sweep = sweep_flag if det >= 0 else (1 - sweep_flag)
    return (new_rx, new_ry, new_rot, new_sweep)


def transform_path_d(path_d: str, matrix: TransformMatrix) -> str:
    """Transform SVG path data and return equivalent absolute-command data."""
    if not path_d.strip():
        return path_d

    tokens = re.findall(
        r"[AaCcHhLlMmQqSsTtVvZz]|[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?",
        path_d,
    )
    if not tokens:
        return path_d

    out: List[str] = []
    i = 0
    cmd: Optional[str] = None
    current = (0.0, 0.0)
    subpath_start = (0.0, 0.0)
    prev_cubic_ctrl: Optional[Tuple[float, float]] = None
    prev_quad_ctrl: Optional[Tuple[float, float]] = None
    prev_cmd: Optional[str] = None

    def has_params(min_count: int) -> bool:
        return i + min_count <= len(tokens)

    while i < len(tokens):
        token = tokens[i]
        if is_command_token(token):
            cmd = token
            i += 1
        elif cmd is None:
            raise ValueError("Path data does not start with a command")

        assert cmd is not None
        absolute = cmd.isupper()
        upper = cmd.upper()

        if upper == "Z":
            out.append("Z")
            current = subpath_start
            prev_cubic_ctrl = None
            prev_quad_ctrl = None
            prev_cmd = "Z"
            cmd = None
            continue

        if upper == "M":
            first = True
            while has_params(2) and not is_command_token(tokens[i]):
                x = float(tokens[i])
                y = float(tokens[i + 1])
                i += 2
                if not absolute:
                    x += current[0]
                    y += current[1]

                tx, ty = apply_point(x, y, matrix)
                if first:
                    out.extend(["M", format_float(tx), format_float(ty)])
                    subpath_start = (x, y)
                    first = False
                else:
                    out.extend(["L", format_float(tx), format_float(ty)])
                current = (x, y)

            prev_cubic_ctrl = None
            prev_quad_ctrl = None
            prev_cmd = "M"
            continue

        if upper == "L":
            while has_params(2) and not is_command_token(tokens[i]):
                x = float(tokens[i])
                y = float(tokens[i + 1])
                i += 2
                if not absolute:
                    x += current[0]
                    y += current[1]
                tx, ty = apply_point(x, y, matrix)
                out.extend(["L", format_float(tx), format_float(ty)])
                current = (x, y)

            prev_cubic_ctrl = None
            prev_quad_ctrl = None
            prev_cmd = "L"
            continue

        if upper == "H":
            while has_params(1) and not is_command_token(tokens[i]):
                x = float(tokens[i])
                i += 1
                if not absolute:
                    x += current[0]
                y = current[1]
                tx, ty = apply_point(x, y, matrix)
                out.extend(["L", format_float(tx), format_float(ty)])
                current = (x, y)

            prev_cubic_ctrl = None
            prev_quad_ctrl = None
            prev_cmd = "H"
            continue

        if upper == "V":
            while has_params(1) and not is_command_token(tokens[i]):
                y = float(tokens[i])
                i += 1
                if not absolute:
                    y += current[1]
                x = current[0]
                tx, ty = apply_point(x, y, matrix)
                out.extend(["L", format_float(tx), format_float(ty)])
                current = (x, y)

            prev_cubic_ctrl = None
            prev_quad_ctrl = None
            prev_cmd = "V"
            continue

        if upper == "C":
            while has_params(6) and not is_command_token(tokens[i]):
                x1 = float(tokens[i])
                y1 = float(tokens[i + 1])
                x2 = float(tokens[i + 2])
                y2 = float(tokens[i + 3])
                x = float(tokens[i + 4])
                y = float(tokens[i + 5])
                i += 6
                if not absolute:
                    x1 += current[0]
                    y1 += current[1]
                    x2 += current[0]
                    y2 += current[1]
                    x += current[0]
                    y += current[1]

                tx1, ty1 = apply_point(x1, y1, matrix)
                tx2, ty2 = apply_point(x2, y2, matrix)
                tx, ty = apply_point(x, y, matrix)
                out.extend(
                    [
                        "C",
                        format_float(tx1),
                        format_float(ty1),
                        format_float(tx2),
                        format_float(ty2),
                        format_float(tx),
                        format_float(ty),
                    ]
                )
                current = (x, y)
                prev_cubic_ctrl = (x2, y2)

            prev_quad_ctrl = None
            prev_cmd = "C"
            continue

        if upper == "S":
            while has_params(4) and not is_command_token(tokens[i]):
                x2 = float(tokens[i])
                y2 = float(tokens[i + 1])
                x = float(tokens[i + 2])
                y = float(tokens[i + 3])
                i += 4

                if prev_cmd in {"C", "S"} and prev_cubic_ctrl is not None:
                    x1 = 2.0 * current[0] - prev_cubic_ctrl[0]
                    y1 = 2.0 * current[1] - prev_cubic_ctrl[1]
                else:
                    x1, y1 = current

                if not absolute:
                    x2 += current[0]
                    y2 += current[1]
                    x += current[0]
                    y += current[1]

                tx1, ty1 = apply_point(x1, y1, matrix)
                tx2, ty2 = apply_point(x2, y2, matrix)
                tx, ty = apply_point(x, y, matrix)
                out.extend(
                    [
                        "C",
                        format_float(tx1),
                        format_float(ty1),
                        format_float(tx2),
                        format_float(ty2),
                        format_float(tx),
                        format_float(ty),
                    ]
                )
                current = (x, y)
                prev_cubic_ctrl = (x2, y2)

            prev_quad_ctrl = None
            prev_cmd = "S"
            continue

        if upper == "Q":
            while has_params(4) and not is_command_token(tokens[i]):
                x1 = float(tokens[i])
                y1 = float(tokens[i + 1])
                x = float(tokens[i + 2])
                y = float(tokens[i + 3])
                i += 4
                if not absolute:
                    x1 += current[0]
                    y1 += current[1]
                    x += current[0]
                    y += current[1]

                tx1, ty1 = apply_point(x1, y1, matrix)
                tx, ty = apply_point(x, y, matrix)
                out.extend(
                    [
                        "Q",
                        format_float(tx1),
                        format_float(ty1),
                        format_float(tx),
                        format_float(ty),
                    ]
                )
                current = (x, y)
                prev_quad_ctrl = (x1, y1)

            prev_cubic_ctrl = None
            prev_cmd = "Q"
            continue

        if upper == "T":
            while has_params(2) and not is_command_token(tokens[i]):
                x = float(tokens[i])
                y = float(tokens[i + 1])
                i += 2

                if prev_cmd in {"Q", "T"} and prev_quad_ctrl is not None:
                    x1 = 2.0 * current[0] - prev_quad_ctrl[0]
                    y1 = 2.0 * current[1] - prev_quad_ctrl[1]
                else:
                    x1, y1 = current

                if not absolute:
                    x += current[0]
                    y += current[1]

                tx1, ty1 = apply_point(x1, y1, matrix)
                tx, ty = apply_point(x, y, matrix)
                out.extend(
                    [
                        "Q",
                        format_float(tx1),
                        format_float(ty1),
                        format_float(tx),
                        format_float(ty),
                    ]
                )
                current = (x, y)
                prev_quad_ctrl = (x1, y1)

            prev_cubic_ctrl = None
            prev_cmd = "T"
            continue

        if upper == "A":
            while has_params(7) and not is_command_token(tokens[i]):
                rx = float(tokens[i])
                ry = float(tokens[i + 1])
                x_rot = float(tokens[i + 2])
                large_arc = int(float(tokens[i + 3]))
                sweep = int(float(tokens[i + 4]))
                x = float(tokens[i + 5])
                y = float(tokens[i + 6])
                i += 7
                if not absolute:
                    x += current[0]
                    y += current[1]

                trx, try_, tr_rot, tr_sweep = transform_arc_parameters(
                    rx=rx,
                    ry=ry,
                    x_axis_rotation=x_rot,
                    sweep_flag=sweep,
                    matrix=matrix,
                )
                tx, ty = apply_point(x, y, matrix)
                out.extend(
                    [
                        "A",
                        format_float(trx),
                        format_float(try_),
                        format_float(tr_rot),
                        str(large_arc),
                        str(tr_sweep),
                        format_float(tx),
                        format_float(ty),
                    ]
                )
                current = (x, y)

            prev_cubic_ctrl = None
            prev_quad_ctrl = None
            prev_cmd = "A"
            continue

        raise ValueError(f"Unsupported path command: {upper}")

    return " ".join(out)


def convert_circle_or_ellipse_to_path(
    elem: ET.Element,
    matrix: TransformMatrix,
    circle: bool,
) -> None:
    """Convert circle/ellipse into a transformed path approximation."""
    cx = float(elem.get("cx", "0"))
    cy = float(elem.get("cy", "0"))
    if circle:
        rx = float(elem.get("r", "0"))
        ry = rx
    else:
        rx = float(elem.get("rx", "0"))
        ry = float(elem.get("ry", "0"))

    points: List[Tuple[float, float]] = []
    samples = 64
    for idx in range(samples):
        theta = (2.0 * math.pi * idx) / samples
        x = cx + rx * math.cos(theta)
        y = cy + ry * math.sin(theta)
        points.append(apply_point(x, y, matrix))

    path_tokens: List[str] = []
    for idx, (px, py) in enumerate(points):
        if idx == 0:
            path_tokens.extend(["M", format_float(px), format_float(py)])
        else:
            path_tokens.extend(["L", format_float(px), format_float(py)])
    path_tokens.append("Z")

    elem.tag = elem.tag.replace("circle" if circle else "ellipse", "path")
    for key in ("cx", "cy", "r", "rx", "ry"):
        if key in elem.attrib:
            del elem.attrib[key]
    elem.set("d", " ".join(path_tokens))


def bake_geometry_for_element(elem: ET.Element, matrix: TransformMatrix, stats: BakeStats) -> bool:
    """Apply a transform matrix to the geometry represented by an SVG element."""
    tag = local_tag(elem.tag)
    changed = False

    if tag in {"g", "svg", "defs", "clipPath", "mask", "pattern", "symbol", "metadata"}:
        return False

    if tag == "path" and "d" in elem.attrib:
        elem.set("d", transform_path_d(elem.get("d", ""), matrix))
        stats.path_elements_rewritten += 1
        changed = True
    elif tag == "line":
        x1 = float(elem.get("x1", "0"))
        y1 = float(elem.get("y1", "0"))
        x2 = float(elem.get("x2", "0"))
        y2 = float(elem.get("y2", "0"))
        tx1, ty1 = apply_point(x1, y1, matrix)
        tx2, ty2 = apply_point(x2, y2, matrix)
        elem.set("x1", format_float(tx1))
        elem.set("y1", format_float(ty1))
        elem.set("x2", format_float(tx2))
        elem.set("y2", format_float(ty2))
        changed = True
    elif tag in {"polyline", "polygon"}:
        changed = transform_points_attribute(elem, matrix)
    elif tag == "rect":
        x = float(elem.get("x", "0"))
        y = float(elem.get("y", "0"))
        w = float(elem.get("width", "0"))
        h = float(elem.get("height", "0"))

        if matrix_is_axis_aligned(matrix):
            p1 = apply_point(x, y, matrix)
            p2 = apply_point(x + w, y + h, matrix)
            min_x = min(p1[0], p2[0])
            min_y = min(p1[1], p2[1])
            elem.set("x", format_float(min_x))
            elem.set("y", format_float(min_y))
            elem.set("width", format_float(abs(p2[0] - p1[0])))
            elem.set("height", format_float(abs(p2[1] - p1[1])))
            changed = True
        else:
            p1 = apply_point(x, y, matrix)
            p2 = apply_point(x + w, y, matrix)
            p3 = apply_point(x + w, y + h, matrix)
            p4 = apply_point(x, y + h, matrix)
            elem.tag = elem.tag.replace("rect", "path")
            for key in ("x", "y", "width", "height", "rx", "ry"):
                if key in elem.attrib:
                    del elem.attrib[key]
            elem.set(
                "d",
                " ".join(
                    [
                        "M",
                        format_float(p1[0]),
                        format_float(p1[1]),
                        "L",
                        format_float(p2[0]),
                        format_float(p2[1]),
                        "L",
                        format_float(p3[0]),
                        format_float(p3[1]),
                        "L",
                        format_float(p4[0]),
                        format_float(p4[1]),
                        "Z",
                    ]
                ),
            )
            changed = True
    elif tag == "circle":
        if matrix_is_axis_aligned(matrix):
            cx = float(elem.get("cx", "0"))
            cy = float(elem.get("cy", "0"))
            r = float(elem.get("r", "0"))
            tcx, tcy = apply_point(cx, cy, matrix)
            rvx, rvy = apply_vector(r, 0.0, matrix)
            sx = math.hypot(rvx, rvy)
            elem.set("cx", format_float(tcx))
            elem.set("cy", format_float(tcy))
            elem.set("r", format_float(sx))
            changed = True
        else:
            convert_circle_or_ellipse_to_path(elem, matrix, circle=True)
            changed = True
    elif tag == "ellipse":
        if matrix_is_axis_aligned(matrix):
            cx = float(elem.get("cx", "0"))
            cy = float(elem.get("cy", "0"))
            rx = float(elem.get("rx", "0"))
            ry = float(elem.get("ry", "0"))
            tcx, tcy = apply_point(cx, cy, matrix)
            rxv = apply_vector(rx, 0.0, matrix)
            ryv = apply_vector(0.0, ry, matrix)
            elem.set("cx", format_float(tcx))
            elem.set("cy", format_float(tcy))
            elem.set("rx", format_float(math.hypot(*rxv)))
            elem.set("ry", format_float(math.hypot(*ryv)))
            changed = True
        else:
            convert_circle_or_ellipse_to_path(elem, matrix, circle=False)
            changed = True
    elif tag in {"text", "tspan", "textPath"}:
        changed = transform_text_like_position_attributes(elem, matrix)
        changed = transform_pair_attributes(elem, matrix) or changed
    else:
        # Best-effort handling for elements with positional point attributes.
        changed = transform_pair_attributes(elem, matrix)

    # Handle common x/y (single-value) position attributes on many elements.
    if "x" in elem.attrib and "y" in elem.attrib and tag not in {"text", "tspan", "textPath"}:
        x_vals = parse_float_list(elem.get("x", ""))
        y_vals = parse_float_list(elem.get("y", ""))
        if x_vals and y_vals and len(x_vals) == len(y_vals):
            tx_vals: List[str] = []
            ty_vals: List[str] = []
            for idx in range(len(x_vals)):
                tx, ty = apply_point(x_vals[idx], y_vals[idx], matrix)
                tx_vals.append(format_float(tx))
                ty_vals.append(format_float(ty))
            elem.set("x", " ".join(tx_vals))
            elem.set("y", " ".join(ty_vals))
            changed = True

    # Handle vector offsets when present.
    if "dx" in elem.attrib and "dy" in elem.attrib:
        dx_vals = parse_float_list(elem.get("dx", ""))
        dy_vals = parse_float_list(elem.get("dy", ""))
        if dx_vals and dy_vals and len(dx_vals) == len(dy_vals):
            tdx_vals: List[str] = []
            tdy_vals: List[str] = []
            for idx in range(len(dx_vals)):
                tdx, tdy = apply_vector(dx_vals[idx], dy_vals[idx], matrix)
                tdx_vals.append(format_float(tdx))
                tdy_vals.append(format_float(tdy))
            elem.set("dx", " ".join(tdx_vals))
            elem.set("dy", " ".join(tdy_vals))
            changed = True

    if changed:
        stats.elements_rewritten += 1
    return changed


def bake_tree(elem: ET.Element, parent_matrix: TransformMatrix, stats: BakeStats) -> None:
    """Walk SVG tree, compose transforms, and rewrite element geometry in place."""
    own_transform = parse_transform_to_matrix(elem.get("transform", ""))
    current_matrix = compose_transforms(parent_matrix, own_transform)

    if "transform" in elem.attrib:
        del elem.attrib["transform"]
        stats.transforms_removed += 1

    bake_geometry_for_element(elem, current_matrix, stats)
    for child in list(elem):
        bake_tree(child, current_matrix, stats)


def register_all_namespaces(svg_path: Path) -> None:
    """Preserve existing namespace prefixes when writing the modified SVG."""
    namespaces: Dict[str, str] = {}
    for _, node in ET.iterparse(str(svg_path), events=("start-ns",)):
        prefix, uri = node
        if prefix not in namespaces:
            namespaces[prefix] = uri

    for prefix, uri in namespaces.items():
        if prefix == "xml":
            continue
        ET.register_namespace(prefix, uri)


def output_path_for(svg_path: Path) -> Path:
    """Return sibling path with `_baked` suffix before extension."""
    return svg_path.with_name(f"{svg_path.stem}_baked{svg_path.suffix}")


def bake_svg_transforms(svg_path: Path) -> Path:
    """Bake all SVG transforms into geometry and write the output file."""
    if not svg_path.exists():
        raise FileNotFoundError(f"Input SVG does not exist: {svg_path}")

    register_all_namespaces(svg_path)
    tree = ET.parse(str(svg_path))
    root = tree.getroot()

    stats = BakeStats()
    bake_tree(root, IDENTITY_MATRIX, stats)

    out_path = output_path_for(svg_path)
    tree.write(str(out_path), encoding="utf-8", xml_declaration=True)

    print(f"Input : {svg_path}")
    print(f"Output: {out_path}")
    print(f"Transforms removed : {stats.transforms_removed}")
    print(f"Elements rewritten : {stats.elements_rewritten}")
    print(f"Path elements baked: {stats.path_elements_rewritten}")
    return out_path


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for one-off transform baking."""
    parser = argparse.ArgumentParser(description="Bake SVG transforms into geometry.")
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to the input SVG.",
    )
    return parser.parse_args()


def main() -> None:
    """Entrypoint for running one-off SVG transform baking."""
    args = parse_args()
    bake_svg_transforms(args.input)


if __name__ == "__main__":
    main()
