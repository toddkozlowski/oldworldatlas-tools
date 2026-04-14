"""Extract Points of Interest from the map SVG into a CSV file.

Produces a CSV with columns: name,type,coordinates
Where `type` is the sub-layer label under the main "Points of Interest" layer
and `coordinates` are the Inkscape/display coordinates (lon,lat) with all
SVG group transforms removed.

This script borrows the transform-handling approach from
`scripts/process_map_svg.py` to ensure coordinates are flattened correctly.
"""
from __future__ import annotations

import csv
import math
import re
from pathlib import Path
from typing import Tuple, List
from xml.etree import ElementTree as ET


NS = {
    'svg': 'http://www.w3.org/2000/svg',
    'inkscape': 'http://www.inkscape.org/namespaces/inkscape'
}


# Inkscape conversion constants are copied from process_map_svg.py
INKSCAPE_VB_X = 428.530
INKSCAPE_SCALE = 0.017504
INKSCAPE_C_Y = 3347.85


TransformMatrix = Tuple[float, float, float, float, float, float]
IDENTITY_MATRIX: TransformMatrix = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)


def svg_to_geo(svg_x: float, svg_y: float) -> Tuple[float, float]:
    lon = (svg_x - INKSCAPE_VB_X) * INKSCAPE_SCALE
    lat = (INKSCAPE_C_Y - svg_y) * INKSCAPE_SCALE
    return lon, lat


def _compose_transforms(outer: TransformMatrix, inner: TransformMatrix) -> TransformMatrix:
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


def _apply_transform_matrix(x: float, y: float, matrix: TransformMatrix) -> Tuple[float, float]:
    a, b, c, d, e, f = matrix
    return (a * x + c * y + e, b * x + d * y + f)


def _parse_transform_to_matrix(transform_str: str) -> TransformMatrix:
    if not transform_str:
        return IDENTITY_MATRIX

    result: TransformMatrix = IDENTITY_MATRIX
    for match in re.finditer(r'([a-zA-Z]+)\s*\(([^)]+)\)', transform_str):
        func = match.group(1)
        args_str = match.group(2).strip()
        try:
            args = [float(v) for v in re.split(r'[,\s]+', args_str) if v]
        except ValueError:
            continue

        if func == 'translate':
            tx = args[0] if len(args) >= 1 else 0.0
            ty = args[1] if len(args) >= 2 else 0.0
            t: TransformMatrix = (1.0, 0.0, 0.0, 1.0, tx, ty)
        elif func == 'scale':
            sx = args[0] if len(args) >= 1 else 1.0
            sy = args[1] if len(args) >= 2 else sx
            t = (sx, 0.0, 0.0, sy, 0.0, 0.0)
        elif func == 'matrix':
            if len(args) >= 6:
                t = (args[0], args[1], args[2], args[3], args[4], args[5])
            else:
                continue
        elif func == 'rotate':
            angle = math.radians(args[0]) if args else 0.0
            cos_a = math.cos(angle)
            sin_a = math.sin(angle)
            if len(args) >= 3:
                cx, cy = args[1], args[2]
                t = (cos_a, sin_a, -sin_a, cos_a,
                     cx - cx * cos_a + cy * sin_a,
                     cy - cx * sin_a - cy * cos_a)
            else:
                t = (cos_a, sin_a, -sin_a, cos_a, 0.0, 0.0)
        else:
            continue

        result = _compose_transforms(outer=result, inner=t)

    return result


def _get_text_element_label(elem: ET.Element) -> str:
    text_content: List[str] = []
    if elem.tag == f"{{{NS['svg']}}}text":
        for tspan in elem.findall(f"{{{NS['svg']}}}tspan"):
            if tspan.text:
                text_content.append(tspan.text.strip())
        if not text_content and elem.text:
            text_content.append(elem.text.strip())
    return " ".join(text_content).strip()


def extract_pois(svg_path: Path) -> List[dict]:
    tree = ET.parse(str(svg_path))
    root = tree.getroot()

    pois = []

    # Find the top-level Points of Interest group by inkscape:label
    poi_layer = None
    for g in root.findall(f".//{{{NS['svg']}}}g"):
        if g.get(f"{{{NS['inkscape']}}}label") == "Points of Interest":
            poi_layer = g
            break

    if poi_layer is None:
        raise RuntimeError("Points of Interest layer not found in SVG")

    # Build initial transform for poi_layer (absorb its transform if any)
    layer_transform_str = poi_layer.get("transform", "")
    layer_matrix = _parse_transform_to_matrix(layer_transform_str) if layer_transform_str else IDENTITY_MATRIX

    # Iterate sub-layers (each sub-layer is a category/type)
    for sub in poi_layer:
        if sub.tag != f"{{{NS['svg']}}}g":
            continue
        sub_label = sub.get(f"{{{NS['inkscape']}}}label") or sub.get("id") or ""

        # Compose sub-layer transform onto layer_matrix
        sub_transform_str = sub.get("transform", "")
        sub_matrix = _compose_transforms(layer_matrix, _parse_transform_to_matrix(sub_transform_str) if sub_transform_str else IDENTITY_MATRIX)

        # Walk descendants recursively to find text elements and absorb group transforms
        def walk(elem: ET.Element, parent_matrix: TransformMatrix):
            for child in elem:
                if child.tag == f"{{{NS['svg']}}}g":
                    grp_transform = child.get("transform", "")
                    child_matrix = _compose_transforms(parent_matrix, _parse_transform_to_matrix(grp_transform) if grp_transform else IDENTITY_MATRIX)
                    walk(child, child_matrix)
                elif child.tag == f"{{{NS['svg']}}}text":
                    name = _get_text_element_label(child)
                    if not name:
                        continue
                    try:
                        svg_x = float(child.get("x", 0))
                        svg_y = float(child.get("y", 0))
                    except (TypeError, ValueError):
                        continue

                    # If the text element has its own transform, apply it first
                    elem_transform = child.get("transform", "")
                    if elem_transform:
                        elem_matrix = _parse_transform_to_matrix(elem_transform)
                        svg_x, svg_y = _apply_transform_matrix(svg_x, svg_y, elem_matrix)

                    # Apply accumulated parent/group transforms to obtain absolute SVG coords
                    abs_x, abs_y = _apply_transform_matrix(svg_x, svg_y, parent_matrix)

                    lon, lat = svg_to_geo(abs_x, abs_y)

                    pois.append({
                        "name": name,
                        "type": sub_label,
                        "lon": lon,
                        "lat": lat,
                    })

        walk(sub, sub_matrix)

    return pois


def main():
    svg_default = Path(__file__).parent.parent.parent / "oldworldatlas-maps" / "OLD_WORLD_ATLAS.svg"
    svg_path = svg_default if svg_default.exists() else Path("OLD_WORLD_ATLAS.svg")

    out_csv = Path(__file__).parent.parent / "output" / "points_of_interest.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    pois = extract_pois(svg_path)

    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "type", "coordinates"])
        for p in pois:
            coord = f"{p['lon']:.6f},{p['lat']:.6f}"
            writer.writerow([p["name"], p["type"], coord])

    print(f"Wrote {len(pois)} POIs to {out_csv}")


if __name__ == '__main__':
    main()
