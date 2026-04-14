"""Create unstretched copies of Settlement and POI text objects in new SVG layers.

This is a one-off utility intended for post-stretch map cleanup/verification.
It duplicates all text elements under the top-level layers:
- Settlements
- Points of Interest

For each duplicated text element, the script:
- Computes absolute SVG coordinates by composing all ancestor transforms.
- Applies element-level transforms before ancestor transforms.
- Writes the copied text with no transform in the output layer hierarchy.
- Normalizes typography to Segoe UI, normal style/weight/stretch, size 1.5.

No existing SVG content is deleted. If the target copy layer already exists from a
previous run, only that generated layer is replaced.
"""

from __future__ import annotations

import argparse
import copy
import math
import re
from pathlib import Path
from typing import Optional, Set, Tuple
from xml.etree import ElementTree as ET


# Register namespaces before parsing/writing so prefixes are stable.
ET.register_namespace("", "http://www.w3.org/2000/svg")
ET.register_namespace("svg", "http://www.w3.org/2000/svg")
ET.register_namespace("inkscape", "http://www.inkscape.org/namespaces/inkscape")
ET.register_namespace("sodipodi", "http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd")
ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")

NS = {
    "svg": "http://www.w3.org/2000/svg",
    "inkscape": "http://www.inkscape.org/namespaces/inkscape",
}

SVG_G = f"{{{NS['svg']}}}g"
SVG_TEXT = f"{{{NS['svg']}}}text"
INKSCAPE_LABEL = f"{{{NS['inkscape']}}}label"
INKSCAPE_GROUP_MODE = f"{{{NS['inkscape']}}}groupmode"

ROOT_COPY_LAYER_LABEL = "Unstretched Text Copies"
SETTLEMENTS_COPY_LAYER_LABEL = "Settlements (Unstretched Copy)"
POI_COPY_LAYER_LABEL = "Points of Interest (Unstretched Copy)"

FONT_SIZE = "1.5"
FONT_FAMILY = "Segoe UI"

TransformMatrix = Tuple[float, float, float, float, float, float]
IDENTITY_MATRIX: TransformMatrix = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)


def default_svg_path() -> Path:
    """Return the default map path used by project scripts."""
    return Path(__file__).parent.parent.parent / "oldworldatlas-maps" / "OLD_WORLD_ATLAS.svg"


def parse_transform_to_matrix(transform_str: str) -> TransformMatrix:
    """Parse SVG transform string into affine matrix (a, b, c, d, e, f)."""
    if not transform_str or not transform_str.strip():
        return IDENTITY_MATRIX

    result: TransformMatrix = IDENTITY_MATRIX

    for match in re.finditer(r"([a-zA-Z]+)\s*\(([^)]+)\)", transform_str):
        func = match.group(1)
        args_str = match.group(2).strip()
        try:
            args = [float(v) for v in re.split(r"[,\s]+", args_str) if v]
        except ValueError:
            continue

        if func == "translate":
            tx = args[0] if len(args) >= 1 else 0.0
            ty = args[1] if len(args) >= 2 else 0.0
            t: TransformMatrix = (1.0, 0.0, 0.0, 1.0, tx, ty)
        elif func == "scale":
            sx = args[0] if len(args) >= 1 else 1.0
            sy = args[1] if len(args) >= 2 else sx
            t = (sx, 0.0, 0.0, sy, 0.0, 0.0)
        elif func == "matrix":
            if len(args) < 6:
                continue
            t = (args[0], args[1], args[2], args[3], args[4], args[5])
        elif func == "rotate":
            angle = math.radians(args[0]) if args else 0.0
            cos_a = math.cos(angle)
            sin_a = math.sin(angle)
            if len(args) >= 3:
                cx, cy = args[1], args[2]
                t = (
                    cos_a,
                    sin_a,
                    -sin_a,
                    cos_a,
                    cx - cx * cos_a + cy * sin_a,
                    cy - cx * sin_a - cy * cos_a,
                )
            else:
                t = (cos_a, sin_a, -sin_a, cos_a, 0.0, 0.0)
        else:
            continue

        result = compose_transforms(outer=result, inner=t)

    return result


def compose_transforms(outer: TransformMatrix, inner: TransformMatrix) -> TransformMatrix:
    """Compose two affine transforms: apply inner first, then outer."""
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


def apply_transform_matrix(x: float, y: float, matrix: TransformMatrix) -> Tuple[float, float]:
    """Apply affine transform matrix (a,b,c,d,e,f) to point (x, y)."""
    a, b, c, d, e, f = matrix
    return (a * x + c * y + e, b * x + d * y + f)


def parse_style(style: str) -> dict[str, str]:
    """Parse a CSS style declaration block into a dictionary."""
    parsed: dict[str, str] = {}
    if not style:
        return parsed

    for part in style.split(";"):
        if not part.strip() or ":" not in part:
            continue
        key, value = part.split(":", 1)
        parsed[key.strip()] = value.strip()
    return parsed


def serialize_style(style_dict: dict[str, str]) -> str:
    """Serialize style dict to deterministic CSS declaration string."""
    if not style_dict:
        return ""
    return ";".join(f"{k}:{style_dict[k]}" for k in sorted(style_dict.keys()))


def parse_first_float(value: Optional[str], default: float = 0.0) -> float:
    """Parse first float token from SVG numeric attribute values."""
    if value is None:
        return default
    token = str(value).strip().replace(",", " ").split()
    if not token:
        return default
    try:
        return float(token[0])
    except ValueError:
        return default


def find_group_by_label(root: ET.Element, label: str) -> Optional[ET.Element]:
    """Find first group with matching inkscape:label."""
    for g in root.iter(SVG_G):
        if g.get(INKSCAPE_LABEL) == label:
            return g
    return None


def build_existing_ids(root: ET.Element) -> Set[str]:
    """Collect all existing id attribute values from document."""
    ids: Set[str] = set()
    for elem in root.iter():
        elem_id = elem.get("id")
        if elem_id:
            ids.add(elem_id)
    return ids


def slugify(value: str) -> str:
    """Return a conservative ID-friendly slug."""
    slug = re.sub(r"[^A-Za-z0-9_\-]+", "_", value.strip())
    slug = slug.strip("_")
    return slug or "item"


def next_unique_id(base: str, used_ids: Set[str]) -> str:
    """Generate a unique element id based on base."""
    candidate = base
    i = 1
    while candidate in used_ids:
        i += 1
        candidate = f"{base}_{i}"
    used_ids.add(candidate)
    return candidate


def find_parent(root: ET.Element, target: ET.Element) -> Optional[ET.Element]:
    """Find parent element of target by walking tree."""
    for elem in root.iter():
        for child in elem:
            if child is target:
                return elem
    return None


def ensure_copy_root_layer(root: ET.Element, used_ids: Set[str]) -> ET.Element:
    """Create or replace the generated copy root layer."""
    existing = find_group_by_label(root, ROOT_COPY_LAYER_LABEL)
    if existing is not None:
        parent = find_parent(root, existing)
        if parent is not None:
            parent.remove(existing)

    new_layer = ET.Element(SVG_G)
    new_layer.set(INKSCAPE_GROUP_MODE, "layer")
    new_layer.set(INKSCAPE_LABEL, ROOT_COPY_LAYER_LABEL)
    new_layer.set("id", next_unique_id("layer_unstretched_text_copies", used_ids))
    root.append(new_layer)
    return new_layer


def create_child_layer(
    parent: ET.Element,
    label: str,
    used_ids: Set[str],
) -> ET.Element:
    """Create a labeled child group configured as an Inkscape layer."""
    group = ET.SubElement(parent, SVG_G)
    group.set(INKSCAPE_LABEL, label)
    group.set(INKSCAPE_GROUP_MODE, "layer")
    group.set("id", next_unique_id(f"copy_{slugify(label)}", used_ids))
    return group


def normalize_text_typography(text_elem: ET.Element) -> None:
    """Force typography to standard unstretched Segoe UI normal, size 1.5."""
    style_dict = parse_style(text_elem.get("style", ""))

    style_dict["font-family"] = FONT_FAMILY
    style_dict["font-size"] = FONT_SIZE
    style_dict["font-style"] = "normal"
    style_dict["font-weight"] = "normal"
    style_dict["font-stretch"] = "normal"

    text_elem.set("style", serialize_style(style_dict))
    text_elem.set("font-family", FONT_FAMILY)
    text_elem.set("font-size", FONT_SIZE)
    text_elem.set("font-style", "normal")
    text_elem.set("font-weight", "normal")


def duplicate_text_element(
    source_text: ET.Element,
    target_parent: ET.Element,
    parent_matrix: TransformMatrix,
    used_ids: Set[str],
) -> None:
    """Copy one text element into target_parent at absolute coordinates."""
    base_x = parse_first_float(source_text.get("x"), 0.0)
    base_y = parse_first_float(source_text.get("y"), 0.0)

    elem_transform = source_text.get("transform", "")
    if elem_transform:
        elem_matrix = parse_transform_to_matrix(elem_transform)
        base_x, base_y = apply_transform_matrix(base_x, base_y, elem_matrix)

    abs_x, abs_y = apply_transform_matrix(base_x, base_y, parent_matrix)

    copied = copy.deepcopy(source_text)

    if "transform" in copied.attrib:
        del copied.attrib["transform"]

    copied.set("x", f"{abs_x:.6f}")
    copied.set("y", f"{abs_y:.6f}")
    copied.set("id", next_unique_id("copy_text", used_ids))

    normalize_text_typography(copied)
    target_parent.append(copied)


def clone_text_hierarchy(
    source_group: ET.Element,
    target_group: ET.Element,
    used_ids: Set[str],
    parent_matrix: TransformMatrix = IDENTITY_MATRIX,
) -> Tuple[int, int]:
    """Clone nested group/text hierarchy from source into target with flattened transforms.

    Returns:
        (group_count, text_count)
    """
    groups_created = 0
    texts_copied = 0

    group_transform = source_group.get("transform", "")
    group_matrix = compose_transforms(
        outer=parent_matrix,
        inner=parse_transform_to_matrix(group_transform),
    ) if group_transform else parent_matrix

    for child in source_group:
        if child.tag == SVG_G:
            child_label = child.get(INKSCAPE_LABEL) or child.get("id") or "Group"
            new_child_group = create_child_layer(target_group, child_label, used_ids)
            groups_created += 1
            sub_groups, sub_texts = clone_text_hierarchy(
                child,
                new_child_group,
                used_ids,
                group_matrix,
            )
            groups_created += sub_groups
            texts_copied += sub_texts
        elif child.tag == SVG_TEXT:
            duplicate_text_element(child, target_group, group_matrix, used_ids)
            texts_copied += 1

    return groups_created, texts_copied


def save_preserving_prefix(svg_path: Path, root: ET.Element) -> None:
    """Save SVG while preserving original pre-<svg> prefix text."""
    raw_text = svg_path.read_text(encoding="utf-8")
    lower = raw_text.lower()
    svg_start = lower.find("<svg")
    prefix = raw_text[:svg_start] if svg_start >= 0 else ""

    body = ET.tostring(root, encoding="unicode", xml_declaration=False)
    output = prefix + body
    svg_path.write_text(output, encoding="utf-8")


def run(svg_path: Path) -> None:
    """Execute the one-off copy operation and save updated SVG."""
    if not svg_path.exists():
        raise FileNotFoundError(f"SVG not found: {svg_path}")

    tree = ET.parse(str(svg_path))
    root = tree.getroot()

    settlements_layer = find_group_by_label(root, "Settlements")
    poi_layer = find_group_by_label(root, "Points of Interest")

    if settlements_layer is None:
        raise RuntimeError("Could not find top-level layer labeled 'Settlements'")
    if poi_layer is None:
        raise RuntimeError("Could not find top-level layer labeled 'Points of Interest'")

    used_ids = build_existing_ids(root)

    copy_root = ensure_copy_root_layer(root, used_ids)
    settlements_copy_layer = create_child_layer(
        copy_root,
        SETTLEMENTS_COPY_LAYER_LABEL,
        used_ids,
    )
    poi_copy_layer = create_child_layer(
        copy_root,
        POI_COPY_LAYER_LABEL,
        used_ids,
    )

    settlements_groups, settlements_texts = clone_text_hierarchy(
        settlements_layer,
        settlements_copy_layer,
        used_ids,
        IDENTITY_MATRIX,
    )
    poi_groups, poi_texts = clone_text_hierarchy(
        poi_layer,
        poi_copy_layer,
        used_ids,
        IDENTITY_MATRIX,
    )

    save_preserving_prefix(svg_path, root)

    print(f"Updated SVG: {svg_path}")
    print(
        "Copied text elements "
        f"(Settlements={settlements_texts}, Points of Interest={poi_texts}, Total={settlements_texts + poi_texts})"
    )
    print(
        "Created sublayers "
        f"(Settlements={settlements_groups}, Points of Interest={poi_groups}, Total={settlements_groups + poi_groups})"
    )
    print(f"Added layer: {ROOT_COPY_LAYER_LABEL}")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Duplicate Settlement and POI text into unstretched copy layers.",
    )
    parser.add_argument(
        "--svg",
        type=Path,
        default=default_svg_path(),
        help="Path to input/output SVG file (modified in place).",
    )
    args = parser.parse_args()

    run(args.svg)


if __name__ == "__main__":
    main()
