"""Diagnose SVG layer transforms and structure."""
import re

SVG_PATH = r"C:\Users\toddc\dev\personal\old-world-atlas\oldworldatlas-maps\OLD_WORLD_ATLAS.svg"

with open(SVG_PATH, encoding="utf-8", errors="replace") as f:
    content = f.read()

print(f"Total chars: {len(content):,}")

# Find all g elements
pattern = r"<g\b([^>]*)>"
layers = []
for m in re.finditer(pattern, content):
    attrs = m.group(1)
    if "groupmode" in attrs and "layer" in attrs:
        id_m = re.search(r'id="([^"]+)"', attrs)
        label_m = re.search(r'inkscape:label="([^"]+)"', attrs)
        transform_m = re.search(r'transform="([^"]+)"', attrs)
        layers.append({
            "id": id_m.group(1) if id_m else "?",
            "label": label_m.group(1) if label_m else "?",
            "transform": transform_m.group(1) if transform_m else None,
        })

print(f"\nTotal layers: {len(layers)}")
print("\n--- Layers WITH transforms ---")
for l in layers:
    if l["transform"]:
        print(f'  [{l["id"]}] "{l["label"]}" => {l["transform"]}')

print("\n--- All layers ---")
for l in layers:
    t = l["transform"] or "NONE"
    print(f'  [{l["id"]}] "{l["label"]}" | {t}')

# Also check root SVG element for any transform
root_match = re.search(r"<svg\b([^>]*)>", content)
if root_match:
    root_attrs = root_match.group(1)
    root_transform = re.search(r'transform="([^"]+)"', root_attrs)
    vb = re.search(r'viewBox="([^"]+)"', root_attrs)
    w = re.search(r'\bwidth="([^"]+)"', root_attrs)
    h = re.search(r'\bheight="([^"]+)"', root_attrs)
    print(f"\n--- Root SVG ---")
    print(f"  viewBox: {vb.group(1) if vb else '?'}")
    print(f"  width: {w.group(1) if w else '?'}")
    print(f"  height: {h.group(1) if h else '?'}")
    print(f"  transform: {root_transform.group(1) if root_transform else 'NONE'}")

# Check for any non-layer g elements with transforms near the top of hierarchy
print("\n--- Non-layer groups with transforms (first 50) ---")
count = 0
for m in re.finditer(pattern, content):
    attrs = m.group(1)
    if "groupmode" not in attrs and 'transform="' in attrs:
        id_m = re.search(r'id="([^"]+)"', attrs)
        transform_m = re.search(r'transform="([^"]+)"', attrs)
        print(f'  [{id_m.group(1) if id_m else "?"}] => {transform_m.group(1) if transform_m else "?"}')
        count += 1
        if count >= 50:
            print("  ... (truncated)")
            break
