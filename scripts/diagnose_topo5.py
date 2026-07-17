"""Targeted diagnosis:
1. Show layer41 content extent (rect3131 width, path8-9 coords)
2. Show actual g70-2 path count (multiline-aware)
3. Show g70-2 coordinate range
4. Confirm sodipodi:insensitive on which layers
"""
import re

SVG = r"C:\Users\toddc\dev\personal\old-world-atlas\oldworldatlas-maps\OLD_WORLD_ATLAS_flattened.svg"
print("Reading…")
with open(SVG, encoding="utf-8", errors="replace") as f:
    content = f.read()
print(f"Read {len(content):,} chars\n")

# --- 1. sodipodi:insensitive in all layers ---
insensitive = re.findall(
    r'<g\b[^>]*sodipodi:insensitive="true"[^>]*>',
    content, re.DOTALL
)
print(f"Layers with sodipodi:insensitive=true: {len(insensitive)}")
for tag in insensitive:
    id_ = re.search(r'id="([^"]+)"', tag)
    label = re.search(r'inkscape:label="([^"]+)"', tag)
    print(f"  [{id_.group(1) if id_ else '?'}] '{label.group(1) if label else '?'}'")

# --- 2. rect3131 in layer41 ---
m_rect = re.search(r'id="rect3131"', content)
if m_rect:
    rs = content.rfind("<rect", 0, m_rect.start())
    rect_tag = content[rs:rs+500].replace('\n', ' ')
    w = re.search(r'width="([^"]+)"', rect_tag)
    h = re.search(r'height="([^"]+)"', rect_tag)
    print(f"\nrect3131: width={w.group(1) if w else '?'} height={h.group(1) if h else '?'}")

# --- 3. g70-2 multiline path count ---
m70 = re.search(r'id="g70-2"', content)
ls70 = content.rfind("<g", 0, m70.start())
# Find the next sibling layer (g71-6 "100") to bound the search
m71 = re.search(r'id="g71-6"', content)
ls71 = content.rfind("<g", 0, m71.start())
layer0_content = content[ls70:ls71]
print(f"\ng70-2 layer '0' content size: {len(layer0_content):,} chars ({len(layer0_content)/1e6:.1f}MB)")

# Count paths with DOTALL so multi-line path tags are found
paths_ml = re.findall(r'<path\b', layer0_content)
print(f"Path elements (multi-line aware): {len(paths_ml)}")

# Get coordinate range using dotall-aware d= extraction
# Extract d=" values — may be multi-line within quotes
all_d_vals = re.findall(r'\bd="((?:[^"\\]|\\.)*)"', layer0_content, re.DOTALL)
print(f"d= values found: {len(all_d_vals)}")

all_nums = []
for dv in all_d_vals:
    nums = re.findall(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", dv)
    all_nums.extend(float(n) for n in nums)

if all_nums:
    print(f"Coordinate range: min={min(all_nums):.1f}  max={max(all_nums):.1f}")
    # X values assumed to be in pairs (rough - alternating with Y but close enough)
    print(f"(viewBox visible: x=428 to 5246, y=-1470 to 3348)")
    print(f"(page: x=0 to 4817, y=0 to 4817)")

# --- 4. Show first path in g70-2 (multi-line) ---
first_path = re.search(r'<path\b(.*?)/?>', layer0_content, re.DOTALL)
if first_path:
    attrs = first_path.group(1).replace('\n', ' ')
    id_ = re.search(r'id="([^"]+)"', attrs)
    d_m = re.search(r'\bd="((?:[^"\\]|\\.){0,200})', attrs, re.DOTALL)
    print(f"\nFirst path in g70-2:")
    print(f"  id={id_.group(1) if id_ else '?'}")
    if d_m:
        coords = re.findall(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", d_m.group(1))[:12]
        print(f"  first coords: {coords}")
