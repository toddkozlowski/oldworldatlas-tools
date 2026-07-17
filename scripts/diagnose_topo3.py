"""Check sodipodi:nodetypes in the topography '0' layer."""
import re
from collections import Counter

SVG_PATH = r"C:\Users\toddc\dev\personal\old-world-atlas\oldworldatlas-maps\OLD_WORLD_ATLAS_flattened.svg"

print("Reading file…")
with open(SVG_PATH, encoding="utf-8", errors="replace") as f:
    content = f.read()
print(f"Read {len(content):,} chars\n")

# Global summary of all sodipodi:nodetypes
all_nt = re.findall(r'sodipodi:nodetypes="([^"]*)"', content)
all_chars = "".join(all_nt)
char_count = Counter(all_chars)
print(f"Global sodipodi:nodetypes count: {len(all_nt)}")
print(f"Node type character distribution: {dict(sorted(char_count.items()))}")
print(f"  'a' (auto-smooth) nodes: {char_count.get('a', 0):,}")
print(f"  's' (smooth) nodes: {char_count.get('s', 0):,}")
print(f"  'c' (cusp) nodes: {char_count.get('c', 0):,}")
print(f"  'z' (symmetric) nodes: {char_count.get('z', 0):,}")

# Now check per topography layer
topo_sub_labels = ["-1000","-500","-250","-100","0","100","250","500","750","1000","1500","2000","2500","3000","4000","5000"]

print("\n--- Per topography layer ---")
for label in topo_sub_labels:
    # Find the layer's opening tag
    m = re.search(r'inkscape:label="' + re.escape(label) + r'"\s*(?:style|id|inkscape)', content)
    if not m:
        m = re.search(r'inkscape:label="' + re.escape(label) + r'"', content)
    if not m:
        continue
    layer_tag_start = content.rfind("<g", 0, m.start())
    # Find closing </g> by counting nesting depth
    pos = layer_tag_start
    depth = 0
    layer_end = len(content)
    i = pos
    while i < min(pos + 50_000_000, len(content)):
        if content[i:i+2] == "<g":
            depth += 1
        elif content[i:i+4] == "</g>":
            depth -= 1
            if depth == 0:
                layer_end = i + 4
                break
        i += 1

    layer_content = content[layer_tag_start:layer_end]
    layer_nt = re.findall(r'sodipodi:nodetypes="([^"]*)"', layer_content)
    layer_chars = "".join(layer_nt)
    layer_count = Counter(layer_chars)
    a_count = layer_count.get('a', 0)
    total = len(layer_chars)
    print(f'  "{label}": {len(layer_nt)} paths with nodetypes, {total} total nodes, {a_count} auto-smooth (a)')

# Additionally check non-topography layers for comparison
print("\n--- Non-topography layers with auto-smooth nodetypes ---")
# Look for nodetypes with 'a' outside the topography group
topo_group_m = re.search(r'id="g82-8"', content)
if topo_group_m:
    topo_start = content.rfind("<g", 0, topo_group_m.start())
    before_topo = content[:topo_start]
    after_topo = content[topo_start + 100:]  # Skip a bit past topo start

    outside_nt = re.findall(r'sodipodi:nodetypes="([^"]*)"', before_topo)
    outside_nt += re.findall(r'sodipodi:nodetypes="([^"]*)"', after_topo)
    outside_chars = "".join(outside_nt)
    outside_count = Counter(outside_chars)
    print(f"  Outside topography: {len(outside_nt)} paths, {outside_count.get('a', 0):,} auto-smooth nodes")
