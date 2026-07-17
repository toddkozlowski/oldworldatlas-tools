"""Inspect topography layer path node counts and command distributions."""
import re
from collections import Counter

SVG_PATH = r"C:\Users\toddc\dev\personal\old-world-atlas\oldworldatlas-maps\OLD_WORLD_ATLAS_flattened.svg"

print("Reading file…")
with open(SVG_PATH, encoding="utf-8", errors="replace") as f:
    content = f.read()
print(f"Read {len(content):,} chars\n")

# Find the topography parent group boundaries
# id=layer44 "new_topography" or id=g82-8 "topography"
topo_parent_ids = ["layer44", "g82-8"]
topo_sub_labels = ["-1000","-500","-250","-100","0","100","250","500","750","1000","1500","2000","2500","3000","4000","5000"]

for label in topo_sub_labels:
    m = re.search(r'inkscape:label="' + re.escape(label) + r'"', content)
    if not m:
        continue
    layer_start = content.rfind("<g", 0, m.start())
    # Find the closing tag - look for next sibling layer or parent close
    # Use a simpler approach: scan 5MB from layer start
    sample_size = 5_000_000
    sample = content[layer_start:layer_start + sample_size]

    # Extract all path d attributes
    path_ds = re.findall(r'\bd="([^"]*)"', sample)

    total_nodes = 0
    cmd_counter = Counter()
    path_info = []
    for d in path_ds:
        tokens = re.findall(r"[AaCcHhLlMmQqSsTtVvZz]", d)
        cmd_counter.update(tokens)
        # Count coordinate pairs as rough node count
        coords = re.findall(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", d)
        path_info.append(len(coords) // 2)
        total_nodes += len(coords) // 2

    if path_info:
        path_info.sort(reverse=True)
        print(f'Layer "{label}":')
        print(f'  Paths: {len(path_info)}, total nodes: {total_nodes:,}')
        print(f'  Largest paths (approx nodes): {path_info[:5]}')
        cmd_str = ", ".join(f"{k}:{v}" for k, v in sorted(cmd_counter.items()))
        print(f'  Commands: {cmd_str}')
