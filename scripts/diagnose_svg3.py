"""Check LPE details - which effects are applied where, and original-d status."""
import re
from collections import defaultdict

SVG_PATH = r"C:\Users\toddc\dev\personal\old-world-atlas\oldworldatlas-maps\OLD_WORLD_ATLAS.svg"

print("Reading file...")
with open(SVG_PATH, encoding="utf-8", errors="replace") as f:
    content = f.read()
print(f"Read {len(content):,} chars\n")

# --- Catalog all LPE definitions in <defs> ---
lpe_defs = {}
for m in re.finditer(r'<inkscape:path-effect\b([^>]*)/?>', content):
    attrs = m.group(1)
    id_m = re.search(r'\bid="([^"]+)"', attrs)
    eff_m = re.search(r'\beffect="([^"]+)"', attrs)
    if id_m:
        lpe_defs[id_m.group(1)] = eff_m.group(1) if eff_m else "unknown"

print(f"LPE definitions in <defs>: {len(lpe_defs)}")
for eid, eff in sorted(lpe_defs.items()):
    print(f"  #{eid} -> {eff}")

# --- Find paths with LPE references and check their original-d ---
# Also track which layer they're in by scanning the file position
print("\n--- Paths with inkscape:path-effect (first 30) ---")
count = 0

# Build layer position map from layer start positions
layer_positions = []
for m in re.finditer(r'<g\b[^>]*groupmode="layer"[^>]*>', content):
    id_m = re.search(r'id="([^"]+)"', m.group())
    label_m = re.search(r'inkscape:label="([^"]+)"', m.group())
    layer_positions.append((m.start(), id_m.group(1) if id_m else "?", label_m.group(1) if label_m else "?"))

def get_layer_at(pos):
    """Find the most recent layer before this position."""
    best = ("?", "?")
    for lpos, lid, llabel in layer_positions:
        if lpos < pos:
            best = (lid, llabel)
        else:
            break
    return best

for m in re.finditer(r'<(?:path|rect)\b([^>]*)>', content):
    attrs = m.group(1)
    if 'inkscape:path-effect=' not in attrs:
        continue
    id_m = re.search(r'\bid="([^"]+)"', attrs)
    lpe_m = re.search(r'inkscape:path-effect="([^"]+)"', attrs)
    has_orig_d = 'inkscape:original-d=' in attrs
    has_d = re.search(r'\bd=', attrs)
    
    # Get LPE effect type
    lpe_id = lpe_m.group(1).lstrip('#') if lpe_m else "?"
    eff_type = lpe_defs.get(lpe_id, "unknown")
    
    layer_id, layer_label = get_layer_at(m.start())
    
    print(f"  [{id_m.group(1) if id_m else '?'}] effect={eff_type} lpe_id={lpe_id} has_original-d={has_orig_d} has_d={bool(has_d)} layer='{layer_label}'")
    count += 1
    if count >= 30:
        print(f"  ... (showing 30 of 1150 total)")
        break

# --- Check specifically near Bretonnia Lakes and Empire Paths content ---
for label in ["Lakes - Bretonnia", "Empire Paths"]:
    m = re.search(r'inkscape:label="' + re.escape(label) + r'"', content)
    if not m:
        print(f"\nLayer '{label}' NOT FOUND")
        continue
    # Get from this layer start to 50000 chars ahead for sampling
    layer_start = content.rfind("<g", 0, m.start())
    sample = content[layer_start:layer_start + 100000]
    
    # Find path elements in this sample
    paths_with_lpe = re.findall(r'<(?:path|rect)\b[^>]*inkscape:path-effect=[^>]*>', sample)
    paths_total = len(re.findall(r'<(?:path|rect)\b', sample))
    
    print(f"\n--- Layer '{label}' ---")
    print(f"  Total <path>/<rect> elements (first 100KB): {paths_total}")
    print(f"  Elements with LPE: {len(paths_with_lpe)}")
    if paths_with_lpe:
        # Show first few
        for p in paths_with_lpe[:3]:
            lpe_m2 = re.search(r'inkscape:path-effect="([^"]+)"', p)
            orig = 'inkscape:original-d' in p
            lpe_id2 = lpe_m2.group(1).lstrip('#') if lpe_m2 else "?"
            eff = lpe_defs.get(lpe_id2, "unknown")
            print(f"    effect={eff} has_original-d={orig}")

# --- Check transform-center details ---
tc_matches = re.findall(r'inkscape:transform-center-[xy]="([^"]+)"', content)
print(f"\ninkscape:transform-center-x/y values (non-zero, first 20):")
nonzero = [v for v in tc_matches if float(v) != 0.0][:20]
for v in nonzero:
    print(f"  {v}")
