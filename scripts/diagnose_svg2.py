"""Deeper SVG diagnostics - LPE, original-d, sodipodi, use elements."""
import re

SVG_PATH = r"C:\Users\toddc\dev\personal\old-world-atlas\oldworldatlas-maps\OLD_WORLD_ATLAS.svg"

print("Reading file...")
with open(SVG_PATH, encoding="utf-8", errors="replace") as f:
    content = f.read()
print(f"Read {len(content):,} chars\n")

# --- LPE and original-d ---
orig_d = len(re.findall(r'inkscape:original-d=', content))
lpe_on_elem = len(re.findall(r'<(?:path|rect|circle|ellipse)[^>]*inkscape:path-effect=', content))
print(f"inkscape:original-d on elements: {orig_d}")
print(f"Elements with inkscape:path-effect: {lpe_on_elem}")

# --- use elements ---
use_elems = re.findall(r'<use\b([^>]*)>', content)
print(f"\n<use> elements: {len(use_elems)}")
for u in use_elems[:20]:
    href_m = re.search(r'(?:xlink:href|href)="([^"]+)"', u)
    id_m = re.search(r'id="([^"]+)"', u)
    t_m = re.search(r'transform="([^"]+)"', u)
    print(f'  id={id_m.group(1) if id_m else "?"} href={href_m.group(1) if href_m else "?"} transform={t_m.group(1) if t_m else "NONE"}')

# --- sodipodi:type elements (basic shapes stored as sodipodi) ---
sp_types = re.findall(r'sodipodi:type="([^"]+)"', content)
from collections import Counter
sp_count = Counter(sp_types)
print(f"\nsodipodi:type counts: {dict(sp_count)}")

# --- Check viewBox vs page dimensions ---
root_m = re.search(r'<svg\b([^>]*)>', content)
if root_m:
    attrs = root_m.group(1)
    vb = re.search(r'viewBox="([^"]+)"', attrs)
    w = re.search(r'\bwidth="([^"]+)"', attrs)
    h = re.search(r'\bheight="([^"]+)"', attrs)
    print(f"\nRoot SVG viewBox: {vb.group(1) if vb else '?'}")
    print(f"Root SVG width: {w.group(1) if w else '?'}, height: {h.group(1) if h else '?'}")
    # Check aspect ratio
    if vb:
        parts = vb.group(1).split()
        vbw, vbh = float(parts[2]), float(parts[3])
        print(f"viewBox aspect ratio W/H: {vbw/vbh:.6f}")

# --- Check for any translate or scale in non-layer groups ---
# Look specifically around Bretonnia Lakes and Empire Paths layers
for label in ["Bretonnia Lakes", "Empire Paths", "Lakes - Bretonnia", "Empire"]:
    # Find the layer definition
    pat = re.compile(r'inkscape:label="' + re.escape(label) + r'"')
    m = pat.search(content)
    if m:
        # Get surrounding context (the opening g tag)
        start = content.rfind("<g", 0, m.start())
        end = content.find(">", m.start()) + 1
        tag_text = content[start:end]
        # Check parent context - go back 5000 chars
        ctx_start = max(0, start - 2000)
        ctx = content[ctx_start:start]
        # Find last opening g tag in context
        parent_gs = list(re.finditer(r'<g\b[^>]*>', ctx))
        print(f"\n--- Layer '{label}' ---")
        print(f"  Tag: {tag_text[:300]}")
        if parent_gs:
            last_parent = parent_gs[-1]
            print(f"  Nearest parent g: {last_parent.group()[:200]}")

# --- Check inkscape:transform-center ---
tc = len(re.findall(r'inkscape:transform-center', content))
print(f"\ninkscape:transform-center occurrences: {tc}")

# --- Check for clip-path on layers ---
clip = re.findall(r'<g\b[^>]*groupmode="layer"[^>]*clip-path="[^"]*"[^>]*>', content)
print(f"Layers with clip-path: {len(clip)}")
for c in clip:
    print(f"  {c[:200]}")

# --- Check for mask on layers ---
mask = re.findall(r'<g\b[^>]*groupmode="layer"[^>]*mask="[^"]*"[^>]*>', content)
print(f"Layers with mask: {len(mask)}")
