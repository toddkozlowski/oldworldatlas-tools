"""Deep inspect topography layer elements for transforms, style transforms, and other metadata."""
import re

SVG_PATH = r"C:\Users\toddc\dev\personal\old-world-atlas\oldworldatlas-maps\OLD_WORLD_ATLAS_flattened.svg"

print("Reading file…")
with open(SVG_PATH, encoding="utf-8", errors="replace") as f:
    content = f.read()
print(f"Read {len(content):,} chars\n")

# Locate the topography parent group (g82-8 "topography")
topo_m = re.search(r'id="g82-8"', content)
if not topo_m:
    print("Could not find topography group g82-8")
else:
    # Sample 10MB from the topography group start
    topo_start = content.rfind("<g", 0, topo_m.start())
    sample = content[topo_start:topo_start + 10_000_000]

    # Check for any transform attributes on non-layer elements
    print("=== Non-layer elements with transform attributes in topography ===")
    for m in re.finditer(r'<(?!g\b)[a-zA-Z:]+\b([^>]*)>', sample):
        attrs = m.group(1)
        if 'transform="' in attrs and 'groupmode' not in attrs:
            id_m = re.search(r'id="([^"]+)"', attrs)
            t_m = re.search(r'transform="([^"]+)"', attrs)
            print(f'  [{id_m.group(1) if id_m else "?"}] {t_m.group(1) if t_m else "?"}')

    # Check for any <g> elements (non-layer) with transforms
    print("\n=== Non-layer groups with transform attributes in topography ===")
    for m in re.finditer(r'<g\b([^>]*)>', sample):
        attrs = m.group(1)
        if 'transform="' in attrs and 'groupmode' not in attrs:
            id_m = re.search(r'id="([^"]+)"', attrs)
            t_m = re.search(r'transform="([^"]+)"', attrs)
            print(f'  [{id_m.group(1) if id_m else "?"}] {t_m.group(1) if t_m else "?"}')

    # Check layer g elements for transforms
    print("\n=== Layer groups in topography ===")
    for m in re.finditer(r'<g\b([^>]*)>', sample):
        attrs = m.group(1)
        if 'groupmode' in attrs:
            id_m = re.search(r'id="([^"]+)"', attrs)
            label_m = re.search(r'inkscape:label="([^"]+)"', attrs)
            t_m = re.search(r'transform="([^"]+)"', attrs)
            style_m = re.search(r'style="([^"]+)"', attrs)
            print(f'  [{id_m.group(1) if id_m else "?"}] "{label_m.group(1) if label_m else "?"}" transform={t_m.group(1) if t_m else "NONE"} style={style_m.group(1) if style_m else "NONE"}')

    # Inspect a single path element from the "250" layer for full attribute list
    print("\n=== First path in '250' layer (full attributes) ===")
    m250 = re.search(r'inkscape:label="250"', sample)
    if m250:
        g_start = sample.rfind("<g", 0, m250.start())
        subsample = sample[g_start:g_start+50000]
        p = re.search(r'<path\b([^>]*)>', subsample)
        if p:
            attrs_raw = p.group(1)
            # Print each attribute on its own line
            for attr_m in re.finditer(r'([a-zA-Z:_-]+)="([^"]*)"', attrs_raw):
                name, val = attr_m.group(1), attr_m.group(2)
                if name == 'd':
                    val = val[:80] + "…"
                print(f'  {name} = "{val}"')

# Check for any elements with inkscape:original-d still remaining (verify flatten worked)
remaining_orig = len(re.findall(r'inkscape:original-d=', content))
remaining_lpe = len(re.findall(r'inkscape:path-effect=', content))
print(f"\n=== Verification ===")
print(f"  inkscape:original-d remaining: {remaining_orig}")
print(f"  inkscape:path-effect remaining: {remaining_lpe}")

# Check for any remaining transform attributes on path elements anywhere
path_transforms = re.findall(r'<path\b[^>]*transform="([^"]+)"[^>]*>', content[:20_000_000])
print(f"  path elements with transform (first 20MB): {len(path_transforms)}")

# Check for style containing matrix() or translate() in topography
if topo_m:
    topo_styles_with_transform = re.findall(r'style="[^"]*(?:matrix|translate|scale)\([^"]*"', sample[:5_000_000])
    print(f"  style attrs with transform functions in topo (first 5MB): {len(topo_styles_with_transform)}")

# Check for sodipodi:nodetypes
nodetypes = len(re.findall(r'sodipodi:nodetypes=', content))
print(f"  sodipodi:nodetypes occurrences: {nodetypes}")
