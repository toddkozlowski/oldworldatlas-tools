import re
with open(r"C:\Users\toddc\dev\personal\old-world-atlas\oldworldatlas-maps\OLD_WORLD_ATLAS.svg", encoding="utf-8", errors="replace") as f:
    content = f.read()

m = re.search(r'<path\b[^>]*\bid="path8-9"[^>]*>', content)
if m:
    print("path8-9 element:")
    print(m.group()[:800])
else:
    print("path8-9 not found")

for eid in ["path-effect3127", "path-effect3087", "path-effect3127-3"]:
    m2 = re.search(r'<inkscape:path-effect\b[^>]*id="' + re.escape(eid) + r'"[^>]*>', content)
    if m2:
        print(f"\nLPE def {eid}:")
        print(m2.group()[:600])

topo_labels = ["100", "250", "500", "2500", "1000", "1500", "2000"]
for label in topo_labels:
    m3 = re.search(r'inkscape:label="' + re.escape(label) + r'"', content)
    if m3:
        start = content.rfind("<g", 0, m3.start())
        sample = content[start:start+50000]
        lpe_count = len(re.findall(r"inkscape:path-effect=", sample))
        print(f'Topography layer "{label}": LPE elements in first 50KB: {lpe_count}')

# Check if topography paths have inkscape:original-d
print("\nTopography 100 layer - first path element:")
m4 = re.search(r'inkscape:label="100"', content)
if m4:
    start = content.rfind("<g", 0, m4.start())
    sample = content[start:start+20000]
    p = re.search(r'<path\b([^>]*)>', sample)
    if p:
        attrs = p.group(1)
        print(f"  has original-d: {'inkscape:original-d' in attrs}")
        print(f"  has path-effect: {'inkscape:path-effect' in attrs}")
        print(f"  transform attr: {re.search(r'transform=\"([^\"]+)\"', attrs)}")
