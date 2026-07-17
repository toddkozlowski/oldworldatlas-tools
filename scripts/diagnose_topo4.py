import re

SVG = r"C:\Users\toddc\dev\personal\old-world-atlas\oldworldatlas-maps\OLD_WORLD_ATLAS_flattened.svg"
with open(SVG, encoding="utf-8", errors="replace") as f:
    content = f.read()

# Layer "0" is g70-2
m = re.search(r'id="g70-2"', content)
ls = content.rfind("<g", 0, m.start())
sample = content[ls:ls+2_000_000]

paths = list(re.finditer(r'<path\b([^>]*)>', sample))
print(f"g70-2 layer '0': {len(paths)} paths in first 2MB")

for pm in paths[:2]:
    a = pm.group(1)
    id_ = re.search(r'id="([^"]+)"', a)
    # d attribute may be long, get it carefully
    d_m = re.search(r'\bd="([^"]{0,300})', a)
    if d_m:
        dval = d_m.group(1)
        coords = re.findall(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", dval)[:20]
        print(f"  id={id_.group(1) if id_ else '?'}")
        print(f"  first coords: {coords}")

# Also get the min/max of ALL coords across the entire g70-2 sample
all_d_text = " ".join(re.findall(r'\bd="([^"]{0,1000000})"', sample))
all_nums = [float(x) for x in re.findall(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", all_d_text)]
if all_nums:
    print(f"\nAll g70-2 coord range in 2MB sample:")
    print(f"  min={min(all_nums):.2f}  max={max(all_nums):.2f}")

# Check The World (layer41) style and path8-9 location
m_world = re.search(r'id="layer41"', content)
if m_world:
    world_tag_start = content.rfind("<g", 0, m_world.start())
    world_tag = content[world_tag_start:world_tag_start+300]
    print(f"\nlayer41 'The World' tag:\n{world_tag}")

# Check path8-9 coords
m8 = re.search(r'id="path8-9"', content)
if m8:
    ps = content.rfind("<path", 0, m8.start())
    path_tag = content[ps:ps+500]
    # extract first coord tokens
    d_m8 = re.search(r'\bd="([^"]{0,200})', path_tag)
    if d_m8:
        nums8 = re.findall(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", d_m8.group(1))[:10]
        print(f"\npath8-9 first coords: {nums8}")
    print(f"path8-9 position in file: {m8.start():,}")
    print(f"g70-2 layer start position: {ls:,}")
    print(f"(path8-9 comes {'before' if m8.start() < ls else 'after'} g70-2)")

# ViewBox
vb = re.search(r'viewBox="([^"]+)"', content[:500])
print(f"\nviewBox: {vb.group(1) if vb else '?'}")
print(f"Page in SVG coords: x=0,y=0 to x=4817.7969,y=4817.7969")
