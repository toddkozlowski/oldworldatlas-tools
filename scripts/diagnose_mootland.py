"""
Diagnostic script to analyze Mootland settlement structure in SVG.
"""

from xml.etree import ElementTree as ET

# Parse the SVG
tree = ET.parse('../oldworldatlas-maps/FULL_MAP_CLEANED.svg')
root = tree.getroot()

NS = {
    'svg': 'http://www.w3.org/2000/svg',
    'inkscape': 'http://www.inkscape.org/namespaces/inkscape'
}

# Find Mootland layer
for g in root.findall(f".//{{{NS['svg']}}}g"):
    label = g.get(f"{{{NS['inkscape']}}}label")
    if label == 'Mootland':
        print(f'Mootland group found')
        print(f'  Transform: {g.get("transform")}')
        print(f'  Has transform: {bool(g.get("transform"))}')
        print(f'\nDirect children of Mootland:')
        
        # Check children for Eicheschatten and Kurbisdorf
        eiche_found = False
        kurbi_found = False
        mootland_children_with_transforms = []
        
        for child in g:
            if child.tag == f"{{{NS['svg']}}}text":
                child_label = child.get(f"{{{NS['inkscape']}}}label")
                if child_label in ['Eicheschatten', 'Kurbisdorf']:
                    print(f'\n  {child_label} (DIRECT TEXT CHILD):')
                    print(f'    x={child.get("x")}, y={child.get("y")}')
                    print(f'    transform={child.get("transform")}')
                    if child_label == 'Eicheschatten':
                        eiche_found = True
                    elif child_label == 'Kurbisdorf':
                        kurbi_found = True
            elif child.tag == f"{{{NS['svg']}}}g":
                child_label = child.get(f"{{{NS['inkscape']}}}label")
                child_transform = child.get('transform')
                if child_transform:
                    mootland_children_with_transforms.append((child_label, child_transform))
                    print(f'\n  Group: {child_label}')
                    print(f'    transform={child_transform}')
                    
                # Check grandchildren
                for grandchild in child:
                    if grandchild.tag == f"{{{NS['svg']}}}text":
                        gc_label = grandchild.get(f"{{{NS['inkscape']}}}label")
                        if gc_label in ['Eicheschatten', 'Kurbisdorf']:
                            print(f'    -> {gc_label} (IN GROUP):')
                            print(f'      x={grandchild.get("x")}, y={grandchild.get("y")}')
                            print(f'      transform={grandchild.get("transform")}')
                            if gc_label == 'Eicheschatten':
                                eiche_found = True
                            elif gc_label == 'Kurbisdorf':
                                kurbi_found = True
        
        if not eiche_found:
            print('\n  Eicheschatten not found in Mootland direct children!')
            print('  Searching deeper...')
            depth = 0
            def search_deep(elem, target_names, depth_level=0):
                for child in elem:
                    if child.tag == f"{{{NS['svg']}}}text":
                        child_label = child.get(f"{{{NS['inkscape']}}}label")
                        if child_label in target_names:
                            print(f'\n  Found {child_label} at depth {depth_level}:')
                            print(f'    x={child.get("x")}, y={child.get("y")}')
                            print(f'    transform={child.get("transform")}')
                            # Walk up to find parent transforms
                            parent = elem
                            print(f'    Parent transform: {parent.get("transform")}')
                    elif child.tag == f"{{{NS['svg']}}}g":
                        group_label = child.get(f"{{{NS['inkscape']}}}label")
                        group_transform = child.get('transform')
                        search_deep(child, target_names, depth_level + 1)
            
            search_deep(g, ['Eicheschatten', 'Kurbisdorf'])
        
        if mootland_children_with_transforms:
            print(f'\n\nMootland child groups with transforms:')
            for label, transform in mootland_children_with_transforms:
                print(f'  {label}: {transform}')
        
        break

print("\n" + "="*60)
print("Analysis Summary:")
print("="*60)
