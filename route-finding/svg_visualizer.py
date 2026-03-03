"""
SVG Visualizer Module
Adds visualization of the pathfinding network to the SVG file.
"""

from lxml import etree
from pathlib import Path
from typing import Dict


class SVGVisualizer:
    """Adds pathfinding visualization to SVG files."""
    
    # SVG namespace
    NS = {
        'svg': 'http://www.w3.org/2000/svg',
        'inkscape': 'http://www.inkscape.org/namespaces/inkscape',
        'sodipodi': 'http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd'
    }
    
    # Node visualization styles (colors)
    NODE_STYLES = {
        'location': {'fill': '#ff0000', 'stroke': '#000000', 'radius': 2.5},  # Red
        'intersection': {'fill': '#00ff00', 'stroke': '#000000', 'radius': 2.0},  # Green
        'endpoint': {'fill': '#0000ff', 'stroke': '#000000', 'radius': 1.5},  # Blue
        'path_node': {'fill': '#888888', 'stroke': '#000000', 'radius': 1.0},  # Gray
    }
    
    def __init__(self, svg_file: str, graph_builder):
        """
        Initialize the visualizer.
        
        Args:
            svg_file: Path to the SVG file
            graph_builder: GraphBuilder instance with the node network
        """
        self.svg_file = svg_file
        self.graph = graph_builder
        self.tree = etree.parse(svg_file)
        self.root = self.tree.getroot()
        
    def add_visualization(self, output_file: str = None):
        """
        Add pathfinding visualization to the SVG.
        
        Args:
            output_file: Path to save the annotated SVG (defaults to input_annotated.svg)
        """
        if output_file is None:
            path = Path(self.svg_file)
            output_file = path.parent / f"{path.stem}_annotated{path.suffix}"
        
        # Remove existing visualization layer if present
        self._remove_existing_layer()
        
        # Create new layer
        layer = self._create_layer()
        
        # Add edges (path segments) first so they appear behind nodes
        self._add_edges(layer)
        
        # Add nodes
        self._add_nodes(layer)
        
        # Add legend
        self._add_legend(layer)
        
        # Save the annotated SVG
        self.tree.write(str(output_file), encoding='utf-8', xml_declaration=True, pretty_print=True)
        
        return output_file
    
    def _remove_existing_layer(self):
        """Remove existing pathfinding visualization layer if present."""
        for layer in self.root.findall('.//svg:g[@inkscape:groupmode="layer"]', self.NS):
            layer_name = layer.get('{http://www.inkscape.org/namespaces/inkscape}label', '')
            if layer_name == 'Pathfinding Nodes':
                self.root.remove(layer)
    
    def _create_layer(self):
        """Create a new layer for pathfinding visualization."""
        layer = etree.SubElement(
            self.root,
            '{http://www.w3.org/2000/svg}g',
            {
                '{http://www.inkscape.org/namespaces/inkscape}label': 'Pathfinding Nodes',
                '{http://www.inkscape.org/namespaces/inkscape}groupmode': 'layer',
                'id': 'pathfinding_layer'
            }
        )
        return layer
    
    def _add_nodes(self, layer):
        """Add node visualizations."""
        for node_id, node in self.graph.nodes.items():
            style_info = self.NODE_STYLES.get(node.node_type, self.NODE_STYLES['path_node'])
            
            # Create circle for node
            circle = etree.SubElement(
                layer,
                '{http://www.w3.org/2000/svg}circle',
                {
                    'cx': str(node.x),
                    'cy': str(node.y),
                    'r': str(style_info['radius']),
                    'style': f"fill:{style_info['fill']};stroke:{style_info['stroke']};stroke-width:0.5",
                    'id': f'node_{node_id}'
                }
            )
            
            # Add label for location and intersection nodes
            if node.node_type in ['location', 'intersection']:
                text = etree.SubElement(
                    layer,
                    '{http://www.w3.org/2000/svg}text',
                    {
                        'x': str(node.x + 3),
                        'y': str(node.y - 2),
                        'style': 'font-size:4px;font-family:sans-serif;fill:#000000',
                        'id': f'label_{node_id}'
                    }
                )
                text.text = node_id
    
    def _add_edges(self, layer):
        """Add edge visualizations (path segments)."""
        for edge in self.graph.edges:
            node1 = self.graph.nodes[edge.node1_id]
            node2 = self.graph.nodes[edge.node2_id]
            
            # Draw a line between nodes
            line = etree.SubElement(
                layer,
                '{http://www.w3.org/2000/svg}line',
                {
                    'x1': str(node1.x),
                    'y1': str(node1.y),
                    'x2': str(node2.x),
                    'y2': str(node2.y),
                    'style': 'stroke:#000000;stroke-width:0.3;opacity:0.5;stroke-dasharray:1,1',
                    'id': f'edge_{edge.node1_id}_{edge.node2_id}'
                }
            )
    
    def _add_legend(self, layer):
        """Add a legend explaining the node types."""
        legend_x = 10
        legend_y = 10
        line_height = 6
        
        # Legend background
        rect = etree.SubElement(
            layer,
            '{http://www.w3.org/2000/svg}rect',
            {
                'x': str(legend_x - 2),
                'y': str(legend_y - 2),
                'width': '40',
                'height': str(len(self.NODE_STYLES) * line_height + 8),
                'style': 'fill:#ffffff;stroke:#000000;stroke-width:0.5;opacity:0.8',
                'id': 'legend_bg'
            }
        )
        
        # Legend title
        title = etree.SubElement(
            layer,
            '{http://www.w3.org/2000/svg}text',
            {
                'x': str(legend_x),
                'y': str(legend_y + 3),
                'style': 'font-size:4px;font-family:sans-serif;font-weight:bold;fill:#000000',
                'id': 'legend_title'
            }
        )
        title.text = 'Node Types'
        
        # Legend entries
        y_offset = legend_y + 8
        for node_type, style_info in self.NODE_STYLES.items():
            # Circle
            circle = etree.SubElement(
                layer,
                '{http://www.w3.org/2000/svg}circle',
                {
                    'cx': str(legend_x + 2),
                    'cy': str(y_offset),
                    'r': str(style_info['radius']),
                    'style': f"fill:{style_info['fill']};stroke:{style_info['stroke']};stroke-width:0.3",
                    'id': f'legend_{node_type}_circle'
                }
            )
            
            # Label
            text = etree.SubElement(
                layer,
                '{http://www.w3.org/2000/svg}text',
                {
                    'x': str(legend_x + 6),
                    'y': str(y_offset + 1.5),
                    'style': 'font-size:3px;font-family:sans-serif;fill:#000000',
                    'id': f'legend_{node_type}_text'
                }
            )
            text.text = node_type.replace('_', ' ').title()
            
            y_offset += line_height
