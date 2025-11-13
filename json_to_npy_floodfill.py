#!/usr/bin/env python3
"""
json_to_npy_floodfill.py

Convert BIM JSON files to GSDiff NPY format using flood fill method.
Creates proper room boundary arrays with wall-based separation and bounding box constraints.

Usage:
    # Single file
    python json_to_npy_floodfill.py --input apartment_1.json --output apartment_1.npy
    
    # Batch process directory
    python json_to_npy_floodfill.py --input_dir ./json_files --output_dir ./npy_files
"""

import json
import argparse
import numpy as np
from pathlib import Path
from collections import deque
from scipy import ndimage


def create_wall_raster(walls, resolution=10.0, wall_thickness=2):
    """
    Rasterize walls into a binary image.
    
    Args:
        walls: List of wall dicts with 'start', 'end', 'type'
        resolution: Millimeters per pixel
        wall_thickness: Width of walls in pixels
    
    Returns:
        wall_mask: Binary image where 1 = wall, 0 = space
        bounds: (x_min, x_max, y_min, y_max)
        to_pixel: Function to convert real coords to pixel coords
    """
    # Get bounds
    all_x = []
    all_y = []
    for wall in walls:
        all_x.extend([wall['start'][0], wall['end'][0]])
        all_y.extend([wall['start'][1], wall['end'][1]])
    
    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)
    
    # Add margin
    margin = 1000  # mm
    x_min -= margin
    x_max += margin
    y_min -= margin
    y_max += margin
    
    # Calculate raster size
    width = int((x_max - x_min) / resolution) + 1
    height = int((y_max - y_min) / resolution) + 1
    
    print(f"  Creating raster: {width}x{height} pixels ({resolution}mm/px)")
    
    # Create coordinate transform
    def to_pixel(coord):
        x, y = coord
        px = int((x - x_min) / resolution)
        py = int((y - y_min) / resolution)
        return px, py
    
    # Create wall mask
    wall_mask = np.zeros((height, width), dtype=np.uint8)
    
    # Draw walls using Bresenham's algorithm
    walls_drawn = 0
    for wall in walls:
        wall_type = wall.get('type', 'wall')
        
        # Draw all wall types including separators
        if wall_type not in {'wall', 'curtain_wall', 'linked_wall', 'separator'}:
            continue
        
        walls_drawn += 1
        
        x1, y1 = to_pixel(wall['start'])
        x2, y2 = to_pixel(wall['end'])
        
        # Bresenham's line algorithm with thickness
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx - dy
        
        x, y = x1, y1
        
        while True:
            # Draw thick point
            for dx_off in range(-wall_thickness, wall_thickness + 1):
                for dy_off in range(-wall_thickness, wall_thickness + 1):
                    px = x + dx_off
                    py = y + dy_off
                    if 0 <= px < width and 0 <= py < height:
                        wall_mask[py, px] = 1
            
            if x == x2 and y == y2:
                break
            
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
    
    print(f"  Drew {walls_drawn} walls")
    
    # Apply smart gap filling
    # Phase 1: Connect aligned nearby endpoints
    endpoint_data = []
    for wall in walls:
        if wall.get('type', 'wall') in {'wall', 'curtain_wall', 'linked_wall', 'separator'}:
            endpoint_data.append(('start', wall['start'], wall['end']))
            endpoint_data.append(('end', wall['end'], wall['start']))
    
    close_gap_mm = 100
    aligned_gap_mm = 300
    alignment_tolerance_mm = 50
    
    close_gap_px = close_gap_mm / resolution
    aligned_gap_px = aligned_gap_mm / resolution
    alignment_tol_px = alignment_tolerance_mm / resolution
    
    connections_made = 0
    connected_pairs = set()
    
    for i, (type1, ep1, other1) in enumerate(endpoint_data):
        px1, py1 = to_pixel(ep1)
        
        for j, (type2, ep2, other2) in enumerate(endpoint_data):
            if i >= j:
                continue
            
            px2, py2 = to_pixel(ep2)
            dist = ((px2 - px1)**2 + (py2 - py1)**2)**0.5
            
            if dist == 0:
                continue
            
            pair = tuple(sorted([i, j]))
            if pair in connected_pairs:
                continue
            
            should_connect = False
            
            if dist < close_gap_px:
                should_connect = True
            elif dist < aligned_gap_px:
                dx = abs(px2 - px1)
                dy = abs(py2 - py1)
                
                if dy < alignment_tol_px and dx < aligned_gap_px:
                    should_connect = True
                elif dx < alignment_tol_px and dy < aligned_gap_px:
                    should_connect = True
            
            if should_connect:
                x, y = px1, py1
                dx_line = abs(px2 - px1)
                dy_line = abs(py2 - py1)
                sx = 1 if px1 < px2 else -1
                sy = 1 if py1 < py2 else -1
                err = dx_line - dy_line
                
                while True:
                    for dx_off in range(-wall_thickness, wall_thickness + 1):
                        for dy_off in range(-wall_thickness, wall_thickness + 1):
                            px = x + dx_off
                            py = y + dy_off
                            if 0 <= px < width and 0 <= py < height:
                                wall_mask[py, px] = 1
                    
                    if x == px2 and y == py2:
                        break
                    
                    e2 = 2 * err
                    if e2 > -dy_line:
                        err -= dy_line
                        x += sx
                    if e2 < dx_line:
                        err += dx_line
                        y += sy
                
                connections_made += 1
                connected_pairs.add(pair)
    
    print(f"  Connected {connections_made} gaps")
    
    # Phase 2: Morphological closing
    structure = np.array([
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [1, 1, 1, 1, 1],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0]
    ], dtype=np.uint8)
    
    wall_mask_closed = ndimage.binary_closing(wall_mask, structure=structure, iterations=2)
    wall_mask = wall_mask_closed.astype(np.uint8)
    
    print(f"  Applied morphological closing")
    
    # Phase 3: Targeted wall extension
    extension_range_mm = 1000
    extension_range_px = extension_range_mm / resolution
    search_radius = 5
    
    extensions_made = 0
    
    for wall in walls:
        if wall.get('type', 'wall') not in {'wall', 'curtain_wall', 'linked_wall', 'separator'}:
            continue
        
        px1, py1 = to_pixel(wall['start'])
        px2, py2 = to_pixel(wall['end'])
        
        dx = px2 - px1
        dy = py2 - py1
        length = (dx**2 + dy**2)**0.5
        
        if length < 1:
            continue
        
        dx_norm = dx / length
        dy_norm = dy / length
        
        is_horizontal = abs(dy_norm) < 0.3
        is_vertical = abs(dx_norm) < 0.3
        
        if not (is_horizontal or is_vertical):
            continue
        
        # Try extending from endpoints
        for start_point, direction in [(px2, py2, 1), (px1, py1, -1)]:
            hit_wall = False
            hit_x, hit_y = None, None
            
            px_start, py_start = start_point
            
            for distance in range(10, int(extension_range_px)):
                test_x = int(px_start + direction * dx_norm * distance)
                test_y = int(py_start + direction * dy_norm * distance)
                
                if not (0 <= test_x < width and 0 <= test_y < height):
                    break
                
                found_wall = False
                for dx_search in range(-search_radius, search_radius + 1):
                    for dy_search in range(-search_radius, search_radius + 1):
                        check_x = test_x + dx_search
                        check_y = test_y + dy_search
                        
                        if 0 <= check_x < width and 0 <= check_y < height:
                            if wall_mask[check_y, check_x] == 1:
                                found_wall = True
                                hit_x, hit_y = check_x, check_y
                                break
                    
                    if found_wall:
                        break
                
                if found_wall:
                    hit_wall = True
                    break
            
            if hit_wall:
                x, y = px_start, py_start
                target_x, target_y = hit_x, hit_y
                
                dx_line = abs(target_x - x)
                dy_line = abs(target_y - y)
                sx = 1 if x < target_x else -1
                sy = 1 if y < target_y else -1
                err = dx_line - dy_line
                
                while True:
                    for dx_off in range(-wall_thickness, wall_thickness + 1):
                        for dy_off in range(-wall_thickness, wall_thickness + 1):
                            px = x + dx_off
                            py = y + dy_off
                            if 0 <= px < width and 0 <= py < height:
                                wall_mask[py, px] = 1
                    
                    if x == target_x and y == target_y:
                        break
                    
                    e2 = 2 * err
                    if e2 > -dy_line:
                        err -= dy_line
                        x += sx
                    if e2 < dx_line:
                        err += dx_line
                        y += sy
                
                extensions_made += 1
    
    print(f"  Extended {extensions_made} walls")
    
    return wall_mask, (x_min, x_max, y_min, y_max), to_pixel


def flood_fill_region(mask, seed_point, bbox_constraint=None):
    """Flood fill from seed point with optional bounding box constraint."""
    height, width = mask.shape
    sx, sy = seed_point
    
    if not (0 <= sx < width and 0 <= sy < height):
        return None
    if mask[sy, sx] == 1:
        return None
    
    region_mask = np.zeros_like(mask, dtype=np.uint8)
    queue = deque([(sx, sy)])
    visited = set()
    
    while queue:
        x, y = queue.popleft()
        
        if (x, y) in visited:
            continue
        if not (0 <= x < width and 0 <= y < height):
            continue
        if mask[y, x] == 1:
            continue
        
        if bbox_constraint is not None:
            x_min_bbox, x_max_bbox, y_min_bbox, y_max_bbox = bbox_constraint
            if not (x_min_bbox <= x <= x_max_bbox and y_min_bbox <= y <= y_max_bbox):
                continue
        
        visited.add((x, y))
        region_mask[y, x] = 1
        
        queue.extend([
            (x + 1, y),
            (x - 1, y),
            (x, y + 1),
            (x, y - 1)
        ])
    
    return region_mask


def json_to_npy(json_path, output_path, resolution=10.0):
    """
    Convert BIM JSON to GSDiff NPY format.
    
    Args:
        json_path: Path to input JSON file
        output_path: Path to output NPY file
        resolution: Millimeters per pixel
    """
    print(f"\nProcessing: {Path(json_path).name}")
    
    # Load JSON
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    rooms = data.get('rooms', [])
    walls_list = data.get('walls', [])
    
    print(f"  Rooms: {len(rooms)}, Walls: {len(walls_list)}")
    
    # Prepare wall data
    walls = []
    for wall in walls_list:
        walls.append({
            'start': (wall['start'][0], wall['start'][1]),
            'end': (wall['end'][0], wall['end'][1]),
            'type': wall.get('type', 'wall')
        })
    
    # Create wall raster
    wall_mask, bounds, to_pixel = create_wall_raster(walls, resolution=resolution)
    x_min, x_max, y_min, y_max = bounds
    height, width = wall_mask.shape
    
    # Create room layout array (0 = wall/unassigned, 1-N = room IDs)
    layout = np.zeros((height, width), dtype=np.int32)
    
    # Mark walls as -1 initially (will be 0 in final output)
    layout[wall_mask == 1] = -1
    
    # Process each room and assign room IDs
    room_id = 1
    room_mapping = {}
    
    for room in rooms:
        room_name = room.get('name', 'Unknown')
        
        # Skip balconies
        if 'balcony' in room_name.lower():
            continue
        
        bbox = room.get('bounding_box', {})
        if not bbox or 'min' not in bbox or 'max' not in bbox:
            continue
        
        # Calculate seed point
        center_x = (bbox['min'][0] + bbox['max'][0]) / 2
        center_y = (bbox['min'][1] + bbox['max'][1]) / 2
        seed_px, seed_py = to_pixel((center_x, center_y))
        
        # Bounding box constraint
        bbox_min_px, bbox_min_py = to_pixel((bbox['min'][0], bbox['min'][1]))
        bbox_max_px, bbox_max_py = to_pixel((bbox['max'][0], bbox['max'][1]))
        bbox_constraint = (bbox_min_px, bbox_max_px, bbox_min_py, bbox_max_py)
        
        # Flood fill
        region_mask = flood_fill_region(wall_mask, (seed_px, seed_py), bbox_constraint=bbox_constraint)
        
        if region_mask is None or region_mask.sum() == 0:
            print(f"    Warning: Could not fill room '{room_name}'")
            continue
        
        # Assign room ID
        layout[region_mask == 1] = room_id
        room_mapping[room_id] = room_name
        
        print(f"    Room {room_id}: {room_name} ({region_mask.sum()} pixels)")
        
        room_id += 1
    
    # Convert walls from -1 to 0
    layout[layout == -1] = 0
    
    # Save as NPY
    np.save(output_path, layout)
    print(f"  ✓ Saved NPY: {output_path}")
    print(f"  Shape: {layout.shape}, Rooms: {room_id - 1}")
    
    return layout, room_mapping


def process_directory(input_dir, output_dir, resolution=10.0):
    """Process all JSON files in a directory."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    json_files = sorted(input_dir.glob('*.json'))
    
    if not json_files:
        print(f"No .json files found in {input_dir}")
        return
    
    print(f"Found {len(json_files)} JSON files")
    print("="*60)
    
    for json_path in json_files:
        output_path = output_dir / f"{json_path.stem}.npy"
        try:
            json_to_npy(json_path, output_path, resolution)
        except Exception as e:
            print(f"Error processing {json_path.name}: {e}")
            import traceback
            traceback.print_exc()
        print()


def main():
    parser = argparse.ArgumentParser(
        description='Convert BIM JSON to GSDiff NPY format using flood fill',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Single file
    python json_to_npy_floodfill.py --input apartment_1.json --output apartment_1.npy
    
    # Batch process directory
    python json_to_npy_floodfill.py --input_dir ./json_files --output_dir ./npy_files
    
    # Custom resolution (default: 10mm/pixel)
    python json_to_npy_floodfill.py --input apt.json --output apt.npy --resolution 5
        """
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--input', type=str, help='Input JSON file')
    group.add_argument('--input_dir', type=str, help='Input directory containing JSON files')
    
    parser.add_argument('--output', type=str, help='Output NPY file (for single file)')
    parser.add_argument('--output_dir', type=str, help='Output directory for NPY files')
    parser.add_argument('--resolution', type=float, default=10.0,
                       help='Millimeters per pixel (default: 10)')
    
    args = parser.parse_args()
    
    if args.input and not args.output:
        parser.error("--output is required when using --input")
    if args.input_dir and not args.output_dir:
        parser.error("--output_dir is required when using --input_dir")
    
    if args.input:
        json_to_npy(args.input, args.output, args.resolution)
    else:
        process_directory(args.input_dir, args.output_dir, args.resolution)


if __name__ == '__main__':
    main()