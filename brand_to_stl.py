#!/usr/bin/env python3
"""
Script to add carved text branding to STL files using Blender Python API
Usage: blender --background --python add_brand_to_stl.py -- input.stl output.stl "YOUR BRAND"
"""

import bpy
import sys
import os
from mathlib import Vector

def clear_scene():
    """Remove all objects from the scene"""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

def import_stl(filepath):
    """Import STL file and return the object"""
    bpy.ops.import_mesh.stl(filepath=filepath)
    obj = bpy.context.selected_objects[0]
    return obj

def get_object_bounds(obj):
    """Get the bounding box dimensions of an object"""
    bbox_corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    
    min_x = min([v.x for v in bbox_corners])
    max_x = max([v.x for v in bbox_corners])
    min_y = min([v.y for v in bbox_corners])
    max_y = max([v.y for v in bbox_corners])
    min_z = min([v.z for v in bbox_corners])
    max_z = max([v.z for v in bbox_corners])
    
    return {
        'min': Vector((min_x, min_y, min_z)),
        'max': Vector((max_x, max_y, max_z)),
        'center': Vector(((min_x + max_x) / 2, (min_y + max_y) / 2, (min_z + max_z) / 2)),
        'size': Vector((max_x - min_x, max_y - min_y, max_z - min_z))
    }

def create_text_object(text, size=1.0):
    """Create a text object"""
    bpy.ops.object.text_add()
    text_obj = bpy.context.active_object
    text_obj.data.body = text
    text_obj.data.size = size
    text_obj.data.align_x = 'CENTER'
    text_obj.data.align_y = 'CENTER'
    
    # Add extrude depth for 3D text
    text_obj.data.extrude = 0.1 * size
    
    # Convert text to mesh for boolean operations
    bpy.context.view_layer.objects.active = text_obj
    bpy.ops.object.convert(target='MESH')
    
    return text_obj

def position_text_on_base(text_obj, model_obj, depth_offset=0.05):
    """Position text on the bottom of the model"""
    # Get model bounds
    bounds = get_object_bounds(model_obj)
    
    # Position text at bottom center of model
    text_obj.location.x = bounds['center'].x
    text_obj.location.y = bounds['center'].y
    text_obj.location.z = bounds['min'].z + depth_offset
    
    # Rotate text to be flat on bottom (facing up)
    text_obj.rotation_euler = (0, 0, 0)
    
    # Scale text to fit the base (80% of width)
    text_bounds = get_object_bounds(text_obj)
    scale_factor = (bounds['size'].x * 0.8) / text_bounds['size'].x
    text_obj.scale = (scale_factor, scale_factor, scale_factor)
    
    # Apply transformations
    bpy.context.view_layer.objects.active = text_obj
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

def carve_text_into_model(model_obj, text_obj, carve_depth=0.5):
    """Use boolean difference to carve text into model"""
    # Make text slightly deeper for clean carving
    text_obj.scale.z = carve_depth
    bpy.ops.object.transform_apply(scale=True)
    
    # Select model and add boolean modifier
    bpy.context.view_layer.objects.active = model_obj
    model_obj.select_set(True)
    
    # Add boolean modifier
    modifier = model_obj.modifiers.new(name="Carve_Text", type='BOOLEAN')
    modifier.operation = 'DIFFERENCE'
    modifier.object = text_obj
    modifier.solver = 'FAST'
    
    # Apply the modifier
    bpy.ops.object.modifier_apply(modifier=modifier.name)
    
    # Delete the text object (no longer needed)
    bpy.data.objects.remove(text_obj, do_unlink=True)

def export_stl(obj, filepath):
    """Export object to STL"""
    # Select only the object to export
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    
    # Export
    bpy.ops.export_mesh.stl(
        filepath=filepath,
        use_selection=True,
        ascii=False
    )

def add_brand_to_stl(input_stl, output_stl, brand_text, text_size=1.0, carve_depth=0.5):
    """
    Main function to add carved branding to STL
    
    Args:
        input_stl: Path to input STL file
        output_stl: Path to output STL file
        brand_text: Text to carve into the model
        text_size: Size of the text (will be auto-scaled to fit)
        carve_depth: How deep to carve the text (in Blender units)
    """
    print(f"Processing: {input_stl}")
    print(f"Brand text: {brand_text}")
    
    # Clear scene
    clear_scene()
    
    # Import STL
    print("Importing STL...")
    model_obj = import_stl(input_stl)
    
    # Create text
    print("Creating text object...")
    text_obj = create_text_object(brand_text, size=text_size)
    
    # Position text on base
    print("Positioning text...")
    position_text_on_base(text_obj, model_obj)
    
    # Carve text into model
    print("Carving text into model...")
    carve_text_into_model(model_obj, text_obj, carve_depth)
    
    # Export result
    print(f"Exporting to: {output_stl}")
    export_stl(model_obj, output_stl)
    
    print("Done!")

if __name__ == "__main__":
    # Parse command line arguments
    # Format: blender --background --python script.py -- input.stl output.stl "Brand Text"
    
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        print("Usage: blender --background --python add_brand_to_stl.py -- input.stl output.stl \"Brand Text\" [text_size] [carve_depth]")
        sys.exit(1)
    
    if len(argv) < 3:
        print("Error: Need at least 3 arguments: input_stl, output_stl, brand_text")
        sys.exit(1)
    
    input_stl = argv[0]
    output_stl = argv[1]
    brand_text = argv[2]
    text_size = float(argv[3]) if len(argv) > 3 else 1.0
    carve_depth = float(argv[4]) if len(argv) > 4 else 0.5
    
    # Validate input file exists
    if not os.path.exists(input_stl):
        print(f"Error: Input file '{input_stl}' not found")
        sys.exit(1)
    
    # Run the branding operation
    add_brand_to_stl(input_stl, output_stl, brand_text, text_size, carve_depth)