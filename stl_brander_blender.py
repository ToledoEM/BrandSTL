#!/usr/bin/env python3
"""
STL Text Carving using Blender Python API
Usage: blender --background --python stl_brander_blender.py -- input.stl output.stl "BRAND TEXT" [carve_depth] [font_size]
"""

import bpy
import sys
import os
from mathutils import Vector

def clear_scene():
    """Remove all objects from the scene"""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

def import_stl(filepath):
    """Import STL file and return the object"""
    try:
        bpy.ops.wm.stl_import(filepath=filepath)
    except:
        # Fallback for older Blender versions
        try:
            bpy.ops.import_mesh.stl(filepath=filepath)
        except:
            raise Exception(f"Could not import STL file: {filepath}")
    
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

def create_text_object(text, font_size=1.0, extrude_depth=1.0):
    """Create a text object with specified font size and extrusion depth"""
    bpy.ops.object.text_add()
    text_obj = bpy.context.active_object
    text_obj.data.body = text
    text_obj.data.size = font_size
    text_obj.data.align_x = 'CENTER'
    text_obj.data.align_y = 'CENTER'
    
    # Set proper spacing for multiple characters
    text_obj.data.space_character = 1.0
    text_obj.data.space_word = 1.0
    
    # Use the specified extrude depth
    text_obj.data.extrude = extrude_depth
    
    # Convert text to mesh for boolean operations
    bpy.context.view_layer.objects.active = text_obj
    bpy.ops.object.convert(target='MESH')
    
    return text_obj

def position_text_on_bottom(text_obj, model_obj, carve_depth=0.5):
    """Position text on the bottom of the model for carving"""
    bounds = get_object_bounds(model_obj)
    text_bounds = get_object_bounds(text_obj)
    
    # Scale text to a small fraction of available space (20% max)
    available_width = bounds['size'].x * 0.2
    available_height = bounds['size'].y * 0.2
    
    width_scale = available_width / text_bounds['size'].x
    height_scale = available_height / text_bounds['size'].y
    
    scale_factor = min(width_scale, height_scale)
    
    print(f"Model dimensions: {bounds['size'].x:.2f} x {bounds['size'].y:.2f}")
    print(f"Text scale factor: {scale_factor:.3f}")
    print(f"Carve depth: {carve_depth}mm")
    
    # Apply scaling
    text_obj.scale = (scale_factor, scale_factor, scale_factor)
    bpy.context.view_layer.objects.active = text_obj
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    
    # Mirror text for correct chirality when carved on bottom
    # Mirror on X-axis so text reads correctly from above
    text_obj.scale.x = -1
    bpy.ops.object.transform_apply(scale=True)
    
    # Position text at bottom center, penetrating into model for carving
    # Position text so it extends from bottom surface into the model
    text_obj.location.x = bounds['center'].x
    text_obj.location.y = bounds['center'].y
    text_obj.location.z = bounds['min'].z + (carve_depth / 2)  # Half above, half below bottom surface
    
    final_bounds = get_object_bounds(text_obj)
    print(f"Final text size: {final_bounds['size'].x:.2f} x {final_bounds['size'].y:.2f}")
    print(f"Text positioned at Z: {text_obj.location.z:.2f} (model bottom at {bounds['min'].z:.2f})")
    print("Text mirrored for correct chirality")

def carve_text_into_model(model_obj, text_obj):
    """Use boolean difference to carve text into model"""
    print(f"Model vertices before: {len(model_obj.data.vertices)}")
    print(f"Text vertices: {len(text_obj.data.vertices)}")
    
    # Select model and add boolean modifier
    bpy.context.view_layer.objects.active = model_obj
    bpy.ops.object.select_all(action='DESELECT')
    model_obj.select_set(True)
    
    # Add boolean modifier for DIFFERENCE (carving)
    modifier = model_obj.modifiers.new(name="Carve_Text", type='BOOLEAN')
    modifier.operation = 'DIFFERENCE'
    modifier.object = text_obj
    modifier.solver = 'FAST'
    
    # Apply the modifier
    bpy.ops.object.modifier_apply(modifier=modifier.name)
    print(f"Model vertices after carving: {len(model_obj.data.vertices)}")
    
    # Delete the text object
    bpy.data.objects.remove(text_obj, do_unlink=True)

def export_stl(obj, filepath):
    """Export object to STL"""
    # Select only the object to export
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    
    # Export using the newer API
    try:
        bpy.ops.wm.stl_export(
            filepath=filepath,
            export_selected_objects=True,
            ascii_format=False
        )
    except:
        # Fallback for older Blender versions
        bpy.ops.export_mesh.stl(
            filepath=filepath,
            use_selection=True,
            ascii=False
        )

def carve_text(input_stl, output_stl, brand_text, carve_depth=1.0, font_size=15):
    """
    Main function to carve text into STL using Blender
    
    Args:
        input_stl: Path to input STL file
        output_stl: Path to output STL file
        brand_text: Text to carve into the model
        carve_depth: How deep to carve the text (in mm)
        font_size: Font size for the text
    """
    print(f"Loading {input_stl}...")
    print(f"Creating text: '{brand_text}'")
    print(f"Carve depth: {carve_depth}mm")
    print(f"Font size: {font_size}")
    
    # Clear scene
    clear_scene()
    
    # Import STL
    model_obj = import_stl(input_stl)
    
    # Create text
    text_obj = create_text_object(brand_text, font_size / 10.0, carve_depth)  # Use carve_depth for extrusion
    
    # Position text on bottom for carving
    print("Positioning text on bottom...")
    position_text_on_bottom(text_obj, model_obj, carve_depth)  # Use mm directly
    
    # Carve text into model
    print("Carving text into model (this may take a moment)...")
    carve_text_into_model(model_obj, text_obj)
    
    # Export result
    print(f"Exporting to {output_stl}...")
    export_stl(model_obj, output_stl)
    
    print("✓ Success!")
    return True

if __name__ == "__main__":
    # Parse command line arguments
    # Format: blender --background --python stl_brander_blender.py -- input.stl output.stl "Brand Text" [carve_depth] [font_size]
    
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        print("Usage: blender --background --python stl_brander_blender.py -- input.stl output.stl \"Brand Text\" [carve_depth] [font_size]")
        print("  carve_depth: depth in mm (default: 1.0)")
        print("  font_size: font size (default: 15)")
        sys.exit(1)
    
    if len(argv) < 3:
        print("Error: Need at least 3 arguments: input_stl, output_stl, brand_text")
        sys.exit(1)
    
    input_file = argv[0]
    output_file = argv[1]
    brand_text = argv[2]
    carve_depth = float(argv[3]) if len(argv) > 3 else 1.0
    font_size = int(argv[4]) if len(argv) > 4 else 15
    
    # Validate input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found")
        sys.exit(1)
    
    # Run the carving operation
    try:
        success = carve_text(input_file, output_file, brand_text, carve_depth, font_size)
        if success:
            print(f"\n✓ Successfully created {output_file}")
        else:
            print(f"\n✗ Failed to process {input_file}")
            sys.exit(1)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        sys.exit(1)