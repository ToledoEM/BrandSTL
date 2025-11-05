# STL Branding Tool

Carve text into STL 3D models using Blender's robust boolean operations engine.

## Features

- **Blender-Powered**: Uses Blender's FAST boolean solver for reliable mesh operations
- **Automatic Text Mirroring**: Text is mirrored for correct chirality when carved on bottom surface
- **Proper Scaling**: Text automatically scales to 20% of available surface area
- **High-Quality Carving**: Boolean difference operation creates clean carved text
- **Mesh Preservation**: Maintains original mesh integrity without destruction

## Requirements

- Blender (installed at `/Applications/Blender.app/Contents/MacOS/Blender` on macOS)

## Usage

### Single File

```bash
/Applications/Blender.app/Contents/MacOS/Blender --background --python stl_brander_blender.py -- input.stl output.stl "BRAND TEXT" [depth] [font_size]
```

Parameters:
- **input.stl**: Input STL file path
- **output.stl**: Output STL file path  
- **text**: Text to carve (carved on bottom surface)
- **depth**: Carve depth in mm (default: 1.0)
- **font_size**: Font size (default: 15)

### Examples

```bash
# Basic usage
/Applications/Blender.app/Contents/MacOS/Blender --background --python stl_brander_blender.py -- model.stl branded.stl "MyBrand"

# With specific depth
/Applications/Blender.app/Contents/MacOS/Blender --background --python stl_brander_blender.py -- model.stl branded.stl "MyBrand" 2.0

# With specific depth and font size
/Applications/Blender.app/Contents/MacOS/Blender --background --python stl_brander_blender.py -- model.stl branded.stl "MyBrand" 2.0 12
```
## Technical Details

### Text Processing

- Text is created using Blender's text object system
- Converted to mesh geometry for boolean operations
- Automatically centered and scaled to fit surface
- Character spacing optimized for readability

### Boolean Operations

- Uses Blender's FAST boolean solver
- DIFFERENCE operation carves text into model
- Preserves mesh integrity and vertex count
- Handles complex geometries reliably

### Positioning

- Text positioned on bottom surface of model
- X-axis mirroring ensures correct chirality
- Penetrates into model for clean carving
- Automatic bounds detection and centering

## Files

- `stl_brander_blender.py` - Blender-based carving implementation
- `stl_brander.py` - Legacy Trimesh implementation (deprecated)

## Dependencies

- Blender 3.0 or higher with Python API
