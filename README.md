# STL Branding Tool

Carve text into STL 3D models programmatically.

## Installation

```bash
uv venv
source .venv/bin/activate
uv pip install trimesh numpy pillow scipy scikit-image
```

## Usage

### Single File

```bash
uv run python stl_brander.py input.stl output.stl "BRAND TEXT" [position] [depth] [font_height_mm]
```

Parameters:
- position: bottom, top, front, back, left, right (default: bottom)
- depth: carve depth in mm (default: 1.0)
- font_height_mm: font height in mm (optional, will auto-fit if not provided)

### Batch Processing

Edit configuration in `batch_carve.py`:

```python
BRAND_TEXT = "YOUR BRAND"
INPUT_FOLDER = "input_stls"
OUTPUT_FOLDER = "branded_stls"
POSITION = "bottom"
TEXT_SCALE = 0.7
CARVE_DEPTH = 1.0
FONT_HEIGHT_MM = None  # Font height in mm (None to prompt)
```

Run:

```bash
uv run python batch_carve.py
```

## Python API

```python
from stl_brander import STLBrander

brander = STLBrander()

brander.carve_text(
    input_stl="model.stl",
    output_stl="branded_model.stl",
    brand_text="YOUR BRAND",
    position="bottom",
    text_scale=0.7,
    carve_depth=1.0,
    font_height_mm=None  # Optional: specific font height in mm (None for auto-fit)
)
```

### Batch Processing

```python
results = brander.batch_carve(
    input_files=["model1.stl", "model2.stl"],
    output_folder="branded_stls",
    brand_text="YOUR BRAND",
    position="bottom",
    carve_depth=1.0
)
```

## Configuration

### Custom Font

```python
brander = STLBrander(font_path="/path/to/font.ttf")
```

### Text Positioning

Available positions: bottom, top, front, back, left, right

### Text Scale

Value between 0.0 and 1.0, relative to model dimensions.

## Files

- `stl_brander.py` - Main carving implementation
- `batch_carve.py` - Batch processing script

## Dependencies

- trimesh - STL file handling and mesh operations
- numpy - Numerical computations
- pillow - Text rendering
- scipy - Mesh processing utilities
- scikit-image - Image processing for text conversion

## Requirements

Python 3.9 or higher