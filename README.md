# Attendance AI

AI-powered attendance sheet recognition for Ontario Continuing Education.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the server
uvicorn server:app --reload --host 0.0.0.0 --port 8000
```

Then open **http://localhost:8000** in your browser.

## How It Works

1. **Upload** a photo of an attendance sheet
2. The system **preprocesses** the image (grayscale, denoise, threshold)
3. **Detects** the document boundary and applies perspective correction
4. **Finds** the table grid (horizontal & vertical lines)
5. **Extracts** individual cells from the latest attendance column
6. **Classifies** each cell as `A` (absent) or `BLANK` (present)
7. Returns **results** with student statuses, summary counts, and debug visualizations

## Project Structure

```
attendance_ai/
├── main.py                  # Pipeline orchestrator
├── config/
│   └── template.json        # Layout & threshold configuration
├── pipeline/
│   ├── preprocess.py        # Grayscale, denoise, threshold, edges
│   ├── detect_document.py   # Contour detection, perspective transform
│   ├── detect_table.py      # Grid line detection, cell boundaries
│   ├── extract_cells.py     # Cell image extraction
│   ├── classify_cell.py     # Baseline + CNN cell classification
│   └── aggregate.py         # Results compilation
├── models/
│   └── cnn_model.py         # PyTorch CNN for cell classification
└── utils/
    └── image_utils.py       # Image I/O, transforms, visualization

server.py                    # FastAPI backend
static/
├── index.html               # Frontend UI
├── style.css                # Dark theme styles
└── app.js                   # Frontend logic
```

## API Endpoints

| Method | Path           | Description                          |
|--------|----------------|--------------------------------------|
| GET    | `/`            | Serve the frontend                   |
| POST   | `/api/process` | Process an attendance sheet image    |
| GET    | `/api/health`  | Health check                         |

## Configuration

Edit `attendance_ai/config/template.json` to adjust:
- Table layout ratios (name column, grid boundaries)
- Classification thresholds (dark pixel ratio)
- Preprocessing parameters (denoise, adaptive threshold)
- Line detection settings

## CNN Training (Optional)

To train the CNN classifier, organize labeled cell images:

```
training_data/
├── A/
│   ├── cell_001.png
│   └── ...
├── BLANK/
│   └── ...
└── UNKNOWN/
    └── ...
```

Then:

```python
from attendance_ai.models.cnn_model import train_model
train_model("training_data/", epochs=30, save_path="models/cell_classifier.pth")
```

Pass the model to the pipeline:

```python
from attendance_ai.main import process_attendance_sheet
result = process_attendance_sheet("photo.jpg", model_path="models/cell_classifier.pth")
```
