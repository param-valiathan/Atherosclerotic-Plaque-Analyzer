# Miguelyser

**Atherosclerotic Plaque Analyzer v2.0**

A desktop application for automated segmentation and quantification of atherosclerotic plaque deposits in Oil Red O-stained mouse aorta microscopy images.

---

## What it does

Miguelyser takes brightfield microscopy images of en-face mouse aortas stained with Oil Red O and automatically:

- Detects the aorta tissue and circular microscope field boundary
- Identifies plaque deposits by color (Oil Red O stain appears orange-red)
- Locates the aortic Y-bifurcation and divides the tissue into three anatomical regions: **Trunk**, **Left Arm**, and **Right Arm**
- Calculates plaque burden (area %) per region and in total
- Exports annotated images and a results spreadsheet (`.xlsx`)

Manual boundary correction with a brush tool is available for images where automated segmentation needs adjustment.

---

## Download

Download the pre-built Windows executable from the [**Releases**](../../releases) page — no Python installation required.

---

## Running from source

### Requirements

- Python 3.9+
- Windows (tested on Windows 10/11)

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run

```bash
python aorta_analyzer.py
```

---

## Usage

1. **Open Folder** — select a directory containing your aorta images (JPEG, PNG, or TIFF)
2. **Process All** — run automated batch analysis
3. **Review** — use ← → arrow keys to navigate images; double-click to zoom
4. **Correct** *(optional)* — click **Edit Boundaries** to manually refine masks with brush tools
5. **Export** — save all results to `output/plaque_results.xlsx`

Annotated images are saved automatically to an `output/` subfolder inside the image directory.

See [**Miguelyser_User_Guide.pdf**](Miguelyser_User_Guide.pdf) for detailed instructions and parameter descriptions.

---

## Output columns

| Column | Description |
|---|---|
| `filename` | Image file name |
| `aorta_area_px` | Total aorta area (pixels) |
| `plaque_area_px` | Total plaque area (pixels) |
| `plaque_pct` | Total plaque burden (%) |
| `trunk_area_px` | Trunk region area |
| `trunk_plaque_px` | Trunk plaque area |
| `trunk_plaque_pct` | Trunk plaque burden (%) |
| `left_arm_area_px` | Left arm region area |
| `left_arm_plaque_px` | Left arm plaque area |
| `left_arm_plaque_pct` | Left arm plaque burden (%) |
| `right_arm_area_px` | Right arm region area |
| `right_arm_plaque_px` | Right arm plaque area |
| `right_arm_plaque_pct` | Right arm plaque burden (%) |

---

## Built with

- [OpenCV](https://opencv.org/) — image processing and segmentation
- [Tkinter](https://docs.python.org/3/library/tkinter.html) — GUI framework
- [Pillow](https://python-pillow.org/) — image rendering
- [pandas](https://pandas.pydata.org/) + [openpyxl](https://openpyxl.readthedocs.io/) — results export
- [NumPy](https://numpy.org/) / [SciPy](https://scipy.org/) — numerical processing

---

## License

This project is currently unlicensed. All rights reserved.

---

*Developed by P. Valiathan, 2026*
