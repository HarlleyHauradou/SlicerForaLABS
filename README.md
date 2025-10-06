# ForaLABS – microCT tools for foraminifera (3D Slicer modules)

[![Slicer 5.8.x](https://img.shields.io/badge/Slicer-5.8.x-blue)](#-installation)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-green)](#-license)
[![Status: Pre-release](https://img.shields.io/badge/Status-Pre--release-orange)](#-status)

**ForaLABS** is a set of **3D Slicer** modules for micro-CT analysis of foraminifera shells, focusing on **morphometric measurements**, **pore statistics**, and **thickness mapping**.

- **Measures** — mesh-based morphometrics (pore count & density, surface area, volume, S/V), MeshLab-like cleaning (remove islands by relative diameter), and **thickness map** (NRRD + mesh colormap) via medial-axis distance.
- **ImportTXM** — import from Zeiss TXM images recon and export to NRRD format.

> Tested with **3D Slicer 5.8.1 (Linux)**. Should also work on recent 5.x builds.

---

## Table of Contents

- [Features](#-features)
- [Installation](#-installation)
- [Quickstart (Measures)](#-quickstart-measures)
- [Outputs](#-outputs)
- [Methods (short)](#-methods-short)
- [Tips](#%EF%B8%8F-tips)
- [Status](#-status)
- [License](#-license)
- [Authors](#-authors)
- [Cite](#-cite)

---

## Features

### Measures
- One-click **morphometric report** (HTML in-app + optional PDF export).
- **Pore count** by mesh genus, **pore density** (count/mm²), **surface area**, **volume**, **S/V**.
- **Mesh cleaning**: *Remove Isolated Pieces (w.r.t. Diameter)* using a threshold as % of the bounding-box diagonal; optional removal of unreferenced vertices.
- **Thickness map**:
  - Computes **2 × distance to medial surface** on a voxel grid (Maurer distance) and exports **NRRD** (in mm).
  - **Colormap on mesh** + **Show in slice views** (Red/Yellow/Green).
- **Save cleaned mesh** as **STL/PLY** (binary).

### ImportTXM
- Import/Load from reconstructed images in Zeiss systems
- Quick TXM metadata parsing (pixel size, image size, binning, bits, frames, etc.) 

---

## Installation

> Requires **3D Slicer 5.8.x** (or recent 5.x), with built-in VTK & SimpleITK.

1. Clone or download this repository:
   ```bash
   git clone https://github.com/<your-org>/<repo>.git
   ```
2. Add the repo as an additional module path in Slicer:
   - **Edit ▸ Application Settings ▸ Modules ▸ Additional module paths** → **Add** the cloned folder and **Restart** Slicer.
   - (Alternatively) copy `ForaLABS/Measures` and `ForaLABS/ImportTXM` into your local Slicer **Modules** directory.
3. After restart, open **ForaLABS ▸ Measures** or **ForaLABS ▸ ImportTXM** from the Modules menu.

---

## Quickstart (Measures)

1. Load your segmentation with a **Closed surface** representation (create it if needed).
2. Open **ForaLABS ▸ Measures**.
3. Select your **Segmentation** and choose the **WALL** (shell) segment.
4. (Optional) Set **Min diameter (% diag)** for island removal and toggle **Remove unreferenced**.
5. Set **Thickness voxel (mm)** (typical microCT shells use ~1–5 µm; enter in **mm**, e.g. `0.0015` = 1.5 µm).
6. Click **Compute** → a metrics **HTML report** appears on the right.
7. Use the action buttons as needed:
   - **Export** → saves **Measures_mesh_metrics.pdf** (rendered from HTML report).
   - **Generate Thickness Map** → colors the mesh by thickness.
   - **Export Thickness (NRRD)** → saves the thickness volume (mm).
   - **Show Thickness in Slices** → displays the thickness volume in Red/Yellow/Green views.
   - **Save Clean Mesh** → exports STL/PLY of the cleaned shell.

---

## Outputs

- **HTML/PDF** report with:
  - Pores (count), Porosity (%)\*, Surface area (mm²), Volume (mm³), S/V (mm⁻¹),
  - Pore density (mm⁻²), Thickness mean/SD (µm), Thickness voxel (µm)
- **NRRD** thickness volume (mm), aligned to the shell’s bounding box.
- **Mesh** with `Thickness_mm` as active scalars (colored range from data).
- **Clean mesh** (largest components kept per threshold) as **STL/PLY**.

\* *Porosity (%) is currently experimental and may be revised in future releases.*

---

## Methods

- **Island removal (MeshLab-like)**: split connected components, compute each component bbox diagonal, **keep only** those ≥ `min_diameter_mm` (defined as a **percentage of the full mesh bbox diagonal**). Safety fallback: if region count is huge, keep **largest component only**.
- **Pore count via mesh genus**: per connected region, estimate Euler characteristic (χ = V − E + F), genus `g = max(0, round((2 − χ)/2))`; sum across regions to get **pore count**.
- **Thickness**: voxelize the shell (spacing = **thickness voxel**), compute **Maurer signed distance** to shell interior; approximate **medial surface** by local maxima (3×3×3). Then compute **2 × distance to medial surface** (inside), zero outside. Export as **NRRD** and map to mesh vertices by nearest-neighbor sampling.

---

## Tips

- **Thickness voxel**: too fine → high memory/time; too coarse → over-smoothed thickness. Start around **1–3 µm** for small shells and adjust.
- **Island removal**: begin with **10–20%** of bbox diagonal; increase if you still see spurious fragments.
- Use **Viridis** or **Rainbow** lookup for the thickness model; auto window/level is enabled.

---

## Status

- **Pre-release (beta)**: features are usable but may change. Please report issues with logs and, if possible, a minimal sample dataset.

---

## License

This project is licensed under the **Apache License 2.0**.  
See [`LICENSE`](./LICENSE) for details.

---

## Authors

- **Harlley Hauradou** (UFRJ – Nuclear Engineering Program)  
- **Thaís Hauradou** (UFRJ – Nuclear Engineering Program)

Academic/industry collaborators are welcome — reach out via Issues.

---

## Cite

If you use ForaLABS in academic work, please cite this repository (and any forthcoming publications). Example:
```text
Hauradou H., Hauradou T. (2025). ForaLABS – microCT tools for foraminifera (3D Slicer modules).
GitHub repository: https://github.com/<your-org>/<repo>
```

---
