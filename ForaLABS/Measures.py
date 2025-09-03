# Measures.py — ScriptedLoadableModule (3D Slicer)
# MESH-only mode (MeshLab-like)
# Metrics: Pore count (genus), Surface area (mm²), Volume (mm³), S/V (mm⁻¹), Pore density (#/mm²)
# Extra: MeshLab-like cleaning (Remove Isolated Pieces wrt Diameter) by **percentage of BBox diagonal**
# Thickness: **Medial method (2×distance)** — simplified
# Provisional extra: **average thickness V/A = Volume/Area** (literature: inverse of specific surface area)
# Now with **thickness map**: exports NRRD volume and paints mesh with per-vertex scalar

import os
import time
import logging
import numpy as np
import SimpleITK as sitk
from vtk.util import numpy_support as ns

import vtk
import qt
import ctk
import slicer

from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin

# =========================================================
# Module
# =========================================================
class Measures(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "Measures"
        self.parent.categories = ["ForaLABS"]
        self.parent.contributors = ["Harlley Hauradou (Nuclear Engineering Program of UFRJ)", "Thaís Hauradou (Nuclear Engineering Program of UFRJ)"]
        self.parent.helpText = (
            "Computes metrics directly from a segment's mesh (Closed surface): Pore count (genus), "
            "Surface area, Volume, S/V and Pore density (count/area). Includes MeshLab-like cleaning: "
            "Remove Isolated Pieces (wrt Diameter) with threshold in % of the BBox diagonal. Measures thickness by "
            "distance to the medial surface (2×distance) and generates a **thickness map** (NRRD volume + mesh colormap)."
        )
        self.parent.acknowledgementText = "VTK (MassProperties/Connectivity/ExtractEdges/CleanPolyData)."

# =========================================================
# Widget (UI)
# =========================================================
class MeasuresWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)
        self.logic = MeasuresLogic()

        # Load external .ui and bind widgets
        try:
            uiWidget = slicer.util.loadUI(self.resourcePath('UI/Measures.ui'))
        except Exception:
            import os
            ui_path = os.path.join(os.path.dirname(__file__), 'Resources', 'UI', 'Measures.ui')
            uiWidget = slicer.util.loadUI(ui_path)
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # Optional aliases (keep old attribute names)
        self.segSelector           = self.ui.segSelector
        self.wallSegmentCombo      = self.ui.wallSegmentCombo
        self.refreshSegsBtn        = self.ui.refreshSegsButton
        self.minDiamPctSpin        = self.ui.minDiamPctSpin
        self.removeUnrefCheck      = self.ui.removeUnrefCheck
        self.poreDensityOuterCheck = self.ui.poreDensityOuterCheck
        self.computeBtn            = self.ui.computeButton
        self.exportBtn             = self.ui.exportButton
        self.saveMeshBtn           = self.ui.saveMeshButton
        self.outputBrowser         = self.ui.outputBrowser
        self.thickEnableCheck      = self.ui.thickEnableCheck
        self.thickVoxelSpin        = self.ui.thickVoxelSpin
        self.genThickBtn           = self.ui.genThickButton
        self.expThickBtn           = self.ui.expThickButton
        self.showThickBtn          = self.ui.showThickButton
        self.statusLabel           = getattr(self.ui, 'statusLabel', qt.QLabel())

        # MRML hookup required for qMRMLNodeComboBox in .ui
        self.segSelector.setMRMLScene(slicer.mrmlScene)

        # Signal connections
        self.refreshSegsBtn.clicked.connect(self.onRefreshSegments)
        self.computeBtn.clicked.connect(self.onCompute)
        self.exportBtn.clicked.connect(self.onExport)
        self.saveMeshBtn.clicked.connect(self.onSaveCleanMesh)
        self.genThickBtn.clicked.connect(self.onGenerateThicknessMap)
        self.expThickBtn.clicked.connect(self.onExportThicknessMap)
        self.showThickBtn.clicked.connect(self.onShowThicknessMapSlices)

        self.layout.addStretch(1)


    # ---- UI helpers ----
    def onRefreshSegments(self):
        self.wallSegmentCombo.clear()
        segNode = self.segSelector.currentNode()
        if not segNode: return
        seg = segNode.GetSegmentation()
        ids = vtk.vtkStringArray(); seg.GetSegmentIDs(ids)
        for i in range(ids.GetNumberOfValues()):
            sid = ids.GetValue(i); name = seg.GetSegment(sid).GetName()
            self.wallSegmentCombo.addItem(f"{name} [{sid}]", sid)

    def _currentSegmentId(self):
        sid = str(self.wallSegmentCombo.currentData)
        if not sid and self.wallSegmentCombo.count() > 0:
            self.wallSegmentCombo.setCurrentIndex(0)
            sid = str(self.wallSegmentCombo.currentData)
        return sid

    # ---- Actions ----
    def onCompute(self):
        segNode = self.segSelector.currentNode(); assert segNode, "Select a SegmentationNode."
        wallSid = self._currentSegmentId(); assert wallSid, "Select the WALL segment."

        # read parameters
        try: minPct = float(self.minDiamPctSpin.value())
        except TypeError: minPct = float(self.minDiamPctSpin.value)
        try: rmUnref = bool(self.removeUnrefCheck.isChecked())
        except TypeError: rmUnref = bool(self.removeUnrefCheck.checked)

        # thickness (medial, optional)
        try:
            thickEnable = bool(self.thickEnableCheck.isChecked())
        except TypeError:
            thickEnable = bool(self.thickEnableCheck.checked)
        try: thickVoxel = float(self.thickVoxelSpin.value())
        except TypeError: thickVoxel = float(self.thickVoxelSpin.value)

        self.statusLabel.text = "Computing…"; slicer.app.processEvents()
        t0 = time.perf_counter()
        metrics = self.logic.compute_metrics(
            segNode, wallSid,
            minDiamPct=minPct, removeUnref=rmUnref,
            thickness_enabled=thickEnable, thickness_voxel_mm=thickVoxel
        )
        dt = time.perf_counter() - t0

        html = self.logic.render_metrics_html(metrics, elapsed=dt)
        self.outputBrowser.setHtml(html)
        self.statusLabel.text = f"Metrics computed (≈ {dt:.2f} s)."

    def onExport(self):
        metrics = self.logic.last_metrics; assert metrics, "Compute metrics first."
        outDir = qt.QFileDialog.getExistingDirectory(None, "Choose output folder")
        if not outDir: return
        htmlPath, csvPath = self.logic.export_results(outDir)
        qt.QMessageBox.information(slicer.util.mainWindow(), "Measures", f"Exported: - {htmlPath} - {csvPath}")

    def onSaveCleanMesh(self):
        # Save cleaned mesh (after island removal) as STL/PLY
        if getattr(self.logic, 'last_clean_poly', None) is None:
            qt.QMessageBox.warning(slicer.util.mainWindow(), "Measures", "Compute metrics first (it generates the cleaned mesh).")
            return
        try:
            fn = qt.QFileDialog.getSaveFileName(None, "Save cleaned mesh", "", "STL (*.stl);;PLY (*.ply)")
            outPath = fn[0] if isinstance(fn, tuple) else fn
        except Exception:
            outPath = None
        if not outPath:
            return
        ok, finalPath = self.logic.save_clean_mesh(outPath)
        if ok:
            qt.QMessageBox.information(slicer.util.mainWindow(), "Measures", f"Mesh saved to: {finalPath}")
        else:
            qt.QMessageBox.critical(slicer.util.mainWindow(), "Measures", "Failed to save cleaned mesh.")

    def onGenerateThicknessMap(self):
        segNode = self.segSelector.currentNode(); assert segNode, "Select a SegmentationNode."
        wallSid = self._currentSegmentId(); assert wallSid, "Select the WALL segment."
        try: voxel = float(self.thickVoxelSpin.value())
        except TypeError: voxel = float(self.thickVoxelSpin.value)
        # robustly read cleaning (CTK may expose value as a property)
        try:
            mdp = float(self.minDiamPctSpin.value())
        except TypeError:
            mdp = float(self.minDiamPctSpin.value)
        try:
            rmUnref = bool(self.removeUnrefCheck.isChecked())
        except TypeError:
            rmUnref = bool(self.removeUnrefCheck.checked)
        self.statusLabel.text = "Generating thickness map…"; slicer.app.processEvents()
        ok, modelNode = self.logic.generate_thickness_map_on_mesh(segNode, wallSid, voxel_mm=voxel, minDiamPct=mdp, removeUnref=rmUnref)
        if ok:
            self.statusLabel.text = "Map applied to mesh (see colored model in the scene)."
        else:
            self.statusLabel.text = "Failed to generate thickness map. Check the log."

    def onExportThicknessMap(self):
        segNode = self.segSelector.currentNode(); assert segNode, "Select a SegmentationNode."
        wallSid = self._currentSegmentId(); assert wallSid, "Select the WALL segment."
        try: voxel = float(self.thickVoxelSpin.value())
        except TypeError: voxel = float(self.thickVoxelSpin.value)
        try:
            fn = qt.QFileDialog.getSaveFileName(None, "Export thickness map (NRRD)", "thickness_map.nrrd", "NRRD (*.nrrd)")
            outPath = fn[0] if isinstance(fn, tuple) else fn
        except Exception:
            outPath = None
        if not outPath:
            return
        # robustly read cleaning
        try:
            mdp = float(self.minDiamPctSpin.value())
        except TypeError:
            mdp = float(self.minDiamPctSpin.value)
        try:
            rmUnref = bool(self.removeUnrefCheck.isChecked())
        except TypeError:
            rmUnref = bool(self.removeUnrefCheck.checked)
        self.statusLabel.text = "Computing & exporting map (NRRD)…"; slicer.app.processEvents()
        ok, finalPath = self.logic.export_thickness_nrrd(segNode, wallSid, outPath, voxel_mm=voxel, minDiamPct=mdp, removeUnref=rmUnref)
        if ok:
            qt.QMessageBox.information(slicer.util.mainWindow(), "Measures", f"Map saved to: {finalPath}")
            self.statusLabel.text = "NRRD map exported."
        else:
            qt.QMessageBox.critical(slicer.util.mainWindow(), "Measures", "Failed to export map.")
            self.statusLabel.text = "Failed to export map."

    def onShowThicknessMapSlices(self):
        segNode = self.segSelector.currentNode(); assert segNode, "Select a SegmentationNode."
        wallSid = self._currentSegmentId(); assert wallSid, "Select the WALL segment."
        try:
            voxel = float(self.thickVoxelSpin.value())
        except TypeError:
            voxel = float(self.thickVoxelSpin.value)
        # cleaning
        try:
            mdp = float(self.minDiamPctSpin.value())
        except TypeError:
            mdp = float(self.minDiamPctSpin.value)
        try:
            rmUnref = bool(self.removeUnrefCheck.isChecked())
        except TypeError:
            rmUnref = bool(self.removeUnrefCheck.checked)
        self.statusLabel.text = "Preparing thickness volume…"; slicer.app.processEvents()
        ok, volNode = self.logic.show_thickness_in_slices(segNode, wallSid, voxel_mm=voxel, minDiamPct=mdp, removeUnref=rmUnref)
        if ok:
            self.statusLabel.text = "Map displayed in slice views (Red/Yellow/Green)."
        else:
            self.statusLabel.text = "Failed to show in slices. Check the log."

    # =========================================================
# Logic (mesh-only)
# =========================================================
class MeasuresLogic(ScriptedLoadableModuleLogic):
    def __init__(self):
        super().__init__()
        self.last_metrics = None
        self.last_clean_poly = None
        self.last_thick_img = None  # SimpleITK Image
        self.last_thick_info = None # dict: {spacing, origin, shape}
        self.last_thick_model = None # vtkMRMLModelNode

    def compute_metrics(self, segNode, wallSegmentId, minDiamPct=10.0, removeUnref=True,
                       thickness_enabled=False, thickness_voxel_mm=0.003):
        poly = self._segment_to_polydata(segNode, wallSegmentId)
        poly = self._prepare_polydata(poly)

        # Pre-clean: remove isolated pieces by diameter relative to BBox diagonal (MeshLab-like)
        bounds = poly.GetBounds(); dx=bounds[1]-bounds[0]; dy=bounds[3]-bounds[2]; dz=bounds[5]-bounds[4]
        bbox_diag = (dx*dx + dy*dy + dz*dz)**0.5 if poly.GetNumberOfPoints()>0 else 0.0
        minDiamMM = (bbox_diag * (float(minDiamPct)/100.0)) if minDiamPct and bbox_diag>0 else 0.0
        poly, clean_info = self._remove_isolated_pieces_by_diameter(poly, minDiamMM, removeUnref=removeUnref)
        clean_info.update({"min_diam_pct": float(minDiamPct), "bbox_diag": float(bbox_diag)})
        self.last_clean_poly = poly

        # Area and volume
        A_mesh, V_mesh = self._surface_area_and_volume(poly)
        # Outer area (only surfaces in contact with exterior), for V/A
        try:
            voxel_for_A = float(thickness_voxel_mm) if thickness_enabled else 0.003
        except Exception:
            voxel_for_A = 0.003
        try:
            A_outer = self._outer_surface_area_mm2_from_poly(poly, voxel_mm=voxel_for_A)
        except Exception as e:
            logging.error(f"A_outer failed, using total area: {e}")
            A_outer = A_mesh
        # Average thickness via V/A (mm)
        thickness_va_mm = (V_mesh / A_mesh) if A_mesh > 0 else float('nan')

        # Pore count via genus
        pore_count = self._genus_based_pore_count(poly)

        # Derived
        S_over_V = (A_mesh / V_mesh) if V_mesh > 0 else float('nan')
        pore_density = (pore_count / A_mesh) if A_mesh > 0 else float('nan')  # #/mm²

        # Thickness (medial)
        thick = None
        if thickness_enabled:
            try:
                # generate map and get stats
                sitk_img, info = self._thickness_map_medial_from_poly(poly, voxel_mm=float(thickness_voxel_mm))
                self.last_thick_img = sitk_img; self.last_thick_info = info
                arr = sitk.GetArrayFromImage(sitk_img)
                vals = arr[np.isfinite(arr) & (arr>0)]
                if vals.size == 0:
                    thick = {"mean_mm": 0.0, "std_mm": 0.0, "voxel_mm": float(thickness_voxel_mm)}
                else:
                    thick = {"mean_mm": float(np.mean(vals)), "std_mm": float(np.std(vals)), "voxel_mm": float(thickness_voxel_mm)}
            except Exception as e:
                logging.error(f"Thickness (medial) error: {e}")
                thick = None

        metrics = {
            "count_pores": int(pore_count),
            "surface_area_mm2": float(A_mesh),
            "outer_area_mm2": float(A_outer),
            "volume_mm3": float(V_mesh),
            "S_over_V": float(S_over_V),
            "thickness_v_over_a_mm": float((V_mesh / A_outer) if A_outer>0 else float('nan')),
            "pore_density_per_mm2": float(pore_density),
            "thickness_v_over_a_mm": float(thickness_va_mm),
            "cleaning": clean_info,
            "thickness_mm": thick
        }
        self.last_metrics = metrics
        return metrics

    # ---------- Render (HTML) ----------
    def render_metrics_html(self, m: dict, elapsed: float = None) -> str:
        def _fmt(x, p=6):
            if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
                return "—"
            return f"{float(x):.{p}f}"
        note = ""
        clean = m.get('cleaning') or {}
        try:
            md = float(clean.get('min_diam_mm', 0.0))
            mp = float(clean.get('min_diam_pct', 0.0))
            bb = float(clean.get('bbox_diag', 0.0))
            if mp > 0.0:
                kept = int(clean.get('regions_kept', 0))
                total = int(clean.get('regions_total', 0))
                overflow = bool(clean.get('overflow_largest_only', False))
                extra = " (fallback: largest component only)" if overflow else ""
                note = f"<div class='sub' style='margin-top:6px;'>Cleaning: removed {max(0,total-kept)} islands &lt; {mp:.1f}% of diag. (≈ {md:.3f} mm of {bb:.6f} mm); kept: {kept}/{total}{extra}.</div>"
        except Exception:
            pass
        thick = m.get('thickness_mm') or {}
        html = f"""
        <html>
        <head>
        <meta charset='utf-8'>
        <style>
          body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Ubuntu, 'Helvetica Neue', Arial, sans-serif; color:#222; }}
          .grid {{ display:grid; grid-template-columns: repeat(2, minmax(280px,1fr)); gap:14px; }}
          .card {{ border:1px solid #e5e7eb; border-radius:12px; padding:12px 14px; box-shadow:0 1px 2px rgba(0,0,0,0.04); }}
          .title {{ font-size:18px; font-weight:600; margin:0 0 6px; }}
          .metric {{ font-size:28px; font-weight:700; margin:2px 0; }}
          .sub {{ color:#555; font-size:12px; }}
          table {{ width:100%; border-collapse:collapse; margin-top:8px; }}
          th, td {{ text-align:left; padding:6px 8px; border-bottom:1px solid #f0f0f0; font-size:14px; }}
        </style>
        </head>
        <body>
          <div class='grid'>
            <div class='card'>
              <div class='title'>Pore count (genus)</div>
              <div class='metric'>{int(m.get('count_pores') or 0)}</div>
              <div class='sub'>Number of handles/tunnels in the triangulated mesh.</div>
            </div>
            <div class='card'>
              <div class='title'>Surface area</div>
              <div class='metric'>{_fmt(m.get('surface_area_mm2'), p=6)} <span class='sub'>mm²</span></div>
              <div class='sub'>Total mesh area (VTK MassProperties).</div>
            </div>
            <div class='card'>
              <div class='title'>Volume</div>
              <div class='metric'>{_fmt(m.get('volume_mm3'), p=6)} <span class='sub'>mm³</span></div>
              <div class='sub'>Solid volume from mesh (VTK MassProperties).</div>
            </div>
            <div class='card'>
              <div class='title'>S/V</div>
              <div class='metric'>{_fmt(m.get('S_over_V'), p=6)} <span class='sub'>mm⁻¹</span></div>
              <div class='sub'>Area/volume ratio (MeshLab-like).</div>
            </div>
            <div class='card'>
              <div class='title'>Pore density</div>
              <div class='metric'>{_fmt(m.get('pore_density_per_mm2'), p=6)} <span class='sub'>mm⁻²</span></div>
              <div class='sub'>Pore count divided by total area.</div>
            </div>
            <div class='card'>
              <div class='title'>Shell thickness</div>
              <table>
                <tr><th>Mean (medial)</th><td>{_fmt(((thick.get('mean_mm')*1000.0) if thick else None), p=3)} µm</td></tr>
                <tr><th>SD (medial)</th><td>{_fmt(((thick.get('std_mm')*1000.0) if thick else None), p=3)} µm</td></tr>
                <tr><th>Voxel</th><td>{_fmt(((thick.get('voxel_mm')*1000.0) if thick else None), p=3)} µm</td></tr>
                <tr><th>Mean (V/A)</th><td>{_fmt((m.get('thickness_v_over_a_mm')*1000.0 if m.get('thickness_v_over_a_mm') is not None else None), p=3)} µm</td></tr>
              </table>
            </div>
          </div>
          <div class='sub' style='margin-top:10px;'>Time {'' if elapsed is None else f'≈ {elapsed:.2f} s'} • Mesh-only</div>
          {note}
        </body>
        </html>
        """
        return html

    # ---------- Export ----------
    def export_results(self, outDir):
        html = self.render_metrics_html(self.last_metrics or {}, elapsed=None)
        htmlPath = os.path.join(outDir, "Measures_mesh_metrics.html")
        with open(htmlPath, 'w', encoding='utf-8') as f:
            f.write(html)

        csvPath = os.path.join(outDir, "Measures_mesh_metrics.csv")
        m = self.last_metrics or {}
        lines = [
            "metric,value",
            f"count_pores,{m.get('count_pores','')}",
            f"surface_area_mm2,{m.get('surface_area_mm2','')}",
            f"outer_area_mm2,{m.get('outer_area_mm2','')}",
            f"volume_mm3,{m.get('volume_mm3','')}",
            f"S_over_V,{m.get('S_over_V','')}",
            f"pore_density_per_mm2,{m.get('pore_density_per_mm2','')}",
            f"thickness_v_over_a_mm,{m.get('thickness_v_over_a_mm','')}",
            f"thickness_v_over_a_um,{(m.get('thickness_v_over_a_mm')*1000.0) if isinstance(m.get('thickness_v_over_a_mm'), (int,float)) else ''}"
        ]
        t = m.get('thickness_mm') or {}
        if t:
            lines += [
                f"thickness_medial_mean_um,{(t.get('mean_mm')*1000.0) if isinstance(t.get('mean_mm'), (int,float)) else ''}",
                f"thickness_medial_std_um,{(t.get('std_mm')*1000.0) if isinstance(t.get('std_mm'), (int,float)) else ''}",
                f"thickness_voxel_um,{(t.get('voxel_mm')*1000.0) if isinstance(t.get('voxel_mm'), (int,float)) else ''}"
            ]
        with open(csvPath, 'w', encoding='utf-8', newline='') as f:
            f.write("".join(lines) + "")
        logging.info(f"Exported: {htmlPath}, {csvPath}")
        return htmlPath, csvPath

    def save_clean_mesh(self, outPath):
        """Saves the cleaned mesh (last generated) as STL or PLY. Returns (ok, finalPath)."""
        try:
            poly_in = getattr(self, 'last_clean_poly', None)
            if poly_in is None:
                return False, outPath
            # Triangulate and clean for safety
            tf = vtk.vtkTriangleFilter(); tf.SetInputData(poly_in); tf.Update()
            clean = vtk.vtkCleanPolyData(); clean.SetInputData(tf.GetOutput()); clean.Update()
            poly = clean.GetOutput()
            ext = os.path.splitext(outPath)[1].lower()
            finalPath = outPath
            if ext == ".stl":
                w = vtk.vtkSTLWriter(); w.SetFileName(finalPath); w.SetInputData(poly); w.SetFileTypeToBinary(); ok = w.Write()
            elif ext == ".ply":
                w = vtk.vtkPLYWriter(); w.SetFileName(finalPath); w.SetInputData(poly); w.SetFileTypeToBinary(); ok = w.Write()
            else:
                finalPath = outPath + ".stl"
                w = vtk.vtkSTLWriter(); w.SetFileName(finalPath); w.SetInputData(poly); w.SetFileTypeToBinary(); ok = w.Write()
            return bool(ok), finalPath
        except Exception as e:
            logging.error(f"Error saving mesh: {e}")
            return False, outPath

    # ---------- Helpers: MESH ----------
    def _segment_to_polydata(self, segNode, segmentId):
        if not segNode.GetSegmentation().ContainsRepresentation("Closed surface"):
            segNode.CreateClosedSurfaceRepresentation()
        poly = vtk.vtkPolyData(); segNode.GetClosedSurfaceRepresentation(segmentId, poly)
        return poly

    def _prepare_polydata(self, poly):
        tf = vtk.vtkTriangleFilter(); tf.SetInputData(poly); tf.Update()
        clean = vtk.vtkCleanPolyData(); clean.SetInputData(tf.GetOutput()); clean.Update()
        return clean.GetOutput()

    def _surface_area_and_volume(self, poly):
        mp = vtk.vtkMassProperties(); mp.SetInputData(poly); mp.Update()
        return float(mp.GetSurfaceArea()), float(mp.GetVolume())

    def _genus_based_pore_count(self, poly):
        conn = vtk.vtkPolyDataConnectivityFilter(); conn.SetInputData(poly)
        conn.SetExtractionModeToAllRegions(); conn.ColorRegionsOn(); conn.Update()
        total_genus = 0
        for rid in range(conn.GetNumberOfExtractedRegions()):
            c2 = vtk.vtkPolyDataConnectivityFilter(); c2.SetInputData(poly)
            c2.SetExtractionModeToSpecifiedRegions(); c2.AddSpecifiedRegion(rid); c2.Update()
            comp = self._prepare_polydata(c2.GetOutput())
            V = comp.GetNumberOfPoints(); F = comp.GetNumberOfCells(); E = self._fast_edge_count(comp)
            chi = V - E + F; g = max(0, int(round((2 - chi) / 2)))
            total_genus += g
        return total_genus

    def _fast_edge_count(self, poly):
        extr = vtk.vtkExtractEdges(); extr.SetInputData(poly); extr.Update()
        return extr.GetOutput().GetNumberOfLines()

    # ---------- Voxelization + map (Medial 2×distance) ----------
    def _voxelize_poly(self, poly, voxel_mm=0.003, margin_vox=1):
        bounds = poly.GetBounds()
        if poly.GetNumberOfPoints() == 0:
            raise RuntimeError("Empty PolyData for voxelization.")
        sp = [float(voxel_mm)]*3
        dx=bounds[1]-bounds[0]; dy=bounds[3]-bounds[2]; dz=bounds[5]-bounds[4]
        dims = [int(np.ceil(dx/sp[0]))+2*margin_vox,
                int(np.ceil(dy/sp[1]))+2*margin_vox,
                int(np.ceil(dz/sp[2]))+2*margin_vox]
        origin = [bounds[0]-margin_vox*sp[0], bounds[2]-margin_vox*sp[1], bounds[4]-margin_vox*sp[2]]
        img = vtk.vtkImageData(); img.SetSpacing(sp); img.SetOrigin(origin); img.SetDimensions(dims); img.AllocateScalars(vtk.VTK_UNSIGNED_CHAR,1)
        # fill with 1
        arr = np.ones((dims[2], dims[1], dims[0]), dtype=np.uint8)
        vtkArr = ns.numpy_to_vtk(arr.ravel(order='C'), deep=1, array_type=vtk.VTK_UNSIGNED_CHAR)
        img.GetPointData().SetScalars(vtkArr)
        # mesh stencil
        p2s = vtk.vtkPolyDataToImageStencil(); p2s.SetInputData(poly); p2s.SetOutputSpacing(sp); p2s.SetOutputOrigin(origin); p2s.SetOutputWholeExtent(img.GetExtent()); p2s.Update()
        st = vtk.vtkImageStencil(); st.SetInputData(img); st.SetStencilConnection(p2s.GetOutputPort()); st.ReverseStencilOff(); st.SetBackgroundValue(0); st.Update()
        out = st.GetOutput(); out_arr = ns.vtk_to_numpy(out.GetPointData().GetScalars()).reshape(dims[2], dims[1], dims[0])
        sitk_img = sitk.GetImageFromArray(out_arr.astype(np.uint8))
        sitk_img.SetSpacing(tuple(sp))
        sitk_img.SetOrigin(tuple(origin))
        return sitk_img, tuple(sp), tuple(origin), (dims[2], dims[1], dims[0])

    def _thickness_map_medial_from_poly(self, poly, voxel_mm=0.003):
        mask, spacing, origin, shape = self._voxelize_poly(poly, voxel_mm=float(voxel_mm))
        mask_u8 = sitk.Cast(mask>0, sitk.sitkUInt8)
        # Internal DT (mm)
        dt = sitk.SignedMaurerDistanceMap(mask_u8, insideIsPositive=True, squaredDistance=False, useImageSpacing=True, backgroundValue=0)
        # approximate medial as local maxima of DT (26-neighborhood)
        gd = sitk.GrayscaleDilate(dt, [1,1,1])
        medial = sitk.Equal(dt, gd)
        # Distance to medial (mm)
        d2m = sitk.SignedMaurerDistanceMap(sitk.Cast(medial, sitk.sitkUInt8), insideIsPositive=False, squaredDistance=False, useImageSpacing=True, backgroundValue=0)
        arr = sitk.GetArrayFromImage(d2m)
        thick_arr = 2.0 * arr
        # zero outside the object
        mask_arr = sitk.GetArrayFromImage(mask_u8) > 0
        thick_arr = np.where(mask_arr, thick_arr, 0.0).astype(np.float32)
        # create SITK image for thickness
        thick_img = sitk.GetImageFromArray(thick_arr)
        thick_img.SetSpacing(spacing)
        thick_img.SetOrigin(origin)
        info = {"spacing": spacing, "origin": origin, "shape": thick_arr.shape}
        return thick_img, info

    def _thickness_stats_medial_from_poly(self, poly, voxel_mm=0.003):
        thick_img, info = self._thickness_map_medial_from_poly(poly, voxel_mm=float(voxel_mm))
        arr = sitk.GetArrayFromImage(thick_img)
        vals = arr[np.isfinite(arr) & (arr>0)]
        if vals.size == 0:
            return {"mean_mm": 0.0, "std_mm": 0.0, "voxel_mm": float(voxel_mm)}
        return {"mean_mm": float(np.mean(vals)), "std_mm": float(np.std(vals)), "voxel_mm": float(voxel_mm)}

    # ---------- Map on mesh / Export ----------
    def generate_thickness_map_on_mesh(self, segNode, wallSegmentId, voxel_mm=0.003, minDiamPct=10.0, removeUnref=True):
        try:
            poly = self._segment_to_polydata(segNode, wallSegmentId)
            poly = self._prepare_polydata(poly)
            # same cleaning as metrics
            bounds = poly.GetBounds(); dx=bounds[1]-bounds[0]; dy=bounds[3]-bounds[2]; dz=bounds[5]-bounds[4]
            bbox_diag = (dx*dx + dy*dy + dz*dz)**0.5 if poly.GetNumberOfPoints()>0 else 0.0
            minDiamMM = (bbox_diag * (float(minDiamPct)/100.0)) if minDiamPct and bbox_diag>0 else 0.0
            poly, _ = self._remove_isolated_pieces_by_diameter(poly, minDiamMM, removeUnref=removeUnref)
            self.last_clean_poly = poly
            # thickness map
            thick_img, info = self._thickness_map_medial_from_poly(poly, voxel_mm=float(voxel_mm))
            self.last_thick_img = thick_img; self.last_thick_info = info
            # paint mesh
            modelNode = self._paint_poly_with_thickness(poly, thick_img, info)
            self.last_thick_model = modelNode
            return True, modelNode
        except Exception as e:
            logging.error(f"generate_thickness_map_on_mesh: {e}")
            return False, None

    def export_thickness_nrrd(self, segNode, wallSegmentId, outPath, voxel_mm=0.003, minDiamPct=10.0, removeUnref=True):
        try:
            # if we already have it, save directly
            if self.last_thick_img is not None:
                sitk.WriteImage(self.last_thick_img, outPath)
                return True, outPath
            # otherwise, generate and save
            ok, _ = self.generate_thickness_map_on_mesh(segNode, wallSegmentId, voxel_mm=float(voxel_mm),
                                                        minDiamPct=minDiamPct, removeUnref=removeUnref)
            if not ok or self.last_thick_img is None:
                return False, outPath
            sitk.WriteImage(self.last_thick_img, outPath)
            return True, outPath
        except Exception as e:
            logging.error(f"export_thickness_nrrd: {e}")
            return False, outPath

    def show_thickness_in_slices(self, segNode, wallSegmentId, voxel_mm=0.003, minDiamPct=10.0, removeUnref=True):
        try:
            # ensure thickness image
            if self.last_thick_img is None:
                poly = self._segment_to_polydata(segNode, wallSegmentId)
                poly = self._prepare_polydata(poly)
                bounds = poly.GetBounds(); dx=bounds[1]-bounds[0]; dy=bounds[3]-bounds[2]; dz=bounds[5]-bounds[4]
                bbox_diag = (dx*dx + dy*dy + dz*dz)**0.5 if poly.GetNumberOfPoints()>0 else 0.0
                minDiamMM = (bbox_diag * (float(minDiamPct)/100.0)) if minDiamPct and bbox_diag>0 else 0.0
                poly, _ = self._remove_isolated_pieces_by_diameter(poly, minDiamMM, removeUnref=removeUnref)
                self.last_clean_poly = poly
                self.last_thick_img, self.last_thick_info = self._thickness_map_medial_from_poly(poly, voxel_mm=float(voxel_mm))
            # create MRML volume from image
            volNode = self._thickness_volume_node_from_sitk(self.last_thick_img, name="Measures_ThicknessNRRD")
            # set as background in the three slice views
            try:
                slicer.util.setSliceViewerLayers(background=volNode, fit=True)
            except Exception:
                lm = slicer.app.layoutManager()
                for viewName in ("Red", "Yellow", "Green"):
                    sliceWidget = lm.sliceWidget(viewName)
                    if sliceWidget:
                        comp = sliceWidget.mrmlSliceCompositeNode()
                        comp.SetBackgroundVolumeID(volNode.GetID())
                        sliceWidget.fitSliceToBackground()
            return True, volNode
        except Exception as e:
            logging.error(f"show_thickness_in_slices: {e}")
            return False, None

    def _thickness_volume_node_from_sitk(self, sitk_img, name="Measures_ThicknessNRRD"):
        arr = sitk.GetArrayFromImage(sitk_img).astype(np.float32)  # (k,j,i)
        volNode = slicer.util.addVolumeFromArray(arr)
        volNode.SetName(name)
        # apply spacing/origin
        sp = sitk_img.GetSpacing(); org = sitk_img.GetOrigin()
        try:
            volNode.SetSpacing(sp)
            volNode.SetOrigin(org)
        except Exception:
            if volNode.GetImageData():
                volNode.GetImageData().SetSpacing(sp)
        # colormap and window/level
        disp = volNode.GetDisplayNode()
        if disp:
            try:
                cn = slicer.util.getNode('Viridis')
                disp.SetAndObserveColorNodeID(cn.GetID())
            except Exception:
                try:
                    cn = slicer.util.getNode('vtkMRMLColorTableNodeRainbow')
                    disp.SetAndObserveColorNodeID(cn.GetID())
                except Exception:
                    pass
            try:
                disp.AutoWindowLevelOn()
            except Exception:
                pass
        # hide zero voxels: apply threshold > 0
        try:
            disp.SetLowerThreshold(1e-12)
            disp.SetUpperThreshold(1e9)
            disp.SetApplyThreshold(True)
        except AttributeError:
            try:
                disp.SetThreshold(1e-12, 1e9)
                disp.ApplyThresholdOn()
            except Exception:
                pass
        return volNode

    def _paint_poly_with_thickness(self, poly, thick_img, info):
        # sample nearest-neighbor thickness volume at each vertex
        spacing = np.array(info["spacing"])  # (sx,sy,sz)
        origin  = np.array(info["origin"])   # (ox,oy,oz)
        arr = sitk.GetArrayFromImage(thick_img)  # (k,j,i) => (z,y,x)
        nz, ny, nx = arr.shape
        # copy mesh for model
        polyCopy = vtk.vtkPolyData(); polyCopy.DeepCopy(poly)
        pts = polyCopy.GetPoints(); npts = pts.GetNumberOfPoints()
        thickVals = vtk.vtkFloatArray(); thickVals.SetName("Thickness_mm"); thickVals.SetNumberOfTuples(npts)
        for pid in range(npts):
            x,y,z = pts.GetPoint(pid)
            i = int(round((x - origin[0]) / spacing[0]))
            j = int(round((y - origin[1]) / spacing[1]))
            k = int(round((z - origin[2]) / spacing[2]))
            if 0 <= i < nx and 0 <= j < ny and 0 <= k < nz:
                val = float(arr[k, j, i])
            else:
                val = 0.0
            thickVals.SetValue(pid, val)
        polyCopy.GetPointData().AddArray(thickVals)
        polyCopy.GetPointData().SetActiveScalars("Thickness_mm")
        # create ModelNode
        modelNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", "Measures_ThicknessModel")
        modelNode.SetAndObservePolyData(polyCopy)
        disp = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelDisplayNode")
        disp.SetScalarVisibility(True)
        disp.SetActiveScalarName("Thickness_mm")
        # use MRML display node flag (not raw vtk)
        try:
            disp.SetScalarRangeFlag(slicer.vtkMRMLDisplayNode.UseDataScalarRange)
        except AttributeError:
            try:
                disp.SetScalarRangeFlagToUseDataScalarRange()
            except AttributeError:
                pass
        modelNode.SetAndObserveDisplayNodeID(disp.GetID())
        return modelNode

    # ---------- Cleaning by relative diameter ----------
    def _remove_isolated_pieces_by_diameter(self, poly, minDiamMM=0.0, removeUnref=True, hard_limit_regions=3000):
        """Remove disconnected components whose bbox diagonal is smaller than minDiamMM.
        If region count is too high (>hard_limit_regions), fallback: keep only largest component.
        Returns (clean_poly, info_dict)."""
        info = {"min_diam_mm": float(minDiamMM), "regions_total": 1, "regions_kept": 1, "overflow_largest_only": False}
        if minDiamMM is None or minDiamMM <= 0.0:
            return poly, info
        conn = vtk.vtkPolyDataConnectivityFilter(); conn.SetInputData(poly); conn.SetExtractionModeToAllRegions(); conn.ColorRegionsOn(); conn.Update()
        n = conn.GetNumberOfExtractedRegions(); info["regions_total"] = int(n)
        if n <= 1:
            info["regions_kept"] = 1
            return poly, info
        if n > hard_limit_regions:
            conn2 = vtk.vtkPolyDataConnectivityFilter(); conn2.SetInputData(poly); conn2.SetExtractionModeToLargestRegion(); conn2.Update()
            out = conn2.GetOutput(); info["regions_kept"] = 1; info["overflow_largest_only"] = True
            if removeUnref:
                clean = vtk.vtkCleanPolyData(); clean.SetInputData(out); clean.Update(); out = clean.GetOutput()
            return out, info
        app = vtk.vtkAppendPolyData(); kept = 0
        for rid in range(n):
            c2 = vtk.vtkPolyDataConnectivityFilter(); c2.SetInputData(poly); c2.SetExtractionModeToSpecifiedRegions(); c2.AddSpecifiedRegion(rid); c2.Update()
            comp = c2.GetOutput(); bounds = comp.GetBounds(); dx=bounds[1]-bounds[0]; dy=bounds[3]-bounds[2]; dz=bounds[5]-bounds[4]
            diam = (dx*dx + dy*dy + dz*dz)**0.5
            if diam >= float(minDiamMM):
                app.AddInputData(comp); kept += 1
        info["regions_kept"] = int(kept)
        if kept == 0:
            conn2 = vtk.vtkPolyDataConnectivityFilter(); conn2.SetInputData(poly); conn2.SetExtractionModeToLargestRegion(); conn2.Update(); out = conn2.GetOutput()
        else:
            app.Update(); out = app.GetOutput()
        if removeUnref:
            clean = vtk.vtkCleanPolyData(); clean.SetInputData(out); clean.Update(); out = clean.GetOutput()
        return out, info

    # ---------- Outer area for V/A ----------
    def _outer_surface_area_mm2_from_poly(self, poly, voxel_mm=0.003):
        """Shell area in contact with the EXTERIOR (mm²), via voxelization and face counting."""
        mask, spacing, origin, shape = self._voxelize_poly(poly, voxel_mm=float(voxel_mm))
        mask_u8 = sitk.Cast(mask>0, sitk.sitkUInt8)
        outside = self._outside_from_mask(mask_u8)
        arr_w = sitk.GetArrayFromImage(mask_u8)>0   # (z,y,x)
        arr_o = sitk.GetArrayFromImage(outside)>0
        sx, sy, sz = spacing  # (x,y,z) mm
        # faces perpendicular to X (pairs along x)
        c1 = np.count_nonzero(arr_w[:, :, :-1] & arr_o[:, :, 1:])
        c2 = np.count_nonzero(arr_o[:, :, :-1] & arr_w[:, :, 1:])
        area_x = (c1 + c2) * (sy * sz)
        # faces perpendicular to Y
        c1 = np.count_nonzero(arr_w[:, :-1, :] & arr_o[:, 1:, :])
        c2 = np.count_nonzero(arr_o[:, :-1, :] & arr_w[:, 1:, :])
        area_y = (c1 + c2) * (sx * sz)
        # faces perpendicular to Z
        c1 = np.count_nonzero(arr_w[:-1, :, :] & arr_o[1:, :, :])
        c2 = np.count_nonzero(arr_o[:-1, :, :] & arr_w[1:, :, :])
        area_z = (c1 + c2) * (sx * sy)
        return float(area_x + area_y + area_z)

    def _outside_from_mask(self, mask_u8):
        inv = sitk.BinaryNot(mask_u8)
        cc = sitk.ConnectedComponent(inv)
        stats = sitk.LabelShapeStatisticsImageFilter(); stats.Execute(cc)
        size = mask_u8.GetSize()  # (x,y,z)
        outside_label = None; outside_count = -1
        for lbl in stats.GetLabels():
            x, sx_, y, sy_, z, sz_ = stats.GetBoundingBox(lbl)
            touches = (x==0 or y==0 or z==0 or (x+sx_==size[0]) or (y+sy_==size[1]) or (z+sz_==size[2]))
            if touches:
                n = stats.GetNumberOfPixels(lbl)
                if n > outside_count:
                    outside_label, outside_count = lbl, n
        if outside_label is None:
            # fallback: largest background component
            lbls = list(stats.GetLabels())
            if not lbls:
                return inv
            outside_label = max(lbls, key=lambda L: stats.GetNumberOfPixels(L))
        outside = sitk.Equal(cc, int(outside_label))
        outside = sitk.Cast(outside, sitk.sitkUInt8)
        outside.CopyInformation(mask_u8)
        return outside

# =========================================================
# Tests (placeholder)
# =========================================================
class MeasuresTest(ScriptedLoadableModuleTest):
    def runTest(self):
        self.setUp(); self.test_All()
    def test_All(self):
        self.delayDisplay("Measures mesh-only test placeholder…"); self.delayDisplay("OK")
