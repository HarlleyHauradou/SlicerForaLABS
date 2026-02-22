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

from string import Template
import os, numpy as np

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
            "Computes measures directly from a segment's mesh (Closed surface): Pore count, and Pore density (count/area)." 
            "\nIncludes MeshLab-like cleaning: "
            "Remove Isolated Pieces (wrt Diameter) with threshold in % of the BBox diagonal. Measures thickness by "
            "distance to the medial surface (2×distance) and generates a **thickness map** (NRRD volume + mesh colormap)."
        )
        self.parent.acknowledgementText = ""

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
        uiWidget.setSizePolicy(qt.QSizePolicy.Expanding, qt.QSizePolicy.Expanding)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # Optional aliases (keep old attribute names)
        self.segSelector           = self.ui.segSelector
        self.wallSegmentCombo      = self.ui.wallSegmentCombo
        self.computeBtn            = self.ui.computeButton
        self.exportBtn             = self.ui.exportButton
        self.saveMeshBtn           = self.ui.saveMeshButton
        self.outputBrowser         = self.ui.outputBrowser
        self.thickVoxelSpin        = self.ui.thickVoxelSpin
        self.genThickBtn           = self.ui.genThickButton
        self.expThickBtn           = self.ui.expThickButton
        self.showThickBtn          = self.ui.showThickButton
        self.refreshSegsBtn        = self.ui.refreshSegsButton
        self.statusLabel           = getattr(self.ui, 'statusLabel', qt.QLabel())

        # Measure-selection checkboxes
        self.chkPores        = self.ui.chkPores
        self.chkPorosity     = self.ui.chkPorosity
        self.chkArea         = self.ui.chkArea
        self.chkVolume       = self.ui.chkVolume
        self.chkSV           = self.ui.chkSV
        self.chkPoreDensity  = self.ui.chkPoreDensity
        self.chkThickness    = self.ui.chkThickness

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

        # --- Settings section (stuck to bottom like Data Probe) ---
        settingsButton = ctk.ctkCollapsibleButton()
        settingsButton.text = "Settings"
        settingsButton.collapsed = True
        self.layout.addWidget(settingsButton)
        settingsLayout = qt.QFormLayout(settingsButton)

        self.minDiamPctSpin = ctk.ctkDoubleSpinBox()
        self.minDiamPctSpin.decimals = 1
        self.minDiamPctSpin.minimum = 0.0
        self.minDiamPctSpin.maximum = 100.0
        self.minDiamPctSpin.value = 10.0
        settingsLayout.addRow("Remove islands: % of BBox diagonal (0=off)", self.minDiamPctSpin)

        self.removeUnrefCheck = qt.QCheckBox("Remove unreferenced vertices")
        self.removeUnrefCheck.checked = True
        settingsLayout.addRow(self.removeUnrefCheck)

        # uiWidget expands to fill all available space (report area grows);
        # Settings stays compact at the bottom.
        self.layout.setStretchFactor(uiWidget, 1)
        self.layout.setStretchFactor(settingsButton, 0)


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
    def _read_include_flags(self):
        """Read the measure-selection checkboxes and return a dict."""
        return {
            "pores": self.chkPores.isChecked(),
            "porosity": self.chkPorosity.isChecked(),
            "area": self.chkArea.isChecked(),
            "volume": self.chkVolume.isChecked(),
            "sv": self.chkSV.isChecked(),
            "pore_density": self.chkPoreDensity.isChecked(),
            "thickness": self.chkThickness.isChecked(),
        }

    def onCompute(self):
        segNode = self.segSelector.currentNode(); assert segNode, "Select a SegmentationNode."
        wallSid = self._currentSegmentId(); assert wallSid, "Select the WALL segment."

        # read parameters
        try: minPct = float(self.minDiamPctSpin.value())
        except TypeError: minPct = float(self.minDiamPctSpin.value)
        try: rmUnref = bool(self.removeUnrefCheck.isChecked())
        except TypeError: rmUnref = bool(self.removeUnrefCheck.checked)

        # thickness 
        try: 
            thickVoxel = float(self.thickVoxelSpin.value())
        except TypeError: 
            thickVoxel = float(self.thickVoxelSpin.value)

        include = self._read_include_flags()

        self.statusLabel.text = "Computing…"; slicer.app.processEvents()
        t0 = time.perf_counter()
        metrics = self.logic.compute_metrics(
            segNode, wallSid,
            minDiamPct=minPct, removeUnref=rmUnref, thickness_voxel_mm=thickVoxel
        )
        dt = time.perf_counter() - t0

        html = self.logic.render_metrics_html(metrics, elapsed=dt, include=include)
        self.outputBrowser.setHtml(html)
        self.statusLabel.text = f"Time (≈ {dt:.2f} s)."

        # Enable buttons that depend on computed results
        self.exportBtn.setEnabled(True)
        self.saveMeshBtn.setEnabled(True)
        self.genThickBtn.setEnabled(True)
        self.expThickBtn.setEnabled(True)
        self.showThickBtn.setEnabled(True)

    def onExport(self):
        metrics = self.logic.last_metrics; assert metrics, "Compute metrics first."
        outDir = qt.QFileDialog.getExistingDirectory(None, "Choose output folder")
        if not outDir: return
        include = self._read_include_flags()
        pdfPath = self.logic.export_results(outDir, include=include)
        qt.QMessageBox.information(slicer.util.mainWindow(), "Measures", f"Exported: - {pdfPath}")

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
                       thickness_enabled=True, thickness_voxel_mm=0.003):
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
                # stats vêm do cálculo (voxels centrais/medial)
                st = (info or {}).get("stats", None)
                if st and st.get("mean_mm", 0.0) > 0:
                    thick = {
                        "mean_mm": float(st.get("mean_mm", 0.0)),
                        "std_mm":  float(st.get("std_mm", 0.0)),
                        "min_mm":  float(st.get("min_mm", 0.0)),
                        "max_mm":  float(st.get("max_mm", 0.0)),
                        "median_mm": float(st.get("median_mm", 0.0)),
                        "voxel_mm": float(thickness_voxel_mm),
                        "central_voxels": int(info.get("central_voxels", 0)),
                    }
                else:
                    logging.warning(f"Thickness stats: {st}, info: {info}")
                    thick = None
            except Exception as e:
                logging.error(f"Thickness error: {e}")
                import traceback
                traceback.print_exc()
                thick = None

        # Porosity (%)
        if pore_count and pore_count > 0:
            porosity_pct = (pore_density / float(pore_count)) * 100.0
        else:
            porosity_pct = float('nan')

        metrics = {
            "count_pores": int(pore_count),
            "surface_area_mm2": float(A_mesh),
            "volume_mm3": float(V_mesh),
            "S_over_V": float(S_over_V),
            "pore_density_per_mm2": float(pore_density),
            "porosity_pct": float(porosity_pct), 
            "cleaning": clean_info,
            "thickness_mm": thick
        }

        self.last_metrics = metrics
        return metrics

    # ---------- Render (HTML) ----------
    def _resource_path(self, rel):
        # Mesmo esquema do self.resourcePath do Slicer, com fallback local
        try:
            return self.resourcePath(rel)
        except Exception:
            return os.path.join(os.path.dirname(__file__), 'Resources', rel)

    def _fmt(self, x, p=6):
        if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
            return "—"
        return f"{float(x):.{p}f}"

    def render_metrics_html(self, m: dict, elapsed: float = None, include: dict = None) -> str:
        # Default: include everything
        if include is None:
            include = {k: True for k in ('pores','porosity','area','volume','sv','pore_density','thickness','cleaning_note')}

        # ---- nota de limpeza ----
        note = ""
        clean = m.get('cleaning') or {}
        if include.get('cleaning_note', True):
            try:
                md = float(clean.get('min_diam_mm', 0.0))
                mp = float(clean.get('min_diam_pct', 0.0))
                bb = float(clean.get('bbox_diag', 0.0))
                if mp > 0.0:
                    kept = int(clean.get('regions_kept', 0))
                    total = int(clean.get('regions_total', 0))
                    overflow = bool(clean.get('overflow_largest_only', False))
                    extra = " (fallback: largest component only)" if overflow else ""
                    note = (f"<div class='sub' style='margin-top:6px;'>"
                            f"Cleaning: removed {max(0,total-kept)} islands &lt; {mp:.1f}% of diag. "
                            f"(≈ {md:.3f} mm of {bb:.6f} mm); kept: {kept}/{total}{extra}.</div>")
            except Exception:
                pass

        # ---- thickness ----
        thick = m.get('thickness_mm') or {}
        thick_mean_um  = self._fmt((thick.get('mean_mm')  * 1000.0) if thick else None, p=3)
        thick_sd_um    = self._fmt((thick.get('std_mm')   * 1000.0) if thick else None, p=3)
        thick_voxel_um = self._fmt((thick.get('voxel_mm') * 1000.0) if thick else None, p=3)

        # ---- build conditional rows ----
        def _row(key, label, value):
            return f"<tr><td class='key'>{label}</td><td class='val'>{value}</td></tr>" if include.get(key, True) else ""

        rows = ""
        rows += _row('pores',        'Pores',                int(m.get('count_pores') or 0))
        rows += _row('porosity',     'Porosity (%)',         self._fmt(m.get('porosity_pct'), p=2))
        rows += _row('area',         'Surface area (mm²)',   self._fmt(m.get('surface_area_mm2'), p=6))
        rows += _row('volume',       'Volume (mm³)',         self._fmt(m.get('volume_mm3'), p=6))
        rows += _row('sv',           'S/V (mm⁻¹)',           self._fmt(m.get('S_over_V'), p=6))
        rows += _row('pore_density', 'Pore density (mm⁻²)',  self._fmt(m.get('pore_density_per_mm2'), p=6))
        if include.get('thickness', True):
            rows += f"<tr><td class='key'>Thickness mean (µm)</td><td class='val'>{thick_mean_um}</td></tr>"
            rows += f"<tr><td class='key'>Thickness SD (µm)</td><td class='val'>{thick_sd_um}</td></tr>"
            rows += f"<tr><td class='key'>Thickness voxel (µm)</td><td class='val'>{thick_voxel_um}</td></tr>"

        ctx = {
            "rows":       rows,
            "foot_left":  "Morphometric measurements using ForaLABS",
            "note":       note,
        }

        # ---- carrega template e substitui ----
        tpl_path = self._resource_path('HTML/MeasuresReport.html')
        try:
            with open(tpl_path, 'r', encoding='utf-8') as f:
                tpl = Template(f.read())
            return tpl.safe_substitute(ctx)
        except Exception as e:
            # Fallback mínimo
            return (f"<html><body><h3>Measures</h3>"
                    f"<table class='kv'>{rows}</table>"
                    f"<div class='sub'>{ctx['foot_left']}</div>"
                    f"{note}</body></html>")


     # ---------- EXPORT Helper ----------
    def _save_html_to_pdf(self, html: str, out_path: str):
        doc = qt.QTextDocument()
        doc.setHtml(html)

        printer = qt.QPrinter()
        printer.setOutputFormat(qt.QPrinter.PdfFormat)
        printer.setOutputFileName(out_path)
        # Página A4 com margens suaves (mm)
        try:
            printer.setPaperSize(qt.QPrinter.A4)
        except Exception:
            pass
        try:
            printer.setPageMargins(10, 10, 10, 10, qt.QPrinter.Millimeter)
        except Exception:
            # Alguns bindings não expõem essa sobrecarga; podemos ignorar
            pass

        # PyQt usa print_, PySide usa print — tratamos os dois:
        if hasattr(doc, "print_"):
            doc.print_(printer)
        else:
            doc.print(printer)

    # ---------- Export ----------
    def export_results(self, outDir, include=None):
        import os, logging
        html = self.render_metrics_html(self.last_metrics or {}, elapsed=None, include=include)
        if not os.path.isdir(outDir):
            os.makedirs(outDir, exist_ok=True)
        pdfPath = os.path.join(outDir, "Measures_mesh_metrics.pdf")
        self._save_html_to_pdf(html, pdfPath)
        logging.info(f"Exported: {pdfPath}")
        return pdfPath


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
    def _voxelize_poly_solid(self, poly, voxel_mm=0.003, margin_vox=2):
        """
        Voxeliza a malha como volume sólido (preenche o interior).
        Para shells, usa dilatação para garantir que há voxels dentro.
        """
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
        """
        Calcula espessura média da parede usando ray casting nos vértices da malha.
        
        Para estruturas shell-like (cascas/paredes), a espessura é estimada 
        lançando raios a partir de cada vértice na direção da normal e 
        medindo a distância até o outro lado da parede.
        
        Retorna: (thick_img, info_dict)
        - thick_img: imagem voxelizada com valores de espessura (para visualização)
        - info_dict: contém estatísticas calculadas diretamente na malha
        """
        # ---------- Método baseado em Ray Casting na malha ----------
        # Calcula espessura diretamente nos vértices da malha
        
        # Garante que temos normais
        normals_filter = vtk.vtkPolyDataNormals()
        normals_filter.SetInputData(poly)
        normals_filter.ComputePointNormalsOn()
        normals_filter.ComputeCellNormalsOff()
        normals_filter.SplittingOff()
        normals_filter.ConsistencyOn()
        normals_filter.AutoOrientNormalsOn()
        normals_filter.Update()
        poly_with_normals = normals_filter.GetOutput()
        
        pts = poly_with_normals.GetPoints()
        normals = poly_with_normals.GetPointData().GetNormals()
        npts = pts.GetNumberOfPoints()
        
        if npts == 0 or normals is None:
            # Fallback: retorna zeros
            info = {
                "spacing": (voxel_mm, voxel_mm, voxel_mm),
                "origin": (0, 0, 0),
                "shape": (1, 1, 1),
                "stats": {"mean_mm": 0.0, "std_mm": 0.0},
                "note": "Malha sem pontos ou normais."
            }
            return sitk.Image(1, 1, 1, sitk.sitkFloat32), info
        
        # Cria locator para ray casting
        cell_locator = vtk.vtkCellLocator()
        cell_locator.SetDataSet(poly_with_normals)
        cell_locator.BuildLocator()
        
        # Estruturas para ray casting
        t_val = vtk.mutable(0.0)
        pcoords = [0.0, 0.0, 0.0]
        sub_id = vtk.mutable(0)
        cell_id = vtk.mutable(0)
        intersection = [0.0, 0.0, 0.0]
        
        # Calcula bounds para determinar alcance máximo do raio
        bounds = poly.GetBounds()
        diag = ((bounds[1]-bounds[0])**2 + (bounds[3]-bounds[2])**2 + (bounds[5]-bounds[4])**2)**0.5
        max_ray_length = diag * 0.5  # metade da diagonal como máximo
        
        # Amostragem: usa todos os pontos ou subsample se muitos
        max_samples = 50000
        if npts > max_samples:
            indices = np.random.choice(npts, max_samples, replace=False)
        else:
            indices = np.arange(npts)
        
        thickness_values = []
        
        for idx in indices:
            pt = pts.GetPoint(idx)
            n = normals.GetTuple(idx)
            
            # Normaliza a normal
            norm_len = (n[0]**2 + n[1]**2 + n[2]**2)**0.5
            if norm_len < 1e-10:
                continue
            n = (n[0]/norm_len, n[1]/norm_len, n[2]/norm_len)
            
            # Lança raio na direção oposta à normal (para dentro da parede)
            # Offset pequeno para evitar auto-interseção
            offset = voxel_mm * 0.1
            start = (pt[0] - n[0]*offset, pt[1] - n[1]*offset, pt[2] - n[2]*offset)
            end = (pt[0] - n[0]*max_ray_length, pt[1] - n[1]*max_ray_length, pt[2] - n[2]*max_ray_length)
            
            hit = cell_locator.IntersectWithLine(start, end, 1e-9, t_val, intersection, pcoords, sub_id, cell_id)
            
            if hit:
                # Distância do ponto original até a interseção
                dist = ((intersection[0]-pt[0])**2 + (intersection[1]-pt[1])**2 + (intersection[2]-pt[2])**2)**0.5
                if dist > voxel_mm * 0.5 and dist < max_ray_length:  # Filtra ruído
                    thickness_values.append(dist)
        
        thickness_arr = np.array(thickness_values)
        
        if thickness_arr.size == 0:
            # Fallback: tenta na direção positiva da normal
            for idx in indices:
                pt = pts.GetPoint(idx)
                n = normals.GetTuple(idx)
                
                norm_len = (n[0]**2 + n[1]**2 + n[2]**2)**0.5
                if norm_len < 1e-10:
                    continue
                n = (n[0]/norm_len, n[1]/norm_len, n[2]/norm_len)
                
                offset = voxel_mm * 0.1
                start = (pt[0] + n[0]*offset, pt[1] + n[1]*offset, pt[2] + n[2]*offset)
                end = (pt[0] + n[0]*max_ray_length, pt[1] + n[1]*max_ray_length, pt[2] + n[2]*max_ray_length)
                
                hit = cell_locator.IntersectWithLine(start, end, 1e-9, t_val, intersection, pcoords, sub_id, cell_id)
                
                if hit:
                    dist = ((intersection[0]-pt[0])**2 + (intersection[1]-pt[1])**2 + (intersection[2]-pt[2])**2)**0.5
                    if dist > voxel_mm * 0.5 and dist < max_ray_length:
                        thickness_values.append(dist)
            
            thickness_arr = np.array(thickness_values)
        
        if thickness_arr.size == 0:
            info = {
                "spacing": (voxel_mm, voxel_mm, voxel_mm),
                "origin": (0, 0, 0),
                "shape": (1, 1, 1),
                "stats": {"mean_mm": 0.0, "std_mm": 0.0},
                "note": "Nenhuma interseção válida encontrada. A malha pode não ser uma shell fechada."
            }
            return sitk.Image(1, 1, 1, sitk.sitkFloat32), info
        
        # Remove outliers (valores muito altos ou muito baixos)
        # Usa IQR para filtrar
        q1, q3 = np.percentile(thickness_arr, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        filtered = thickness_arr[(thickness_arr >= max(lower_bound, voxel_mm)) & (thickness_arr <= upper_bound)]
        
        if filtered.size < 10:
            filtered = thickness_arr  # Usa todos se filtro removeu demais
        
        # Estatísticas
        stats = {
            "mean_mm": float(np.mean(filtered)),
            "std_mm": float(np.std(filtered)),
            "min_mm": float(np.min(filtered)),
            "max_mm": float(np.max(filtered)),
            "median_mm": float(np.median(filtered)),
        }
        
        # Cria imagem de espessura para visualização (opcional)
        # Voxeliza e preenche com valor médio
        try:
            mask, spacing, origin, shape = self._voxelize_poly_solid(poly, voxel_mm=float(voxel_mm))
            mask_arr = sitk.GetArrayFromImage(mask)
            thick_vol = np.zeros_like(mask_arr, dtype=np.float32)
            thick_vol[mask_arr > 0] = stats["mean_mm"]
            thick_img = sitk.GetImageFromArray(thick_vol)
            thick_img.SetSpacing(spacing)
            thick_img.SetOrigin(origin)
        except Exception:
            thick_img = sitk.Image(1, 1, 1, sitk.sitkFloat32)
            spacing = (voxel_mm, voxel_mm, voxel_mm)
            origin = (0, 0, 0)
            shape = (1, 1, 1)
        
        info = {
            "spacing": spacing,
            "origin": origin,
            "shape": shape,
            "stats": stats,
            "samples": int(len(indices)),
            "valid_measurements": int(filtered.size),
            "method": "ray_casting",
        }
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
            volNode = self._thickness_volume_node_from_sitk(self.last_thick_img*1000, name="Thickness")
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
    def _remove_isolated_pieces_by_diameter(self, poly, minDiamMM=0.0, removeUnref=True, hard_limit_regions=500):
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

# =========================================================
# Tests (placeholder)
# =========================================================
class MeasuresTest(ScriptedLoadableModuleTest):
    def runTest(self):
        self.setUp(); self.test_All()
    def test_All(self):
        self.delayDisplay("Measures mesh-only test placeholder…"); self.delayDisplay("OK")
