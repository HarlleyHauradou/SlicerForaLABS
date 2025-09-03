# -*- coding: utf-8 -*-
#
# ImportTXM.py  (OLE-only version)
# ScriptedLoadableModule to import .txm volumes (ZEISS XRM) directly into 3D Slicer
# using only 'olefile' (OLE reading) and SimpleITK (NRRD writing). No dxchange.
#
# Installation: place this file in any Slicer ScriptedLoadable modules folder
# (for example, inside your ForaLABS package). Restart Slicer.
# The module will appear under "MicroCT" as "Import TXM".
#
# Author: Harlley + Chica (ForaLAB)

import os
import json
import logging
import traceback
import struct
import html

import slicer
import SimpleITK as sitk
from slicer.ScriptedLoadableModule import *
import qt
import ctk


# -----------------------------------------------------------------------------
# Module
# -----------------------------------------------------------------------------

class ImportTXM(ScriptedLoadableModule):
    """Slicer module definitions."""
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "Import TXM"
        self.parent.categories = ["ForaLABS"]
        self.parent.dependencies = []
        self.parent.contributors = ["Harlley Hauradou (Nuclear Engineering Program of UFRJ)", "Thaís Hauradou (Nuclear Engineering Program of UFRJ)"]
        self.parent.helpText = (
            "Import ZEISS reconstructed volumes (.txm) using Python only in Slicer. "
            "Converts spacing from µm to mm and loads as NRRD."
        )
        self.parent.acknowledgementText = (
            "OLE reading with 'olefile'. Developed in the ForaLAB project."
        )


# -----------------------------------------------------------------------------
# Widget (GUI)
# -----------------------------------------------------------------------------

class ImportTXMWidget(ScriptedLoadableModuleWidget):

    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)
        self.logic = ImportTXMLogic()

        # Load external .ui and bind widgets
        try:
            uiWidget = slicer.util.loadUI(self.resourcePath('UI/ImportTXM.ui'))
        except Exception:
            ui_path = os.path.join(os.path.dirname(__file__), 'Resources', 'UI', 'ImportTXM.ui')
            uiWidget = slicer.util.loadUI(ui_path)
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # Aliases (optional, keep old attribute names)
        self.txmPathEdit  = self.ui.txmPathEdit
        self.outDirEdit   = self.ui.outDirEdit
        self.unitCombo    = self.ui.unitCombo
        self.flipZCheck   = self.ui.flipZCheck
        self.importBtn    = self.ui.importBtn
        self.metaBtn      = self.ui.metaBtn
        self.spacingLabel = self.ui.spacingLabel
        self.infoText     = self.ui.infoText
        self.progress     = self.ui.progress

        # Configure ctkPathLineEdit filters (API compatibility)
        try:
            self.txmPathEdit.filters = ctk.ctkPathLineEdit().Files
        except Exception:
            try:
                self.txmPathEdit.setFilters(ctk.ctkPathLineEdit().Files)
            except Exception:
                pass
        try:
            self.outDirEdit.filters = ctk.ctkPathLineEdit().Dirs
        except Exception:
            try:
                self.outDirEdit.setFilters(ctk.ctkPathLineEdit().Dirs)
            except Exception:
                pass
        try:
            self.txmPathEdit.nameFilters = ["ZEISS TXM (*.txm)"]
        except Exception:
            pass
        try:
            self.unitCombo.setCurrentIndex(0)  # micrometers (µm)
        except Exception:
            pass

        # Signal connections
        self.importBtn.clicked.connect(self.onImport)
        self.metaBtn.clicked.connect(self.onShowMeta)

        # Internal state
        self._lastMeta = None
        self._lastNRRD = None
        self._lastNode = None

        self.layout.addStretch(1)


    # ---------------------------- GUI helpers ---------------------------- #

    def _msg(self, text):
        logging.info(text)
        # Show as plain text (no escaping) to avoid entities like &#x27;
        self.infoText.append(str(text))
        slicer.util.showStatusMessage(str(text), 3000)

    def _err(self, text):
        logging.error(text)
        # Use HTML only for color, keeping regular quotes (quote=False)
        safe = html.escape(str(text), quote=False)
        self.infoText.append("<span style='color:#e53935;'>%s</span>" % safe)
        slicer.util.showStatusMessage(str(text), 5000)

    def _progress(self, val):
        self.progress.setValue(int(val))
        slicer.app.processEvents()

    def _updateSpacingLabel(self, spacing_mm):
        if not spacing_mm:
            self.spacingLabel.setText("Applied spacing (mm): —")
            return
        sx, sy, sz = spacing_mm
        self.spacingLabel.setText(
            "Applied spacing (mm): X={:.7f}, Y={:.7f}, Z={:.7f}".format(sx, sy, sz)
        )

    # ----------------------------- actions -------------------------------- #

    def onImport(self):
        txmPath = self.txmPathEdit.currentPath.strip()
        outDir = self.outDirEdit.currentPath.strip()

        if not txmPath or not os.path.isfile(txmPath):
            self._err("Select a valid .txm file.")
            return

        if not outDir:
            outDir = os.path.dirname(txmPath)
            self.outDirEdit.currentPath = outDir

        # safe index (method or property)
        idxGetter = getattr(self.unitCombo, "currentIndex")
        idx = idxGetter() if callable(idxGetter) else idxGetter
        unit = "um" if idx == 0 else "mm"

        flipZ = self.flipZCheck.isChecked()

        try:
            self._progress(5)
            self._msg("Checking/installing dependency 'olefile'...")
            self.logic.ensurePackages()
            self._progress(15)

            self._msg(f"Reading TXM (OLE): {txmPath}")
            arr, meta = self.logic.readTXM(txmPath)
            if arr is None:
                raise RuntimeError("Failed to read the .txm file")

            self._progress(45)

            self._msg("Inferring spacing (voxel size) and converting units...")
            spacing_mm = self.logic.inferSpacingMM(meta, unit_hint=unit)
            if spacing_mm is None:
                self._msg("Voxel/pixel metadata not found; assuming 1.0000000 mm isotropic.")
                spacing_mm = (1.0, 1.0, 1.0)

            if flipZ:
                self._msg("Applying Z flip...")
                arr = arr[::-1, :, :]

            self._progress(70)

            self._msg("Writing NRRD with mm spacing and loading into Slicer...")
            nrrdPath, node = self.logic.saveAndLoadNRRD(arr, spacing_mm, txmPath, outDir)

            self._progress(90)

            self._updateSpacingLabel(spacing_mm)
            self._msg("OK! NRRD: {}".format(nrrdPath))
            self._lastMeta = meta
            self._lastNRRD = nrrdPath
            self._lastNode = node
            self.metaBtn.enabled = True

            # Show volume in 2D viewers (do not enable 3D rendering automatically)
            if node:
                lm = slicer.app.layoutManager()
                if lm:
                    for name in ("Red", "Yellow", "Green"):
                        w = lm.sliceWidget(name)
                        if w:
                            cn = w.mrmlSliceCompositeNode()
                            cn.SetBackgroundVolumeID(node.GetID())
                    slicer.util.setSliceViewerLayers(background=node)
                    slicer.util.resetSliceViews()

            self._progress(100)
            self._msg("Done.")

        except Exception as e:
            tb = traceback.format_exc()
            self._err(f"Error: {e}\n{tb}")
            self._progress(0)

    def onShowMeta(self):
        if not self._lastMeta:
            self._err("No metadata to display.")
            return

        # Dialog with metadata table
        dlg = qt.QDialog(slicer.util.mainWindow())
        dlg.setWindowTitle("TXM Metadata")
        dlg.resize(760, 540)
        v = qt.QVBoxLayout(dlg)

        table = qt.QTableWidget()
        table.setColumnCount(2)
        table.setHorizontalHeaderLabels(["Key", "Value"])
        items = sorted([(str(k), self._safeVal(vv)) for k, vv in self._lastMeta.items()], key=lambda x: x[0].lower())
        table.setRowCount(len(items))
        for i, (k, val) in enumerate(items):
            table.setItem(i, 0, qt.QTableWidgetItem(k))
            table.setItem(i, 1, qt.QTableWidgetItem(val))
        table.horizontalHeader().setStretchLastSection(True)
        table.horizontalHeader().setSectionResizeMode(0, qt.QHeaderView.ResizeToContents)
        v.addWidget(table)

        if self._lastNRRD:
            lbl = qt.QLabel(f"NRRD file: <code>{html.escape(self._lastNRRD)}</code>")
            v.addWidget(lbl)

        btn = qt.QPushButton("Close")
        btn.clicked.connect(dlg.accept)
        v.addWidget(btn, 0, qt.Qt.AlignRight)

        dlg.exec_()

    def _safeVal(self, v):
        try:
            if isinstance(v, (dict, list, tuple)):
                return json.dumps(v)
            return str(v)
        except Exception:
            return repr(v)


# -----------------------------------------------------------------------------
# Logic
# -----------------------------------------------------------------------------

class ImportTXMLogic(ScriptedLoadableModuleLogic):
    """Implements TXM (OLE) reading, spacing inference, and NRRD writing/usage."""

    # ------------------------- dependencies ------------------------- #
    def ensurePackages(self):
        try:
            import olefile  # noqa: F401
        except Exception:
            slicer.util.pip_install("olefile")

    # ---------------------------- reading --------------------------- #
    def readTXM(self, txmPath):
        """
        Pure OLE reader: returns (arr, meta) with arr shape (Z, Y, X) and meta dict.
        Supports DataType 1 (uint8), 5 (uint16) and 10 (float32).
        """
        import numpy as np
        import olefile

        if not olefile.isOleFile(txmPath):
            raise RuntimeError("TXM file is not a valid OLE container.")

        ole = olefile.OleFileIO(txmPath)

        # ---- read helpers ---- #
        def _read_stream(path):
            if not ole.exists(path):
                return None
            with ole.openstream(path) as s:
                return s.read()

        def _read_int(paths):
            """Reads integer (little-endian) from one of the paths, using stream size heuristics."""
            for p in paths:
                b = _read_stream(p)
                if b is None:
                    continue
                n = len(b)
                try:
                    if n >= 4:
                        # use first 4 bytes as uint32 LE
                        return int.from_bytes(b[:4], byteorder="little", signed=False)
                    elif n == 2:
                        return int.from_bytes(b, byteorder="little", signed=False)
                    elif n >= 8:
                        # some store as int64; convert to int
                        return int.from_bytes(b[:8], byteorder="little", signed=False)
                except Exception:
                    pass
            return None

        def _read_float(paths):
            """Try double then float; return float or None."""
            for p in paths:
                b = _read_stream(p)
                if b is None:
                    continue
                try:
                    if len(b) >= 8:
                        return struct.unpack('<d', b[:8])[0]
                except Exception:
                    pass
                try:
                    if len(b) >= 4:
                        return struct.unpack('<f', b[:4])[0]
                except Exception:
                    pass
            return None

        def _read_image(stream_name, dtype, h, w):
            b = _read_stream(stream_name)
            if b is None:
                raise RuntimeError(f"Missing stream: {stream_name}")
            arr = np.frombuffer(b, dtype=dtype)
            expected = int(h) * int(w)
            if arr.size != expected:
                raise RuntimeError(f"Unexpected size in stream {stream_name}: {arr.size} vs {expected}")
            return arr.reshape((int(h), int(w)))

        # --- Basic metadata ---
        n_imgs  = _read_int(["ImageInfo/NoOfImages", "ImageInfo/NumberOfImages"])  # uint32
        width   = _read_int(["ImageInfo/ImageWidth", "ImageInfo/Width"])           # uint32
        height  = _read_int(["ImageInfo/ImageHeight", "ImageInfo/Height"])         # uint32
        dt_code = _read_int(["ImageInfo/DataType"])                                 # uint32

        # try to read pixel size as double (higher precision) and, as fallback, float/int
        px_um = _read_float([
            "ImageInfo/PixelSize", "ImageInfo/pixelsize", "ImageInfo/Pixel Size",
            "ImageInfo/VoxelSize", "ImageInfo/Voxel Size"
        ])
        if px_um is None:
            ival = _read_int([
                "ImageInfo/PixelSize", "ImageInfo/pixelsize", "ImageInfo/Pixel Size",
                "ImageInfo/VoxelSize", "ImageInfo/Voxel Size"
            ])
            if ival is not None:
                px_um = float(ival)

        if any(v is None for v in (n_imgs, width, height, dt_code)):
            ole.close()
            raise RuntimeError("Incomplete TXM metadata (NoOfImages/ImageWidth/ImageHeight/DataType).")

        # --- dtype by DataType ---
        dt_code = int(dt_code)
        if dt_code == 1:
            np_dtype = np.uint8
        elif dt_code == 5:
            np_dtype = np.uint16
        elif dt_code == 10:
            np_dtype = np.float32
        else:
            ole.close()
            raise RuntimeError(f"Unsupported TXM DataType {dt_code} (expected 1=uint8, 5=uint16, or 10=float32).")

        # --- Read all slices ---
        vol = np.empty((int(n_imgs), int(height), int(width)), dtype=np_dtype)
        for idx in range(int(n_imgs)):
            # Folders in blocks of 100 images: ImageData1, ImageData2, ...
            block = (idx + 1 + 99) // 100  # ceil((idx+1)/100)
            stream = f"ImageData{block}/Image{idx+1}"
            vol[idx] = _read_image(stream, np_dtype, height, width)

        ole.close()

        meta = {
            "image_width": int(width),
            "image_height": int(height),
            "number_of_images": int(n_imgs),
            "data_type": int(dt_code),
            "pixel_size": float(px_um) if px_um is not None else None,   # µm
            "source": "ole-only",
        }
        return vol, meta

    # ---------------------- spacing inference ---------------------- #
    def inferSpacingMM(self, meta: dict, unit_hint="um"):
        """
        Returns tuple (sx, sy, sz) in millimeters.
        Looks for common pixel/voxel size keys in meta.
        """
        def _to_float(x):
            try:
                if isinstance(x, str):
                    x = x.strip().replace(",", ".")
                return float(x)
            except Exception:
                return None

        # Explicit voxel size XYZ
        vx = _to_float(meta.get("voxel_size_x")) or _to_float(meta.get("VoxelSizeX"))
        vy = _to_float(meta.get("voxel_size_y")) or _to_float(meta.get("VoxelSizeY"))
        vz = _to_float(meta.get("voxel_size_z")) or _to_float(meta.get("VoxelSizeZ"))

        # Isotropic pixel/voxel
        if vx is None or vy is None or vz is None:
            for key in ("pixel_size", "pixelsize", "voxel_size", "VoxelSize", "PixelSize"):
                val = meta.get(key)
                f = _to_float(val)
                if f:
                    vx = vy = vz = f
                    break

        if vx is None or vy is None or vz is None:
            return None

        # Unit (µm by default)
        unit = None
        for key in meta.keys():
            lk = str(key).lower()
            if "unit" in lk and "size" in lk:
                try:
                    unit = str(meta.get(key)).lower()
                except Exception:
                    unit = None
                break
        if unit is None:
            unit = unit_hint.lower()

        in_um = any(u in unit for u in ["µm", "um", "micrometer", "micron", "micrometre", "micrometro"])
        mul = 0.001 if in_um else 1.0  # µm → mm
        return (vx * mul, vy * mul, vz * mul)

    # ---------------------- save and load -------------------------- #
    def saveAndLoadNRRD(self, arr, spacing_mm, srcPath, outDir):
        """
        Convert numpy -> SimpleITK, set spacing (mm), write compressed NRRD,
        and load into Slicer. Returns (nrrdPath, volumeNode).
        """
        base = os.path.splitext(os.path.basename(srcPath))[0]
        safeBase = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in base)
        nrrdPath = os.path.join(outDir, f"{safeBase}.nrrd")

        # numpy (Z,Y,X) -> SimpleITK Image
        img = sitk.GetImageFromArray(arr)  # keeps (Z,Y,X)
        img.SetSpacing(tuple(float(x) for x in spacing_mm))

        writer = sitk.ImageFileWriter()
        writer.SetFileName(nrrdPath)
        writer.UseCompressionOn()
        writer.Execute(img)

        props = {"name": safeBase}
        node = slicer.util.loadVolume(nrrdPath, properties=props)
        return nrrdPath, node


# -----------------------------------------------------------------------------
# (Optional) Basic test
# -----------------------------------------------------------------------------

class ImportTXMTest(ScriptedLoadableModuleTest):
    """Simple import test (requires a valid path)."""

    def runTest(self):
        self.setUp()
        self.test_import_txm_smoke()

    def test_import_txm_smoke(self):
        logic = ImportTXMLogic()
        logic.ensurePackages()
        samplePath = ""  # "/path/to/a/file.txm"
        if not samplePath:
            logging.info("Test skipped (set samplePath).")
            self.assertTrue(True)
            return
        arr, meta = logic.readTXM(samplePath)
        spacing = logic.inferSpacingMM(meta, unit_hint="um")
        nrrd, node = logic.saveAndLoadNRRD(arr, spacing or (1.0, 1.0, 1.0), samplePath, os.path.dirname(samplePath))
        self.assertTrue(os.path.isfile(nrrd))
        self.assertIsNotNone(node)
