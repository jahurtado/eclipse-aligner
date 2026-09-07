# -*- mode: python ; coding: utf-8 -*-
"""How the macOS bundle is built.

The whole cost here is size. A stock PySide6 install is 1.2 GB because it
ships every Qt module; this editor imports three -- QtCore, QtGui, QtWidgets --
so everything else is excluded by name. The scientific wheels bring their own
bulk and cannot be pruned the same way: imagecodecs and scipy load extensions
by path, and dropping one only shows up as a crash on the frame that needs it.

Windows does not go through here. PyInstaller's bootloader trips a Defender
false positive (`Program:Win32/Vigram.A`) that silently truncates the compiled
installer; the Windows build instead ships an embeddable CPython with the wheels
laid beside it and no frozen binary at all. See packaging/make-win.ps1.
"""

import os
import sys

from PyInstaller.utils.hooks import collect_dynamic_libs, collect_submodules

# The version, from the one place that has it. Without it PyInstaller writes
# `CFBundleShortVersionString: 0.0.0`, which is what the Finder shows under
# Get Info and what every installer and updater reads -- a build that says
# 0.0.0 is a build nobody can tell apart from the last one.
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "..", "src")))
from eclipse_aligner import __version__ as VERSION

# `collect_submodules` returns the package itself first, and excluding that
# takes Qt out altogether -- which builds, and is 34 MB lighter, and dies the
# moment the window is asked for. Keep the package and the three modules used.
QT_KEEP = {"PySide6", "PySide6.QtCore", "PySide6.QtGui", "PySide6.QtWidgets"}
QT_DROP = [m for m in collect_submodules("PySide6")
           if m not in QT_KEEP and not m.startswith("PySide6.support")]

a = Analysis(
    ["app.py"],
    pathex=[".."],
    binaries=collect_dynamic_libs("imagecodecs"),
    datas=[("../src/eclipse_aligner/icons", "eclipse_aligner/icons")],
    # imagecodecs reaches its sixty extensions with a computed __import__, so
    # the graph sees none of them. Naming the ones a DNG needs looked tidy and
    # was wrong: the chain broke on one that was not on the list, and the only
    # symptom was a decode failing on the first frame. The whole package goes
    # in -- 37 MB to never have to guess again.
    hiddenimports=collect_submodules("imagecodecs") + ["skimage.registration"],
    # setuptools is build tooling and the app never imports it; PyInstaller
    # pulls a stub of it in behind some dependency, and with it a vendored
    # `Lorem ipsum.txt` -- filler text, the only file in the bundle that is
    # documentation of nothing. Out it goes, and pkg_resources with it, which
    # is the other half of the same package.
    excludes=QT_DROP + [
        "matplotlib", "tkinter", "PyQt5", "PyQt6", "IPython", "pytest",
        "pandas", "notebook", "sphinx", "setuptools", "pkg_resources",
    ],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(pyz, a.scripts, [], exclude_binaries=True,
          name="eclipse-aligner", console=False,
          disable_windowed_traceback=False, argv_emulation=True,
          target_arch=None, codesign_identity=None, entitlements_file=None)

coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False,
               name="eclipse-aligner")

app = BUNDLE(coll, name="eclipse-aligner.app",
             icon="../src/eclipse_aligner/icons/icon.icns",
             bundle_identifier="com.elcacharrista.eclipse-aligner",
             info_plist={
                 "CFBundleName": "eclipse-aligner",
                 "CFBundleDisplayName": "eclipse-aligner",
                 "CFBundleShortVersionString": VERSION,
                 "CFBundleVersion": VERSION,
                 "NSHighResolutionCapable": True,
                 # Opening a clip means opening its folder, so the app says it
                 # takes one -- that is what lets a folder be dropped on the
                 # icon, and what `open -a` needs to pass one through.
                 "CFBundleDocumentTypes": [{
                     "CFBundleTypeName": "Clip folder",
                     "CFBundleTypeRole": "Editor",
                     "LSItemContentTypes": ["public.folder"],
                     "CFBundleTypeOSTypes": ["fold"],
                 }],
             })
