"""PyInstaller recipe for the ChatLab macOS application bundle."""

import sys

from PyInstaller.utils.hooks import collect_all, collect_submodules

sys.path.insert(0, SPECPATH)
from version import BUNDLE_IDENTIFIER, __version__  # noqa: E402


datas = []
binaries = []
hiddenimports = []

# Gradio ships its browser client as package data. Transformers and diffusers
# both discover model implementations lazily, so include their built-in
# architectures for downloaded Hugging Face models rather than limiting the
# desktop app to OLMo and one pipeline.
for package in (
    "diffusers",
    "gradio",
    "gradio_client",
    "groovy",
    "huggingface_hub",
    # Metal quantization imports the Hub kernel loader lazily. collect_all
    # retains its distribution metadata as well as kernels_data's native code.
    "kernels",
    "kernels_data",
    "safehttpx",
    "safetensors",
    "tokenizers",
):
    package_datas, package_binaries, package_imports = collect_all(package)
    datas += package_datas
    binaries += package_binaries
    hiddenimports += package_imports

hiddenimports += collect_submodules("transformers.models", on_error="warn once")
hiddenimports += collect_submodules("transformers.quantizers", on_error="warn once")
# DiffusionPipeline builds itself from the class names in model_index.json, so
# every pipeline and every component class a downloaded repo might name has to
# be in the bundle; none of them is imported by any line of ChatLab's own.
hiddenimports += collect_submodules("diffusers.pipelines", on_error="warn once")
hiddenimports += collect_submodules("diffusers.schedulers", on_error="warn once")
hiddenimports += collect_submodules("diffusers.models", on_error="warn once")
hiddenimports += [
    "transformers.integrations.metal_quantization",
    "transformers.integrations.hub_kernels",
]

a = Analysis(
    ["desktop_launcher.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["IPython", "jupyter", "matplotlib", "notebook", "pytest"],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="ChatLab",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="ChatLab",
)
app = BUNDLE(
    coll,
    name="ChatLab.app",
    icon=None,
    bundle_identifier=BUNDLE_IDENTIFIER,
    info_plist={
        "CFBundleDisplayName": "ChatLab",
        "CFBundleName": "ChatLab",
        "CFBundleShortVersionString": __version__,
        "CFBundleVersion": __version__,
        "LSMinimumSystemVersion": "13.0",
        "NSHighResolutionCapable": True,
    },
)
