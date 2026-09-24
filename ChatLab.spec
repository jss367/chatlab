"""PyInstaller recipe for the ChatLab macOS application bundle."""

import os
import sys

from PyInstaller.utils.hooks import collect_all, collect_submodules, copy_metadata

sys.path.insert(0, SPECPATH)
from version import BUNDLE_IDENTIFIER, __version__  # noqa: E402


datas = []
binaries = []
hiddenimports = []

# The mark: the .icns macOS draws in the Dock and the Finder, and the PNG
# branding.py hands Gradio for the window's own tab.
ICON = os.path.join(SPECPATH, "assets", "ChatLab.icns")
datas += [(os.path.join(SPECPATH, "assets", "icon.png"), "assets")]

# Diffusers checks these distributions' versions when imported. Their modules
# alone are insufficient in a frozen bundle; retain the package metadata too.
for package in ("requests", "filelock", "numpy"):
    datas += copy_metadata(package)

# The startup record in logs.py reads these versions from package metadata
# rather than by importing the packages, which would cost the launch several
# seconds. Metadata a bundle was built without reads as "absent", and a
# memory report naming the wrong torch is worse than one naming none. The
# rest of logs.RECORDED_PACKAGES come with the collect_all calls below.
for package in ("torch", "transformers", "accelerate"):
    try:
        datas += copy_metadata(package)
    except Exception:  # noqa: BLE001 - a version this cannot read is not a build failure
        pass

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
    # Imported only when a LoRA adapter is loaded, and it builds each tuner
    # type from a registry of its submodules, so analysis alone misses them.
    "peft",
    "safehttpx",
    "safetensors",
    # The tokenizer converters. Transformers imports both lazily, inside the
    # branch that reads a repo shipping no tokenizer.json, and each is a
    # compiled extension: analysis alone would leave the bundle able to read
    # a tokenizer.json and nothing else.
    "sentencepiece",
    "tiktoken",
    "tokenizers",
):
    package_datas, package_binaries, package_imports = collect_all(package)
    datas += package_datas
    binaries += package_binaries
    hiddenimports += package_imports

# MLX is an Apple silicon dependency: the Metal library and the compiled core
# ship as package data, and mlx-lm resolves each architecture's module by the
# model_type in a downloaded config, so every one of them has to be present.
if sys.platform == "darwin":
    for package in ("mlx", "mlx_lm"):
        try:
            package_datas, package_binaries, package_imports = collect_all(package)
        except Exception:  # noqa: BLE001 - a bundle without mlx still runs Transformers
            continue
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
# Optional first-party pages are imported only when enabled at runtime.
hiddenimports += collect_submodules("extensions")
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
    icon=ICON,
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
