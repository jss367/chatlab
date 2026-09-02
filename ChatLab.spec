"""PyInstaller recipe for the ChatLab macOS application bundle."""

from PyInstaller.utils.hooks import collect_all, collect_submodules


datas = []
binaries = []
hiddenimports = []

# Gradio ships its browser client as package data. Transformers discovers model
# implementations lazily, so include its built-in architectures for downloaded
# Hugging Face models rather than limiting the desktop app to OLMo alone.
for package in (
    "gradio",
    "gradio_client",
    "groovy",
    "huggingface_hub",
    "safehttpx",
    "safetensors",
    "tokenizers",
):
    package_datas, package_binaries, package_imports = collect_all(package)
    datas += package_datas
    binaries += package_binaries
    hiddenimports += package_imports

hiddenimports += collect_submodules("transformers.models", on_error="warn once")

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
    bundle_identifier="build.chatlab.app",
    info_plist={
        "CFBundleDisplayName": "ChatLab",
        "CFBundleName": "ChatLab",
        "CFBundleShortVersionString": "0.1.0",
        "CFBundleVersion": "1",
        "LSMinimumSystemVersion": "13.0",
        "NSHighResolutionCapable": True,
    },
)
