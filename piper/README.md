# Piper

MiraSpeak module. Adds the Piper voice engine: local English voices that run on the CPU. Download one speaker at a time from the module page.

It needs MiraSpeak with module API 1, engine modules (`voice_engine`), and per-module packages (CetiModule 0.3.0 or later).

## Install

In MiraSpeak, open **Integrations → Modules → Install from URL…** and paste:

https://github.com/miramocha/MiraSpeak-modules/tree/v0.3.0/piper

The link sets the tag and the folder.

It asks for **voice engine** and **network**. It also installs these Python packages into its own folder:
- `piper-tts` 1.8.0
- `pathvalidate` 3.3.1

The download from PyPI is about 33 MB. A Python module runs inside MiraSpeak with the same access as the app.

Then open **Voice Engines → In-App** and pick **Piper** in the Engine list, the same place the built-in Piper was. Pick a voice and press **Download**. MiraSpeak adds its own speech cache and generate-ahead settings under the page.

## Coming from the built-in Piper

Older MiraSpeak had Piper built in. The engine id is still `piper`, so personas, presets, and voice settings carry over. Downloaded voices stay in the same folder, and lines Piper already cached still play from the cache.

Until the module is installed and enabled, a line that uses Piper speaks with Kokoro, and the line says so.

## Packages

`module.toml` pins the packages, and each `pylock.<python>-<platform>.toml` file locks the exact wheels and hashes for one target.

piper-tts also needs onnxruntime and numpy. MiraSpeak already ships both for Kokoro, so the locks leave them out (`--no-deps`) and Piper uses the app's copies. That saves about 25 MB per install. If a new piper-tts needs another package MiraSpeak does not ship, add it to `requirements.in` and `requires`.

After changing a pin in `requirements.in` and `module.toml`, rebuild every lock:

```sh
uv pip compile requirements.in --no-deps --python-version 3.12 --python-platform x86_64-pc-windows-msvc --only-binary :all: --format pylock.toml --no-header -o pylock.cp312-win_amd64.toml
uv pip compile requirements.in --no-deps --python-version 3.12 --python-platform x86_64-manylinux_2_28 --only-binary :all: --format pylock.toml --no-header -o pylock.cp312-linux_x86_64.toml
MACOSX_DEPLOYMENT_TARGET=14.0 uv pip compile requirements.in --no-deps --python-version 3.12 --python-platform aarch64-apple-darwin --only-binary :all: --format pylock.toml --no-header -o pylock.cp312-macos_arm64.toml
```

macOS 14 matches MiraSpeak's own onnxruntime, which has no arm64 wheel for older macOS.

## License

The module code is [MIT](../LICENSE). Piper itself (piper-tts) is GPL-3.0-or-later and is installed from PyPI, not shipped here. Each voice keeps the license published with its files.
