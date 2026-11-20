# Discord

MiraSpeak module. A Discord bot joins your voice channel and plays everything MiraSpeak speaks, so you need no virtual audio cable. You make the bot yourself; see the [Discord guide](https://github.com/miramocha/MiraSpeak-docs/blob/main/guides/discord.md).

It needs MiraSpeak with module API 1 and per-module packages (CetiModule 0.2.0 or later).

## Install

In MiraSpeak, open **Integrations → Modules → Install from URL…** and paste:

https://github.com/miramocha/MiraSpeak-modules/tree/v0.2.0/discord

The link sets the tag and the folder.

It asks for **audio**, **network** and **observe lines**. It also installs these Python packages into its own folder:
- `discord.py[voice]` 2.7.1
- `davey` 0.1.7
- the packages they need, such as aiohttp and PyNaCl

The download from PyPI is under 4 MB. A Python module runs inside MiraSpeak with the same access as the app.

If you used the Discord module that came with older MiraSpeak, your bot token and settings carry over. The module id is still `discord`.

## Packages

`module.toml` pins the packages, and each `pylock.<python>-<platform>.toml` file locks the exact wheels and hashes for one target. After changing a pin in `requirements.in` and `module.toml`, rebuild every lock:

```sh
uv pip compile requirements.in --python-version 3.12 --python-platform x86_64-pc-windows-msvc --only-binary :all: --format pylock.toml --no-header -o pylock.cp312-win_amd64.toml
uv pip compile requirements.in --python-version 3.12 --python-platform x86_64-manylinux_2_28 --only-binary :all: --format pylock.toml --no-header -o pylock.cp312-linux_x86_64.toml
uv pip compile requirements.in --python-version 3.12 --python-platform aarch64-apple-darwin --only-binary :all: --format pylock.toml --no-header -o pylock.cp312-macos_arm64.toml
```

Keep `davey` below 0.2 for discord.py 2.7.1. davey 0.2 stopped setting the voice keys the way discord.py 2.7.1 expects, so the bot sends audio no one can decrypt.

## License

[MIT](../LICENSE)
