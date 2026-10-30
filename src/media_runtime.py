"""Gateway-managed media generation: media-gen behind this gateway
(docs/adr/0004-local-image-and-video-generation.md section 3).

``add MODEL`` runs in the foreground: it builds media-gen and its engines from
the locked project in ``media-runtime/`` with uv, downloads the model's pinned,
hash-verified files through media-gen, installs the ``com.local.media-gen``
LaunchAgent (loopback only, with a generated API key), adds the ``media``
provider to this gateway, loads the model from the shipped catalog layer and
checks that media-gen serves it. ``remove`` takes one model away, or the whole
runtime with its files and gateway routes. ``custom`` manages the owner's own LoRA
profiles on a curated model (media-gen's ``custom`` command), routed only once enabled.

Ownership follows src.local_runtime: the LaunchAgent records this gateway's
``ModelGatewayRoot``, a ``com.local.media-gen`` without it is never modified,
and launchd jobs are stopped only when loaded from this gateway's plist. The
launchctl, sysctl, network and subprocess edges are local_runtime's, or
module-level functions here, so tests replace them.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import http.client
import json
import os
import plistlib
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

from src import config_io
from src import local_runtime as lr
from src.config_lock import config_write_lock
from src.local_runtime import GiB, LocalRuntimeError
from src.secret_files import resolve_api_key_file

RUNTIME_PROJECT = lr.PACKAGE_ROOT / "media-runtime"
MEDIA_LABEL = "com.local.media-gen"
DEFAULT_PORT = 9112
# Memory the rest of the Mac keeps beyond a model's own floor.
SYSTEM_MEMORY = 16 * GiB
# Disk kept free beyond a model's files and the engines (about 6 GB).
DISK_MARGIN = 16 * GiB
METAL_SMOKE = "import mlx.core as m; assert m.metal.is_available()"


@dataclass(frozen=True)
class MediaModel:
    name: str  # gateway model name, and media-gen's id
    modality: str  # "image" or "video"
    display_name: str
    desc: str
    memory_gib: int  # media-gen's floor for loading and running it
    download_bytes: int
    license: str
    license_url: str
    # Gated on Hugging Face: the owner accepts its terms there and signs in (`hf auth login`) first.
    gated: bool = False


# The curated models of the media-gen release media-runtime/ pins; each matches its catalog.
MODELS = {
    model.name: model
    for model in (
        MediaModel("flux3-image-klein", "image", "FLUX 3 Image Klein",
                   "FLUX 3 Image Klein text-to-image, served by Model Gateway media", 28, 20_321_000_000,
                   "FLUX Kommunity License", "https://huggingface.co/black-forest-labs/flux-3-image-klein", gated=True),
        MediaModel("qwen-image-2.1", "image", "Qwen-Image 2.1",
                   "Qwen-Image 2.1 text-to-image with strong text rendering, served by Model Gateway media",
                   74, 33_132_000_000,
                   "Qwen Research License (non-commercial)", "https://huggingface.co/Qwen/Qwen-Image-2.1"),
        MediaModel("wan2.2-i2v-a14b", "video", "Wan 2.2 I2V-A14B (MLX Q8)",
                   "Silent image-to-video; requires a reference image, up to 5 seconds",
                   128, 42_702_440_820,
                   "Apache-2.0", "https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B"),
        MediaModel("ltx-2.5", "video", "LTX-2.5",
                   "LTX-2.5 distilled text/image-to-video with audio, served by Model Gateway media",
                   40, 45_224_000_000,
                   "LTX-2 Community License", "https://huggingface.co/dgrauet/ltx-2.5-mlx-q8", gated=True),
    )
}


# ── paths ────────────────────────────────────────────────────────────────────


def root() -> Path:
    return lr.state_dir() / "media"


def venv_dir() -> Path:
    return root() / "venv"


def home_dir() -> Path:
    """MEDIA_GEN_HOME: media-gen's models, outputs and job database."""
    return root() / "home"


def api_key_path() -> Path:
    return root() / "api.key"


def installed_path() -> Path:
    return root() / "installed.json"


def custom_path() -> Path:
    """The owner's enabled custom profiles, as media-gen last listed them."""
    return root() / "custom.json"


def build_path() -> Path:
    """The digest of the media-runtime lock the venv was built from."""
    return root() / "build.sha256"


def plist_path() -> Path:
    return lr.launch_agents_dir() / f"{MEDIA_LABEL}.plist"


def port() -> int:
    raw = os.environ.get("MODEL_GATEWAY_MEDIA_PORT", "").strip() or str(DEFAULT_PORT)
    if not raw.isdigit() or not 1024 <= int(raw) <= 65535:
        raise LocalRuntimeError("MODEL_GATEWAY_MEDIA_PORT must be an integer between 1024 and 65535")
    return int(raw)


def base_url() -> str:
    return f"http://127.0.0.1:{port()}"


# ── edges (replaced in tests) ────────────────────────────────────────────────


def _run(*args: str, env: dict | None = None) -> None:
    subprocess.run(args, check=True, env=env)


def _output(*args: str, env: dict | None = None) -> str:
    return subprocess.run(args, check=True, env=env, capture_output=True, text=True).stdout


# ── state ────────────────────────────────────────────────────────────────────


def model(name: str) -> MediaModel:
    if name not in MODELS:
        raise LocalRuntimeError(f"Unknown media model {name!r}; choose one of: {', '.join(sorted(MODELS))}")
    return MODELS[name]


def installed() -> list[str]:
    """The installed models, in MODELS order."""
    path = installed_path()
    try:
        lr._regular(path, root())
        names = json.loads(path.read_text()) if path.exists() else []
    except (OSError, ValueError, LocalRuntimeError):
        return []
    return [name for name in MODELS if name in names]


def _write_installed(names: list[str]) -> None:
    lr._atomic(installed_path(), json.dumps(sorted(set(names))) + "\n", root())


def custom_profiles() -> list[dict]:
    """The enabled custom profiles on a curated base model: ``name``, ``base`` and ``display_name``."""
    path = custom_path()
    try:
        lr._regular(path, root())
        rows = json.loads(path.read_text()) if path.exists() else []
    except (OSError, ValueError, LocalRuntimeError):
        return []
    return [
        row for row in rows if isinstance(row, dict) and row.get("base") in MODELS
        and all(isinstance(row.get(key), str) for key in ("name", "base", "display_name"))
        and row["name"].startswith("custom-")
    ] if isinstance(rows, list) else []


def owner() -> str | None:
    """``gateway`` when this gateway owns com.local.media-gen, ``external`` for anyone else, None if absent."""
    path = plist_path()
    if path.is_symlink() or (path.exists() and not path.is_file()):
        return "external"
    if path.exists():
        value = lr._plist(path)
        return "gateway" if value is not None and value.get(lr.OWNER_KEY) == lr.owner_value() else "external"
    return "external" if lr._loaded(MEDIA_LABEL) else None


def _require_owned() -> None:
    if owner() == "external":
        raise LocalRuntimeError(
            f"{MEDIA_LABEL} is managed outside Model Gateway ({plist_path()}); Model Gateway will not modify it")


def _our_provider(block: object) -> bool:
    if not isinstance(block, dict) or not isinstance(block.get("api_key_file"), str):
        return False
    configured = resolve_api_key_file(block["api_key_file"], config_io.CONFIG_PATH)
    return configured == Path(os.path.realpath(api_key_path()))


def catalog_rows(config: dict, *, installed_only: bool = True) -> list[dict]:
    """This runtime's models for the shipped catalog layer, while this gateway manages ``media``."""
    configured = config.get("providers")
    if installed_only and not (isinstance(configured, dict) and _our_provider(configured.get("media"))):
        return []
    names = set(installed()) if installed_only else set(MODELS)
    rows = [
        {"name": m.name, "provider": "media", "provider_model_id": m.name, "modality": m.modality,
         "display_name": m.display_name, "desc": m.desc, "pricing_status": "unmetered"}
        for m in MODELS.values() if m.name in names
    ]
    # The owner's own profiles are never part of a release, so they load only once enabled here,
    # each under its own name and marked with the curated model it builds on (custom_base).
    for profile in custom_profiles() if installed_only else []:
        base = MODELS[profile["base"]]
        if base.name in names:
            rows.append({
                "name": profile["name"], "provider": "media", "provider_model_id": profile["name"],
                "modality": base.modality, "display_name": profile["display_name"][:80],
                "desc": f"Owner-supplied LoRA on {base.display_name}, on this Mac only",
                "pricing_status": "unmetered", "custom_base": base.name,
            })
    return rows


def _api_key(*, create: bool = False) -> str:
    return lr._key(api_key_path(), create=create, base=root())


def _media_gen(path: str, *, timeout: int = 15, body: dict | None = None) -> dict:
    return lr._request(f"{base_url()}{path}", _api_key(), body, timeout=timeout)


def status() -> dict:
    names = installed()
    memory = lr._memory_bytes()
    healthy = False
    if owner() == "gateway" and lr._loaded(MEDIA_LABEL):
        with contextlib.suppress(OSError, ValueError, http.client.HTTPException):
            healthy = lr._request(f"{base_url()}/health", timeout=3).get("status") == "ok"
    return {
        "managed": owner() == "gateway",
        "running": healthy,
        "base_url": f"{base_url()}/v1" if names else None,
        "models": [
            {"name": m.name, "modality": m.modality, "installed": m.name in names,
             "eligible": memory >= m.memory_gib * GiB + SYSTEM_MEMORY,
             "memory_gib": m.memory_gib, "download_gb": round(m.download_bytes / 1e9, 1)}
            for m in MODELS.values()
        ],
        "custom": custom_profiles(),
    }


# ── install ──────────────────────────────────────────────────────────────────


def media_plist() -> dict:
    environment = {
        # ffmpeg (Homebrew) muxes generated audio into videos.
        "PATH": f"{Path.home()}/.local/bin:/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin",
        "MEDIA_GEN_HOME": str(home_dir()),
        "MEDIA_GEN_HOST": "127.0.0.1",
        "MEDIA_GEN_PORT": str(port()),
        "MEDIA_GEN_API_KEY_FILE": str(api_key_path()),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }
    # media-gen unloads idle oMLX models to make room: this gateway's own oMLX, or one the
    # owner names (with a key file, never a key) for an oMLX run outside Model Gateway.
    external_url = os.environ.get("MODEL_GATEWAY_MEDIA_OMLX_URL", "").strip()
    external_key = os.environ.get("MODEL_GATEWAY_MEDIA_OMLX_API_KEY_FILE", "").strip()
    if external_url:
        environment["MEDIA_GEN_OMLX_URL"] = external_url
        if external_key:
            environment["MEDIA_GEN_OMLX_API_KEY_FILE"] = str(Path(external_key).expanduser().resolve())
    elif lr.omlx_owner() == "gateway" and lr.api_key_path().is_file():
        environment["MEDIA_GEN_OMLX_URL"] = f"http://127.0.0.1:{lr.port()}"
        environment["MEDIA_GEN_OMLX_API_KEY_FILE"] = str(lr.api_key_path())
    return {
        "Label": MEDIA_LABEL, lr.OWNER_KEY: lr.owner_value(),
        "ProgramArguments": [str(venv_dir() / "bin" / "media-gen"), "serve"],
        "WorkingDirectory": str(root()),
        "EnvironmentVariables": environment,
        "StandardOutPath": str(root() / "logs" / "media-gen.log"),
        "StandardErrorPath": str(root() / "logs" / "media-gen.log"),
        "Umask": 0o77, "RunAtLoad": True, "KeepAlive": True, "ThrottleInterval": 10,
        "ProcessType": "Interactive",
    }


def _preflight(media: MediaModel) -> None:
    if not lr._apple_silicon():
        raise LocalRuntimeError("Media generation needs a Mac with Apple silicon")
    if lr._memory_bytes() < media.memory_gib * GiB + SYSTEM_MEMORY:
        raise LocalRuntimeError(
            f"{media.name} needs a Mac with at least {media.memory_gib + SYSTEM_MEMORY // GiB} GB of memory")
    free = lr._disk_free(lr.state_dir() if lr.state_dir().exists() else Path.home())
    if free < media.download_bytes + DISK_MARGIN:
        raise LocalRuntimeError(
            f"{media.name} needs {(media.download_bytes + DISK_MARGIN) / 1e9:.0f} GB of free disk space")
    if owner() is None and lr._port_in_use(port()):
        raise LocalRuntimeError(f"Port {port()} is in use; set MODEL_GATEWAY_MEDIA_PORT to a free port")
    _require_owned()
    gateway = config_io.load_config_full().get("providers") or {}
    for provider_id, block in gateway.items() if isinstance(gateway, dict) else ():
        if str(provider_id).lower() == "media" and not _our_provider(block):
            raise LocalRuntimeError("The gateway already has a 'media' provider that Model Gateway does not "
                                    "manage; remove it first")


def _lock_digest() -> str:
    return hashlib.sha256((RUNTIME_PROJECT / "uv.lock").read_bytes()).hexdigest()


def _built() -> bool:
    """Whether the venv was built from this release's lock."""
    with contextlib.suppress(OSError):
        return (venv_dir() / "bin" / "media-gen").is_file() and build_path().read_text().strip() == _lock_digest()
    return False


def _build() -> None:
    """Build (or update) media-gen and its engines from the locked project."""
    env = dict(os.environ, UV_PROJECT_ENVIRONMENT=str(venv_dir()))
    _run(lr._uv(), "sync", "--locked", "--no-dev", "--project", str(RUNTIME_PROJECT), env=env)
    _run(str(venv_dir() / "bin" / "python"), "-c", METAL_SMOKE)
    lr._atomic(build_path(), _lock_digest() + "\n", root())


def _pull(media: MediaModel) -> None:
    env = dict(os.environ, MEDIA_GEN_HOME=str(home_dir()))
    limit = os.environ.get("MODEL_GATEWAY_MEDIA_DOWNLOAD_MBPS", "").strip() or "150"
    _run(str(venv_dir() / "bin" / "media-gen"), "pull", media.name, "--limit-mbps", limit, env=env)


def _running() -> bool:
    with contextlib.suppress(OSError, ValueError, http.client.HTTPException):
        return lr._loaded(MEDIA_LABEL) and lr._request(f"{base_url()}/health", timeout=3).get("status") == "ok"
    return False


def _start(*, rebuilt: bool) -> None:
    """Install or update the LaunchAgent and (re)start media-gen when its build or plist changed.

    A restart fails any video media-gen is generating, so a running, current service is left alone.
    """
    (root() / "logs").mkdir(parents=True, exist_ok=True, mode=0o700)
    path = plist_path()
    current = owner()
    _require_owned()
    data = plistlib.dumps(media_plist())
    if current == "gateway" and not rebuilt and path.read_bytes() == data and _running():
        return
    if current is None:
        lr._create_new(path, data)
    else:
        lr._atomic(path, data, base=path.parent)
    lr._bootstrap(path, MEDIA_LABEL)
    lr._wait_health(f"{base_url()}/health")


def _configure_gateway() -> None:
    """Add the ``media`` provider and signal the gateway to reload; roll back on failure."""
    lr._require_gateway_state_dir()
    with config_write_lock(config_io.CONFIG_PATH):
        snapshot = config_io.snapshot_writable_files()
        try:
            config = config_io.load_config_full()
            if not isinstance(config.get("providers"), dict):
                config["providers"] = {}
            config["providers"]["media"] = {
                "base_url": f"{base_url()}/v1", "protocol": "openai", "api_key": "",
                "api_key_file": str(api_key_path()), "enabled": True,
                "managed_by": config_io.MEDIA_MANAGED,
            }
            lr._write_config(config)
            lr.bump_generation()
        except BaseException:
            config_io.restore_writable_files(snapshot)
            with contextlib.suppress(Exception):
                lr.bump_generation()
            raise


def _unconfigure_gateway() -> bool:
    with config_write_lock(config_io.CONFIG_PATH):
        snapshot = config_io.snapshot_writable_files()
        try:
            config = config_io.load_config_full()
            configured = config.get("providers")
            if not (isinstance(configured, dict) and _our_provider(configured.get("media"))):
                return False
            del configured["media"]
            lr._write_config(config)
            return True
        except BaseException:
            config_io.restore_writable_files(snapshot)
            raise


def _check(media: MediaModel) -> None:
    """media-gen serves the model: listed as installed, and an image model really generates."""
    served = {row["id"]: row for row in _media_gen("/v1/models").get("data", [])}
    if not served.get(media.name, {}).get("installed"):
        raise LocalRuntimeError(f"media-gen does not report {media.name} as installed")
    if media.modality == "image":
        result = _media_gen("/v1/images/generations", timeout=900, body={
            "model": media.name, "prompt": "a red apple on a white table", "size": "512x512", "seed": 1})
        if not result.get("data") or not result["data"][0].get("b64_json"):
            raise LocalRuntimeError(f"media-gen did not generate an image with {media.name}")


def _lock():
    return lr._lock_at(root() / ".provision.lock", 0, "Media generation is being installed or removed")


def add(name: str) -> dict:
    media = model(name)
    _preflight(media)
    root().mkdir(parents=True, exist_ok=True, mode=0o700)
    lr._no_symlinks(root(), root())
    with _lock():
        first = owner() is None
        _api_key(create=True)
        rebuilt = not _built()
        if rebuilt:
            print("Building media-gen and its engines...", flush=True)
            _build()
            # A release can retire models: their files go, and installed.json forgets them.
            _run(str(venv_dir() / "bin" / "media-gen"), "prune",
                 env=dict(os.environ, MEDIA_GEN_HOME=str(home_dir())))
            _write_installed(installed())
        print(f"Downloading {media.name} ({media.download_bytes / 1e9:.1f} GB)...", flush=True)
        _pull(media)
        before = installed()
        try:
            _start(rebuilt=rebuilt)
            _check(media)
            # Listed first: the gateway loads the rows when _configure_gateway signals it.
            _write_installed([*before, media.name])
            _configure_gateway()
        except BaseException:
            with contextlib.suppress(Exception):
                _write_installed(before)
            if first and owner() == "gateway":
                # Nothing is installed yet: stop the service this run started. The build and the
                # verified download stay, so a retry resumes.
                with contextlib.suppress(Exception):
                    lr._bootout(MEDIA_LABEL, plist_path())
                    plist_path().unlink()
            raise
    return status()


def _has_custom_profiles() -> bool:
    """Whether the owner keeps any adapter in media-gen's home (media-gen's ``custom/adapters``)."""
    return any((home_dir() / "custom" / "adapters").glob("*.safetensors"))


def _teardown() -> None:
    _unconfigure_gateway()
    if owner() == "gateway":
        lr._bootout(MEDIA_LABEL, plist_path())
        plist_path().unlink()
    shutil.rmtree(root())
    lr.bump_generation()


def remove(name: str | None = None) -> dict:
    """Remove one model, or (without ``name``, or for the last installed model) the whole runtime."""
    _require_owned()
    if not root().exists():
        raise LocalRuntimeError("Media generation is not installed")
    lr._no_symlinks(root(), root())
    with _lock():
        names = installed()
        if name is not None:
            model(name)
            if name not in names:
                raise LocalRuntimeError(f"{name} is not installed")
            if names == [name] and _has_custom_profiles():
                raise LocalRuntimeError(
                    f"{name} is the last media model, and removing it removes media generation with your custom "
                    f"profiles: remove them first (`model-gateway media custom remove`), or remove everything "
                    f"with `model-gateway media remove`")
            if names != [name]:
                _write_installed([n for n in names if n != name])
                lr.bump_generation()  # the gateway stops routing to it before its files go
                env = dict(os.environ, MEDIA_GEN_HOME=str(home_dir()))
                _run(str(venv_dir() / "bin" / "media-gen"), "remove", name, env=env)
                return status()
        _teardown()
    return status()


# ── custom profiles ──────────────────────────────────────────────────────────


def _media_gen_cli(*args: str, capture: bool = False) -> str:
    env = dict(os.environ, MEDIA_GEN_HOME=str(home_dir()))
    command = (str(venv_dir() / "bin" / "media-gen"), "custom", *args)
    if capture:
        return _output(*command, env=env)
    _run(*command, env=env)
    return ""


def _listed() -> list[dict]:
    rows = json.loads(_media_gen_cli("list", "--json", capture=True))
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise LocalRuntimeError("media-gen listed its custom profiles in an unexpected format")
    return rows


def _sync_custom() -> list[dict]:
    """Record media-gen's enabled profiles and reload the gateway: the routes follow that list."""
    rows = _listed()
    enabled = [{"name": row["id"], "base": row["base"], "display_name": row["display_name"]}
               for row in rows if row.get("enabled")]
    lr._atomic(custom_path(), json.dumps(enabled, indent=2) + "\n", root())
    lr.bump_generation()
    return rows


def custom(action: str, *args: str) -> list[dict]:
    """Run ``media-gen custom ACTION ARGS`` on this gateway's media-gen, then route what it enabled."""
    _require_owned()
    if not installed():
        raise LocalRuntimeError("Media generation is not installed; add a model first")
    lr._no_symlinks(root(), root())
    with _lock():
        if not _built():
            raise LocalRuntimeError(
                f"media-gen is older than this release; update it first with `model-gateway media add "
                f"{installed()[0]}`")
        if action == "list":
            return _listed()
        if action == "enable":
            (profile_id,) = args
            row = next((row for row in _listed() if row.get("id") == profile_id), None)
            if row is None:
                raise LocalRuntimeError(f"No custom profile {profile_id!r}")
            if row.get("base") not in installed():
                raise LocalRuntimeError(f"{profile_id} builds on {row.get('base')}, which is not installed")
        _media_gen_cli(action, *args)
        if action == "enable":
            # Routed only once media-gen serves it; any failure on the way leaves it disabled.
            try:
                served = {row.get("id") for row in _media_gen("/v1/models").get("data", [])}
                if args[0] not in served:
                    raise LocalRuntimeError(f"media-gen does not serve {args[0]}; it was left disabled")
                return _sync_custom()
            except BaseException:
                with contextlib.suppress(Exception):
                    _media_gen_cli("disable", args[0])
                    _sync_custom()
                raise
        return _sync_custom()


def _print_custom(rows: list[dict]) -> None:
    if not rows:
        print("No custom profiles")
    for row in rows:
        state = "enabled" if row.get("enabled") else "disabled"
        print(f"  {row.get('id', ''):<28}{state:<10}on {row.get('base')}, strength {row.get('strength')}  "
              f"{row.get('license')} ({row.get('source')})")


def _print_status(value: dict) -> None:
    print(f"{'managed:':<12}{'yes' if value['managed'] else 'no'}")
    print(f"{'running:':<12}{'yes' if value['running'] else 'no'}")
    print(f"{'base_url:':<12}{value['base_url'] or '-'}")
    print("models:")
    for row in value["models"]:
        state = "installed" if row["installed"] else "available" if row["eligible"] else "needs more memory"
        print(f"  {row['name']:<20}{row['modality']:<7}{state} ({row['download_gb']} GB, "
              f"{row['memory_gib']} GiB to run)")


def _license_accepted(media: MediaModel, accepted: bool) -> bool:
    """Each owner downloads the weights and is bound by the model's own license, so they accept it first."""
    print(f"{media.display_name} is licensed under the {media.license}: {media.license_url}")
    if media.gated:
        print("Its weights are gated: accept its terms on that page while signed in to Hugging Face, "
              "then sign in here with `hf auth login` (or set HF_TOKEN).")
    if accepted:
        return True
    if not sys.stdin.isatty():
        print("error: pass --accept-license to accept the license and install it", file=sys.stderr)
        return False
    if input("Accept its terms and install it? [y/N] ").strip().lower() in ("y", "yes"):
        return True
    print("Not installed")
    return False


def _custom_command(args: argparse.Namespace) -> int:
    if args.action == "list":
        rows = custom("list")
        print(json.dumps(rows, indent=2)) if args.json else _print_custom(rows)
        return 0
    if args.action == "import":
        print("The adapter's own license and its base model's license both apply to your use of it; "
              "Model Gateway keeps it on this Mac and never publishes or syncs it.")
        options = ["--base", args.base, "--name", args.name, "--license", args.license, "--source", args.source,
                   "--strength", str(args.strength)]
        if args.display_name:
            options += ["--display-name", args.display_name]
        custom("import", str(Path(args.path).expanduser().resolve()), *options)
        print(f"custom-{args.name} is imported but not routed; enable it with "
              f"`model-gateway media custom enable custom-{args.name}`")
        return 0
    custom(args.action, args.id)
    print({"enable": f"{args.id} is enabled: request it by name",
           "disable": f"{args.id} is disabled", "remove": f"{args.id} removed"}[args.action])
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="model-gateway media", description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    show = commands.add_parser("status", help="installed models and the service")
    show.add_argument("--json", action="store_true")
    add_parser = commands.add_parser("add", help="install a model (and media-gen, the first time)")
    add_parser.add_argument("model", choices=sorted(MODELS))
    add_parser.add_argument("--accept-license", action="store_true",
                            help="accept the model's license without being asked")
    drop = commands.add_parser("remove", help="remove one model, or (without a model) everything")
    drop.add_argument("model", nargs="?", choices=sorted(MODELS))
    own = commands.add_parser("custom", help="your own LoRA profiles on a curated model, on this Mac only")
    actions = own.add_subparsers(dest="action", required=True)
    listing = actions.add_parser("list", help="the custom profiles")
    listing.add_argument("--json", action="store_true")
    adding = actions.add_parser("import", help="check and copy in a LoRA as a new, disabled profile")
    adding.add_argument("path", help="a .safetensors LoRA adapter")
    adding.add_argument("--base", required=True, choices=sorted(MODELS),
                        help="the curated model it was made for (LTX-2.5 accepts LoRAs)")
    adding.add_argument("--name", required=True, help="served as custom-NAME")
    adding.add_argument("--license", required=True, help="the adapter's license, as its publisher states it")
    adding.add_argument("--source", required=True, help="where it came from (a URL, or your own training)")
    adding.add_argument("--display-name")
    adding.add_argument("--strength", type=float, default=1.0)
    for action in ("enable", "disable", "remove"):
        actions.add_parser(action).add_argument("id", help="custom-NAME")
    args = parser.parse_args(argv)
    os.umask(0o077)
    try:
        if args.command == "status":
            value = status()
            print(json.dumps(value, indent=2)) if args.json else _print_status(value)
        elif args.command == "add":
            if not _license_accepted(MODELS[args.model], args.accept_license):
                return 1
            add(args.model)
            print(f"{args.model} is ready: POST /v1/images/generations or /v1/videos on this gateway "
                  f"(list it with GET /v1/models?include=modality)")
        elif args.command == "custom":
            return _custom_command(args)
        else:
            remove(args.model)
            print(f"Media model {args.model} removed" if args.model else "Media generation removed")
        return 0
    except (ValueError, yaml.YAMLError, OSError, RuntimeError, subprocess.SubprocessError,
            http.client.HTTPException) as exc:
        print(f"error: {lr.safe_message(exc)}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
