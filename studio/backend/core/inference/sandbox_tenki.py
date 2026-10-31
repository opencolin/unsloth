# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Opt-in remote backend: each tool call runs in a disposable Tenki Sandbox VM. Off unless UNSLOTH_TENKI_SANDBOX=1.

The launch is still a local process, a relay (sandbox_tenki_relay.py) that copies the workdir up, streams the
command's output and copies changed files back. tools.py keeps its own drain, timeout and killpg paths unchanged;
a relay killed mid-call leaves its VM to the cleanup callback here, then to the session's max_duration.
"""

from __future__ import annotations

import importlib.util
import json
import os
import posixpath
import sys
import tempfile

from loggers import get_logger

from .os_sandbox import (
    PROFILE_VERSION,
    PreparedSandboxLaunch,
    SandboxBuildError,
    ToolLaunchPlan,
)

logger = get_logger(__name__)

ENABLE_ENV = "UNSLOTH_TENKI_SANDBOX"
ALLOW_OUTBOUND_ENV = "UNSLOTH_TENKI_ALLOW_OUTBOUND"
ALLOW_DOMAINS_ENV = "UNSLOTH_TENKI_ALLOW_DOMAINS"
MAX_SYNC_MB_ENV = "UNSLOTH_TENKI_MAX_SYNC_MB"
MAX_SESSION_SECONDS_ENV = "UNSLOTH_TENKI_MAX_SESSION_SECONDS"
# The SDK's own resolution order; only these reach the relay, never the VM.
CREDENTIAL_ENV = ("TENKI_AUTH_TOKEN", "TENKI_API_KEY")
_ENDPOINT_ENV = ("TENKI_API_ENDPOINT", "TENKI_API_URL")
# What the relay needs to reach the API through a corporate proxy or a private CA.
_TRANSPORT_ENV = (
    "HTTPS_PROXY",
    "https_proxy",
    "HTTP_PROXY",
    "http_proxy",
    "NO_PROXY",
    "no_proxy",
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
)
# Host paths mean nothing in the VM, so the tool env is reduced to these.
_FORWARDED_TOOL_ENV = (
    "PYTHONIOENCODING",
    "PYTHONUNBUFFERED",
    "PYTHONDONTWRITEBYTECODE",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "TERM",
    "NO_COLOR",
    "MPLBACKEND",
)

BACKEND_NAME = "tenki-vm"
PROFILE_ID = f"tenki-vm-{PROFILE_VERSION}"
LIMITATIONS = (
    "remote_execution",
    "workdir_copied_not_shared",
    "deletions_not_synced_back",
    "only_regular_files_synced_back",
    "model_library_not_available",
    "gpu_not_available",
    "requires_network_and_credentials",
    "per_call_vm_startup",
)

REMOTE_HOME = "/home/tenki"
REMOTE_WORKDIR = f"{REMOTE_HOME}/work"
_RELAY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sandbox_tenki_relay.py")
# Studio's timer fires first; the remote budget only bounds a relay nobody is waiting on.
_REMOTE_TIMEOUT_GRACE_SECONDS = 30
# VM create (3 min SDK default) plus both copies.
_SESSION_OVERHEAD_SECONDS = 600
_DEFAULT_MAX_SESSION_SECONDS = 3600
_DEFAULT_MAX_SYNC_MB = 256
_SHELLS = ("bash", "sh")


def _flag(name: str, default: bool) -> bool:
    value = os.environ.get(name, "").strip().lower()
    if not value:
        return default
    return value in ("1", "true", "yes", "on")


def _int_env(name: str, default: int) -> int:
    try:
        value = int(os.environ.get(name, "").strip() or default)
    except ValueError:
        return default
    return value if value > 0 else default


def enabled() -> bool:
    """Read on every launch, so turning it off takes effect without a restart."""
    return _flag(ENABLE_ENV, False)


def available() -> tuple[bool, str]:
    """No network round trip: an unreachable API surfaces as the call's own output."""
    if sys.platform == "win32":
        return False, "the Tenki backend maps POSIX shells only; Windows hosts keep MXC."
    if sys.version_info < (3, 10):
        return False, "the Tenki SDK needs Python 3.10 or newer."
    if importlib.util.find_spec("tenki") is None:
        return False, "the Tenki SDK is not installed (pip install tenki)."
    if not any(os.environ.get(name) for name in CREDENTIAL_ENV):
        return False, "no Tenki API key: set TENKI_API_KEY to a workspace key (tk_...)."
    return True, "Tool calls run in a disposable Tenki Sandbox VM."


def limitations() -> tuple[str, ...]:
    if allow_outbound() and not allowed_domains():
        return (*LIMITATIONS, "unrestricted_network")
    return LIMITATIONS


def allow_outbound() -> bool:
    # On by default so pip install works as it does locally; the VM never reaches this host's loopback.
    return _flag(ALLOW_OUTBOUND_ENV, True)


def allowed_domains() -> list[str]:
    raw = os.environ.get(ALLOW_DOMAINS_ENV, "")
    return [part.strip() for part in raw.split(",") if part.strip()]


def _remote_path(arg: str, workdir: str) -> str:
    """A workdir path becomes the same path under the VM's workdir; anything else passes through as text."""
    if not os.path.isabs(arg):
        return arg
    relative = os.path.relpath(os.path.realpath(arg), os.path.realpath(workdir))
    if relative == os.curdir:
        return REMOTE_WORKDIR
    if relative == os.pardir or relative.startswith(os.pardir + os.sep):
        return arg
    return posixpath.join(REMOTE_WORKDIR, *relative.split(os.sep))


def remote_argv(argv: tuple[str, ...], workdir: str) -> list[str]:
    """Studio's own interpreter becomes the VM's python3; a shell keeps its name. Anything else is refused."""
    if not argv:
        raise SandboxBuildError("the Tenki backend got an empty command")
    program = argv[0]
    if os.path.realpath(program) == os.path.realpath(sys.executable):
        head = "python3"
    elif os.path.basename(program) in _SHELLS:
        head = os.path.basename(program)
    else:
        raise SandboxBuildError(
            f"the Tenki backend runs Python and POSIX shells only, not {os.path.basename(program)!r}"
        )
    return [head, *(_remote_path(arg, workdir) for arg in argv[1:])]


def _relay_env(private_dir: str) -> dict[str, str]:
    env = {"PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1", "HOME": private_dir}
    for name in (*CREDENTIAL_ENV, *_ENDPOINT_ENV, *_TRANSPORT_ENV):
        if os.environ.get(name):
            env[name] = os.environ[name]
    return env


def _max_session_seconds(timeout_seconds: int | None) -> int:
    ceiling = _int_env(MAX_SESSION_SECONDS_ENV, _DEFAULT_MAX_SESSION_SECONDS)
    if timeout_seconds is None:
        return ceiling
    return min(ceiling, timeout_seconds + _SESSION_OVERHEAD_SECONDS)


def terminate_leftover_session(session_file: str) -> None:
    """The relay removes this file once it has closed its VM; a file still here means it was killed first."""
    try:
        with open(session_file, encoding = "utf-8") as handle:
            session_id = handle.read().strip()
    except FileNotFoundError:
        return
    if not session_id:
        return
    from tenki import Client
    client = Client(timeout = 15)
    try:
        client.get(session_id).close_if_open()
        logger.info("Terminated Tenki session %s left by a killed tool call", session_id)
    finally:
        client.close()


def prepare(plan: ToolLaunchPlan) -> PreparedSandboxLaunch:
    if not os.path.isdir(plan.workdir):
        raise SandboxBuildError(f"the tool workdir does not exist: {plan.workdir}")
    command = remote_argv(plan.argv, plan.workdir)
    # Outside the workdir, so neither the copy up nor the copy back can see it.
    private_dir = tempfile.mkdtemp(prefix = "unsloth-tenki-")
    session_file = os.path.join(private_dir, "session")
    spec = {
        "argv": command,
        "local_workdir": os.path.realpath(plan.workdir),
        "remote_workdir": REMOTE_WORKDIR,
        "env": {
            name: plan.env[name] for name in _FORWARDED_TOOL_ENV if plan.env.get(name) is not None
        },
        "private_dir": private_dir,
        "session_file": session_file,
        "timeout_seconds": (
            None
            if plan.timeout_seconds is None
            else plan.timeout_seconds + _REMOTE_TIMEOUT_GRACE_SECONDS
        ),
        "max_session_seconds": _max_session_seconds(plan.timeout_seconds),
        "max_sync_bytes": _int_env(MAX_SYNC_MB_ENV, _DEFAULT_MAX_SYNC_MB) * 1024 * 1024,
        "allow_outbound": allow_outbound(),
        "allow_domains": allowed_domains(),
        "execution_kind": plan.execution_kind or "",
    }
    spec_path = os.path.join(private_dir, "spec.json")
    with open(spec_path, "w", encoding = "utf-8") as handle:
        json.dump(spec, handle)
    return PreparedSandboxLaunch(
        # -I: the relay's cwd is the workdir, which tool code wrote; nothing there may shadow an import.
        argv = (sys.executable, "-I", _RELAY, spec_path),
        workdir = plan.workdir,
        env = _relay_env(private_dir),
        preexec_fn = plan.preexec_fn,
        backend = BACKEND_NAME,
        timeout_seconds = plan.timeout_seconds,
        close_fds = plan.close_fds,
        terminate_descendants = plan.terminate_descendants,
        cleanup_callbacks = [lambda: terminate_leftover_session(session_file)],
        cleanup_paths = [private_dir],
    )
