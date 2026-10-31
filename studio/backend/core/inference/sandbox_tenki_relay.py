# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Relay for one Tenki tool call, run as `python -I sandbox_tenki_relay.py SPEC`. Stdlib and the Tenki SDK only.

Copies the workdir up, runs the command with stdout and stderr merged, streams it to this process's stdout,
copies regular files back and exits with the command's exit code. 125 means the relay itself failed.
"""

from __future__ import annotations

import json
import os
import posixpath
import stat
import sys
import tarfile
import threading

RELAY_FAILED = 125
# Host-only state: packages built for this OS and Studio's scratch dir.
NOT_SYNCED = frozenset(("unsloth-tmp", ".unsloth-packages"))
_MERGE_STDERR = 'exec "$@" 2>&1'


class SyncError(RuntimeError):
    pass


def pack_workdir(workdir: str, archive: str, max_bytes: int) -> None:
    total = 0
    with tarfile.open(archive, "w") as tar:
        for base, dirs, names in os.walk(workdir, followlinks = False):
            relative_base = os.path.relpath(base, workdir)
            if relative_base == os.curdir:
                dirs[:] = [name for name in dirs if name not in NOT_SYNCED]
                names = [name for name in names if name not in NOT_SYNCED]
            for name in (*dirs, *names):
                path = os.path.join(base, name)
                arcname = posixpath.normpath(posixpath.join(*relative_base.split(os.sep), name))
                info = os.lstat(path)
                # Sockets, FIFOs and devices stay on the host; the VM has its own.
                if not (
                    stat.S_ISREG(info.st_mode)
                    or stat.S_ISDIR(info.st_mode)
                    or stat.S_ISLNK(info.st_mode)
                ):
                    continue
                total += info.st_size if stat.S_ISREG(info.st_mode) else 0
                if total > max_bytes:
                    raise SyncError(
                        f"the workdir is over the {max_bytes // (1024 * 1024)} MB Tenki sync limit"
                    )
                tar.add(path, arcname = arcname, recursive = False)


def _real_directories_to(workdir: str, parts: list[str]) -> bool:
    """Every existing ancestor is a real directory, never a link: a copy back must stay inside the workdir."""
    path = workdir
    for part in parts[:-1]:
        path = os.path.join(path, part)
        try:
            info = os.lstat(path)
        except FileNotFoundError:
            continue
        if not stat.S_ISDIR(info.st_mode):
            return False
    return True


def unpack_into_workdir(archive: str, workdir: str, max_bytes: int) -> int:
    """Regular files and directories only, by name, never through a link. Returns the entries skipped."""
    skipped = 0
    total = 0
    with tarfile.open(archive, "r") as tar:
        for member in tar:
            name = posixpath.normpath(member.name)
            if name in (os.curdir, ""):
                continue
            parts = name.split("/")
            if name.startswith("/") or os.pardir in parts or parts[0] in NOT_SYNCED:
                skipped += 1
                continue
            if not _real_directories_to(workdir, parts):
                skipped += 1
                continue
            target = os.path.join(workdir, *parts)
            if member.isdir():
                if os.path.lexists(target) and not (
                    os.path.isdir(target) and not os.path.islink(target)
                ):
                    skipped += 1
                    continue
                os.makedirs(target, exist_ok = True)
                continue
            if not member.isfile() or (os.path.isdir(target) and not os.path.islink(target)):
                skipped += 1
                continue
            total += member.size
            if total > max_bytes:
                raise SyncError(
                    f"the files copied back are over the {max_bytes // (1024 * 1024)} MB sync limit"
                )
            source = tar.extractfile(member)
            staging = f"{target}.unsloth-sync-{os.getpid()}"
            # O_EXCL: never write through something already at the staging name.
            fd = os.open(
                staging,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o755 if member.mode & 0o111 else 0o644,
            )
            with os.fdopen(fd, "wb") as handle:
                while True:
                    chunk = source.read(1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
            # Replaces a link at target with the file rather than writing through it.
            os.replace(staging, target)
    return skipped


def _drain(stream) -> None:
    for _ in stream:
        pass


def _write_session_file(path: str, session_id: str) -> None:
    staging = f"{path}.tmp"
    with open(staging, "w", encoding = "utf-8") as handle:
        handle.write(session_id)
    os.replace(staging, path)


def _exit_code(result) -> int:
    if result.signal:
        # The guest names the signal rather than numbering it; 137 reads as "killed" in the transcript.
        return 137
    return result.exit_code if result.exit_code >= 0 else 1


def run(spec: dict, out) -> int:
    from tenki import Sandbox

    private_dir = spec["private_dir"]
    local_workdir = spec["local_workdir"]
    remote_workdir = spec["remote_workdir"]
    remote_home = posixpath.dirname(remote_workdir)
    max_bytes = spec["max_sync_bytes"]
    archive_in = os.path.join(private_dir, "in.tar")
    archive_out = os.path.join(private_dir, "out.tar")
    sandbox = None
    try:
        pack_workdir(local_workdir, archive_in, max_bytes)
        sandbox = Sandbox.create(
            name = f"unsloth-studio-{spec.get('execution_kind') or 'tool'}",
            allow_inbound = False,
            allow_outbound = spec["allow_outbound"],
            allow_domains = spec["allow_domains"] or None,
            max_duration = spec["max_session_seconds"],
            metadata = {"client": "unsloth-studio"},
        )
        _write_session_file(spec["session_file"], sandbox.id)

        remote_in = posixpath.join(remote_home, ".unsloth-in.tar")
        sandbox.fs.mkdir(remote_workdir)
        sandbox.fs.upload(archive_in, remote_in)
        sandbox.exec("tar", "-xf", remote_in, "-C", remote_workdir, check = True)
        sandbox.fs.remove(remote_in)

        proc = sandbox.start(
            "sh",
            "-c",
            _MERGE_STDERR,
            "sh",
            *spec["argv"],
            cwd = remote_workdir,
            env = spec["env"],
            timeout = spec["timeout_seconds"],
        )
        proc.close_stdin()
        # Merged into stdout above; drained anyway so a full stderr buffer cannot stall the stream.
        threading.Thread(target = _drain, args = (proc.stderr,), daemon = True).start()
        for chunk in proc.stdout:
            out.write(chunk)
            out.flush()
        result = proc.wait()

        remote_out = posixpath.join(remote_home, ".unsloth-out.tar")
        sandbox.exec("tar", "-cf", remote_out, "-C", remote_workdir, ".", check = True)
        if sandbox.fs.stat(remote_out).size > max_bytes + 10 * 1024 * 1024:
            raise SyncError("the workdir grew past the Tenki sync limit; nothing was copied back")
        sandbox.fs.download(remote_out, archive_out)
        skipped = unpack_into_workdir(archive_out, local_workdir, max_bytes)
        if skipped:
            out.write(
                f"\n[Tenki sandbox] {skipped} links or special files were not copied back.\n".encode()
            )
        if getattr(result, "timed_out", False):
            out.write(b"\n[Tenki sandbox] the command hit its time limit in the VM.\n")
        return _exit_code(result)
    except Exception as exc:  # noqa: BLE001 - reported as the call's output, never a host fallback
        out.write(f"\n[Tenki sandbox] {type(exc).__name__}: {exc}\n".encode())
        out.flush()
        return RELAY_FAILED
    finally:
        if sandbox is not None:
            try:
                sandbox.close_if_open()
                os.unlink(spec["session_file"])
            except Exception:  # noqa: BLE001 - the server's cleanup callback retries from the file
                pass


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        sys.stderr.write("usage: sandbox_tenki_relay.py SPEC\n")
        return RELAY_FAILED
    with open(argv[1], encoding = "utf-8") as handle:
        spec = json.load(handle)
    return run(spec, sys.stdout.buffer)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
