# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The opt-in Tenki backend, against a fake SDK whose VM is a local directory: no network, no credentials."""

import io
import json
import os
import posixpath
import sys
import tarfile
import types
from dataclasses import dataclass

import pytest

from core.inference import os_sandbox, sandbox_tenki, sandbox_tenki_relay as relay

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason = "the Tenki backend is POSIX only")


@dataclass
class _Result:
    exit_code: int = 0
    signal: "str | None" = None
    timed_out: bool = False


@dataclass
class _Stat:
    size: int


class _FakeProcess:
    def __init__(self, chunks, result):
        self.stdout = iter(chunks)
        self.stderr = iter(())
        self._result = result

    def close_stdin(self):
        pass

    def wait(self):
        return self._result


class _FakeFS:
    def __init__(self, vm):
        self.vm = vm

    def mkdir(self, path):
        os.makedirs(self.vm.local(path), exist_ok = True)

    def upload(self, local_path, remote_path):
        with open(local_path, "rb") as src, open(self.vm.local(remote_path), "wb") as dst:
            dst.write(src.read())

    def download(self, remote_path, local_path):
        with open(self.vm.local(remote_path), "rb") as src, open(local_path, "wb") as dst:
            dst.write(src.read())

    def remove(self, path):
        os.unlink(self.vm.local(path))

    def stat(self, path):
        return _Stat(os.path.getsize(self.vm.local(path)))


class _FakeSandbox:
    """/home/tenki lives under *root*; `tool` stands in for the command and edits the VM's workdir."""

    instances: list = []
    root = ""
    tool = None
    chunks = (b"hello\n",)
    result = _Result()
    fail_on_start = False

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.id = f"sess-{len(_FakeSandbox.instances)}"
        self.closed = False
        self.started = None
        self.fs = _FakeFS(self)
        _FakeSandbox.instances.append(self)

    @classmethod
    def create(cls, **kwargs):
        return cls(**kwargs)

    def local(self, remote):
        assert remote.startswith(sandbox_tenki.REMOTE_HOME)
        return os.path.join(self.root, *remote.split("/"))

    def exec(self, *argv, check = False):
        assert argv[0] == "tar" and argv[3] == "-C"
        if argv[1] == "-xf":
            with tarfile.open(self.local(argv[2])) as tar:
                tar.extractall(self.local(argv[4]))
        else:
            with tarfile.open(self.local(argv[2]), "w") as tar:
                tar.add(self.local(argv[4]), arcname = ".")
        return _Result()

    def start(self, *argv, cwd, env, timeout):
        if _FakeSandbox.fail_on_start:
            raise RuntimeError("data plane unavailable")
        self.started = {"argv": argv, "cwd": cwd, "env": env, "timeout": timeout}
        if _FakeSandbox.tool is not None:
            _FakeSandbox.tool(self.local(cwd))
        return _FakeProcess(_FakeSandbox.chunks, _FakeSandbox.result)

    def close_if_open(self):
        self.closed = True


class _FakeClient:
    gets: list = []

    def __init__(self, **kwargs):
        pass

    def get(self, session_id):
        sandbox = _FakeSandbox()
        sandbox.id = session_id
        _FakeClient.gets.append(sandbox)
        return sandbox

    def close(self):
        pass


@pytest.fixture
def fake_tenki(monkeypatch, tmp_path):
    module = types.ModuleType("tenki")
    module.Sandbox = _FakeSandbox
    module.Client = _FakeClient
    monkeypatch.setitem(sys.modules, "tenki", module)
    vm = tmp_path / "vm"
    (vm / "home" / "tenki").mkdir(parents = True)
    monkeypatch.setattr(_FakeSandbox, "instances", [])
    monkeypatch.setattr(_FakeSandbox, "root", str(vm))
    monkeypatch.setattr(_FakeSandbox, "tool", None)
    monkeypatch.setattr(_FakeSandbox, "chunks", (b"hello\n",))
    monkeypatch.setattr(_FakeSandbox, "result", _Result())
    monkeypatch.setattr(_FakeSandbox, "fail_on_start", False)
    monkeypatch.setattr(_FakeClient, "gets", [])
    return module


@pytest.fixture
def enabled(monkeypatch):
    monkeypatch.setenv(sandbox_tenki.ENABLE_ENV, "1")
    monkeypatch.setenv("TENKI_API_KEY", "tk_test_not_a_real_key")
    monkeypatch.setattr(sandbox_tenki.importlib.util, "find_spec", lambda name: object())


def _plan(workdir, mode = "auto", argv = None, env = None):
    return os_sandbox.ToolLaunchPlan(
        argv = tuple(argv or (sys.executable, "-u", os.path.join(workdir, "studio_exec_x.py"))),
        workdir = str(workdir),
        env = env if env is not None else {"PYTHONIOENCODING": "utf-8", "PATH": "/host/bin"},
        requested_mode = mode,
        timeout_seconds = 60,
        execution_kind = "python",
    )


def _spec(prepared):
    with open(prepared.argv[-1], encoding = "utf-8") as handle:
        return json.load(handle)


# --- off by default -----------------------------------------------------------------------------


def test_off_by_default_leaves_the_local_path_untouched(monkeypatch, tmp_path):
    monkeypatch.delenv(sandbox_tenki.ENABLE_ENV, raising = False)
    assert not sandbox_tenki.enabled()
    reached = []

    def local_snapshot(**kwargs):
        reached.append(kwargs)
        raise RuntimeError("stop at the local capability check")

    monkeypatch.setattr(os_sandbox, "capability_snapshot", local_snapshot)
    monkeypatch.setattr(
        sandbox_tenki, "available", lambda: pytest.fail("probed Tenki while disabled")
    )
    with pytest.raises(RuntimeError, match = "local capability"):
        os_sandbox.prepare_tool_launch(_plan(tmp_path))
    assert reached


@pytest.mark.parametrize("value", ["", "0", "false", "no"])
def test_only_an_explicit_yes_enables_it(monkeypatch, value):
    monkeypatch.setenv(sandbox_tenki.ENABLE_ENV, value)
    assert not sandbox_tenki.enabled()


@pytest.mark.parametrize("mode", ["full", "software"])
def test_full_and_software_modes_never_go_remote(monkeypatch, tmp_path, enabled, mode):
    monkeypatch.setattr(sandbox_tenki, "prepare", lambda plan: pytest.fail("went remote"))
    prepared = os_sandbox.prepare_tool_launch(_plan(tmp_path, mode = mode))
    assert prepared.backend != sandbox_tenki.BACKEND_NAME


# --- enabled but unavailable ---------------------------------------------------------------------


def test_required_refuses_without_a_key(monkeypatch, tmp_path, enabled):
    monkeypatch.delenv("TENKI_API_KEY")
    monkeypatch.delenv("TENKI_AUTH_TOKEN", raising = False)
    with pytest.raises(os_sandbox.SandboxUnavailableError, match = "TENKI_SANDBOX_UNAVAILABLE"):
        os_sandbox.prepare_tool_launch(_plan(tmp_path, mode = "required"))


def test_auto_keeps_local_isolation_without_the_sdk(monkeypatch, tmp_path, enabled):
    monkeypatch.setattr(sandbox_tenki.importlib.util, "find_spec", lambda name: None)
    monkeypatch.setattr(
        os_sandbox,
        "capability_snapshot",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("local path reached")),
    )
    with pytest.raises(RuntimeError, match = "local path reached"):
        os_sandbox.prepare_tool_launch(_plan(tmp_path))


# --- prepare ---------------------------------------------------------------------------------------


def test_prepare_routes_through_the_relay(tmp_path, enabled):
    prepared = os_sandbox.prepare_tool_launch(_plan(tmp_path))
    try:
        assert prepared.backend == "tenki-vm"
        assert prepared.argv[:3] == (sys.executable, "-I", sandbox_tenki._RELAY)
        record = prepared.execution_record
        assert record.effective_mode == "remote_vm" and record.os_isolation
        assert "model_library_not_available" in record.limitations
        assert record.network_policy == "unrestricted"
        spec = _spec(prepared)
        assert spec["argv"] == ["python3", "-u", f"{sandbox_tenki.REMOTE_WORKDIR}/studio_exec_x.py"]
        assert spec["timeout_seconds"] == 60 + sandbox_tenki._REMOTE_TIMEOUT_GRACE_SECONDS
        # Host PATH stays on the host; the key reaches the relay only, never the spec or the VM env.
        assert spec["env"] == {"PYTHONIOENCODING": "utf-8"}
        assert "tk_test_not_a_real_key" not in json.dumps(spec)
        assert prepared.env["TENKI_API_KEY"] == "tk_test_not_a_real_key"
        assert not os.path.commonpath([spec["private_dir"], str(tmp_path)]) == str(tmp_path)
    finally:
        prepared.cleanup()
    assert not os.path.exists(spec["private_dir"])


def test_allowlist_drops_the_unrestricted_network_limitation(monkeypatch, tmp_path, enabled):
    monkeypatch.setenv(sandbox_tenki.ALLOW_DOMAINS_ENV, "pypi.org, files.pythonhosted.org")
    prepared = os_sandbox.prepare_tool_launch(_plan(tmp_path))
    try:
        assert _spec(prepared)["allow_domains"] == ["pypi.org", "files.pythonhosted.org"]
        assert prepared.execution_record.network_policy == "allowlist"
        assert "unrestricted_network" not in prepared.execution_record.limitations
    finally:
        prepared.cleanup()


def test_terminal_keeps_the_shell_and_the_command_text(tmp_path):
    argv = sandbox_tenki.remote_argv(("bash", "-c", "ls /etc"), str(tmp_path))
    assert argv == ["bash", "-c", "ls /etc"]


def test_host_paths_outside_the_workdir_are_not_remapped(tmp_path):
    inside = str(tmp_path / "a" / "b.py")
    argv = sandbox_tenki.remote_argv((sys.executable, inside, "/etc/passwd"), str(tmp_path))
    assert argv == ["python3", f"{sandbox_tenki.REMOTE_WORKDIR}/a/b.py", "/etc/passwd"]


def test_other_programs_are_refused(tmp_path):
    with pytest.raises(os_sandbox.SandboxBuildError, match = "Python and POSIX shells"):
        sandbox_tenki.remote_argv(("/usr/bin/curl", "example.com"), str(tmp_path))


# --- the relay against a fake VM ---------------------------------------------------------------------


def _run_relay(tmp_path, workdir, **overrides):
    private = tmp_path / "private"
    private.mkdir(exist_ok = True)
    spec = {
        "argv": ["python3", "-u", f"{sandbox_tenki.REMOTE_WORKDIR}/job.py"],
        "local_workdir": str(workdir),
        "remote_workdir": sandbox_tenki.REMOTE_WORKDIR,
        "env": {"PYTHONIOENCODING": "utf-8"},
        "private_dir": str(private),
        "session_file": str(private / "session"),
        "timeout_seconds": 90,
        "max_session_seconds": 700,
        "max_sync_bytes": 1024 * 1024,
        "allow_outbound": False,
        "allow_domains": [],
        "execution_kind": "python",
        **overrides,
    }
    out = io.BytesIO()
    code = relay.run(spec, out)
    return code, out.getvalue(), spec


def test_relay_round_trip(tmp_path, fake_tenki):
    workdir = tmp_path / "work"
    workdir.mkdir()
    (workdir / "job.py").write_text("print('hi')")
    (workdir / "unsloth-tmp").mkdir()
    (workdir / "unsloth-tmp" / "host-only").write_text("x")

    def tool(remote_workdir):
        assert os.path.exists(os.path.join(remote_workdir, "job.py"))
        assert not os.path.exists(os.path.join(remote_workdir, "unsloth-tmp"))
        os.makedirs(os.path.join(remote_workdir, "out"))
        with open(os.path.join(remote_workdir, "out", "plot.txt"), "w") as handle:
            handle.write("made in the VM")

    _FakeSandbox.tool = staticmethod(tool)
    _FakeSandbox.result = _Result(exit_code = 3)
    code, output, spec = _run_relay(tmp_path, workdir)

    assert code == 3
    assert output == b"hello\n"
    assert (workdir / "out" / "plot.txt").read_text() == "made in the VM"
    (sandbox,) = _FakeSandbox.instances
    assert sandbox.kwargs["allow_inbound"] is False and sandbox.kwargs["allow_outbound"] is False
    assert sandbox.kwargs["max_duration"] == 700
    assert sandbox.started["argv"][:4] == ("sh", "-c", relay._MERGE_STDERR, "sh")
    assert sandbox.started["argv"][4:] == tuple(spec["argv"])
    assert sandbox.started["timeout"] == 90
    assert sandbox.closed
    assert not os.path.exists(spec["session_file"])


def test_relay_failure_is_output_and_still_closes_the_vm(tmp_path, fake_tenki):
    workdir = tmp_path / "work"
    workdir.mkdir()
    _FakeSandbox.fail_on_start = True
    code, output, _ = _run_relay(tmp_path, workdir)
    assert code == relay.RELAY_FAILED
    assert b"[Tenki sandbox] RuntimeError: data plane unavailable" in output
    assert _FakeSandbox.instances[0].closed


def test_relay_refuses_an_oversized_workdir_before_creating_a_vm(tmp_path, fake_tenki):
    workdir = tmp_path / "work"
    workdir.mkdir()
    (workdir / "big.bin").write_bytes(b"\0" * 2048)
    code, output, _ = _run_relay(tmp_path, workdir, max_sync_bytes = 1024)
    assert code == relay.RELAY_FAILED and b"sync limit" in output
    assert _FakeSandbox.instances == []


def test_a_killed_relay_leaves_its_session_to_the_cleanup_callback(tmp_path, fake_tenki):
    session_file = tmp_path / "session"
    session_file.write_text("sess-abandoned")
    sandbox_tenki.terminate_leftover_session(str(session_file))
    (sandbox,) = _FakeClient.gets
    assert sandbox.id == "sess-abandoned" and sandbox.closed


def test_no_session_file_means_nothing_to_terminate(tmp_path, fake_tenki):
    sandbox_tenki.terminate_leftover_session(str(tmp_path / "missing"))
    assert _FakeClient.gets == []


# --- copying back is the host-facing edge -------------------------------------------------------------


def _hostile_archive(path):
    with tarfile.open(path, "w") as tar:

        def add(name, data = b"x", kind = tarfile.REGTYPE, linkname = ""):
            info = tarfile.TarInfo(name)
            info.type = kind
            info.linkname = linkname
            info.size = len(data) if kind == tarfile.REGTYPE else 0
            info.mode = 0o4777
            tar.addfile(info, io.BytesIO(data) if kind == tarfile.REGTYPE else None)

        add("./ok.txt", b"fine")
        add("../escape.txt")
        add("/abs.txt")
        add("./link", kind = tarfile.SYMTYPE, linkname = "/etc/passwd")
        add("./hard", kind = tarfile.LNKTYPE, linkname = "ok.txt")
        add("./dev", kind = tarfile.CHRTYPE)
        add("./via/inner.txt")
        add("./.unsloth-packages/evil.py")


def test_unpack_takes_regular_files_only_and_never_follows_links(tmp_path):
    workdir = tmp_path / "work"
    outside = tmp_path / "outside"
    workdir.mkdir()
    outside.mkdir()
    # A link already in the workdir must not carry a copied-back file out of it.
    os.symlink(outside, workdir / "via")
    archive = tmp_path / "out.tar"
    _hostile_archive(archive)

    skipped = relay.unpack_into_workdir(str(archive), str(workdir), 1024 * 1024)

    assert (workdir / "ok.txt").read_bytes() == b"fine"
    assert os.stat(workdir / "ok.txt").st_mode & 0o7777 == 0o755
    assert sorted(os.listdir(workdir)) == ["ok.txt", "via"]
    assert os.listdir(outside) == []
    assert not (tmp_path / "escape.txt").exists()
    assert skipped == 7


def test_unpack_replaces_a_link_instead_of_writing_through_it(tmp_path):
    workdir = tmp_path / "work"
    workdir.mkdir()
    target = tmp_path / "host-file"
    target.write_text("untouched")
    os.symlink(target, workdir / "notes.txt")
    archive = tmp_path / "out.tar"
    with tarfile.open(archive, "w") as tar:
        info = tarfile.TarInfo("./notes.txt")
        info.size = 3
        tar.addfile(info, io.BytesIO(b"new"))

    relay.unpack_into_workdir(str(archive), str(workdir), 1024)

    assert target.read_text() == "untouched"
    assert not os.path.islink(workdir / "notes.txt")
    assert (workdir / "notes.txt").read_text() == "new"
