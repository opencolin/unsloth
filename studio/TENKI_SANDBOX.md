# Running Studio tool calls in a Tenki Sandbox VM

Status: prototype, off by default. Design note plus a minimal backend
(`backend/core/inference/sandbox_tenki.py`, `sandbox_tenki_relay.py`) and mocked
tests (`backend/tests/test_sandbox_tenki.py`). Nothing has been run against the
live Tenki API yet, so latency and SDK behaviour under Studio's rlimits are
unverified.

## Question

Studio runs the `python` and `terminal` tool calls of local models behind an OS
sandbox: bubblewrap on Linux, Seatbelt on macOS and MXC on Windows. These share
the host kernel, and their `LIMITATIONS` list what still gets through. For
example, `unrestricted_network` includes this machine's loopback services,
`system_paths_readable` applies, and `model_cache_writable` applies too. A
disposable [Tenki Sandbox](https://tenki.cloud/docs/sandbox/quickstart) VM
removes the host from what the tool code can reach. The question is whether
this fits Studio's launch contract without changing anything for current users.

**Answer: yes, as an opt-in backend.** The contract is shaped for local
processes, but a local *relay* process drives the remote VM and satisfies that
contract unchanged.

## How it plugs in

`os_sandbox.prepare_tool_launch(plan)` turns a `ToolLaunchPlan` into a
`PreparedSandboxLaunch`. `tools.py` then spawns that launch with
`spawn_prepared_launch` and owns everything after the spawn: it drains merged
stdout, applies its own timeout, cancels through `killpg(SIGKILL)`, verifies
completion and runs `prepared.cleanup()`.

The Tenki backend keeps all of that. `prepare` returns a launch whose argv is

```
<studio python> -I sandbox_tenki_relay.py <private-dir>/spec.json
```

The relay is an ordinary child process. `tools.py` reads its stdout and kills
its process group exactly as it would for bwrap. The relay does the remote work:

1. Pack the workdir into a tar. It skips `unsloth-tmp/` and
   `.unsloth-packages/`, which are host-only, and refuses if the files exceed
   the size cap.
2. `Sandbox.create(allow_inbound=False, allow_outbound=…, allow_domains=…,
   max_duration=…)`, then write the session id to `<private-dir>/session`.
3. `fs.upload` the tar, then `exec("tar", "-xf", …)` into `/home/tenki/work`.
4. `start("sh", "-c", 'exec "$@" 2>&1', "sh", *argv, cwd=…, env=…, timeout=…)`.
   Each `proc.stdout` chunk is written to the relay's stdout as it arrives.
5. Run `tar -cf` remotely, `fs.download`, then unpack locally. Only regular
   files and directories come back. Entries are taken by name, never through a
   link, with mode reduced to 0644 or 0755.
6. `close_if_open()`, delete the session file, and exit with the command's exit
   code. 125 means the relay itself failed, and the reason is printed as output.

`ToolLaunchPlan.argv` is rewritten for the VM. Studio's own interpreter becomes
`python3`, and `bash`/`sh` keep their names. Workdir paths become
`/home/tenki/work/...`. Any other program is refused with `SandboxBuildError`.
The tool env is reduced to locale and Python I/O variables, because host
`PATH`, `HOME` and so on mean nothing in the VM.

### Contract map

| Studio contract point | Tenki call / mechanism | Notes |
|---|---|---|
| `ToolLaunchPlan.argv` | `Sandbox.start(*argv, cwd=…)` | Remapped as above. stderr is merged by an `sh -c` wrapper, because the SDK yields stdout and stderr as separate streams. |
| `ToolLaunchPlan.workdir` | `fs.upload` + `tar -x` before; `tar -c` + `fs.download` after | Copied, not shared. See the gaps below. |
| `ToolLaunchPlan.env` | `start(env=…)` | Allowlist only. The API key is never forwarded. |
| `timeout_seconds` | `start(timeout=t+30)`; `create(max_duration=t+600)` | Studio's timer still fires first and kills the relay. The remote budget only bounds an orphan. |
| `cancel_event` / killpg | the relay dies; `cleanup_callbacks` → `Client().get(id).close_if_open()` | The relay cannot clean up after SIGKILL, so the server process terminates the VM from the session file. `max_duration` is the backstop if that also fails. |
| streaming | `for chunk in proc.stdout` → relay stdout | Byte stream, flushed per chunk. |
| exit code | `CommandResult.exit_code`; a signal maps to 137 | `tools.py` formats `Exit code N` as usual. |
| `PreparedSandboxLaunch.cleanup_paths` | private `mkdtemp` dir (spec, tars, session file) | It lives outside the workdir, so neither copy can see it. |
| `SandboxCapability` / probe | `sandbox_tenki.available()` | SDK importable, key present, POSIX host, Python ≥ 3.10. No network probe. |
| `ToolExecutionRecord` | `effective_mode="remote_vm"`, `backend="tenki-vm"`, `environment="tenki"` | `network_policy` is `unrestricted`, `allowlist` or `none` from the create options. |
| `BACKEND_NAME` / `PROFILE_ID` / `LIMITATIONS` | `tenki-vm`, `tenki-vm-unsloth-sandbox-v1`, see the module | Plus `unrestricted_network` unless outbound is off or allowlisted. |

### Enabling it

| Variable | Default | Effect |
|---|---|---|
| `UNSLOTH_TENKI_SANDBOX` | off | `1` routes `auto`/`required` tool calls to Tenki. `full` and `software` never go remote. |
| `TENKI_API_KEY` / `TENKI_AUTH_TOKEN` | — | Workspace key (`tk_...`), read by the relay only. |
| `TENKI_API_ENDPOINT` | SDK default | Forwarded to the relay. |
| `UNSLOTH_TENKI_ALLOW_OUTBOUND` | on | `0` creates the VM with outbound access off. |
| `UNSLOTH_TENKI_ALLOW_DOMAINS` | — | Comma list for the session egress allowlist, e.g. `pypi.org,files.pythonhosted.org`. |
| `UNSLOTH_TENKI_MAX_SYNC_MB` | 256 | Cap on each direction of the copy. |
| `UNSLOTH_TENKI_MAX_SESSION_SECONDS` | 3600 | Ceiling on the VM's `max_duration`. |

`pip install tenki` goes into Studio's environment. With the variable unset,
`_tenki_launch` returns `None` before probing anything, and the local backends
run exactly as before; a test pins this. If Tenki is enabled but unavailable
(no SDK, no key), `required` refuses with `TENKI_SANDBOX_UNAVAILABLE` and
`auto` logs a warning and keeps local isolation.

## Gaps

**Workdir sync.** The workdir is copied, not mounted, so several things follow.
Deletions in the VM are not mirrored back, and neither are symlinks, hard links
or special files; the relay reports how many it skipped. A host file the tool
"edits" is replaced, never written through, which is the safe direction. Large
workdirs hit the cap and fail with a clear message. A real version wants a
manifest diff (hash before and after) instead of a full tar both ways, and
mirrored deletions limited to files that were uploaded.

**Model library and HF cache.** These are not available. The local backends
bind the model cache in because tool code may load a model; at tens of GB it
cannot be uploaded per call. Options, none built: a Tenki
[volume](https://tenki.cloud/docs/sandbox/volumes.md) holding a curated subset;
a [snapshot](https://tenki.cloud/docs/sandbox/snapshots.md) or template with
common packages and small models baked in; or letting the VM download from HF
over its own network, with the user's HF token as an explicit opt-in, never by
default. The VM has no GPU, which matches bubblewrap's `gpu_devices_hidden`.

**Cancellation and timeouts.** Studio cancels with `killpg(SIGKILL)`, so the
relay never gets to clean up. The server-side cleanup callback covers this:
`PreparedSandboxLaunch.cleanup()` runs in `tools.py`'s `finally` and terminates
the session named in the file. `max_duration` bounds the case where the server
also dies. A prompt cancel still waits for the HTTP terminate call, with a 15 s
client timeout, inside cleanup. That is synchronous on the tool thread today.

**Latency per tool call.** Not measured; no live key was used here. Every call
pays for VM create (the SDK waits up to 180 s by default), the upload, two tar
execs, the download and the terminate. That suits long-running or risky calls,
not a model that fires 20 quick `ls` calls per turn. The obvious next step is
one VM per chat session: a `sticky` session or a long `max_duration`, reused
across calls and paused when the chat goes idle, with the workdir synced
incrementally. That needs a session registry keyed like `tools._workdirs` and a
hook on chat deletion, so it is beyond a prototype.

**Offline use.** No network means no VM. `available()` does not probe the API
(so it adds no startup cost), so offline the call fails at launch with a relay
error as its output. It does not fall back to the host, by design: a remote
backend that silently became local would defeat the opt-in. An `auto`-mode
fallback to local isolation on connection errors would be easy to add if
maintainers want it.

**Credentials.** The workspace key is read from Studio's environment and passed
only in the relay's environment. It never goes into the spec file, the VM env
or the workdir, and tests check this. The relay runs on the host under the
same uid as Studio. Tool code never runs on the host in this mode, so nothing
there can read the relay's `/proc/<pid>/environ`. A Studio settings UI and
encrypted storage for the key would be needed before this is user-facing. VMs
spend the workspace's credits, so a per-chat or per-day budget should come
with any UI.

**Other gaps.**
- Python packages: `.unsloth-packages` is host-built and not synced, so
  `pip install` inside the VM lasts one call.
- User project folders and managed-account confinement are decided in
  `tools.py` before `os_sandbox` runs, so those calls keep their current path.
  Windows is excluded, because the relay maps POSIX shells only.
- The relay runs under `_sandbox_preexec`'s rlimits (`RLIMIT_NPROC`,
  `RLIMIT_AS`, `RLIMIT_FSIZE`). Whether the SDK's gRPC threads fit under them
  has not been checked on a live run. The relay may need its own, looser limit
  set.
- `-I` keeps the model-writable cwd off `sys.path` on Python 3.11 and later. On
  3.10, the script's directory (`core/inference`) is on the path, but none of its
  modules has a stdlib module name.

## Backend module vs. MCP server

Tenki also hosts an MCP server at `https://mcp.tenki.cloud/mcp`, and
`mcp_config_import.py` already accepts `streamableHttp` URLs with `oauth`.

| | Backend (`sandbox_tenki.py`) | MCP server |
|---|---|---|
| What it changes | Where the existing `python`/`terminal` tools run | Adds new tools (`tenki_run_code`, `tenki_exec`, …) beside them |
| Model behaviour | None; the same tools, transparently isolated | The model must choose the Tenki tools, and `python`/`terminal` stay local |
| Workdir and file sentinels | Synced back, so Studio's file cards keep working | No access; files stay in the VM unless the model reads them out |
| Auth | Workspace API key | OAuth; Tenki's docs say Claude Code is the only supported client today |
| Blast radius | One VM per call, created and destroyed by Studio | The model gets the whole workspace surface (terminate any sandbox, volumes, snapshots) and spends credits at will |
| Code in Studio | ~500 lines plus tests, in security-critical code | None |

**Recommendation: build the backend module.** The goal is stronger isolation
for the tools Studio already runs, and only a backend gives that. An MCP server
does not isolate `python`/`terminal`; it adds a second, unconstrained way to
run code. It also hands a local model control of a billing workspace, through
a client Tenki does not yet support. The MCP route makes sense later, as an
independent "cloud sandbox" integration, once Tenki confirms OAuth from
non-Claude-Code clients and offers a key scoped to a single session.

Before proposing the backend upstream:
1. Measure latency on a real workspace, and decide per-call VMs vs. a per-chat VM.
2. Run under `_sandbox_preexec`'s rlimits on Linux and macOS.
3. Add a settings toggle and key storage instead of environment variables.
4. Decide the model-cache story (volume, template, or none).
