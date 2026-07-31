# B300 HTTP Verifier

This directory is a standalone reference implementation of the HTTP service
used to evaluate kernel submissions on a B300 GPU. It is intentionally
separate from the repository's top-level `cute_harness` package: the top-level
package is the evaluator client, while this project is the remote execution
service.

An authenticated HTTP service that executes one complete PyTorch/CuTe Python
file, captures its stdout/stderr, measures CUDA device time with
`torch.profiler`, and optionally stores a PyTorch or Nsight Systems report.
Debug runs reuse a long-lived Python/CUDA worker by default, avoiding repeated
interpreter, import, and compiler startup.

## API

### Run the service locally

Set a non-default API key and start the service:

```bash
export CUTE_HARNESS_API_KEY="$(openssl rand -hex 32)"
python3 -m pip install -e '.[test]'
python3 -m cute_harness.app
```

Run source:

```bash
curl -sS http://localhost:18080/v1/runs \
  -H "X-API-Key: $CUTE_HARNESS_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "filename": "submission.py",
    "profiler": "pytorch",
    "iterations": 3,
    "exclusive": false,
    "code": "import torch\nif __name__ == \"__main__\":\n    x = torch.ones(1024, device=\"cuda\")\n    print(x.sum())"
  }'
```

Or upload the Python file directly:

```bash
curl -sS http://localhost:18080/v1/runs/file \
  -H "X-API-Key: $CUTE_HARNESS_API_KEY" \
  -F "file=@examples/submission.py" \
  -F "profiler=pytorch" \
  -F "iterations=3" \
  -F "exclusive=false"
```

`iterations` is 1 by default and may be set from 1 through 100. Each request
gets one unmeasured warmup, followed by the requested measured iterations.
`device_times_ms` contains every measured CUDA/device time; the backward-
compatible `device_time_ms` is their median.

`exclusive=false` is the default fast-debug path. Requests are serialized and
reuse one long-lived Python/CUDA worker, so imported modules, CUDA context, and
compiler caches stay warm across submissions. Set `exclusive=true` for a
benchmark: the shared worker is stopped first and the request runs in a fresh,
isolated process. The next debug request starts a new warm worker.

`profiler` may be omitted, `"pytorch"`, or `"nsys"`. Every successful primary
run uses `torch.profiler`; each timing is the sum of self CUDA/device times
reported by its key averages. A requested PyTorch trace captures the final
measured iteration. Nsight Systems requires `nsys` on `PATH` and executes the
source a second time because PyTorch profiler and Nsight cannot safely own
CUPTI concurrently.

Download a report:

```bash
curl -OJ http://localhost:18080/v1/profiles/PROFILE_ID \
  -H "X-API-Key: $CUTE_HARNESS_API_KEY"
```

Configuration uses the `CUTE_HARNESS_*` environment variables defined in
`cute_harness/config.py`. Runs are serialized to one GPU slot and default to a
300-second timeout. The timeout applies to the full warmup-plus-iterations
request.

When operating the service remotely, keep it behind an authenticated private
network or SSH tunnel. Do not send source code or API credentials over a
public, unencrypted HTTP connection.

## Container deployment

The production launcher runs the service in a read-only, non-root container
with all Linux capabilities dropped, no outbound network, one explicitly
mapped GPU, bounded memory/PIDs/logs, and Docker-managed volumes for profiles
and compilation caches:

```bash
docker build -t cute-harness:latest .
cp deploy/service.env.example deploy/service.env
# Edit deploy/service.env and set a unique CUTE_HARNESS_API_KEY.
./deploy/docker-run.sh
```

The host must provide Docker and NVIDIA GPU container support. On hosts without
NVIDIA Container Toolkit, the launcher explicitly maps the NVIDIA device nodes
and mounts only `/usr/lib/x86_64-linux-gnu` read-only for driver libraries. It
does not mount a home directory, project directory, or Docker socket.

## Security

The AST policy rejects direct subprocess, shell, process-spawn, dynamic-code,
`ctypes`, and common Python object-graph escape APIs while allowing `os.environ`.
The child receives a reduced environment, an isolated temporary `HOME`, and
never receives the API key. Logs are returned in full up to the configurable
10 MiB safety limit.

Static source inspection is **not a security boundary** for hostile Python.
Before publishing this service, run each job in a fresh container/VM with:

- no host mounts or Docker socket;
- an unprivileged user, read-only root filesystem, and a small writable tmpfs;
- disabled network access;
- PID, memory, CPU, disk, and wall-clock limits;
- seccomp/AppArmor restrictions appropriate for CUDA;
- a separate artifact volume controlled by the API process.

The included Dockerfile is packaging, not a per-request sandbox.
