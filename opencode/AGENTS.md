# Remote GPU runner

When CUDA-kernel correctness or performance needs to be checked, you may submit a
self-contained Python file to the shared remote GPU service with `curl`.

- Base URL: `http://109.236.57.62:18080`
- Authentication: read `CUTE_HARNESS_API_KEY` from the environment. Never print,
  embed, or commit the key.
- A submission should define `main()` and call it from
  `if __name__ == "__main__":`.

## Submission template

Save a self-contained submission such as the following as `submission.py`:

```python
import torch


def main() -> None:
    torch.manual_seed(0)
    left = torch.randn((1024, 1024), device="cuda")
    right = torch.randn((1024, 1024), device="cuda")
    result = left @ right
    torch.cuda.synchronize()
    print(f"result={result[0, 0].item():.6f}")


if __name__ == "__main__":
    main()
```

Submit and profile the file:

```bash
curl -sS 'http://109.236.57.62:18080/v1/runs/file' \
  -H "X-API-Key: ${CUTE_HARNESS_API_KEY}" \
  -F 'file=@path/to/submission.py' \
  -F 'profiler=pytorch'
```

The JSON response includes correctness/process information and performance fields
such as `success`, `exit_code`, `stdout`, `stderr`, `device_time_ms`,
`profile_id`, `profile_error`, and `timed_out`.

Download a profile when useful:

```bash
curl -fOJ 'http://109.236.57.62:18080/v1/profiles/<profile_id>' \
  -H "X-API-Key: ${CUTE_HARNESS_API_KEY}"
```

Establish correctness first, then optimize and remeasure promising versions.
Current `device_time_ms` comes from PyTorch Profiler without a controlled warmup,
so treat a single result as directional and repeat final candidates. Keep use of
the shared B300 modest and do not start an uncontrolled optimization loop.
