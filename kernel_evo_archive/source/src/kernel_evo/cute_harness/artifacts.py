"""Extract inspectable artifacts from compiled Python CuTe DSL executors."""

from __future__ import annotations

import io
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any


def _decode_mlir_escaped_bytes(payload: bytes) -> bytes:
    converted = bytearray()
    index = 0
    hexdigits = b"0123456789abcdefABCDEF"
    while index < len(payload):
        if payload[index] == 0x5C and index + 2 < len(payload):
            first, second = payload[index + 1], payload[index + 2]
            if first in hexdigits and second in hexdigits:
                converted.extend(bytes.fromhex(payload[index + 1 : index + 3].decode("ascii")))
                index += 3
                continue
            if first == 0x5C:
                converted.append(0x5C)
                index += 2
                continue
        converted.append(payload[index])
        index += 1
    return bytes(converted)


def _embedded_cubins(compiled: Any) -> list[bytes]:
    module = getattr(compiled, "ir_module", None)
    if module is None:
        return []
    cubins: list[bytes] = []

    def visit(operation):
        if operation.name == "gpu.binary":
            stream = io.BytesIO()
            operation.write_bytecode(stream)
            bytecode = stream.getvalue()
            marker = b'bin = "'
            if marker in bytecode:
                payload = bytecode.split(marker, 1)[1].split(b'">', 1)[0]
                cubins.append(_decode_mlir_escaped_bytes(payload))
        from cutlass._mlir import ir

        return ir.WalkResult.ADVANCE

    module.operation.walk(visit)
    return cubins


def _write_optional(path: Path, value: Any, *, binary: bool = False) -> bool:
    if value is None:
        return False
    if binary:
        if not isinstance(value, (bytes, bytearray)):
            return False
        path.write_bytes(bytes(value))
    else:
        path.write_text(str(value), encoding="utf-8")
    return True


def dump_compiled_artifacts(
    compiled: Any,
    output_dir: str | Path,
    *,
    prefix: str = "kernel",
    disassemble: bool = True,
) -> dict[str, Any]:
    """Dump artifacts across old (4.2) and newer CuTe DSL executor interfaces."""
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    files: list[Path] = []

    if _write_optional(destination / f"{prefix}.ptx", getattr(compiled, "__ptx__", None)):
        files.append(destination / f"{prefix}.ptx")
    if _write_optional(destination / f"{prefix}.mlir", getattr(compiled, "__mlir__", None)):
        files.append(destination / f"{prefix}.mlir")

    cubin_value = getattr(compiled, "__cubin__", None)
    cubins = (
        [bytes(cubin_value)]
        if isinstance(cubin_value, (bytes, bytearray))
        else _embedded_cubins(compiled)
    )
    for index, cubin in enumerate(cubins):
        suffix = "" if len(cubins) == 1 else f"_{index}"
        cubin_path = destination / f"{prefix}{suffix}.cubin"
        cubin_path.write_bytes(cubin)
        files.append(cubin_path)

        if disassemble and (nvdisasm := shutil.which("nvdisasm")):
            completed = subprocess.run(
                [nvdisasm, str(cubin_path)],
                text=True,
                capture_output=True,
                timeout=60,
                check=False,
            )
            if completed.returncode == 0:
                sass_path = cubin_path.with_suffix(".sass")
                sass_path.write_text(completed.stdout, encoding="utf-8")
                files.append(sass_path)
        if cuobjdump := shutil.which("cuobjdump"):
            completed = subprocess.run(
                [cuobjdump, "--dump-resource-usage", str(cubin_path)],
                text=True,
                capture_output=True,
                timeout=30,
                check=False,
            )
            resource_path = cubin_path.with_suffix(".resources.txt")
            resource_path.write_text(completed.stdout or completed.stderr, encoding="utf-8")
            files.append(resource_path)

    manifest = {
        "dialect": "cute_dsl_python",
        "executor_type": f"{type(compiled).__module__}.{type(compiled).__qualname__}",
        "files": [str(path) for path in files],
        "cubin_count": len(cubins),
    }
    manifest_path = destination / f"{prefix}.artifacts.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    manifest["manifest"] = str(manifest_path)
    return manifest

