"""Audit the cce-mlir artifact against PyPTO's kernel ABI."""

from __future__ import annotations

import re
from pathlib import Path

_KERNEL_NAME = re.compile(r'@[^\s]+_name\("([^"]+)\\00"\)')
_ACL_APIS = (
    "aclrtBinaryLoadFromData",
    "aclrtBinaryGetFunction",
    "aclrtKernelArgsAppend",
    "aclrtLaunchKernelWithConfig",
)


def audit(path: str | Path) -> dict[str, object]:
    text = Path(path).read_text()
    symbols = _KERNEL_NAME.findall(text)
    apis = [api for api in _ACL_APIS if f'"{api}\\00"' in text]
    entry_present = "kernel_entry" in text
    return {
        "artifact": str(path),
        "cce_device_symbols": symbols,
        "cce_acl_launcher_apis": apis,
        "pypto_required_entry": "kernel_entry(__gm__ int64_t*)",
        "pypto_entry_present": entry_present,
        "compatible_with_pypto_external_source": bool(entry_present and not apis),
        "status": "compatible" if entry_present and not apis else "adapter_required",
    }
