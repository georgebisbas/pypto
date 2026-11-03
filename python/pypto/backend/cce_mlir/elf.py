# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING, BUT NOT LIMITED TO, NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Minimal little-endian ELF64 inspection for generated device objects.

The route needs to know, without shelling out, whether cce-mlir produced a
relocatable object and whether it defines the expected device symbol. Only the
few header and symbol-table fields needed for that answer are parsed.
"""

from __future__ import annotations

import struct

__all__ = [
    "ElfError",
    "entry_point",
    "symbol_names",
    "symbol_values",
    "is_relocatable",
]

_ELF_MAGIC = b"\x7fELF"
_ET_REL = 1
_SHT_SYMTAB = 2
_SHT_DYNSYM = 11


class ElfError(ValueError):
    """The bytes are not a little-endian ELF64 object."""


def _parse_header(data: bytes) -> tuple[int, int, int, int, int]:
    if len(data) < 64 or data[:4] != _ELF_MAGIC:
        raise ElfError("missing ELF64 magic")
    if data[4] != 2 or data[5] != 1:
        raise ElfError("expected a little-endian 64-bit ELF")
    e_type, e_machine = struct.unpack_from("<HH", data, 16)
    e_shoff = struct.unpack_from("<Q", data, 0x28)[0]
    e_shentsize, e_shnum, e_shstrndx = struct.unpack_from("<HHH", data, 0x3A)
    if e_shentsize < 64 or e_shoff + e_shentsize * e_shnum > len(data):
        raise ElfError("truncated or malformed section table")
    return e_type, e_machine, e_shoff, e_shentsize, e_shnum


def _section(data: bytes, offset: int) -> tuple[int, int, int, int, int]:
    sh_name, sh_type = struct.unpack_from("<II", data, offset)
    sh_offset = struct.unpack_from("<Q", data, offset + 0x18)[0]
    sh_size = struct.unpack_from("<Q", data, offset + 0x20)[0]
    sh_link = struct.unpack_from("<I", data, offset + 0x28)[0]
    return sh_name, sh_type, sh_offset, sh_size, sh_link


def is_relocatable(data: bytes) -> bool:
    """Whether the object is an ``ET_REL`` ELF.

    Args:
        data: Raw ELF bytes.

    Returns:
        ``True`` for a relocatable object.

    Raises:
        ElfError: When the bytes are not a supported ELF.
    """
    return _parse_header(data)[0] == _ET_REL


def entry_point(data: bytes) -> int:
    """Return the ELF entry address (``e_entry``)."""
    if len(data) < 32 or data[:4] != _ELF_MAGIC:
        raise ElfError("missing ELF64 magic")
    return struct.unpack_from("<Q", data, 0x18)[0]


def symbol_values(data: bytes) -> dict[str, int]:
    """Return each symbol's ``st_value`` (section-relative in an ``ET_REL``).

    Args:
        data: Raw ELF bytes.

    Returns:
        Symbol name -> ``st_value``. Later duplicate names overwrite earlier
        ones, matching the symbol table's own visibility order.

    Raises:
        ElfError: When the bytes are not a supported ELF or the tables are
            truncated.
    """
    _, _, shoff, shentsize, shnum = _parse_header(data)
    values: dict[str, int] = {}
    entry_size = 24
    for index in range(shnum):
        _, sh_type, offset, size, link = _section(data, shoff + index * shentsize)
        if sh_type not in (_SHT_SYMTAB, _SHT_DYNSYM):
            continue
        _, _, str_offset, str_size, _ = _section(data, shoff + link * shentsize)
        if offset + size > len(data) or str_offset + str_size > len(data):
            raise ElfError("truncated symbol or string table")
        for entry in range(size // entry_size):
            base = offset + entry * entry_size
            st_name, st_info = struct.unpack_from("<IB", data, base)
            st_value = struct.unpack_from("<Q", data, base + 8)[0]
            if st_name == 0:
                continue
            start = str_offset + st_name
            end = data.find(b"\x00", start, str_offset + str_size)
            name = data[start : end if end >= 0 else str_offset + str_size].decode("ascii", "replace")
            values[name] = st_value
    return values


def symbol_names(data: bytes) -> tuple[str, ...]:
    """Return every defined and undefined symbol name in the ELF.

    Args:
        data: Raw ELF bytes.

    Returns:
        Symbol names in symbol-table order.

    Raises:
        ElfError: When the bytes are not a supported ELF or the tables are
            truncated.
    """
    _, _, shoff, shentsize, shnum = _parse_header(data)
    names: list[str] = []
    for index in range(shnum):
        _, sh_type, offset, size, link = _section(data, shoff + index * shentsize)
        if sh_type not in (_SHT_SYMTAB, _SHT_DYNSYM):
            continue
        _, _, str_offset, str_size, _ = _section(data, shoff + link * shentsize)
        if offset + size > len(data) or str_offset + str_size > len(data):
            raise ElfError("truncated symbol or string table")
        entry_size = 24
        for entry in range(size // entry_size):
            st_name = struct.unpack_from("<I", data, offset + entry * entry_size)[0]
            if st_name == 0:
                continue
            start = str_offset + st_name
            end = data.find(b"\x00", start, str_offset + str_size)
            names.append(data[start : end if end >= 0 else str_offset + str_size].decode("ascii", "replace"))
    return tuple(names)
