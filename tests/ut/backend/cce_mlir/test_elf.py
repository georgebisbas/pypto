# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING, BUT NOT LIMITED TO, NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""ELF inspection tests (no cce-mlir dependency)."""

import pytest
from pypto.backend.cce_mlir.elf import ElfError, is_relocatable, symbol_names


def test_non_elf_bytes_are_rejected():
    for data in (b"", b"not an elf", b"\x7fEL"):
        with pytest.raises(ElfError):
            is_relocatable(data)
        with pytest.raises(ElfError):
            symbol_names(data)


def test_32_bit_or_big_endian_elf_is_rejected():
    header = bytearray(b"\x7fELF" + b"\x01" + b"\x01" + b"\x01" + b"\x00" * 57)
    with pytest.raises(ElfError):
        is_relocatable(bytes(header))
    header = bytearray(b"\x7fELF" + b"\x02" + b"\x02" + b"\x01" + b"\x00" * 57)
    with pytest.raises(ElfError):
        is_relocatable(bytes(header))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
