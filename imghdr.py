"""Compatibility shim for the removed ``imghdr`` stdlib module.

Python 3.13 removed `imghdr` (see PEP 695).  Streamlit still imports it and
blows up when it's missing.  Provide the minimal API it uses so the app can
start.  The implementation prefers Pillow if it's available, otherwise it
falls back to a few simple magic-byte checks.

The real ``imghdr`` was very small; Streamlit only ever calls ``what`` with a
filename, so we only implement that variant.
"""

from __future__ import annotations

import os
from typing import Optional

try:
    from PIL import Image
except ImportError:  # if pillow isn't installed we still try header sniffing
    Image = None


def what(filename: str, h: bytes | None = None) -> Optional[str]:
    """Return a string describing the image type, or ``None`` if unknown.

    ``filename`` may be a path or a file-like object; only the filename
    variant is used by Streamlit.  ``h`` is optional header bytes.
    """

    # read header bytes if not supplied
    if h is None:
        try:
            with open(filename, "rb") as f:
                h = f.read(32)
        except OSError:
            return None

    # prefer Pillow when available; it handles a wide range of formats
    if Image is not None:
        try:
            if hasattr(filename, "read"):
                img = Image.open(filename)
            else:
                img = Image.open(filename)
            fmt = img.format
            if fmt is not None:
                return fmt.lower()
        except Exception:  # fall through to simple sniffing
            pass

    # simple header checks borrowed from original imghdr
    if h.startswith(b"\xff\xd8"):
        return "jpeg"
    if h.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if h[:6] in (b"GIF87a", b"GIF89a"):
        return "gif"
    if h[:2] == b"BM":
        return "bmp"
    # WebP, TIFF, etc. could be added if needed.
    return None
