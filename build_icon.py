#!/usr/bin/env python3
"""Render the Flow Dictate app icon (🎙 on a dark gradient squircle) as AppIcon.icns.

Usage: .venv/bin/python build_icon.py [output-dir]
Requires macOS (AppKit + iconutil).
"""

import pathlib
import subprocess
import sys
import tempfile

import AppKit
from Foundation import NSMakeRect

SIZES = [16, 32, 128, 256, 512]  # each also gets an @2x variant


def draw_icon(px: int) -> bytes:
    img = AppKit.NSImage.alloc().initWithSize_((px, px))
    img.lockFocus()

    inset = px * 0.08  # macOS icons leave a margin around the squircle
    rect = NSMakeRect(inset, inset, px - 2 * inset, px - 2 * inset)
    radius = (px - 2 * inset) * 0.22
    path = AppKit.NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(rect, radius, radius)

    gradient = AppKit.NSGradient.alloc().initWithStartingColor_endingColor_(
        AppKit.NSColor.colorWithCalibratedRed_green_blue_alpha_(0.13, 0.14, 0.18, 1.0),
        AppKit.NSColor.colorWithCalibratedRed_green_blue_alpha_(0.24, 0.20, 0.35, 1.0),
    )
    gradient.drawInBezierPath_angle_(path, 90.0)

    text = "🎙"
    font = AppKit.NSFont.systemFontOfSize_(px * 0.52)
    attrs = {AppKit.NSFontAttributeName: font}
    s = AppKit.NSAttributedString.alloc().initWithString_attributes_(text, attrs)
    size = s.size()
    s.drawAtPoint_(((px - size.width) / 2, (px - size.height) / 2))

    img.unlockFocus()

    tiff = img.TIFFRepresentation()
    rep = AppKit.NSBitmapImageRep.imageRepWithData_(tiff)
    rep.setSize_((px, px))
    return rep.representationUsingType_properties_(AppKit.NSBitmapImageFileTypePNG, None)


def main():
    out_dir = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path(__file__).parent
    with tempfile.TemporaryDirectory() as tmp:
        iconset = pathlib.Path(tmp) / "AppIcon.iconset"
        iconset.mkdir()
        for size in SIZES:
            for scale, suffix in ((1, ""), (2, "@2x")):
                png = draw_icon(size * scale)
                png.writeToFile_atomically_(
                    str(iconset / f"icon_{size}x{size}{suffix}.png"), True
                )
        target = out_dir / "AppIcon.icns"
        subprocess.run(["iconutil", "-c", "icns", str(iconset), "-o", str(target)], check=True)
        print(f"wrote {target}")


if __name__ == "__main__":
    main()
