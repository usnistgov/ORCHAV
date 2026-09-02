"""Generate ORCHAV branding derivatives from the canonical SVG artwork.

Editable SVG sources live beneath ``visualizer/resources/branding``. This
script preserves those sources, generates the dark-background wordmark and
platform derivatives beside them, and writes review-only contact sheets to
``.artifacts/branding-review``.
"""

from __future__ import annotations

import argparse
import re
import shutil
import struct
import xml.etree.ElementTree as ET
from io import BytesIO
from pathlib import Path

from PIL import IcnsImagePlugin, IcoImagePlugin, Image, ImageChops, ImageDraw, ImageFont
from PySide6.QtCore import QRectF
from PySide6.QtGui import QImage, QPainter
from PySide6.QtSvg import QSvgRenderer

ICON_SIZES = (
    16,
    20,
    22,
    24,
    30,
    32,
    36,
    40,
    48,
    60,
    64,
    72,
    80,
    96,
    128,
    180,
    192,
    256,
    512,
    1024,
)
WINDOWS_ICON_SIZES = (16, 20, 24, 30, 32, 36, 40, 48, 60, 64, 72, 80, 96, 128, 256)
LINUX_ICON_SIZES = (16, 22, 24, 32, 48, 64, 128, 256, 512)

LIGHT_WORDMARK_PRIMARY = "#011A46"
DARK_WORDMARK_PRIMARY = "#F3F6FB"
LIGHT_WORDMARK_NODE_SHELL = "#00163B"
DARK_WORDMARK_NODE_SHELL = "#CBD8EA"


def _update_docname(svg: str, filename: str) -> str:
    return re.sub(r'sodipodi:docname="[^"]+"', f'sodipodi:docname="{filename}"', svg)


def _wordmark_variant(source: str, filename: str, dark: bool) -> str:
    result = _update_docname(source, filename)
    background = "dark" if dark else "light"
    result = re.sub(
        r'aria-label="ORCHAV logo[^"]*"',
        f'aria-label="ORCHAV logo for {background} backgrounds"',
        result,
    )
    if not dark:
        return result

    result = re.sub(
        re.escape(LIGHT_WORDMARK_PRIMARY),
        DARK_WORDMARK_PRIMARY,
        result,
        flags=re.IGNORECASE,
    )
    return re.sub(
        re.escape(LIGHT_WORDMARK_NODE_SHELL),
        DARK_WORDMARK_NODE_SHELL,
        result,
        flags=re.IGNORECASE,
    )


def _render_svg(
    svg_path: Path,
    output_path: Path,
    width: int,
    height: int,
    content_scale: float = 1.0,
) -> None:
    renderer = QSvgRenderer(str(svg_path))
    if not renderer.isValid():
        raise ValueError(f"Qt could not load SVG: {svg_path}")

    image = QImage(width, height, QImage.Format_RGBA8888)
    image.fill(0)
    painter = QPainter(image)
    painter.setRenderHint(QPainter.Antialiasing, True)
    painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
    rendered_width = width * content_scale
    rendered_height = height * content_scale
    renderer.render(
        painter,
        QRectF(
            (width - rendered_width) / 2,
            (height - rendered_height) / 2,
            rendered_width,
            rendered_height,
        ),
    )
    painter.end()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not image.save(str(output_path), "PNG"):
        raise OSError(f"Could not write PNG: {output_path}")


def _save_ico(path: Path, images: dict[int, Image.Image]) -> None:
    sizes = [(size, size) for size in WINDOWS_ICON_SIZES]
    base = images[max(WINDOWS_ICON_SIZES)]
    appended = [images[size] for size in WINDOWS_ICON_SIZES if size != base.width]
    path.parent.mkdir(parents=True, exist_ok=True)
    base.save(path, format="ICO", sizes=sizes, append_images=appended)


def _png_bytes(image: Image.Image) -> bytes:
    stream = BytesIO()
    image.save(stream, format="PNG")
    return stream.getvalue()


def _save_icns(path: Path, images: dict[int, Image.Image]) -> None:
    """Write a modern ICNS with 1x and Retina PNG resources."""
    resources = (
        (b"icp4", 16),
        (b"icp5", 32),
        (b"icp6", 64),
        (b"ic07", 128),
        (b"ic08", 256),
        (b"ic09", 512),
        (b"ic10", 1024),
        (b"ic11", 32),
        (b"ic12", 64),
        (b"ic13", 256),
        (b"ic14", 512),
    )
    payloads = [(code, _png_bytes(images[size])) for code, size in resources]
    chunks = [code + struct.pack(">I", len(data) + 8) + data for code, data in payloads]
    toc_data = b"".join(code + struct.pack(">I", len(data) + 8) for code, data in payloads)
    toc = b"TOC " + struct.pack(">I", len(toc_data) + 8) + toc_data
    body = toc + b"".join(chunks)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"icns" + struct.pack(">I", len(body) + 8) + body)


def _load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    names = (
        "C:/Windows/Fonts/seguisb.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
        (
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
            if bold
            else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        ),
    )
    for name in names:
        if Path(name).exists():
            return ImageFont.truetype(name, size)
    return ImageFont.load_default()


def _checkerboard(size: tuple[int, int], cell: int = 16) -> Image.Image:
    image = Image.new("RGB", size, "#F5F7FA")
    draw = ImageDraw.Draw(image)
    for y in range(0, size[1], cell):
        for x in range(0, size[0], cell):
            if (x // cell + y // cell) % 2:
                draw.rectangle((x, y, x + cell - 1, y + cell - 1), fill="#D9DEE7")
    return image


def _create_wordmark_review(
    path: Path, light_wordmark: Image.Image, dark_wordmark: Image.Image
) -> None:
    width, height = 1600, 760
    panel_height = height // 2
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, width, panel_height), fill="#F6F8FA")
    draw.rectangle((0, panel_height, width, height), fill="#0D1117")

    target_width = 1400
    target_height = round(light_wordmark.height * target_width / light_wordmark.width)
    for row, image in enumerate((light_wordmark, dark_wordmark)):
        rendered = image.resize((target_width, target_height), Image.Resampling.LANCZOS)
        x = (width - target_width) // 2
        y = row * panel_height + (panel_height - target_height) // 2
        canvas.paste(rendered, (x, y), rendered)

    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def _create_icon_size_review(path: Path, images: dict[int, Image.Image]) -> None:
    sizes = (16, 20, 24, 32, 40, 48, 64, 128, 256)
    columns = 3
    cell_width, cell_height = 550, 300
    margin, header = 45, 85
    rows = (len(sizes) + columns - 1) // columns
    canvas = Image.new(
        "RGB",
        (margin * 2 + columns * cell_width, header + margin + rows * cell_height),
        "#F6F8FA",
    )
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (margin, 24),
        "ORCHAV exported application icons (nearest-neighbor enlargement)",
        fill="#011A46",
        font=_load_font(30, bold=True),
    )
    label_font = _load_font(23, bold=True)

    for index, size in enumerate(sizes):
        row, column = divmod(index, columns)
        left = margin + column * cell_width
        top = header + row * cell_height
        background = _checkerboard((260, 260), 20)
        enlarged = images[size].resize((220, 220), Image.Resampling.NEAREST)
        background.paste(enlarged, (20, 20), enlarged)
        canvas.paste(background, (left, top))
        draw.text(
            (left + 285, top + 14),
            f"{size} px — {'optical-size' if size <= 48 else 'full master'}",
            fill="#011A46",
            font=label_font,
        )
        native_x = left + 285 + (250 - size) // 2
        native_y = top + 42 + (256 - size) // 2
        canvas.paste(images[size], (native_x, native_y), images[size])

    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def _create_symbolic_review(path: Path, symbolic: Image.Image) -> None:
    canvas = Image.new("RGB", (1280, 560), "white")
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((640, 0, 1280, 560), fill="#0D1117")
    dark = symbolic.resize((400, 400), Image.Resampling.LANCZOS)
    light = Image.new("RGBA", dark.size, (243, 246, 251, 0))
    alpha = dark.getchannel("A")
    light.putalpha(alpha)
    canvas.paste(dark, (120, 80), dark)
    canvas.paste(light, (760, 80), light)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def _create_social_preview(path: Path, app_icon: Image.Image, dark_wordmark: Image.Image) -> None:
    width, height = 1280, 640
    canvas = Image.new("RGB", (width, height))
    draw = ImageDraw.Draw(canvas)
    start = (1, 26, 70)
    end = (10, 67, 138)
    for x in range(width):
        ratio = x / (width - 1)
        color = tuple(round(a + (b - a) * ratio) for a, b in zip(start, end))
        draw.line((x, 0, x, height), fill=color)

    icon = app_icon.resize((330, 330), Image.Resampling.LANCZOS)
    wordmark_width = 720
    wordmark_height = round(dark_wordmark.height * wordmark_width / dark_wordmark.width)
    wordmark = dark_wordmark.resize((wordmark_width, wordmark_height), Image.Resampling.LANCZOS)
    canvas.paste(icon, (90, (height - icon.height) // 2), icon)
    canvas.paste(
        wordmark,
        (470, (height - wordmark.height) // 2),
        wordmark,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def _validate_assets(output_dir: Path, source_wordmark: Path, review_dir: Path) -> None:
    svg_paths = sorted(output_dir.rglob("*.svg"))
    for svg_path in svg_paths:
        ET.parse(svg_path)
        if not QSvgRenderer(str(svg_path)).isValid():
            raise ValueError(f"Qt could not validate SVG: {svg_path}")

    png_dir = output_dir / "png"
    for size in ICON_SIZES:
        png_path = png_dir / f"orchav-app-icon-{size}.png"
        with Image.open(png_path) as image:
            rgba = image.convert("RGBA")
            if rgba.size != (size, size):
                raise ValueError(f"Unexpected dimensions for {png_path}: {rgba.size}")
            if rgba.getchannel("A").getextrema() != (0, 255):
                raise ValueError(f"Expected full alpha range in {png_path}")
            if any(
                rgba.getpixel(point)[3]
                for point in ((0, 0), (size - 1, 0), (0, size - 1), (size - 1, size - 1))
            ):
                raise ValueError(f"Expected transparent corners in {png_path}")

    light_path = png_dir / "orchav-wordmark-on-light.png"
    dark_path = png_dir / "orchav-wordmark-on-dark.png"
    with Image.open(light_path) as image:
        light_wordmark = image.convert("RGBA")
    with Image.open(dark_path) as image:
        dark_wordmark = image.convert("RGBA")
    if ImageChops.difference(
        light_wordmark.getchannel("A"), dark_wordmark.getchannel("A")
    ).getbbox():
        raise ValueError("Light and dark wordmarks do not share the same silhouette")

    source_render = review_dir / "source-wordmark-validation.png"
    _render_svg(source_wordmark, source_render, 1415, 285)
    try:
        with Image.open(source_render) as source_image:
            if ImageChops.difference(source_image.convert("RGBA"), light_wordmark).getbbox():
                raise ValueError("On-light wordmark differs visually from its source")
    finally:
        source_render.unlink(missing_ok=True)

    ico_path = output_dir / "windows" / "orchav.ico"
    with ico_path.open("rb") as stream:
        ico_sizes = {
            width for width, height in IcoImagePlugin.IcoFile(stream).sizes() if width == height
        }
    if ico_sizes != set(WINDOWS_ICON_SIZES):
        raise ValueError(f"Unexpected ICO sizes: {sorted(ico_sizes)}")

    icns_path = output_dir / "macos" / "orchav.icns"
    with icns_path.open("rb") as stream:
        icns = IcnsImagePlugin.IcnsFile(stream)
        icns_sizes = set(icns.itersizes())
        for size in icns_sizes:
            if icns.getimage(size).convert("RGBA").getchannel("A").getextrema() != (0, 255):
                raise ValueError(f"Expected alpha in ICNS representation {size}")
    expected_icns_sizes = {
        (16, 16, 1),
        (16, 16, 2),
        (32, 32, 1),
        (32, 32, 2),
        (64, 64, 1),
        (128, 128, 1),
        (128, 128, 2),
        (256, 256, 1),
        (256, 256, 2),
        (512, 512, 1),
        (512, 512, 2),
    }
    if icns_sizes != expected_icns_sizes:
        raise ValueError(f"Unexpected ICNS sizes: {sorted(icns_sizes)}")

    with Image.open(output_dir / "social" / "orchav-github-social-preview.png") as image:
        if image.size != (1280, 640):
            raise ValueError(f"Unexpected social preview size: {image.size}")


def generate(repo_root: Path) -> tuple[Path, Path]:
    output_dir = repo_root / "visualizer" / "resources" / "branding"
    review_dir = repo_root / ".artifacts" / "branding-review"

    review_dir.mkdir(parents=True, exist_ok=True)

    wordmark_source_path = output_dir / "orchav-wordmark-on-light.svg"
    icon_svg_path = output_dir / "orchav-app-icon.svg"
    small_svg_path = output_dir / "orchav-app-icon-small.svg"
    symbolic_svg_path = output_dir / "orchav-app-icon-symbolic.svg"
    canonical_sources = (
        wordmark_source_path,
        icon_svg_path,
        small_svg_path,
        symbolic_svg_path,
    )
    missing_sources = [path for path in canonical_sources if not path.is_file()]
    if missing_sources:
        missing = ", ".join(str(path) for path in missing_sources)
        raise FileNotFoundError(f"Missing canonical branding source: {missing}")

    wordmark_source = wordmark_source_path.read_text(encoding="utf-8")
    dark_name = "orchav-wordmark-on-dark.svg"
    light_svg_path = wordmark_source_path
    dark_svg_path = output_dir / dark_name
    dark_svg_path.write_text(
        _wordmark_variant(wordmark_source, dark_name, dark=True),
        encoding="utf-8",
    )

    png_dir = output_dir / "png"
    light_png_path = png_dir / "orchav-wordmark-on-light.png"
    dark_png_path = png_dir / "orchav-wordmark-on-dark.png"
    _render_svg(light_svg_path, light_png_path, 1415, 285)
    _render_svg(dark_svg_path, dark_png_path, 1415, 285)

    icon_images: dict[int, Image.Image] = {}
    for size in ICON_SIZES:
        raster_source = small_svg_path if size <= 48 else icon_svg_path
        raster_path = png_dir / f"orchav-app-icon-{size}.png"
        content_scale = 1.0 if size <= 48 else 0.96
        _render_svg(
            raster_source,
            raster_path,
            size,
            size,
            content_scale=content_scale,
        )
        with Image.open(raster_path) as image:
            icon_images[size] = image.convert("RGBA")

    _save_ico(output_dir / "windows" / "orchav.ico", icon_images)
    _save_icns(output_dir / "macos" / "orchav.icns", icon_images)

    iconset_dir = output_dir / "macos" / "orchav.iconset"
    iconset_files = {
        "icon_16x16.png": 16,
        "icon_16x16@2x.png": 32,
        "icon_32x32.png": 32,
        "icon_32x32@2x.png": 64,
        "icon_128x128.png": 128,
        "icon_128x128@2x.png": 256,
        "icon_256x256.png": 256,
        "icon_256x256@2x.png": 512,
        "icon_512x512.png": 512,
        "icon_512x512@2x.png": 1024,
    }
    iconset_dir.mkdir(parents=True, exist_ok=True)
    for filename, size in iconset_files.items():
        shutil.copyfile(
            png_dir / f"orchav-app-icon-{size}.png",
            iconset_dir / filename,
        )

    linux_dir = output_dir / "linux" / "hicolor"
    for size in LINUX_ICON_SIZES:
        destination = linux_dir / f"{size}x{size}" / "apps" / "orchav.png"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(png_dir / f"orchav-app-icon-{size}.png", destination)
    scalable_dir = linux_dir / "scalable" / "apps"
    symbolic_dir = linux_dir / "symbolic" / "apps"
    scalable_dir.mkdir(parents=True, exist_ok=True)
    symbolic_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(icon_svg_path, scalable_dir / "orchav.svg")
    shutil.copyfile(symbolic_svg_path, symbolic_dir / "orchav-symbolic.svg")

    web_dir = output_dir / "web"
    web_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(small_svg_path, web_dir / "favicon.svg")
    shutil.copyfile(
        png_dir / "orchav-app-icon-16.png",
        web_dir / "favicon-16.png",
    )
    shutil.copyfile(
        png_dir / "orchav-app-icon-32.png",
        web_dir / "favicon-32.png",
    )
    shutil.copyfile(output_dir / "windows" / "orchav.ico", web_dir / "favicon.ico")
    shutil.copyfile(
        png_dir / "orchav-app-icon-192.png",
        web_dir / "orchav-web-app-icon-192.png",
    )
    shutil.copyfile(
        png_dir / "orchav-app-icon-512.png",
        web_dir / "orchav-web-app-icon-512.png",
    )
    apple_touch_icon = Image.new("RGB", (180, 180), "#0C4C9B")
    apple_touch_icon.paste(icon_images[180], (0, 0), icon_images[180])
    apple_touch_icon.save(web_dir / "apple-touch-icon.png")

    symbolic_png_path = review_dir / "symbolic.png"
    _render_svg(symbolic_svg_path, symbolic_png_path, 512, 512)
    with Image.open(light_png_path) as image:
        light_wordmark = image.convert("RGBA")
    with Image.open(dark_png_path) as image:
        dark_wordmark = image.convert("RGBA")
    with Image.open(symbolic_png_path) as image:
        symbolic = image.convert("RGBA")
    symbolic_png_path.unlink()

    social_path = output_dir / "social" / "orchav-github-social-preview.png"
    _create_social_preview(
        social_path,
        icon_images[1024],
        dark_wordmark,
    )
    _create_wordmark_review(
        review_dir / "01-wordmark-light-dark.png",
        light_wordmark,
        dark_wordmark,
    )
    _create_icon_size_review(
        review_dir / "02-app-icon-export-sizes.png",
        icon_images,
    )
    _create_symbolic_review(
        review_dir / "03-symbolic-icon-light-dark.png",
        symbolic,
    )
    shutil.copyfile(social_path, review_dir / "04-github-social-preview.png")
    _validate_assets(output_dir, wordmark_source_path, review_dir)
    return output_dir, review_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="ORCHAV repository root",
    )
    args = parser.parse_args()
    output_dir, review_dir = generate(args.repo_root.resolve())
    print(f"Brand assets: {output_dir}")
    print(f"Review sheets: {review_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
