"""pygfx image-based lighting and scene environment helpers.

``PygfxIBLManager`` is intentionally backend-local: it discovers ORCHAV HDR
assets, converts equirectangular maps into pygfx cubemaps, persists a local
numpy cache, and applies the result either at scene level or per material
depending on the installed pygfx feature set.
"""

from __future__ import annotations

import hashlib
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Optional
from weakref import WeakSet

import numpy as np

from ...backends.pygfx_pbr import pbr_material_classes
from ...lighting.assets import cached_environment, environment_catalog, ibl_cache_dir
from .canvas import _env_flag

logger = logging.getLogger(__name__)

__all__ = ["DEFAULT_PYGFX_IBL_INTENSITY", "PygfxIBLManager"]

DEFAULT_PYGFX_IBL_INTENSITY = 2000.0


def _supports_scene_environment(gfx: Any) -> bool:
    """Return whether this pygfx version exposes ``Scene.environment``."""
    try:
        return hasattr(gfx.Scene(), "environment")
    except Exception:
        return False


# IBL (Image-Based Lighting) manager

IBL_FACE_SIZE_DEFAULT = 512
IBL_CACHE_VERSION = "linear_v2"


class PygfxIBLManager:
    """Manages IBL environment maps for the pygfx renderer.

    Loads equirectangular HDR images from ``libraries/ibl/``, converts them to
    cubemap textures via vectorised numpy projection, and exposes the result
    through pygfx's scene-level environment map when available, falling back to
    per-material ``env_map`` binding on older stacks.
    """

    def __init__(self, gfx: Any, ibl_dir: Path) -> None:
        """Initialize IBL caches and detect the active scene-environment mode."""
        self._gfx = gfx
        self._ibl_dir = ibl_dir
        self._texture_cache: dict[str, Any] = {}
        self._texture_map_cache: dict[str, Any] = {}
        self._current_name: str = "default"
        self._current_intensity: float = 1.0
        self._background: Optional[Any] = None
        # Lighting observes live materials; it must not own removed scenes or
        # replaced materials and their GPU buffers across scenario switches.
        self._tracked_materials: WeakSet[Any] = WeakSet()
        self._scene: Optional[Any] = None
        self._use_scene_environment: bool = _env_flag(
            "ORCHAV_PYGFX_USE_SCENE_ENVIRONMENT", False
        ) and _supports_scene_environment(gfx)
        self._face_size: int = int(
            os.environ.get("ORCHAV_IBL_FACE_SIZE", str(IBL_FACE_SIZE_DEFAULT))
        )

    # Discovery

    def discover_available(self) -> list[str]:
        """Return sorted list of available IBL environment names."""
        if not self._ibl_dir.is_dir():
            return []
        names: set[str] = set()
        for p in self._ibl_dir.glob("*.hdr"):
            stem = p.stem
            for suffix in ("_4k_zup", "_1k_zup", "_zup", "_4k", "_1k"):
                if stem.endswith(suffix):
                    stem = stem[: -len(suffix)]
                    break
            names.add(stem)
        return sorted(names)

    # HDR resolution

    def _resolve_hdr_path(self, name: str) -> Optional[Path]:
        """Find the best HDR file for *name*.

        Standard equirectangular HDRs (row 0 = zenith, row H-1 = nadir) are
        preferred. The ``_zup`` variants encode an alternate pre-rotated
        cubemap convention whose vertical orientation is incompatible with the
        Khronos-correct face mapping, so they are fallback candidates only.
        """
        asset = environment_catalog().get(name)
        if asset is not None:
            return cached_environment(asset)
        direct = Path(name)
        if direct.suffix.lower() == ".hdr" and direct.is_file():
            return direct
        candidates = [
            self._ibl_dir / f"{name}.hdr",
            self._ibl_dir / f"{name}_4k.hdr",
            self._ibl_dir / f"{name}_1k.hdr",
            self._ibl_dir / f"{name}_zup.hdr",
            self._ibl_dir / f"{name}_4k_zup.hdr",
            self._ibl_dir / f"{name}_1k_zup.hdr",
        ]
        for c in candidates:
            if c.is_file():
                return c
        return None

    def _cache_path(self, name: str) -> Path:
        """Cache outside the checkout, keyed by source identity and preparation."""
        asset = environment_catalog().get(name)
        if asset is not None:
            identity = asset.cache_key
        else:
            path = self._resolve_hdr_path(name) or self._ibl_dir / name
            try:
                stat = path.stat()
                identity = f"{path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}:{stat.st_ctime_ns}"
            except OSError:
                identity = str(path.resolve())
            identity = hashlib.sha256(identity.encode()).hexdigest()[:32]
        return (
            ibl_cache_dir()
            / "cubemaps"
            / f"{identity}_face{self._face_size}_{IBL_CACHE_VERSION}_rgba.npy"
        )

    def _load_cached_cubemap(self, name: str) -> Optional[np.ndarray]:
        """Load a validated cubemap cache produced by this manager."""
        path = self._cache_path(name)
        if not path.is_file():
            return None
        try:
            cubemap = np.load(path, allow_pickle=False)
        except (OSError, ValueError) as exc:
            logger.warning("IBL '%s': failed to read pygfx cache %s: %s", name, path, exc)
            return None

        if cubemap.ndim != 4 or cubemap.shape[0] != 6:
            logger.warning("IBL '%s': ignoring invalid pygfx cache shape %s", name, cubemap.shape)
            return None
        if cubemap.shape[1] != self._face_size or cubemap.shape[2] != self._face_size:
            logger.warning(
                "IBL '%s': ignoring cache with face size %s (expected %s)",
                name,
                cubemap.shape[1:3],
                (self._face_size, self._face_size),
            )
            return None
        if cubemap.shape[3] not in (3, 4):
            logger.warning("IBL '%s': ignoring cache with %d channels", name, cubemap.shape[3])
            return None

        logger.info("IBL '%s': loaded pygfx cubemap cache %s", name, path.name)
        return np.asarray(cubemap, dtype=np.float32)

    def _write_cubemap_cache(self, name: str, cubemap: np.ndarray) -> None:
        """Persist a converted cubemap cache without making IBL loading depend on it."""
        path = self._cache_path(name)
        temporary = None
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as fh:
                temporary = Path(fh.name)
                np.save(fh, np.asarray(cubemap, dtype=np.float32), allow_pickle=False)
            temporary.replace(path)
            logger.info("IBL '%s': wrote pygfx cubemap cache %s", name, path.name)
        except OSError as exc:
            logger.debug("IBL '%s': could not write pygfx cache %s: %s", name, path, exc)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    # HDR loading (cv2 → pure-Python RGBE fallback)

    @staticmethod
    def _read_rgbe(path: Path) -> np.ndarray:
        """Read a Radiance .hdr/.pic file and return ``(H, W, 3)`` float32.

        Implements the RGBE (Ward) format: 4-byte pixels (R, G, B, exponent)
        with optional new-style run-length encoding.
        """
        with open(path, "rb") as f:
            signature = f.readline().strip()
            if signature.startswith(b"version https://git-lfs.github.com/"):
                raise ValueError("This file is a Git LFS pointer, not a downloaded HDR image")
            if signature not in (b"#?RADIANCE", b"#?RGBE"):
                raise ValueError("Not a Radiance HDR image")
            # --- header ---
            while True:
                line = f.readline()
                if not line or line.strip() == b"":
                    break
            # Resolution line, e.g. ``-Y 1024 +X 2048``
            res_line = f.readline().decode("ascii", errors="replace").strip()
            parts = res_line.split()
            if len(parts) != 4:
                raise ValueError(f"Bad resolution line: {res_line!r}")
            height, width = int(parts[1]), int(parts[3])

            # --- scanline data ---
            img = np.empty((height, width, 3), dtype=np.float32)
            for y in range(height):
                scanline = PygfxIBLManager._read_rgbe_scanline(f, width)
                # Decode RGBE → float RGB
                r, g, b, e = (
                    scanline[:, 0],
                    scanline[:, 1],
                    scanline[:, 2],
                    scanline[:, 3],
                )
                scale = np.where(e == 0, 0.0, np.ldexp(1.0, e.astype(np.int32) - 128 - 8))
                img[y, :, 0] = r * scale
                img[y, :, 1] = g * scale
                img[y, :, 2] = b * scale
        return img

    @staticmethod
    def _read_rgbe_scanline(f, width: int) -> np.ndarray:  # type: ignore[override]
        """Read one RGBE scanline (handles new-style RLE)."""
        header = f.read(4)
        if len(header) < 4:
            raise ValueError("Unexpected EOF in HDR scanline header")
        # New-style RLE: sentinel bytes 2, 2, then width high/low
        if header[0] == 2 and header[1] == 2:
            scan_w = (header[2] << 8) | header[3]
            if scan_w != width:
                raise ValueError(f"Scanline width mismatch: {scan_w} vs {width}")
            buf = np.empty((width, 4), dtype=np.uint8)
            for ch in range(4):
                ptr = 0
                while ptr < width:
                    byte = f.read(1)
                    if not byte:
                        raise ValueError("Unexpected EOF in RLE data")
                    count = byte[0]
                    if count > 128:
                        # Run of repeated value
                        count -= 128
                        val = f.read(1)
                        if not val:
                            raise ValueError("Unexpected EOF in RLE data")
                        buf[ptr : ptr + count, ch] = val[0]
                    else:
                        # Run of literal values
                        data = f.read(count)
                        if len(data) < count:
                            raise ValueError("Unexpected EOF in RLE data")
                        buf[ptr : ptr + count, ch] = np.frombuffer(data, dtype=np.uint8)
                    ptr += count
            return buf
        # Old-style (uncompressed): header is the first pixel
        buf = np.empty((width, 4), dtype=np.uint8)
        buf[0] = np.frombuffer(header, dtype=np.uint8)
        rest = f.read((width - 1) * 4)
        if len(rest) < (width - 1) * 4:
            raise ValueError("Unexpected EOF in uncompressed scanline")
        buf[1:] = np.frombuffer(rest, dtype=np.uint8).reshape(width - 1, 4)
        return buf

    def _load_hdr(self, path: Path, name: str) -> Optional[np.ndarray]:
        """Load an HDR file as ``(H, W, 3)`` float32.

        Tries OpenCV first (fastest), falls back to pure-Python RGBE reader.
        """
        # Attempt 1: OpenCV (fastest, handles all HDR variants)
        try:
            import cv2

            bgr = cv2.imread(str(path), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
            if bgr is not None:
                return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
            logger.debug("IBL '%s': cv2 returned None for %s, trying fallback", name, path)
        except ImportError:
            pass

        # Attempt 2: pure-Python RGBE reader (no external deps)
        try:
            return self._read_rgbe(path)
        except (OSError, ValueError) as exc:
            logger.warning("IBL '%s': failed to read %s: %s", name, path, exc)
            return None

    # Equirectangular -> cubemap conversion (vectorised numpy)

    @staticmethod
    def _equirect_to_cubemap(equirect: np.ndarray, face_size: int = 512) -> np.ndarray:
        """Convert an equirectangular HDR image to a 6-face cubemap.

        Args:
            equirect: ``(H, W, C)`` standard equirectangular image (top is zenith).
            face_size: Pixel resolution of each cubemap face.

        Returns:
            ``(6, face_size, face_size, C)`` float32 cubemap array.
            Face order: +X, -X, +Y, -Y, +Z, -Z.
        """
        H, W, C = equirect.shape
        fs = face_size

        # Normalised pixel coordinates in [-1, 1] for each face pixel.
        grid = np.linspace(-1.0, 1.0, fs, dtype=np.float32)
        uu, vv = np.meshgrid(grid, -grid)  # v flipped so top = +1

        # Build (fs, fs, 3) direction arrays per face.
        #
        # These are Z-up world-space directions that, after the pygfx background
        # shader's right-to-left-handed x-flip (texcoord.x = -texcoord.x), produce
        # the correct cubemap lookup vectors per the Khronos cubemap spec.
        #
        # Derivation: for each face the Khronos spec defines (sc, tc, ma) →
        # lookup direction.  Converting to (uu, vv) coordinates and undoing the
        # x-flip gives the world direction used to sample the Z-up equirect.
        # Keep this mapping aligned with the renderer coordinate-convention notes.
        ones = np.ones_like(uu)
        face_dirs = np.empty((6, fs, fs, 3), dtype=np.float32)
        face_dirs[0] = np.stack([-ones, vv, -uu], axis=-1)  # +X
        face_dirs[1] = np.stack([ones, vv, uu], axis=-1)  # -X
        face_dirs[2] = np.stack([-uu, ones, -vv], axis=-1)  # +Y
        face_dirs[3] = np.stack([-uu, -ones, vv], axis=-1)  # -Y
        face_dirs[4] = np.stack([-uu, vv, ones], axis=-1)  # +Z (up)
        face_dirs[5] = np.stack([uu, vv, -ones], axis=-1)  # -Z (down)

        # Flatten to (6*fs*fs, 3) for vectorised spherical mapping.
        dirs = face_dirs.reshape(-1, 3)
        x, y, z = dirs[:, 0], dirs[:, 1], dirs[:, 2]
        length = np.sqrt(x * x + y * y + z * z)
        length = np.maximum(length, 1e-8)

        # Z-up spherical: theta = atan2(y, x), phi = asin(z / len)
        theta = np.arctan2(y, x)
        phi = np.arcsin(np.clip(z / length, -1.0, 1.0))

        # Map to equirect UV in [0, 1].
        u_eq = theta / (2.0 * np.pi) + 0.5
        v_eq = 0.5 - phi / np.pi  # row 0 = top = +Z

        # Bilinear sample from equirect image.
        u_px = np.clip(u_eq * W - 0.5, 0, W - 1).astype(np.float32)
        v_px = np.clip(v_eq * H - 0.5, 0, H - 1).astype(np.float32)

        u0 = np.floor(u_px).astype(np.int32)
        v0 = np.floor(v_px).astype(np.int32)
        u1 = np.minimum(u0 + 1, W - 1)
        v1 = np.minimum(v0 + 1, H - 1)

        fu = (u_px - u0.astype(np.float32))[:, None]
        fv = (v_px - v0.astype(np.float32))[:, None]

        s00 = equirect[v0, u0]
        s01 = equirect[v0, u1]
        s10 = equirect[v1, u0]
        s11 = equirect[v1, u1]

        sampled = (
            s00 * (1 - fu) * (1 - fv) + s01 * fu * (1 - fv) + s10 * (1 - fu) * fv + s11 * fu * fv
        )

        return sampled.reshape(6, fs, fs, C)

    # Texture loading

    def load_ibl(self, name: str) -> Optional[Any]:
        """Load (or return cached) cubemap texture for *name*.

        Returns a ``pygfx.Texture`` cubemap, or ``None`` if the HDR file
        cannot be found or loaded.
        """
        if name in self._texture_cache:
            self._current_name = name
            return self._texture_cache[name]

        asset = environment_catalog().get(name)
        if asset is not None and cached_environment(asset) is None:
            logger.warning(
                "IBL '%s' is not installed; use Rendering > Lighting to download it", name
            )
            return None

        cubemap = self._load_cached_cubemap(name)
        hdr_path: Optional[Path] = None
        source_name: str
        source_width: int
        source_height: int
        if cubemap is None:
            hdr_path = self._resolve_hdr_path(name)
            if hdr_path is None:
                logger.warning("IBL '%s': no HDR file found in %s", name, self._ibl_dir)
                return None

            equirect = self._load_hdr(hdr_path, name)
            if equirect is None:
                return None

            if asset is not None:
                # Panoramas are already zenith-up. Only rotate their azimuth;
                # this keeps the visible sun aligned with the lighting recipe.
                shift = round(asset.yaw_degrees / 360 * equirect.shape[1])
                equirect = np.roll(equirect, shift, axis=1) * asset.radiance_scale

            cubemap = self._equirect_to_cubemap(equirect, self._face_size)
            source_name = hdr_path.name
            source_width = int(equirect.shape[1])
            source_height = int(equirect.shape[0])
        else:
            source_name = self._cache_path(name).name
            source_width = int(cubemap.shape[2])
            source_height = int(cubemap.shape[1])

        # pygfx expects RGBA for some backends; pad alpha channel if needed.
        if cubemap.shape[-1] == 3:
            alpha = np.ones((*cubemap.shape[:-1], 1), dtype=np.float32)
            cubemap = np.concatenate([cubemap, alpha], axis=-1)
        elif cubemap.dtype != np.float32:
            cubemap = np.asarray(cubemap, dtype=np.float32)

        if hdr_path is not None:
            self._write_cubemap_cache(name, cubemap)

        gfx = self._gfx
        texture = gfx.Texture(
            cubemap,
            dim=2,
            size=(self._face_size, self._face_size, 6),
            generate_mipmaps=True,
            colorspace="physical",
        )

        self._texture_cache[name] = texture
        self._current_name = name
        logger.info(
            "IBL '%s': loaded %s (%dx%d, face=%d)",
            name,
            source_name,
            source_width,
            source_height,
            self._face_size,
        )
        return texture

    # Scene integration

    @property
    def uses_scene_environment(self) -> bool:
        """Return whether IBL is applied through ``Scene.environment``."""
        return self._use_scene_environment

    def _current_environment_map(self) -> Optional[Any]:
        """Return the current environment object in the form this pygfx stack wants."""
        texture = self._texture_cache.get(self._current_name)
        if texture is None:
            return None
        if not self._use_scene_environment:
            return texture
        cached = self._texture_map_cache.get(self._current_name)
        if cached is not None:
            return cached
        texture_map_cls = getattr(self._gfx, "TextureMap", None)
        if texture_map_cls is None:
            return texture
        cached = texture_map_cls(texture)
        self._texture_map_cache[self._current_name] = cached
        return cached

    def _background_attached_to(self, scene: Any) -> bool:
        """Return True when the cached skybox background is attached to *scene*."""
        if scene is None or self._background is None:
            return False
        if getattr(self._background, "parent", None) is scene:
            return True
        children = getattr(scene, "children", None)
        try:
            return children is not None and self._background in children
        except (TypeError, ValueError):
            return False

    def _remove_background(self, scene: Any) -> None:
        """Detach the cached skybox background if it is currently in a scene."""
        if self._background is None:
            return
        target_scene = getattr(self._background, "parent", None)
        if target_scene is None and self._background_attached_to(scene):
            target_scene = scene
        if target_scene is not None and hasattr(target_scene, "remove"):
            try:
                target_scene.remove(self._background)
            except (ValueError, RuntimeError):
                pass
        self._background = None

    def apply_to_scene(self, scene: Any) -> bool:
        """Add or replace the skybox background in *scene*."""
        texture = self._texture_cache.get(self._current_name)
        if texture is None:
            return False

        gfx = self._gfx
        applied_environment = False
        self._scene = scene

        if self._use_scene_environment and scene is not None and hasattr(scene, "environment"):
            try:
                scene.environment = self._current_environment_map()
                applied_environment = True
            except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
                logger.warning(
                    "IBL '%s': failed to set scene.environment: %s",
                    self._current_name,
                    exc,
                )

        self._remove_background(scene)

        try:
            _, _, skybox_cls = pbr_material_classes(gfx)
            bg_mat = skybox_cls(map=texture)
            self._background = gfx.Background(None, bg_mat)
            if scene is not None:
                scene.add(self._background)
            return applied_environment or True
        except (AttributeError, RuntimeError, TypeError) as exc:
            logger.warning("IBL: failed to create skybox background: %s", exc)
            self._background = None
            return applied_environment

    def apply_to_material(self, material: Any) -> None:
        """Apply IBL-related material state for the active environment."""
        if material is None:
            return
        # Paper/color-inspection materials must preserve albedo. MeshBasicMaterial
        # can still reflect an env_map despite having no direct-light shader.
        # Use exact type: the lit material classes inherit from MeshBasicMaterial.
        if type(material) is getattr(self._gfx, "MeshBasicMaterial", None):
            material.env_map = None
            self._tracked_materials.discard(material)
            return
        if self._texture_cache.get(self._current_name) is None:
            return
        try:
            if hasattr(material, "env_map"):
                material.env_map = (
                    None if self._use_scene_environment else self._current_environment_map()
                )
            if hasattr(material, "env_map_intensity"):
                material.env_map_intensity = self._current_intensity
        except (AttributeError, RuntimeError):
            pass
        self._tracked_materials.add(material)

    def set_intensity(self, value: float) -> None:
        """Update ``env_map_intensity`` on all tracked materials."""
        self._current_intensity = value
        for mat in self._tracked_materials:
            try:
                if hasattr(mat, "env_map_intensity"):
                    mat.env_map_intensity = value
            except (AttributeError, RuntimeError):
                continue

    def set_skybox_visible(self, show: bool, scene: Any) -> bool:
        """Add or remove the skybox Background from *scene*."""
        if scene is None:
            return False
        if show:
            if self._background is None:
                return self.apply_to_scene(scene)
            if not self._background_attached_to(scene):
                try:
                    scene.add(self._background)
                except (ValueError, RuntimeError):
                    pass
            return True
        else:
            self._remove_background(scene)
            return True

    def cleanup_material(self, material: Any) -> None:
        """Remove a material from the tracked list."""
        self._tracked_materials.discard(material)
