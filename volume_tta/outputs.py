"""Implementation subsystem extracted from the v17.0.5 volume TTA runtime.

This physical split intentionally preserves the original numerical and scheduling
behavior. Public coordination contracts live under ``inference_backends``.
"""

from __future__ import annotations

from ._stdlib import *
import numpy as np
from ._deps import _numba, cv2, tifffile, tqdm

from .config import (
    DEFAULT_CHANNEL_FORMAT,
    GIB,
)
from .runtime import (
    runtime_telemetry,
    runtime_telemetry_phase,
)

_OVERLAY_BLUE_U16 = np.array([0, 0, 255], dtype=np.uint16)  # RGB blue in the uint16 blend domain

def _overlay_blend_blue_inplace(frame_rgb: np.ndarray, mask2d: np.ndarray) -> None:
    """50% blue blend on mask pixels, restricted to the mask's bbox, in place.

 Byte-identical to the legacy full-frame masked path ((frame + blue) // 2 in uint16 on
 mask pixels, untouched elsewhere) but allocates only bbox-sized temps: no full-frame
 bool cast, no masked gather/scatter over the whole frame."""
    m = np.asarray(mask2d)
    ys = np.flatnonzero(m.any(axis=1))
    if ys.size == 0:
        return
    y0, y1 = int(ys[0]), int(ys[-1]) + 1
    xs = np.flatnonzero(m[y0:y1].any(axis=0))
    x0, x1 = int(xs[0]), int(xs[-1]) + 1
    sub = frame_rgb[y0:y1, x0:x1]
    msub = m[y0:y1, x0:x1] != 0
    blended = ((sub.astype(np.uint16) + _OVERLAY_BLUE_U16) // 2).astype(np.uint8)
    np.copyto(sub, blended, where=msub[:, :, None])

def _gray_frame_into_rgb_buffer(frame_gray: np.ndarray, frame_buf: np.ndarray) -> np.ndarray:
    """Expand a gray frame into the reused (H, W, 3) buffer, no per-frame alloc.

 Returns the buffer actually written (cv2 falls back to allocating only if dst were
 incompatible, which the caller's preallocated buffer never is)."""
    gray = np.ascontiguousarray(np.asarray(frame_gray, dtype=np.uint8))
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB, dst=frame_buf)

def ffv1_segment_count(total_frames: int) -> int:
    """Number of concurrent contiguous FFV1 encode shards for a final video.

 Each shard has identical codec parameters and contains whole, independently decodable
 FFV1 frames. ``YOLO_TTA_FFV1_SEGMENTS=1`` preserves the original single-pipe path.
 Explicit values are bounded only by the number of frames; the automatic default targets
 one 30-thread FFV1 encoder per roughly 32 allocated CPUs, capped at six encoders."""
    total = max(0, int(total_frames))
    automatic = min(6, max(1, int(math.ceil(float(_cpu_count()) / 32.0))))
    requested = max(1, _env_int('YOLO_TTA_FFV1_SEGMENTS', int(automatic)))
    return 1 if total <= 1 else min(int(total), int(requested))

def _ffv1_contiguous_segments(total_frames: int, segment_count: int) -> List[Tuple[int, int]]:
    """Return balanced, gap-free ``[start, stop)`` frame ranges."""
    total = max(0, int(total_frames))
    count = max(1, min(int(segment_count), max(1, total)))
    base, extra = divmod(total, count)
    ranges: List[Tuple[int, int]] = []
    start = 0
    for idx in range(count):
        stop = start + int(base) + (1 if idx < int(extra) else 0)
        ranges.append((int(start), int(stop)))
        start = stop
    return ranges

def _atomic_publication_stage_path(
    out_path: Path,
    *,
    publication_root: Optional[Path] = None,
) -> Tuple[Path, Path]:
    """Return a same-filesystem staging path outside the publication tree.

 The staging directory is a sibling of the run's publication root. Recursive rsync or
 watchers rooted at ``publication_root`` therefore see neither shards nor a growing
 ``.assembling`` inode; only the final atomic rename enters that tree."""
    final_path = Path(out_path)
    root = Path(publication_root) if publication_root is not None else final_path.parent
    final_path.parent.mkdir(parents=True, exist_ok=True)
    root.parent.mkdir(parents=True, exist_ok=True)
    # Every publisher owns its directory. A shared empty staging directory can be
    # removed by one concurrent writer after another has selected (but not yet
    # created) its stage file.
    stage_dir = Path(tempfile.mkdtemp(
        prefix=f'.{root.name}.atomic-result-',
        dir=str(root.parent),
    ))
    if int(stage_dir.stat().st_dev) != int(final_path.parent.stat().st_dev):
        shutil.rmtree(stage_dir, ignore_errors=True)
        raise RuntimeError(
            f'Atomic publication staging and destination are on different filesystems: '
            f'{stage_dir} vs {final_path.parent}'
        )
    token = f'{os.getpid()}.{threading.get_ident()}.{time.time_ns()}'
    stage_path = stage_dir / f'.{final_path.stem}.{token}.assembling{final_path.suffix}'
    return stage_path, stage_dir

def _publish_staged_file_atomically(stage_path: Path, out_path: Path) -> None:
    """Durably replace ``out_path`` with a completed same-filesystem staging file."""
    stage = Path(stage_path)
    final_path = Path(out_path)
    if not stage.is_file() or int(stage.stat().st_size) <= 0:
        raise RuntimeError(f'Atomic publication stage is missing or empty: {stage}')
    # Windows rejects FlushFileBuffers (os.fsync) on a read-only handle with EBADF.
    # Open update-capable without modifying the completed payload.
    with open(stage, 'rb+') as stage_fh:
        os.fsync(stage_fh.fileno())
    os.replace(stage, final_path)
    try:
        parent_fd = os.open(str(final_path.parent), os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    except OSError:
        pass


@contextlib.contextmanager
def _same_directory_atomic_output(out_path: Path) -> Iterator[Path]:
    """Yield a private sibling path and publish it only after a successful write.

    Keeping the temporary inode in the destination directory guarantees that the final
    ``os.replace`` cannot cross filesystems.  The previous output remains intact if the
    producer raises, and the private file is removed on every exit path.
    """
    final_path = Path(out_path)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    token = f'{os.getpid()}.{threading.get_ident()}.{time.time_ns()}'
    stage_path = final_path.with_name(
        f'.{final_path.name}.{token}.assembling'
    )
    try:
        yield stage_path
        _publish_staged_file_atomically(stage_path, final_path)
    finally:
        try:
            stage_path.unlink(missing_ok=True)
        except Exception:
            pass


def _write_json_atomically(out_path: Path, payload: object) -> Path:
    """Serialize JSON completely before atomically replacing its public sidecar."""
    final_path = Path(out_path)
    with _same_directory_atomic_output(final_path) as stage_path:
        stage_path.write_text(json.dumps(payload, indent=2))
    return final_path

def _run_sharded_ffv1_encode(
    *,
    out_path: Path,
    total_frames: int,
    description: str,
    show_progress: bool,
    encode_range: Callable[[int, int, Path, threading.Event, Callable[[subprocess.Popen], None], Callable[[subprocess.Popen], None], Callable[[], None]], None],
    scratch_dir: Optional[Path] = None,
    publication_root: Optional[Path] = None,
) -> None:
    """Encode contiguous FFV1 shards into memory, then stream-concat atomically to disk.

 Each range callback owns one ffmpeg producer/process whose Matroska output is an anonymous
 memfd. On the first shard failure all registered encoders are terminated and every worker
 is reaped. The only filesystem write is the ordered concat stream into a same-filesystem
 atomic publication stage, so the previous destination remains untouched on failure.
 ``scratch_dir`` remains in the public call contract but is no longer used for encoded shards."""
    _ = scratch_dir
    total = int(total_frames)
    segments = int(ffv1_segment_count(total))
    cancel_event = threading.Event()
    active_lock = threading.Lock()
    active_processes: set[subprocess.Popen] = set()

    def _register(proc: subprocess.Popen) -> None:
        with active_lock:
            active_processes.add(proc)
            cancelled = cancel_event.is_set()
        if cancelled:
            try:
                proc.terminate()
            except Exception:
                pass

    def _unregister(proc: subprocess.Popen) -> None:
        with active_lock:
            active_processes.discard(proc)

    def _terminate_active() -> None:
        cancel_event.set()
        with active_lock:
            processes = list(active_processes)
        for proc in processes:
            try:
                proc.terminate()
            except Exception:
                pass
        deadline = time.monotonic() + 5.0
        for proc in processes:
            try:
                proc.wait(timeout=max(0.0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                try:
                    proc.kill()
                except Exception:
                    pass
            except Exception:
                pass

    progress = tqdm(total=max(0, total), desc=description, disable=not show_progress)

    def _progress_one() -> None:
        progress.update(1)

    joined_path: Optional[Path] = None
    publication_stage_dir: Optional[Path] = None
    executor: Optional[ThreadPoolExecutor] = None
    futures: List[Future] = []
    shard_files: List[object] = []
    first_error: Optional[BaseException] = None
    try:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        ranges = _ffv1_contiguous_segments(total, segments)
        for idx in range(len(ranges)):
            shard_files.append(_open_memory_backed_encoded_chunk(
                f'{Path(out_path).stem}.ffv1.segment_{idx:04d}.mkv',
                require_fileno=True,
            ))
        shard_paths = [_memory_backed_encoded_chunk_path(chunk) for chunk in shard_files]
        executor = ThreadPoolExecutor(max_workers=len(ranges), thread_name_prefix='ffv1-segment')
        futures = [
            executor.submit(
                encode_range,
                int(start),
                int(stop),
                shard_path,
                cancel_event,
                _register,
                _unregister,
                _progress_one,
            )
            for (start, stop), shard_path in zip(ranges, shard_paths)
        ]
        for fut in as_completed(futures):
            try:
                fut.result()
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
                    _terminate_active()
        if first_error is not None:
            raise RuntimeError(f'FFV1 segmented encode failed for {Path(out_path).name}') from first_error

        missing_indices: List[int] = []
        for idx, chunk in enumerate(shard_files):
            chunk.flush()
            if _memory_backed_encoded_chunk_size(chunk) <= 0:
                missing_indices.append(int(idx))
        if missing_indices:
            raise RuntimeError(
                f'FFV1 encoder did not produce {len(missing_indices)} in-memory shard(s): '
                f'{missing_indices[:3]}'
            )

        joined_path, publication_stage_dir = _atomic_publication_stage_path(
            Path(out_path), publication_root=publication_root,
        )
        if len(shard_files) == 1:
            shard_files[0].seek(0)
            with open(joined_path, 'wb', buffering=0) as dst_fh:
                shutil.copyfileobj(shard_files[0], dst_fh, length=16 * 1024 * 1024)
                dst_fh.flush()
                os.fsync(dst_fh.fileno())
        else:
            for chunk in shard_files:
                chunk.seek(0)
            concat_text = 'ffconcat version 1.0\n' + ''.join(
                f"file 'file:/proc/self/fd/{int(chunk.fileno())}'\n"
                for chunk in shard_files
            )
            cmd = [
                'ffmpeg', '-y', '-v', 'error',
                '-f', 'concat', '-safe', '0',
                '-protocol_whitelist', 'file,pipe', '-i', 'pipe:0',
                '-map', '0:v:0', '-c', 'copy', '-f', 'matroska', str(joined_path),
            ]
            completed = _spawn_subprocess_with_retry(
                lambda: subprocess.run(
                    cmd,
                    input=concat_text.encode('utf-8'),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=False,
                    pass_fds=tuple(int(chunk.fileno()) for chunk in shard_files),
                ),
                f'ffmpeg FFV1 concat ({Path(out_path).name})',
            )
            if int(completed.returncode) != 0:
                stderr = (
                    completed.stderr.decode('utf-8', errors='ignore')
                    if isinstance(completed.stderr, (bytes, bytearray))
                    else str(completed.stderr)
                )
                raise RuntimeError(f'FFV1 lossless concat failed for {Path(out_path).name}: {stderr}')
            with open(joined_path, 'rb') as joined_fh:
                os.fsync(joined_fh.fileno())
        _publish_staged_file_atomically(joined_path, Path(out_path))
        joined_path = None
    except BaseException:
        _terminate_active()
        raise
    finally:
        if executor is not None:
            try:
                executor.shutdown(wait=True, cancel_futures=True)
            except TypeError:  # Python < 3.9 compatibility
                executor.shutdown(wait=True)
        progress.close()
        for chunk in shard_files:
            try:
                chunk.close()
            except Exception:
                pass
        if joined_path is not None:
            try:
                joined_path.unlink(missing_ok=True)
            except Exception:
                pass
        if publication_stage_dir is not None:
            try:
                publication_stage_dir.rmdir()
            except OSError:
                pass

def write_overlay_video(
    volume_rgb: np.memmap,  # (T,H,W) gray/luma
    mask_u8: np.ndarray,    # (T,H,W) 0/1
    out_path: Path,
    fps: float,
    show_progress: bool = True,
    scratch_dir: Optional[Path] = None,
    publication_root: Optional[Path] = None,
) -> None:
    """Encode a source/mask overlay, using sharded FFV1 and memory-backed concatenation when admitted."""
    T, H, W = volume_rgb.shape
    assert mask_u8.shape == (T, H, W)

    def _encode_range(
        start: int,
        stop: int,
        shard_path: Path,
        cancel_event: threading.Event,
        register_proc: Callable[[subprocess.Popen], None],
        unregister_proc: Callable[[subprocess.Popen], None],
        progress_one: Callable[[], None],
    ) -> None:
        proc = ffmpeg_ffv1_rgb_writer(
            shard_path,
            width=W,
            height=H,
            fps=fps,
        )
        register_proc(proc)
        frame_buf = np.empty((int(H), int(W), 3), dtype=np.uint8)
        primary_error: Optional[BaseException] = None
        primary_traceback = None
        try:
            assert proc.stdin is not None
            for t in range(int(start), int(stop)):
                if cancel_event.is_set():
                    break
                frame = _gray_frame_into_rgb_buffer(volume_rgb[t], frame_buf)
                _overlay_blend_blue_inplace(frame, mask_u8[t])
                proc.stdin.write(memoryview(frame).cast('B'))
                progress_one()
        except BaseException as exc:
            primary_error = exc
            primary_traceback = exc.__traceback__
        try:
            close_ffmpeg_writer(proc)
        except BaseException as exc:
            if primary_error is None:
                primary_error = exc
                primary_traceback = exc.__traceback__
        finally:
            unregister_proc(proc)
        if primary_error is not None:
            raise primary_error.with_traceback(primary_traceback)

    _run_sharded_ffv1_encode(
        out_path=Path(out_path),
        total_frames=int(T),
        description=f"Writing overlay video ({out_path.name})",
        show_progress=bool(show_progress),
        encode_range=_encode_range,
        scratch_dir=scratch_dir,
        publication_root=publication_root,
    )

def mask_to_yolo_polygons(mask01: np.ndarray) -> List[List[Tuple[float, float]]]:
    """Convert a binary mask (H,W) to a list of external polygons with normalized coords."""
    h, w = mask01.shape
    m = (mask01.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    polys: List[List[Tuple[float, float]]] = []
    for cnt in contours:
        if cnt is None or len(cnt) < 3:
            continue
        approx = cv2.approxPolyDP(cnt, epsilon=1.0, closed=True)
        if approx is None or len(approx) < 3:
            continue
        pts = approx.reshape(-1, 2)
        polys.append([(float(x) / float(w), float(y) / float(h)) for (x, y) in pts])
    return polys

DEFAULT_LABEL_PATTERN = "labels/{Filename}_%04d.txt"

DEFAULT_BINARY_PATTERN = "binary_masks/{Filename}_Binary_%04d.tiff"

def _resolve_output_pattern(pattern_value: Optional[str], default_pattern: str, out_dir: Path, stem: str) -> Optional[Path]:
    if pattern_value is None:
        return None
    pattern = default_pattern if str(pattern_value) == "__DEFAULT__" else str(pattern_value)
    pattern = pattern.replace("{Filename}", stem)
    path = Path(pattern)
    if not path.is_absolute():
        path = out_dir / path
    return path

def _tag_frame_pattern(path: Path, tag: str) -> Path:
    parent = path.parent
    if parent.name:
        parent = parent.with_name(f"{parent.name}_{tag.lower()}")

    name = path.name
    m = re.search(r"(%0\d+d)", name)
    if m is not None:
        prefix = name[:m.start()]
        if prefix.endswith("_"):
            name = prefix + f"{tag}_" + name[m.start():]
        else:
            name = prefix + f"_{tag}_" + name[m.start():]
    else:
        suffix = "".join(path.suffixes)
        base = name[:-len(suffix)] if suffix else name
        name = f"{base}_{tag}{suffix}"
    return parent / name

def _format_frame_path(pattern_path: Path, frame_idx_1based: int) -> Path:
    as_str = str(pattern_path)
    if "%" in as_str:
        try:
            return Path(as_str % int(frame_idx_1based))
        except TypeError:
            pass
    return pattern_path.with_name(f"{pattern_path.stem}_{int(frame_idx_1based):04d}{pattern_path.suffix}")


_FRAME_NUMBER_PLACEHOLDER_RE = re.compile(r'%(?:0\d+)?d')


def _frame_sequence_name_matcher(
    pattern_path: Path,
    extensions: Sequence[str],
) -> re.Pattern[str]:
    """Match files belonging to a frame pattern, independent of frame count/extension."""
    pattern = Path(pattern_path)
    name = pattern.name
    placeholder = _FRAME_NUMBER_PLACEHOLDER_RE.search(name)
    suffix = pattern.suffix
    if placeholder is not None:
        prefix = name[:placeholder.start()]
        tail = name[placeholder.end():]
        tail_without_suffix = tail[:-len(suffix)] if suffix and tail.lower().endswith(suffix.lower()) else tail
    else:
        prefix = f'{pattern.stem}_'
        tail_without_suffix = ''

    def _normalize_extension(extension: object) -> str:
        value = str(extension)
        value = value if value.startswith('.') else f'.{value}'
        return value.lower() if os.name == 'nt' else value

    normalized_extensions = {
        _normalize_extension(ext)
        for ext in extensions
        if str(ext)
    }
    if suffix:
        normalized_extensions.add(_normalize_extension(suffix))
    extension_expr = (
        '(?:' + '|'.join(re.escape(ext) for ext in sorted(normalized_extensions)) + ')'
        if normalized_extensions else ''
    )
    return re.compile(
        rf'^{re.escape(prefix)}\d+{re.escape(tail_without_suffix)}{extension_expr}$',
        # Windows paths are conventionally case-insensitive; POSIX paths are not.  On
        # Linux, folding case here could delete a distinct sequence in the same directory.
        flags=(re.IGNORECASE if os.name == 'nt' else 0),
    )


def _publish_staged_frame_sequence(
    stage_pattern: Path,
    final_pattern: Path,
    total: int,
    *,
    stale_extensions: Sequence[str],
) -> None:
    """Publish a fully rendered sequence, then remove tails/obsolete extensions.

    Rendering happens in a hidden directory under the destination directory.  Therefore a
    rendering failure exposes none of the new generation and preserves every prior frame.
    Each completed frame enters the public sequence with an atomic same-filesystem replace.
    """
    count = max(0, int(total))
    pairs = [
        (
            _format_frame_path(Path(stage_pattern), idx),
            _format_frame_path(Path(final_pattern), idx),
        )
        for idx in range(1, count + 1)
    ]
    missing = [stage for stage, _final in pairs if not stage.is_file()]
    if missing:
        raise RuntimeError(f'Frame sequence staging is incomplete; missing {missing[0]}')

    for stage_path, final_path in pairs:
        final_path.parent.mkdir(parents=True, exist_ok=True)
        os.replace(stage_path, final_path)

    expected = {
        os.path.normcase(os.path.abspath(str(final_path)))
        for _stage_path, final_path in pairs
    }
    matcher = _frame_sequence_name_matcher(Path(final_pattern), stale_extensions)
    for candidate in Path(final_pattern).parent.iterdir():
        if not candidate.is_file() or matcher.fullmatch(candidate.name) is None:
            continue
        normalized = os.path.normcase(os.path.abspath(str(candidate)))
        if normalized not in expected:
            candidate.unlink()


@contextlib.contextmanager
def _staged_frame_sequence(pattern_path: Path) -> Iterator[Path]:
    """Yield an equivalent frame pattern rooted in a private destination-side directory."""
    final_pattern = Path(pattern_path)
    final_pattern.parent.mkdir(parents=True, exist_ok=True)
    stage_dir = Path(tempfile.mkdtemp(
        prefix=f'.{final_pattern.parent.name}.frame-generation-',
        dir=str(final_pattern.parent),
    ))
    try:
        yield stage_dir / final_pattern.name
    finally:
        shutil.rmtree(stage_dir, ignore_errors=True)

def _write_label_file_from_mask(mask2d: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    m = np.asarray(mask2d) > 0
    if not np.any(m):
        out_path.write_text("")
        return

    polys = mask_to_yolo_polygons(m.astype(np.uint8))
    if not polys:
        out_path.write_text("")
        return

    lines: List[str] = []
    for poly in polys:
        coords: List[str] = []
        for x, y in poly:
            coords.append(f"{x:.6f}")
            coords.append(f"{y:.6f}")
        lines.append("0 " + " ".join(coords))
    out_path.write_text("\n".join(lines) + "\n")

def _write_binary_tiff_frame(mask2d: np.ndarray, out_path: Path) -> None:
    """Write one true bilevel binary TIFF frame with DEFLATE compression."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mask_bool = np.asarray(mask2d, dtype=bool)
    tifffile.imwrite(
        str(out_path),
        mask_bool,
        photometric='minisblack',
        compression='deflate',
    )

def write_yolo_labels_from_pattern(
    mask_u8: np.ndarray,
    pattern_path: Path,
    workers: int = 1,
    show_progress: bool = True,
) -> Path:
    pattern_path.parent.mkdir(parents=True, exist_ok=True)
    total = int(mask_u8.shape[0])
    with _staged_frame_sequence(pattern_path) as stage_pattern:
        def _write_frame(t: int) -> None:
            fp = _format_frame_path(stage_pattern, int(t) + 1)
            _write_label_file_from_mask(np.asarray(mask_u8[int(t)]), fp)

        parallel_for_indices(
            total,
            _write_frame,
            max_workers=choose_slice_parallel_workers(int(workers), total),
            desc=f"Writing YOLO labels ({pattern_path.parent.name})",
            show_progress=show_progress,
        )
        _publish_staged_frame_sequence(
            stage_pattern,
            pattern_path,
            total,
            stale_extensions=(pattern_path.suffix, '.txt'),
        )
    return pattern_path.parent

def write_binary_tiff_sequence_from_pattern(
    mask_u8: np.ndarray,
    pattern_path: Path,
    workers: int = 1,
    show_progress: bool = True,
) -> Path:
    pattern_path.parent.mkdir(parents=True, exist_ok=True)
    total = int(mask_u8.shape[0])
    with _staged_frame_sequence(pattern_path) as stage_pattern:
        def _write_frame(t: int) -> None:
            fp = _format_frame_path(stage_pattern, int(t) + 1)
            _write_binary_tiff_frame(np.asarray(mask_u8[int(t)]), fp)

        parallel_for_indices(
            total,
            _write_frame,
            max_workers=choose_slice_parallel_workers(int(workers), total),
            desc=f"Writing binary TIFF sequence ({pattern_path.parent.name})",
            show_progress=show_progress,
        )
        _publish_staged_frame_sequence(
            stage_pattern,
            pattern_path,
            total,
            stale_extensions=(pattern_path.suffix, '.tif', '.tiff'),
        )
    return pattern_path.parent

def write_binary_video_from_mask_volume(
    mask_u8: np.ndarray,
    video_path: Path,
    fps: float,
    show_progress: bool = True,
    scratch_dir: Optional[Path] = None,
    publication_root: Optional[Path] = None,
) -> Path:
    """Write a binary FFV1 MKV, optionally as concurrent losslessly concatenated t shards."""
    T, H, W = mask_u8.shape
    def _encode_range(
        start: int,
        stop: int,
        shard_path: Path,
        cancel_event: threading.Event,
        register_proc: Callable[[subprocess.Popen], None],
        unregister_proc: Callable[[subprocess.Popen], None],
        progress_one: Callable[[], None],
    ) -> None:
        proc = ffmpeg_ffv1_gray_writer(
            shard_path,
            width=W,
            height=H,
            fps=fps,
        )
        register_proc(proc)
        # One buffer per producer; the K producers only read the shared mask volume.
        frame_buf = np.empty((int(H), int(W)), dtype=np.uint8)
        primary_error: Optional[BaseException] = None
        primary_traceback = None
        try:
            assert proc.stdin is not None
            for t in range(int(start), int(stop)):
                if cancel_event.is_set():
                    break
                np.multiply(np.asarray(mask_u8[t]), 255, out=frame_buf)
                proc.stdin.write(memoryview(frame_buf).cast('B'))
                progress_one()
        except BaseException as exc:
            primary_error = exc
            primary_traceback = exc.__traceback__
        try:
            close_ffmpeg_writer(proc)
        except BaseException as exc:
            if primary_error is None:
                primary_error = exc
                primary_traceback = exc.__traceback__
        finally:
            unregister_proc(proc)
        if primary_error is not None:
            raise primary_error.with_traceback(primary_traceback)

    _run_sharded_ffv1_encode(
        out_path=Path(video_path),
        total_frames=int(T),
        description=f"Writing binary MKV ({video_path.name})",
        show_progress=bool(show_progress),
        encode_range=_encode_range,
        scratch_dir=scratch_dir,
        publication_root=publication_root,
    )
    return video_path

def _nrrd_ascii_header_text(value: object) -> str:
    """Return an ASCII-safe representation for pynrrd header fields."""
    text = str(value)
    replacements = {
        '°': 'deg',
        '±': '+/-',
        'µ': 'u',
        '–': '-',
        '—': '-',
        '−': '-',
        '\u00a0': ' ',
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    text = text.encode('ascii', 'ignore').decode('ascii')
    return text

def _nrrd_space_directions_matrix(
    spatial_axes: int = 3,
    list_axis: bool = False,
    list_axis_position: str = 'first',
) -> np.ndarray:
    """Return a NRRD ``space directions`` matrix with optional non-spatial list axis.

 Non-spatial axes are represented by a row of NaNs, which serializes as
 ``none``. ``list_axis_position='last'`` is used by decomposed segmentation
 NRRDs so the on-disk byte stream is one complete ``(t,Y,X)`` layer block
 followed by the next layer."""
    spatial_axes_i = max(1, int(spatial_axes))
    if bool(list_axis):
        position = str(list_axis_position).strip().lower()
        mat = np.full((spatial_axes_i + 1, spatial_axes_i), np.nan, dtype=np.float64)
        if position == 'last':
            mat[:spatial_axes_i, :] = np.eye(spatial_axes_i, dtype=np.float64)
        elif position == 'first':
            mat[1:, :] = np.eye(spatial_axes_i, dtype=np.float64)
        else:
            raise ValueError("list_axis_position must be 'first' or 'last'")
        return mat
    return np.eye(spatial_axes_i, dtype=np.float64)

def nrrd_slicer_header(mask_shape_zyx: Tuple[int, int, int]) -> Dict[str, object]:
    t_dim, h, w = (int(mask_shape_zyx[0]), int(mask_shape_zyx[1]), int(mask_shape_zyx[2]))
    return {
        "space": NRRD_SPACE,
        "kinds": ["domain", "domain", "domain"],
        "space directions": _nrrd_space_directions_matrix(spatial_axes=3, list_axis=False),
        "space origin": np.zeros((3,), dtype=np.float64),
        "content": f"binary segmentation mask; source_shape_tyx=({t_dim},{h},{w}); exported_axes=(X,Y,t)",
    }

_SLICER_SEGMENT_COLOR_PALETTE: Tuple[Tuple[float, float, float], ...] = (
    (0.83, 0.16, 0.16),  # red
    (0.12, 0.47, 0.71),  # blue
    (1.00, 0.60, 0.07),  # orange
    (0.17, 0.63, 0.17),  # green
    (0.58, 0.40, 0.74),  # purple
    (0.09, 0.75, 0.81),  # cyan
    (0.89, 0.47, 0.76),  # pink
    (0.74, 0.74, 0.13),  # olive
    (0.55, 0.34, 0.29),  # brown
    (0.98, 0.75, 0.37),  # light orange
    (0.68, 0.78, 0.91),  # light blue
    (0.60, 0.87, 0.54),  # light green
    (1.00, 0.60, 0.59),  # salmon
    (0.77, 0.69, 0.84),  # lavender
    (0.62, 0.85, 0.90),  # pale cyan
    (0.95, 0.90, 0.45),  # yellow
    (0.78, 0.58, 0.45),  # tan
    (0.96, 0.71, 0.82),  # light pink
    (0.47, 0.05, 0.53),  # violet
    (0.10, 0.35, 0.20),  # dark green
)

def _stable_layer_color_index(token: str) -> int:
    """Deterministic FNV-1a hash of a layer token.

 Python's builtin hash is salted per process, which would shuffle segment colors
 between runs; the palette pick must be reproducible for the same layer suffix."""
    h = 0x811C9DC5
    for b in str(token).encode('utf-8', errors='ignore'):
        h = ((h ^ int(b)) * 0x01000193) & 0xFFFFFFFF
    return int(h)

def slicer_segment_palette_color(token: str) -> Tuple[float, float, float]:
    """Deterministic palette color for a layer token (no in-run collision probing)."""
    palette = _SLICER_SEGMENT_COLOR_PALETTE
    return palette[_stable_layer_color_index(str(token)) % len(palette)]

def _slicer_segment_name_for_out_path(out_path: Path) -> str:
    name = Path(out_path).name
    for ext in ('.seg.nrrd', '.nrrd'):
        if name.lower().endswith(ext):
            return name[:-len(ext)]
    return Path(out_path).stem

def _slicer_segment_extent_for_output(
    ref: 'NrrdLayerRef',
    output_shape_tyx: Tuple[int, int, int],
) -> NrrdSegmentExtent:
    """Map the layer's backing-store segment extent into the output geometry.

 Returns the inclusive Slicer (minX maxX minY maxY minT maxT) extent in the exported
 (X,Y,t) axis order. Extents are expanded outward when the output geometry is scaled
 (e.g. low-quality downbins) and fall back to the full output extent when the backing
 extent is unknown or empty, so the header never understates where mask voxels live."""
    out_t, out_h, out_w = (int(output_shape_tyx[0]), int(output_shape_tyx[1]), int(output_shape_tyx[2]))
    full: NrrdSegmentExtent = (0, max(0, out_w - 1), 0, max(0, out_h - 1), 0, max(0, out_t - 1))
    extent = _coerce_segment_extent(getattr(ref, 'segment_extent_ijk', None))
    if extent is None:
        return full
    x0, x1, y0, y1, t0, t1 = (int(v) for v in extent)
    if x1 < x0 or y1 < y0 or t1 < t0:
        return full
    in_t, in_h, in_w = (int(v) for v in getattr(ref, 'segment_extent_shape_tyx', (0, 0, 0)))
    if in_t <= 0 or in_h <= 0 or in_w <= 0:
        return full

    def _axis(lo: int, hi: int, n_in: int, n_out: int) -> Tuple[int, int]:
        if int(n_in) != int(n_out):
            scale = float(n_out) / float(n_in)
            lo = int(math.floor(float(lo) * scale))
            hi = int(math.ceil((float(hi) + 1.0) * scale)) - 1
        return (max(0, min(int(lo), int(n_out) - 1)), max(0, min(int(hi), int(n_out) - 1)))

    x_lo, x_hi = _axis(x0, x1, in_w, out_w)
    y_lo, y_hi = _axis(y0, y1, in_h, out_h)
    # The payload restore is endpoint-aligned nearest-neighbour when temporal
    # upscaling and interval-OR when downscaling. A ratio-only extent transform
    # understates support (for example, source 1 in 3 -> 100 maps to 25..74,
    # not 33..66), and Slicer may discard payload outside Segment0_Extent.
    exact_t = _nrrd_output_z_extent_for_source_extent(
        int(in_t), int(out_t), int(t0), int(t1),
    )
    if exact_t is None:
        return full
    t_lo, t_hi = (int(exact_t[0]), int(exact_t[1]))
    return (x_lo, x_hi, y_lo, y_hi, t_lo, t_hi)

def _nrrd_output_z_extent_for_source_extent(
    in_t: int,
    out_t: int,
    source_z0: int,
    source_z1: int,
) -> Optional[Tuple[int, int]]:
    """Invert the exact payload restore map for a non-empty source-z interval.

 Temporal upscaling is endpoint-aligned nearest-neighbour, while downscaling ORs a
 floor/ceil coverage range. Ratio-only extent scaling is not equivalent (3 -> 100 with
 source z=1 is the canonical counterexample), so extent metadata uses the same mapping
 as the payload writer."""
    in_t_i = max(1, int(in_t))
    out_t_i = max(1, int(out_t))
    source_lo = max(0, min(int(source_z0), in_t_i - 1))
    source_hi = max(0, min(int(source_z1), in_t_i - 1))
    if source_hi < source_lo:
        return None
    first = -1
    last = -1
    for out_z in range(int(out_t_i)):
        source_indices = _restore_source_indices_for_output_z(
            int(in_t_i), int(out_t_i), int(out_z),
        )
        if not source_indices:
            continue
        if int(source_indices[-1]) < int(source_lo) or int(source_indices[0]) > int(source_hi):
            continue
        if first < 0:
            first = int(out_z)
        last = int(out_z)
    return None if first < 0 else (int(first), int(last))

def _nrrd_raster_plan(
    ref: 'NrrdLayerRef',
    output_shape_tyx: Tuple[int, int, int],
) -> NrrdRasterPlan:
    """Map a layer extent into an always-full output reference raster."""
    out_t, out_h, out_w = (int(v) for v in output_shape_tyx)
    raw_extent = _coerce_segment_extent(getattr(ref, 'segment_extent_ijk', None))
    empty = bool(
        raw_extent is not None
        and (
            int(raw_extent[1]) < int(raw_extent[0])
            or int(raw_extent[3]) < int(raw_extent[2])
            or int(raw_extent[5]) < int(raw_extent[4])
        )
    )
    segment_extent = (
        _nrrd_empty_segment_extent()
        if bool(empty)
        else _slicer_segment_extent_for_output(ref, output_shape_tyx)
    )
    return NrrdRasterPlan(
        stored_shape_tyx=(out_t, out_h, out_w),
        segment_extent_xyt=segment_extent,
        empty_segment=bool(empty),
    )

def _nrrd_mapped_extent_preserve_empty(
    ref: 'NrrdLayerRef', output_shape_tyx: Tuple[int, int, int],
) -> NrrdSegmentExtent:
    extent = _coerce_segment_extent(getattr(ref, 'segment_extent_ijk', None))
    if extent is not None and (
        int(extent[1]) < int(extent[0])
        or int(extent[3]) < int(extent[2])
        or int(extent[5]) < int(extent[4])
    ):
        return _nrrd_empty_segment_extent()
    return _slicer_segment_extent_for_output(ref, output_shape_tyx)

def slicer_segmentation_header_fields(
    *,
    segment_name: str,
    color_rgb: Tuple[float, float, float],
    extent_xyt: NrrdSegmentExtent,
) -> Dict[str, object]:
    """3D Slicer.seg.nrrd custom header fields for one single-segment binary labelmap.

 All keys are non-standard NRRD fields, so _write_nrrd_ascii_header emits them with the
 key:=value separator Slicer expects. Segment0_LabelValue matches the uint8 mask value 1."""
    r, g, b = (float(color_rgb[0]), float(color_rgb[1]), float(color_rgb[2]))
    return {
        'Segmentation_ContainedRepresentationNames': 'Binary labelmap|',
        'Segmentation_MasterRepresentation': 'Binary labelmap',
        'Segmentation_SourceRepresentation': 'Binary labelmap',
        'Segment0_ID': 'Segment_1',
        'Segment0_Name': str(segment_name),
        'Segment0_NameAutoGenerated': '0',
        'Segment0_Color': f'{r:.6g} {g:.6g} {b:.6g}',
        'Segment0_ColorAutoGenerated': '0',
        'Segment0_LabelValue': '1',
        'Segment0_Layer': '0',
        'Segment0_Extent': ' '.join(str(int(v)) for v in extent_xyt),
        'Segment0_Tags': (
            'TerminologyEntry:Segmentation category and type - 3D Slicer General Anatomy list'
            '~SCT^85756007^Tissue~SCT^85756007^Tissue~^^~Anatomic codes - DICOM master list~^^~^^|'
        ),
    }

def nrrd_gzip_compresslevel() -> int:
    """Return the NRRD member-gzip compression level.

    QAT maps this intent by hardware generation; ISA-L maps it to 0..3.
    """
    return int(np.clip(_env_int('YOLO_TTA_NRRD_GZIP_LEVEL', 3), 0, 9))

def nrrd_z_chunk_cap() -> int:
    """Cap on one NRRD payload block, in output t-slices.

 The RAM-scaled buffer previously resolved to the WHOLE payload on large-RAM nodes, so the
 writer filled an ~payload-sized block before compression began. Bounded
 blocks pipeline parallel fill with the selected gzip backend; two blocks
 per in-flight write are resident (double buffering)."""
    return max(1, _env_int('YOLO_TTA_NRRD_Z_CHUNK_SLICES', 128))

def nrrd_fill_workers() -> int:
    """Return the process-wide worker count used to fill NRRD payload blocks.

    Raw-bbox decoding and OpenCV restores release the GIL. The bounded shared pool prevents
    concurrent layer writers from multiplying memory-scanning threads.
    """
    return max(2, _env_int('YOLO_TTA_NRRD_FILL_WORKERS', max(2, min(32, _cpu_count() // 4))))

_NRRD_FILL_EXECUTOR: Optional[ThreadPoolExecutor] = None

_NRRD_FILL_EXECUTOR_LOCK = threading.Lock()

def _nrrd_fill_executor() -> ThreadPoolExecutor:
    global _NRRD_FILL_EXECUTOR
    with _NRRD_FILL_EXECUTOR_LOCK:
        if _NRRD_FILL_EXECUTOR is None:
            _NRRD_FILL_EXECUTOR = ThreadPoolExecutor(
                max_workers=int(nrrd_fill_workers()),
                thread_name_prefix='nrrd-fill',
            )
        return _NRRD_FILL_EXECUTOR

def _nrrd_parallel_fill_indices(
    count: int,
    func: Callable[[int], None],
    *,
    requested_workers: int,
) -> None:
    total = max(0, int(count))
    lanes = max(1, min(int(requested_workers), int(total))) if total > 0 else 0
    if lanes <= 1:
        for idx in range(total):
            func(int(idx))
        return
    # At most one range per requested lane prevents one sink writer from flooding the
    # shared FIFO with thousands of tiny tasks while preserving GIL-releasing slice work.
    chunk = max(1, int(math.ceil(float(total) / float(lanes))))

    def _run(start: int, stop: int) -> None:
        for idx in range(int(start), int(stop)):
            func(int(idx))

    executor = _nrrd_fill_executor()
    futures = [
        executor.submit(_run, int(start), int(min(total, start + chunk)))
        for start in range(0, total, chunk)
    ]
    first_error: Optional[BaseException] = None
    for fut in futures:
        try:
            fut.result()
        except BaseException as exc:
            if first_error is None:
                first_error = exc
    if first_error is not None:
        raise first_error

def nrrd_stream_buffer_bytes(required_bytes: Optional[int] = None) -> int:
    """Resolve the RAM budget for one NRRD payload slab.
    
    An explicit environment limit wins; otherwise the budget is a bounded fraction of current anonymous-memory headroom."""
    explicit = os.environ.get('YOLO_TTA_NRRD_STREAM_BUFFER_MIB', '').strip()
    if explicit:
        mib = max(1, _env_int('YOLO_TTA_NRRD_STREAM_BUFFER_MIB', 4096))
        target = int(mib) * 1024 * 1024
    else:
        min_mib = max(1, _env_int('YOLO_TTA_NRRD_STREAM_BUFFER_MIN_MIB', 4096))
        reserve_gib = max(1.0, _env_float('YOLO_TTA_NRRD_STREAM_BUFFER_RESERVE_GIB', 192.0))
        max_gib = max(1.0, _env_float('YOLO_TTA_NRRD_STREAM_BUFFER_MAX_GIB', 384.0))
        fraction = min(0.90, max(0.01, _env_float('YOLO_TTA_NRRD_STREAM_BUFFER_FRACTION', 0.35)))
        avail = int(available_anon_work_bytes())
        usable = max(0, int(avail) - int(reserve_gib * GIB))
        target = int(max(int(min_mib) * 1024 * 1024, min(float(max_gib) * float(GIB), float(usable) * float(fraction))))
    if required_bytes is not None and int(required_bytes) > 0:
        target = min(int(target), int(required_bytes))
    return max(1, int(target))

def _nrrd_full_slice_z_chunk(layer_count: int, width: int, height: int, depth: int) -> int:
    full_slice_bytes = max(1, int(layer_count) * int(width) * int(height) * np.dtype(np.uint8).itemsize)
    full_payload_bytes = full_slice_bytes * max(1, int(depth))
    target = nrrd_stream_buffer_bytes(full_payload_bytes)
    if target < full_slice_bytes:
        return 1
    # current behavior: bound the block so parallel
    # fill and the selected gzip backend pipeline instead of materializing the
    # entire payload before compression starts.
    return max(1, min(int(depth), int(target // full_slice_bytes), int(nrrd_z_chunk_cap())))

def nrrd_madvise_dontneed_interval() -> int:
    # projected cvol pages are reread by the final union. Run 126080
    # spent ~40 s on its first completion frontier and stalled again at z≈1280 after
    # concurrent NRRD readers had discarded those pages. Keep them warm by default;
    # memory-constrained jobs can restore the former interval with an explicit value.
    return max(0, _env_int('YOLO_TTA_NRRD_MADVISE_DONTNEED_INTERVAL', 0))

def nrrd_gzip_workers() -> int:
    """Size of the shared complete-member compression pool."""
    cores = max(1, _cpu_count())
    return max(2, _env_int('YOLO_TTA_NRRD_GZIP_WORKERS', max(4, cores // 2)))

def nrrd_gzip_chunk_bytes() -> int:
    return max(1, _env_int('YOLO_TTA_NRRD_GZIP_CHUNK_MIB', 16)) * 1024 * 1024

# Historical single-pool state is retained as an inert compatibility symbol for the
# refactor inventory; v17.0.8 uses the backend-sized pool map below.
_NRRD_GZIP_EXECUTOR: Optional[ThreadPoolExecutor] = None

_NRRD_GZIP_EXECUTORS: Dict[Tuple[str, int], ThreadPoolExecutor] = {}

_NRRD_GZIP_EXECUTOR_LOCK = threading.Lock()

_NRRD_GZIP_EXECUTOR_ATEXIT_REGISTERED = False

def _nrrd_codec_worker_count(
    codec_spec: Tuple[str, int, Callable[[object], bytes]],
) -> int:
    """Bound hardware sessions by the binding's advertised usable capacity."""
    backend, _level, compressor = codec_spec
    requested = int(nrrd_gzip_workers())
    if not bool(getattr(compressor, 'hardware_backend', False)):
        return max(1, requested)
    limit = max(1, int(getattr(compressor, 'max_concurrency', 1)))
    env_name = (
        'YOLO_TTA_NRRD_QAT_WORKERS'
        if str(backend) == 'qat' else
        'YOLO_TTA_NRRD_IAA_WORKERS'
    )
    configured = max(1, _env_int(str(env_name), min(requested, limit)))
    return max(1, min(int(configured), int(limit)))

def _nrrd_gzip_executor(
    codec_spec: Tuple[str, int, Callable[[object], bytes]],
) -> ThreadPoolExecutor:
    """Return a shared pool sized for the selected CPU or hardware backend."""
    global _NRRD_GZIP_EXECUTOR_ATEXIT_REGISTERED
    backend = str(codec_spec[0])
    workers = int(_nrrd_codec_worker_count(codec_spec))
    key = (backend, workers)
    with _NRRD_GZIP_EXECUTOR_LOCK:
        executor = _NRRD_GZIP_EXECUTORS.get(key)
        if executor is None:
            executor = ThreadPoolExecutor(
                max_workers=int(workers), thread_name_prefix=f'nrrd-gzip-{backend}',
            )
            _NRRD_GZIP_EXECUTORS[key] = executor
        if not _NRRD_GZIP_EXECUTOR_ATEXIT_REGISTERED:
            atexit.register(shutdown_nrrd_gzip_executors)
            _NRRD_GZIP_EXECUTOR_ATEXIT_REGISTERED = True
        return executor

def _run_on_every_executor_thread(
    executor: ThreadPoolExecutor,
    workers: int,
    callback: Callable[[], None],
) -> None:
    """Use a barrier so one task is resident on every pool thread before callback."""
    count = max(1, int(workers))
    barrier = threading.Barrier(count)

    def _call() -> None:
        barrier.wait(timeout=60.0)
        callback()

    futures = [executor.submit(_call) for _ in range(count)]
    first_error: Optional[BaseException] = None
    for fut in futures:
        try:
            fut.result()
        except BaseException as exc:
            if first_error is None:
                first_error = exc
    if first_error is not None:
        raise first_error

def _report_nrrd_gzip_cleanup_failure(
    operation: str,
    exc: BaseException,
) -> None:
    """Best-effort cleanup diagnostics that cannot replace a pipeline exception."""
    try:
        runtime_telemetry().fallback(str(operation), exc)
    except BaseException:
        pass
    try:
        print(
            'Warning: Intel compression cleanup failed during '
            f'{operation} ({type(exc).__name__}: {exc}).'
        )
    except BaseException:
        pass

def _shutdown_nrrd_gzip_executor_safely(executor: ThreadPoolExecutor) -> None:
    """Drain an executor without allowing teardown failures to escape cleanup."""
    try:
        executor.shutdown(wait=True, cancel_futures=True)
    except TypeError:
        # Python/runtime-compatible fallback for executors without cancel_futures.
        try:
            executor.shutdown(wait=True)
        except BaseException as exc:
            _report_nrrd_gzip_cleanup_failure(
                'nrrd.compression.executor_shutdown', exc,
            )
    except BaseException as exc:
        _report_nrrd_gzip_cleanup_failure(
            'nrrd.compression.executor_shutdown', exc,
        )

def _record_nrrd_native_codec_stats(backend: str) -> None:
    """Publish and reset native hardware counters after a codec pool drains."""
    name = str(backend)
    if name not in {'qat', 'iaa'}:
        return
    try:
        from .intel_compression import native_stats

        stats = dict(native_stats(name, reset=True))
        if not stats:
            return
        telemetry = runtime_telemetry()
        telemetry.gauge(f'nrrd.compression.{name}.native_stats', stats)
        cumulative = {
            'input_bytes',
            'output_bytes',
            'logical_requests',
            'physical_members',
            'hardware_requests',
            'software_fallback_requests',
            'queue_busy_events',
            'failures',
            'sessions_created',
            'sessions_closed',
            'elapsed_ns',
        }
        for key in cumulative:
            value = stats.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            telemetry.add(f'nrrd.compression.{name}.native.{key}', value)
    except BaseException as exc:
        _report_nrrd_gzip_cleanup_failure(
            f'nrrd.compression.{name}.native_stats', exc,
        )

def _close_nrrd_gzip_executor_entry(
    backend: str,
    workers: int,
    executor: ThreadPoolExecutor,
) -> None:
    """Close one idle pool, including native state on hardware worker threads."""
    try:
        # CPython's private ThreadPoolExecutor exit hook may already have joined the
        # pool before ordinary atexit callbacks run. Native TLS destructors have then
        # run too, so there is no worker left on which to schedule explicit cleanup.
        if (
            str(backend) in {'qat', 'iaa'}
            and not bool(getattr(executor, '_shutdown', False))
        ):
            from .intel_compression import close_current_thread_state

            _run_on_every_executor_thread(
                executor, int(workers), close_current_thread_state,
            )
    except BaseException as exc:
        _report_nrrd_gzip_cleanup_failure(
            'nrrd.compression.thread_cleanup', exc,
        )
    finally:
        _shutdown_nrrd_gzip_executor_safely(executor)
        _record_nrrd_native_codec_stats(str(backend))

def _retire_nrrd_codec_executor(codec_spec: NrrdMemberCodecSpec) -> None:
    """Release a failed hardware codec's sessions immediately after preflight/KAT."""
    backend = str(codec_spec[0])
    workers = int(_nrrd_codec_worker_count(codec_spec))
    with _NRRD_GZIP_EXECUTOR_LOCK:
        executor = _NRRD_GZIP_EXECUTORS.pop((backend, workers), None)
    if executor is not None:
        _close_nrrd_gzip_executor_entry(backend, workers, executor)

def shutdown_nrrd_gzip_executors() -> None:
    """Drain pools and close native TLS sessions on their owning worker threads."""
    global _NRRD_MEMBER_GZIP_ANNOUNCED, _NRRD_CPU_DEFLATE_BACKEND_ANNOUNCED
    with _NRRD_GZIP_EXECUTOR_LOCK:
        entries = list(_NRRD_GZIP_EXECUTORS.items())
        _NRRD_GZIP_EXECUTORS.clear()
    try:
        for (backend, workers), executor in entries:
            _close_nrrd_gzip_executor_entry(str(backend), int(workers), executor)
    finally:
        # A subsequent embedded run gets a fresh hardware probe, KAT, thread preflight,
        # and selected-backend announcement even when its environment policy changed.
        with _NRRD_MEMBER_GZIP_TEST_LOCK:
            _NRRD_MEMBER_GZIP_OK.clear()
            _NRRD_MEMBER_GZIP_FAILURE_REASONS.clear()
        _NRRD_MEMBER_CODEC_FAILURES_ANNOUNCED.clear()
        _NRRD_MEMBER_GZIP_ANNOUNCED = False
        _NRRD_CPU_DEFLATE_BACKEND_ANNOUNCED = False

_GZIP_MEMBER_HEADER = b'\x1f\x8b\x08\x00\x00\x00\x00\x00\x00\x03'

_NRRD_ZERO_MEMBER_CACHE: Dict[int, bytes] = {}

_NRRD_ZERO_MEMBER_LOCK = threading.Lock()

def _gzip_member_from_raw_stream(raw_stream: bytes, crc: int, length: int) -> bytes:
    """Frame one raw RFC-1951 stream as a complete RFC-1952 gzip member."""
    return (
        _GZIP_MEMBER_HEADER
        + raw_stream
        + struct.pack('<II', int(crc) & 0xFFFFFFFF, int(length) & 0xFFFFFFFF)
    )

def _zero_gzip_member(length: int) -> bytes:
    """Return a cached complete gzip member representing ``length`` zero bytes."""
    ln = max(0, int(length))
    with _NRRD_ZERO_MEMBER_LOCK:
        cached = _NRRD_ZERO_MEMBER_CACHE.get(ln)
    if cached is not None:
        return cached
    zeros = bytes(ln)
    cobj = zlib.compressobj(1, zlib.DEFLATED, -15)
    raw = cobj.compress(zeros) + cobj.flush(zlib.Z_FINISH)
    member = _gzip_member_from_raw_stream(raw, zlib.crc32(zeros), ln)
    with _NRRD_ZERO_MEMBER_LOCK:
        _NRRD_ZERO_MEMBER_CACHE[ln] = member
    return member

_NRRD_MEMBER_CODEC_SETTING_WARNED = False

NrrdMemberCodecSpec = Tuple[str, int, Callable[[object], bytes]]

def nrrd_member_codec_requested() -> str:
    """NRRD gzip policy; ``cpu`` is the opt-out from preferred automatic QAT."""
    global _NRRD_MEMBER_CODEC_SETTING_WARNED
    raw = os.environ.get('YOLO_TTA_NRRD_MEMBER_CODEC', 'auto').strip().lower()
    if raw in {'auto', 'cpu', 'qat', 'iaa', 'libdeflate', 'isal', 'zlib'}:
        return str(raw)
    if not _NRRD_MEMBER_CODEC_SETTING_WARNED:
        _NRRD_MEMBER_CODEC_SETTING_WARNED = True
        print(
            f'Warning: invalid YOLO_TTA_NRRD_MEMBER_CODEC={raw!r}; using auto '
            '(QAT -> libdeflate -> ISA-L -> zlib).'
        )
    return 'auto'

def nrrd_libdeflate_level() -> int:
    """Resolve the libdeflate complete-member compression level.
    
    Level zero remains valid input to codec self-test so unsupported store mode can fall through safely."""
    return int(np.clip(
        _env_int('YOLO_TTA_NRRD_LIBDEFLATE_LEVEL', int(nrrd_gzip_compresslevel())),
        0,
        12,
    ))

def _nrrd_member_codec_candidates() -> Tuple[str, ...]:
    requested = str(nrrd_member_codec_requested())
    if requested == 'auto':
        return ('qat', 'libdeflate', 'isal', 'zlib')
    if requested == 'cpu':
        return ('libdeflate', 'isal', 'zlib')
    return (requested,)

def _nrrd_optional_numa_id(environment_name: str) -> Optional[int]:
    raw = os.environ.get(str(environment_name), '').strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f'{environment_name} must be an integer, got {raw!r}') from exc
    return None if value < 0 else int(value)

def _nrrd_qat_level(capabilities: Dict[str, object]) -> int:
    """Map zlib intent to Intel's generation-dependent QAT compression levels."""
    raw_supported = capabilities.get('supported_levels', tuple(range(1, 10)))
    try:
        if isinstance(raw_supported, (str, bytes)):
            supported = tuple(
                sorted({int(token.strip()) for token in str(raw_supported).split(',') if token.strip()})
            )
        else:
            supported = tuple(sorted({int(value) for value in raw_supported}))  # type: ignore[union-attr]
    except Exception:
        supported = tuple(range(1, 10))

    def _validated(level: int) -> int:
        if int(level) not in supported:
            raise ValueError(
                f'QAT level {int(level)} is unsupported; binding reports {supported}'
            )
        return int(level)

    explicit = os.environ.get('YOLO_TTA_NRRD_QAT_LEVEL', '').strip()
    if explicit:
        try:
            level = int(explicit)
        except ValueError as exc:
            raise ValueError(
                f'YOLO_TTA_NRRD_QAT_LEVEL must be an integer, got {explicit!r}'
            ) from exc
        if not 1 <= int(level) <= 9:
            raise ValueError('YOLO_TTA_NRRD_QAT_LEVEL must be in [1, 9]')
        return _validated(int(level))
    configured = int(nrrd_gzip_compresslevel())
    if configured <= 0:
        raise RuntimeError(
            'QAT auto selection is ineligible for gzip level 0; choose '
            'YOLO_TTA_NRRD_MEMBER_CODEC=cpu or set an explicit QAT level'
        )
    generation = str(
        capabilities.get('hardware_generation', capabilities.get('qat_generation', ''))
    ).strip().lower()
    qat_1x = bool(
        generation.startswith('1')
        or generation.startswith('qat1')
        or 'qat 1.' in generation
    )
    qat_2plus = bool(
        generation.startswith(('2', '3', '4'))
        or generation.startswith(('qat2', 'qat3', 'qat4'))
        or any(f'qat {major}.' in generation for major in (2, 3, 4))
    )
    if configured <= 4:
        return _validated(1)
    if not qat_1x and not qat_2plus:
        raise RuntimeError(
            'QAT binding must report hardware_generation to map gzip levels above 4; '
            'set YOLO_TTA_NRRD_QAT_LEVEL explicitly to override'
        )
    if configured == 5:
        return _validated(5 if qat_1x else 1)
    if configured <= 8:
        return _validated(5 if qat_1x else 6)
    return _validated(9)

def _nrrd_iaa_level(capabilities: Dict[str, object]) -> int:
    configured = int(nrrd_gzip_compresslevel())
    default = 1 if configured <= 5 else 3
    level = int(_env_int('YOLO_TTA_NRRD_IAA_LEVEL', int(default)))
    raw_supported = capabilities.get('supported_levels', (1, 3))
    try:
        supported = tuple(sorted({int(value) for value in raw_supported}))  # type: ignore[union-attr]
    except Exception:
        supported = (1, 3)
    if int(level) not in supported:
        raise ValueError(
            f'YOLO_TTA_NRRD_IAA_LEVEL={level} is unsupported; binding reports {supported}'
        )
    return int(level)

def _nrrd_member_codec_spec(
    codec_name: str,
) -> NrrdMemberCodecSpec:
    """Resolve one gzip-member-sequence codec without silently substituting."""
    name = str(codec_name).strip().lower()
    configured_level = int(nrrd_gzip_compresslevel())
    if name in {'qat', 'iaa'}:
        from .intel_compression import create_gzip_compressor, probe_capabilities

        if name == 'qat' and _env_flag('YOLO_TTA_NRRD_QAT_SW_FALLBACK', False):
            raise RuntimeError(
                'YOLO_TTA_NRRD_QAT_SW_FALLBACK must remain disabled; '
                'automatic fallback occurs only before an NRRD opens'
            )
        capabilities = probe_capabilities(name)
        if name == 'qat':
            level = int(_nrrd_qat_level(capabilities))
            numa_id = _nrrd_optional_numa_id('YOLO_TTA_NRRD_QAT_NUMA_ID')
        else:
            level = int(_nrrd_iaa_level(capabilities))
            numa_id = _nrrd_optional_numa_id('YOLO_TTA_NRRD_IAA_NUMA_ID')
        compressor = create_gzip_compressor(
            name, int(level), numa_id=numa_id, capabilities=capabilities,
        )
        return str(name), int(level), compressor
    if name == 'libdeflate':
        import deflate  # type: ignore

        level = int(nrrd_libdeflate_level())
        gzip_compress = getattr(deflate, 'gzip_compress')

        def _compress_libdeflate(payload: bytes) -> bytes:
            # Positional level is intentional: this is the python-deflate public API and
            # ensures the exact one-shot call shape is covered by the known-answer test.
            return bytes(gzip_compress(payload, int(level)))

        return 'libdeflate', int(level), _compress_libdeflate
    if name == 'isal':
        from isal import isal_zlib  # type: ignore

        level = 0 if configured_level <= 0 else (
            1 if configured_level <= 4 else (2 if configured_level <= 7 else 3)
        )
        level = int(np.clip(_env_int('YOLO_TTA_NRRD_ISAL_LEVEL', int(level)), 0, 3))

        def _compress_isal(payload: bytes) -> bytes:
            cobj = isal_zlib.compressobj(
                int(level),
                isal_zlib.DEFLATED,
                16 + int(getattr(isal_zlib, 'MAX_WBITS', 15)),
            )
            return bytes(cobj.compress(payload) + cobj.flush())

        return 'isal', int(level), _compress_isal
    if name == 'zlib':
        level = int(configured_level)

        def _compress_zlib(payload: bytes) -> bytes:
            cobj = zlib.compressobj(int(level), zlib.DEFLATED, 16 + int(zlib.MAX_WBITS))
            return bytes(cobj.compress(payload) + cobj.flush())

        return 'zlib', int(level), _compress_zlib
    raise ValueError(f'Unsupported NRRD member codec {codec_name!r}')

def nrrd_member_gzip_window_bytes() -> int:
    """Uncompressed bytes one layer writer may hold in flight (copied chunks awaiting deflate)."""
    return max(64, _env_int('YOLO_TTA_NRRD_MEMBER_GZIP_WINDOW_MIB', 512)) * 1024 * 1024

class _MemberParallelGzipPayloadWriter:
    """Pipelined encoder emitting one complete gzip member sequence per chunk.

 ``write(data)`` detaches chunks before deferring them, so callers may immediately reuse
 their buffers. Completion is unordered, but a sequence-indexed ready map drains the
 longest available ordered prefix. This prevents a slow early member from hiding later
 completed work while preserving byte-exact payload order."""

    def __init__(
        self,
        fh: object,
        *,
        chunk_bytes: Optional[int] = None,
        codec_spec: Optional[NrrdMemberCodecSpec] = None,
    ) -> None:
        resolved = codec_spec if codec_spec is not None else _select_nrrd_member_codec()
        if resolved is None:
            raise RuntimeError('No validated complete-member NRRD gzip codec is available')
        self.codec_spec = resolved
        self.fh = fh
        self.backend = str(resolved[0])
        self.eff_level = int(resolved[1])
        self.member_compress = resolved[2]
        self.minimum_input_bytes = max(
            1, int(getattr(self.member_compress, 'minimum_input_bytes', 1))
        )
        self.chunk_bytes = int(chunk_bytes) if chunk_bytes else int(nrrd_gzip_chunk_bytes())
        self.window_bytes = int(nrrd_member_gzip_window_bytes())
        self.closed = False
        self._next_sequence = 0
        self._next_write_sequence = 0
        self._pending: Dict[Future, Tuple[int, int]] = {}
        self._completed: Dict[int, Tuple[bytes, int]] = {}
        self._inflight_bytes = 0
        self._small_pending = bytearray()
        self._hardware_lookbehind: Optional[object] = None
        self._hardware_lookbehind_is_zero = False

    def _compress_member(self, payload: object) -> Tuple[bytes, int]:
        """Encode in one native gzip pass; CRC is produced by ISA-L/zlib itself.

 Known zeros enter through ``write_zeros`` from extent/cvol metadata. Unknown data is
 deliberately not pre-scanned: native DEFLATE handles an unexpected zero member more
 cheaply than adding a full memory pass to every nonzero member."""
        mv = payload if isinstance(payload, memoryview) else memoryview(payload)  # type: ignore[arg-type]
        mv = mv.cast('B')
        ln = int(len(mv))
        telemetry = runtime_telemetry()
        telemetry.add(f'nrrd.compression.{self.backend}.input_bytes', int(ln))
        try:
            with telemetry.span(f'nrrd.compression.{self.backend}.compress'):
                try:
                    member_sequence = bytes(self.member_compress(mv))
                except TypeError:
                    if bool(getattr(self.member_compress, 'hardware_backend', False)):
                        # A hardware call may already have completed before a binding
                        # raises while validating status/output. Never resubmit it.
                        raise
                    # A codec wheel without generic buffer-protocol support is still
                    # valid; make its one required copy inside the worker, never on the
                    # sparse assembly thread.
                    member_sequence = bytes(self.member_compress(bytes(mv)))
        except Exception:
            telemetry.add(f'nrrd.compression.{self.backend}.failures', 1)
            raise
        telemetry.add(f'nrrd.compression.{self.backend}.output_bytes', len(member_sequence))
        telemetry.add(f'nrrd.compression.{self.backend}.logical_requests', 1)
        return member_sequence, int(ln)

    def _enqueue_completed(self, member: bytes) -> None:
        seq = int(self._next_sequence)
        self._next_sequence += 1
        self._completed[seq] = (member, 0)

    def _enqueue_chunk(self, chunk_mv: memoryview) -> None:
        seq = int(self._next_sequence)
        self._next_sequence += 1
        payload = bytes(chunk_mv)  # detach once; classification no longer adds a NumPy scan
        ln = int(len(payload))
        fut = _nrrd_gzip_executor(self.codec_spec).submit(
            self._compress_member, payload,
        )
        self._pending[fut] = (int(seq), int(ln))
        self._inflight_bytes += int(ln)

    def _enqueue_owned_chunk(self, owner: object) -> None:
        """Transfer one caller-owned contiguous allocation to a compressor worker.

 The Future retains ``mv`` (and therefore its NumPy exporter) until native DEFLATE
 finishes. This removes the former 8--16 MiB ``bytes`` detachment copy for every
 sparse cvol member while preserving the ordinary write buffer-reuse contract."""
        mv = owner if isinstance(owner, memoryview) else memoryview(owner)  # type: ignore[arg-type]
        mv = mv.cast('B')
        seq = int(self._next_sequence)
        self._next_sequence += 1
        ln = int(len(mv))
        fut = _nrrd_gzip_executor(self.codec_spec).submit(
            self._compress_member, mv,
        )
        self._pending[fut] = (int(seq), int(ln))
        self._inflight_bytes += int(ln)

    def _hardware_lookbehind_enabled(self) -> bool:
        return bool(
            getattr(self.member_compress, 'hardware_backend', False)
            and int(self.minimum_input_bytes) > 1
        )

    def _submit_hardware_lookbehind(self) -> None:
        owner = self._hardware_lookbehind
        if owner is None:
            return
        is_zero = bool(self._hardware_lookbehind_is_zero)
        self._hardware_lookbehind = None
        self._hardware_lookbehind_is_zero = False
        if is_zero:
            self._enqueue_completed(_zero_gzip_member(len(memoryview(owner).cast('B'))))
        else:
            self._enqueue_owned_chunk(owner)

    def _queue_data_chunk(self, chunk: object, *, owned: bool = False) -> None:
        """Queue data, retaining one final hardware request for cross-write tails."""
        mv = chunk if isinstance(chunk, memoryview) else memoryview(chunk)  # type: ignore[arg-type]
        mv = mv.cast('B')
        if not self._hardware_lookbehind_enabled():
            if owned:
                self._enqueue_owned_chunk(mv)
            else:
                self._enqueue_chunk(mv)
            return
        self._submit_hardware_lookbehind()
        # Ordinary write() promises that its caller may immediately reuse the source;
        # the owned path deliberately retains its exporter instead.
        self._hardware_lookbehind = mv if owned else bytes(mv)
        self._hardware_lookbehind_is_zero = False

    def _collect_completions(self, *, block: bool) -> int:
        if not self._pending:
            return 0
        if bool(block):
            done, _not_done = wait(set(self._pending), return_when=FIRST_COMPLETED)
        else:
            done = {fut for fut in self._pending if fut.done()}
        for fut in done:
            seq, charged = self._pending.pop(fut)
            member, ln = fut.result()
            if int(ln) != int(charged):
                raise RuntimeError(f'NRRD member length mismatch: {ln} != {charged}')
            self._completed[int(seq)] = (member, int(charged))
        return int(len(done))

    def _write_ready_prefix(self) -> int:
        written = 0
        while int(self._next_write_sequence) in self._completed:
            member, charged = self._completed.pop(int(self._next_write_sequence))
            self.fh.write(member)
            self._inflight_bytes -= int(charged)
            self._next_write_sequence += 1
            written += 1
        return int(written)

    def _drain(self, *, block: bool) -> None:
        self._collect_completions(block=False)
        self._write_ready_prefix()
        while self._pending and (bool(block) or self._inflight_bytes > self.window_bytes):
            self._collect_completions(block=True)
            self._write_ready_prefix()
        if bool(block):
            self._write_ready_prefix()

    def _write_impl(self, data: bytes | bytearray | memoryview) -> int:
        if self.closed:
            raise RuntimeError('Cannot write to a closed gzip payload stream')
        mv = data if isinstance(data, memoryview) else memoryview(data)
        mv = mv.cast('B')
        total = len(mv)
        off = 0
        if self._small_pending:
            take = min(total, int(self.minimum_input_bytes) - len(self._small_pending))
            if take > 0:
                self._small_pending.extend(mv[:take])
                off += int(take)
            if len(self._small_pending) >= int(self.minimum_input_bytes):
                self._queue_data_chunk(memoryview(bytes(self._small_pending)))
                self._small_pending.clear()
                self._drain(block=False)
        remaining = int(total - off)
        if 0 < remaining < int(self.minimum_input_bytes):
            self._small_pending.extend(mv[off:])
            return int(total)
        while off < total:
            remaining = int(total - off)
            ln = min(int(self.chunk_bytes), remaining)
            tail = int(remaining - ln)
            # QATzip cannot prove hardware execution for sub-threshold tails. Merge a
            # tiny tail into the preceding logical request instead of allowing SW work.
            if 0 < tail < int(self.minimum_input_bytes):
                ln = int(remaining)
            self._queue_data_chunk(mv[off:off + ln])
            off += int(ln)
            self._drain(block=False)
        return int(total)

    def write(self, data: bytes | bytearray | memoryview) -> int:
        try:
            return self._write_impl(data)
        except BaseException:
            self.closed = True
            self._abandon_and_settle()
            raise

    def write_known_nonzero(self, data: bytes | bytearray | memoryview) -> int:
        """Encode cvol-index-classified bytes without a redundant all-zero scan."""
        return self.write(data)

    def write_owned_known_nonzero(self, data: object) -> int:
        """Ownership-transfer fast path for one already slice-aligned member."""
        if self.closed:
            raise RuntimeError('Cannot write to a closed gzip payload stream')
        mv = data if isinstance(data, memoryview) else memoryview(data)  # type: ignore[arg-type]
        mv = mv.cast('B')
        if len(mv) <= 0:
            return 0
        if self._small_pending or len(mv) < int(self.minimum_input_bytes):
            return self.write(mv)
        try:
            self._queue_data_chunk(mv, owned=True)
            self._drain(block=False)
        except BaseException:
            self.closed = True
            self._abandon_and_settle()
            raise
        return int(len(mv))

    def write_aligned_zeros(self, nbytes: int) -> int:
        """Emit one complete cached zero member matching a whole-slice sparse member."""
        if self.closed:
            raise RuntimeError('Cannot write to a closed gzip payload stream')
        requested = int(nbytes)
        ln = int(requested)
        if ln <= 0:
            return 0
        if self._small_pending:
            take = min(ln, int(self.minimum_input_bytes) - len(self._small_pending))
            self._small_pending.extend(bytes(int(take)))
            ln -= int(take)
            if len(self._small_pending) >= int(self.minimum_input_bytes):
                self._queue_data_chunk(memoryview(bytes(self._small_pending)))
                self._small_pending.clear()
                self._drain(block=False)
        if ln <= 0:
            return int(nbytes)
        if self._hardware_lookbehind_enabled() and ln < int(self.minimum_input_bytes):
            if self._hardware_lookbehind is None:
                self._small_pending.extend(bytes(int(ln)))
            else:
                self._hardware_lookbehind = (
                    bytes(memoryview(self._hardware_lookbehind).cast('B'))
                    + bytes(int(ln))
                )
            return int(requested)
        self._submit_hardware_lookbehind()
        if self._hardware_lookbehind_enabled():
            # Retain the final hardware-sized zero span so a later sub-threshold
            # nonzero write can be appended without reordering bytes or falling back
            # to software. Most of a large zero run still uses the cached member.
            cached = int(ln) - int(self.minimum_input_bytes)
            if cached > 0:
                self._enqueue_completed(_zero_gzip_member(int(cached)))
            self._hardware_lookbehind = bytes(int(self.minimum_input_bytes))
            self._hardware_lookbehind_is_zero = True
        else:
            self._enqueue_completed(_zero_gzip_member(int(ln)))
        self._drain(block=False)
        return int(requested)

    def write_zeros(self, nbytes: int) -> int:
        """Emit cached all-zero members without materializing the zeros."""
        if self.closed:
            raise RuntimeError('Cannot write to a closed gzip payload stream')
        remaining = int(nbytes)
        if self._small_pending and remaining > 0:
            take = min(remaining, int(self.minimum_input_bytes) - len(self._small_pending))
            self._small_pending.extend(bytes(int(take)))
            remaining -= int(take)
            if len(self._small_pending) >= int(self.minimum_input_bytes):
                self._queue_data_chunk(memoryview(bytes(self._small_pending)))
                self._small_pending.clear()
                self._drain(block=False)
        if (
            self._hardware_lookbehind_enabled()
            and 0 < remaining < int(self.minimum_input_bytes)
        ):
            if self._hardware_lookbehind is None:
                self._small_pending.extend(bytes(int(remaining)))
            else:
                self._hardware_lookbehind = (
                    bytes(memoryview(self._hardware_lookbehind).cast('B'))
                    + bytes(int(remaining))
                )
            remaining = 0
        if remaining > 0:
            self._submit_hardware_lookbehind()
        if self._hardware_lookbehind_enabled() and remaining > 0:
            # Keep the logical tail of the zero run unresolved. If the next call is
            # tiny, close() merges it into this hardware-eligible span; if there is no
            # next call, _submit_hardware_lookbehind() emits it from the zero cache.
            retained = int(self.minimum_input_bytes)
            cached_remaining = int(remaining) - retained
            while cached_remaining > 0:
                ln = min(self.chunk_bytes, cached_remaining)
                self._enqueue_completed(_zero_gzip_member(int(ln)))
                cached_remaining -= int(ln)
                self._drain(block=False)
            self._hardware_lookbehind = bytes(retained)
            self._hardware_lookbehind_is_zero = True
            remaining = 0
        while remaining > 0:
            ln = min(self.chunk_bytes, remaining)
            self._enqueue_completed(_zero_gzip_member(int(ln)))
            remaining -= int(ln)
            self._drain(block=False)
        return int(nbytes)

    def close(self) -> None:
        if self.closed:
            return
        if self._small_pending:
            if self._hardware_lookbehind is not None:
                combined = bytes(memoryview(self._hardware_lookbehind).cast('B')) + bytes(
                    self._small_pending
                )
                self._hardware_lookbehind = None
                self._hardware_lookbehind_is_zero = False
                self._small_pending.clear()
                self._queue_data_chunk(memoryview(combined), owned=True)
            elif len(self._small_pending) < int(self.minimum_input_bytes):
                pending_bytes = int(len(self._small_pending))
                self.closed = True
                self._abandon_and_settle()
                raise RuntimeError(
                    f'{self.backend} cannot encode the final {pending_bytes}-byte '
                    f'payload entirely in hardware (minimum {self.minimum_input_bytes})'
                )
            else:
                self._queue_data_chunk(memoryview(bytes(self._small_pending)))
                self._small_pending.clear()
        self._submit_hardware_lookbehind()
        self.closed = True
        try:
            self._drain(block=True)
        except BaseException:
            self._abandon_and_settle()
            raise
        if self._pending or self._completed or int(self._next_write_sequence) != int(self._next_sequence):
            raise RuntimeError('NRRD member completion map did not drain completely')

    def _abandon_and_settle(self) -> None:
        """Retain inputs and wait until no failed native/DMA request can still run."""
        pending = list(self._pending)
        for fut in pending:
            fut.cancel()
        for fut in pending:
            try:
                fut.result()
            except BaseException:
                pass
        self._pending.clear()
        self._completed.clear()
        self._small_pending.clear()
        self._hardware_lookbehind = None
        self._hardware_lookbehind_is_zero = False
        self._inflight_bytes = 0

    def __enter__(self) -> '_MemberParallelGzipPayloadWriter':
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        if exc_type is None:
            self.close()
            return
        # Broken stream: the file is already failed — abandon pending members.
        self.closed = True
        self._abandon_and_settle()

_NRRD_MEMBER_GZIP_OK: Dict[Tuple[str, int], bool] = {}

_NRRD_MEMBER_GZIP_FAILURE_REASONS: Dict[Tuple[str, int], str] = {}

_NRRD_MEMBER_GZIP_TEST_LOCK = threading.Lock()

_NRRD_MEMBER_GZIP_ANNOUNCED = False

_NRRD_MEMBER_CODEC_FAILURES_ANNOUNCED: set = set()

_NRRD_CPU_DEFLATE_BACKEND_ANNOUNCED = False

_NRRD_CPU_DEFLATE_BACKEND_ANNOUNCE_LOCK = threading.Lock()

def _after_nrrd_compression_fork_child() -> None:
    """Discard parent compressor pools, locks, and policy caches in a fork child."""
    global _NRRD_GZIP_EXECUTOR, _NRRD_GZIP_EXECUTORS
    global _NRRD_GZIP_EXECUTOR_LOCK, _NRRD_ZERO_MEMBER_LOCK
    global _NRRD_ZERO_MEMBER_CACHE, _NRRD_MEMBER_CODEC_SETTING_WARNED
    global _NRRD_MEMBER_GZIP_OK, _NRRD_MEMBER_GZIP_FAILURE_REASONS
    global _NRRD_MEMBER_GZIP_TEST_LOCK, _NRRD_MEMBER_GZIP_ANNOUNCED
    global _NRRD_MEMBER_CODEC_FAILURES_ANNOUNCED
    global _NRRD_CPU_DEFLATE_BACKEND_ANNOUNCED
    global _NRRD_CPU_DEFLATE_BACKEND_ANNOUNCE_LOCK

    # No ThreadPoolExecutor worker survives fork(), and any inherited lock may have
    # been held by a vanished parent thread. Never call shutdown() or acquire one of
    # those locks in the child; replace the complete process-local state directly.
    _NRRD_GZIP_EXECUTOR = None
    _NRRD_GZIP_EXECUTORS = {}
    _NRRD_GZIP_EXECUTOR_LOCK = threading.Lock()
    _NRRD_ZERO_MEMBER_CACHE = {}
    _NRRD_ZERO_MEMBER_LOCK = threading.Lock()
    _NRRD_MEMBER_CODEC_SETTING_WARNED = False
    _NRRD_MEMBER_GZIP_OK = {}
    _NRRD_MEMBER_GZIP_FAILURE_REASONS = {}
    _NRRD_MEMBER_GZIP_TEST_LOCK = threading.Lock()
    _NRRD_MEMBER_GZIP_ANNOUNCED = False
    _NRRD_MEMBER_CODEC_FAILURES_ANNOUNCED = set()
    _NRRD_CPU_DEFLATE_BACKEND_ANNOUNCED = False
    _NRRD_CPU_DEFLATE_BACKEND_ANNOUNCE_LOCK = threading.Lock()
    # _NRRD_GZIP_EXECUTOR_ATEXIT_REGISTERED intentionally remains unchanged:
    # Python's atexit registry is inherited, so the child already owns that callback.

if hasattr(os, 'register_at_fork'):
    os.register_at_fork(after_in_child=_after_nrrd_compression_fork_child)

def _announce_nrrd_cpu_deflate_backend(
    tier: str,
    *,
    codec_spec: NrrdMemberCodecSpec,
) -> None:
    """Report the actual encoder selected after hardware proof/KAT, once per process."""
    global _NRRD_CPU_DEFLATE_BACKEND_ANNOUNCED
    if _NRRD_CPU_DEFLATE_BACKEND_ANNOUNCED:
        return
    with _NRRD_CPU_DEFLATE_BACKEND_ANNOUNCE_LOCK:
        if _NRRD_CPU_DEFLATE_BACKEND_ANNOUNCED:
            return
        backend, level, _compress = codec_spec
        display = {
            'qat': 'Intel QAT/QATzip hardware',
            'iaa': 'Intel IAA/QPL hardware',
            'libdeflate': 'libdeflate (python-deflate)',
            'isal': 'ISA-L (python-isal)',
            'zlib': 'zlib',
        }.get(str(backend), str(backend))
        requested = str(nrrd_member_codec_requested())
        print(
            f'NRRD DEFLATE backend selected: {display}, level={int(level)}, '
            f'policy={requested}, tier={tier}.'
        )
        telemetry = runtime_telemetry()
        telemetry.gauge('nrrd.compression.requested_backend', requested)
        telemetry.gauge('nrrd.compression.selected_backend', str(backend))
        telemetry.gauge('nrrd.compression.effective_level', int(level))
        capabilities = getattr(_compress, 'capabilities', None)
        if isinstance(capabilities, dict):
            telemetry.gauge(f'nrrd.compression.{backend}.capabilities', capabilities)
        _NRRD_CPU_DEFLATE_BACKEND_ANNOUNCED = True

def _preflight_nrrd_codec_threads(codec_spec: NrrdMemberCodecSpec) -> None:
    """Initialize every hardware pool worker before any logical NRRD is opened."""
    compressor = codec_spec[2]
    if not bool(getattr(compressor, 'hardware_backend', False)):
        return
    preflight = getattr(compressor, 'preflight_thread_state', None)
    if not callable(preflight):
        raise RuntimeError(
            f'{codec_spec[0]} hardware codec does not expose thread-session preflight'
        )
    workers = int(_nrrd_codec_worker_count(codec_spec))
    executor = _nrrd_gzip_executor(codec_spec)
    _run_on_every_executor_thread(executor, workers, preflight)

def _nrrd_member_codec_test_key(codec_spec: NrrdMemberCodecSpec) -> Tuple[str, int]:
    compressor_identity = getattr(codec_spec[2], 'cache_key', str(codec_spec[0]))
    return (repr(compressor_identity), int(codec_spec[1]))

def _nrrd_member_codec_self_test(
    codec_spec: NrrdMemberCodecSpec,
) -> bool:
    """KAT one codec's complete framing plus the member writer's ordered/zero paths."""
    key = _nrrd_member_codec_test_key(codec_spec)
    cached = _NRRD_MEMBER_GZIP_OK.get(key)
    if cached is not None:
        return bool(cached)
    with _NRRD_MEMBER_GZIP_TEST_LOCK:
        cached = _NRRD_MEMBER_GZIP_OK.get(key)
        if cached is not None:
            return bool(cached)
        failure_reason = ''
        try:
            import gzip as _gzip
            _preflight_nrrd_codec_threads(codec_spec)
            minimum_input = max(
                1, int(getattr(codec_spec[2], 'minimum_input_bytes', 1))
            )
            kat_chunk = max(4096, int(minimum_input))
            # Scale both nonzero phases from the binding's hardware threshold. Fixed
            # kilobyte-sized KAT data would incorrectly reject a valid accelerator whose
            # minimum request is larger. The short tail makes the writer merge it into
            # the preceding chunk instead of accidentally exercising software fallback.
            short_tail = 1 if int(minimum_input) <= 1 else min(3000, minimum_input - 1)
            part_a_size = int(kat_chunk) * 2 + int(short_tail)
            part_a_seed = bytes(range(256)) + b'\x00' * 31
            part_a = (
                part_a_seed * ((int(part_a_size) + len(part_a_seed) - 1) // len(part_a_seed))
            )[:int(part_a_size)]
            part_b = b'\x00' * 8192  # write_zeros: cached members only
            part_c_size = int(kat_chunk) + int(short_tail)
            part_c_seed = b'nrrd-member-gzip-self-test\x00'
            part_c = (
                part_c_seed * ((int(part_c_size) + len(part_c_seed) - 1) // len(part_c_seed))
            )[:int(part_c_size)]
            sink = io.BytesIO()
            writer = _MemberParallelGzipPayloadWriter(
                sink,
                chunk_bytes=int(kat_chunk),
                codec_spec=codec_spec,
            )
            writer.write(part_a)
            writer.write_zeros(len(part_b))
            writer.write(part_c)
            writer.close()
            ok = bool(_gzip.decompress(sink.getvalue()) == part_a + part_b + part_c)
            if not ok:
                failure_reason = 'known-answer gzip round trip returned different bytes'
        except Exception as exc:
            ok = False
            failure_reason = f'{type(exc).__name__}: {exc}'
        if not bool(ok) and bool(getattr(codec_spec[2], 'hardware_backend', False)):
            _retire_nrrd_codec_executor(codec_spec)
        _NRRD_MEMBER_GZIP_OK[key] = bool(ok)
        if ok:
            _NRRD_MEMBER_GZIP_FAILURE_REASONS.pop(key, None)
        else:
            _NRRD_MEMBER_GZIP_FAILURE_REASONS[key] = (
                str(failure_reason) or 'known-answer/round-trip self-test failed'
            )
        return bool(ok)

def _select_nrrd_member_codec(
    *,
    expected_input_bytes: Optional[int] = None,
) -> Optional[NrrdMemberCodecSpec]:
    """First available and KAT-validated configured complete-member codec wins."""
    requested = str(nrrd_member_codec_requested())
    for name in _nrrd_member_codec_candidates():
        codec_loaded = False
        module_missing = False
        try:
            spec = _nrrd_member_codec_spec(str(name))
            codec_loaded = True
            minimum_input = max(1, int(getattr(spec[2], 'minimum_input_bytes', 1)))
            if (
                expected_input_bytes is not None
                and int(expected_input_bytes) < int(minimum_input)
            ):
                reason = (
                    f'logical payload is {int(expected_input_bytes)} bytes, below the '
                    f'{minimum_input}-byte hardware-only minimum'
                )
                raise RuntimeError(reason)
            if _nrrd_member_codec_self_test(spec):
                runtime_telemetry().gauge('nrrd.compression.requested_backend', requested)
                runtime_telemetry().gauge('nrrd.compression.selected_backend', str(spec[0]))
                return spec
            reason = _NRRD_MEMBER_GZIP_FAILURE_REASONS.get(
                _nrrd_member_codec_test_key(spec),
                'known-answer/round-trip self-test failed',
            )
        except Exception as exc:
            reason = str(exc) or type(exc).__name__
            module_missing = bool(getattr(exc, 'module_missing', False))
        # Missing optional codecs are expected in auto mode. Forced selection and actual
        # codec corruption are announced once, without preventing the lower gzip tiers.
        policy_chain = requested in {'auto', 'cpu'}
        announce = bool(
            not policy_chain
            or codec_loaded
            or str(name) == 'zlib'
            or (str(name) in {'qat', 'iaa'} and not module_missing)
        )
        runtime_telemetry().fallback(f'nrrd.compression.{name}', RuntimeError(str(reason)))
        failure_key = (str(name), str(reason))
        if announce and failure_key not in _NRRD_MEMBER_CODEC_FAILURES_ANNOUNCED:
            _NRRD_MEMBER_CODEC_FAILURES_ANNOUNCED.add(failure_key)
            if policy_chain:
                print(
                    f'Warning: NRRD member codec {name!r} unavailable ({reason}); '
                    'using the next validated NRRD gzip tier.'
                )
            else:
                print(f'Warning: requested NRRD member codec {name!r} failed ({reason}).')
    return None

def _require_nrrd_member_codec(
    *,
    expected_input_bytes: Optional[int] = None,
) -> NrrdMemberCodecSpec:
    member_codec = _select_nrrd_member_codec(
        expected_input_bytes=expected_input_bytes,
    )
    if member_codec is None:
        requested = nrrd_member_codec_requested()
        raise RuntimeError(
            'No validated complete-member NRRD gzip codec is available '
            f'(YOLO_TTA_NRRD_MEMBER_CODEC={requested!r}; tried '
            f'{list(_nrrd_member_codec_candidates())}).'
        )
    return member_codec

def _open_nrrd_payload_writer(
    fh: object,
    *,
    codec_spec: Optional[NrrdMemberCodecSpec] = None,
) -> _MemberParallelGzipPayloadWriter:
    """Open one already-proven or newly selected complete gzip codec.

 Automatic selection tries hardware-only QAT, libdeflate, ISA-L, and zlib in order.
 If every configured codec fails validation, output stops with a specific
 error rather than switching to an unvalidated framing implementation."""
    member_codec = codec_spec if codec_spec is not None else _require_nrrd_member_codec()
    global _NRRD_MEMBER_GZIP_ANNOUNCED
    if not _NRRD_MEMBER_GZIP_ANNOUNCED:
        _NRRD_MEMBER_GZIP_ANNOUNCED = True
        print(
            f'Member-parallel NRRD gzip active: codec={member_codec[0]}, '
            f'level={int(member_codec[1])}, '
            f'{max(1, nrrd_gzip_chunk_bytes() // (1024 * 1024))} MiB members, '
            'whole-layer pipelining.'
        )
    _announce_nrrd_cpu_deflate_backend(
        'member-parallel gzip', codec_spec=member_codec,
    )
    return _MemberParallelGzipPayloadWriter(fh, codec_spec=member_codec)

def _madvise_array_mmap(arr: object, advice_name: str) -> None:
    try:
        import mmap as _mmap_module  # local import so non-POSIX platforms remain unaffected
        advice = getattr(_mmap_module, str(advice_name), None)
        if advice is None:
            return
        mmap_obj = getattr(arr, '_mmap', None)
        if mmap_obj is None:
            base = getattr(arr, 'base', None)
            mmap_obj = getattr(base, '_mmap', None)
        madvise_fn = getattr(mmap_obj, 'madvise', None)
        if callable(madvise_fn):
            madvise_fn(advice)
    except Exception:
        pass

def _nrrd_float_text(value: float) -> str:
    value_f = float(value)
    if math.isnan(value_f):
        return 'none'
    if math.isinf(value_f):
        return 'inf' if value_f > 0 else '-inf'
    return f'{value_f:.17g}'

def _nrrd_vector_text(values: Sequence[object]) -> str:
    return '(' + ','.join(_nrrd_float_text(float(v)) for v in values) + ')'

def _nrrd_space_directions_text(value: object) -> str:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 1:
        return _nrrd_vector_text(arr.tolist())
    if arr.ndim != 2:
        raise ValueError(f'NRRD space directions must be 1D or 2D, got shape {arr.shape}')
    parts: List[str] = []
    for row in arr:
        if np.all(np.isnan(row)):
            parts.append('none')
        else:
            parts.append(_nrrd_vector_text(row.tolist()))
    return ' '.join(parts)

def _nrrd_header_value_text(key: str, value: object) -> str:
    key_l = str(key).strip().lower()
    if key_l == 'space directions':
        return _nrrd_space_directions_text(value)
    if key_l in {'space origin', 'measurement frame'}:
        arr = np.asarray(value, dtype=np.float64)
        if arr.ndim == 1:
            return _nrrd_vector_text(arr.tolist())
    if isinstance(value, np.ndarray):
        arr = value
        if arr.ndim == 1:
            return ' '.join(_nrrd_float_text(float(v)) for v in arr.tolist())
        return ' '.join(_nrrd_vector_text(row) for row in arr.tolist())
    if isinstance(value, (list, tuple)):
        return ' '.join(_nrrd_ascii_header_text(v) for v in value)
    if isinstance(value, bool):
        return 'true' if value else 'false'
    return _nrrd_ascii_header_text(value).replace('\n', ' ').replace('\r', ' ')

_NRRD_STANDARD_FIELDS = {
    'type', 'dimension', 'sizes', 'spacings', 'thicknesses', 'axis mins', 'axis maxs',
    'centers', 'labels', 'units', 'min', 'max', 'old min', 'old max', 'endian',
    'encoding', 'line skip', 'byte skip', 'data file', 'content', 'sample units',
    'space', 'space dimension', 'space units', 'space origin', 'space directions',
    'measurement frame', 'kinds', 'block size',
}

def _nrrd_field_separator(key: str) -> str:
    return ':' if str(key).strip().lower() in _NRRD_STANDARD_FIELDS else ':='

def _write_nrrd_ascii_header(
    fh: object,
    *,
    header: Dict[str, object],
    sizes: Sequence[int],
    dimension: int,
    data_type: str = 'uint8',
    encoding: str = 'gzip',
) -> None:
    header_copy: Dict[str, object] = dict(header)
    for reserved in ('type', 'dimension', 'sizes'):
        header_copy.pop(reserved, None)
    header_copy['encoding'] = str(encoding)

    standard_order = [
        'space', 'space dimension', 'kinds', 'space directions', 'space origin',
        'measurement frame', 'content', 'encoding', 'endian',
    ]

    lines: List[str] = [
        'NRRD0005',
        f'# Complete NRRD file generated by {SCRIPT_BASENAME}',
        f'type: {str(data_type)}',
        f'dimension: {int(dimension)}',
        'sizes: ' + ' '.join(str(int(v)) for v in sizes),
    ]

    emitted: set[str] = set()
    for key in standard_order:
        if key in header_copy:
            lines.append(f'{key}: {_nrrd_header_value_text(key, header_copy[key])}')
            emitted.add(key)

    for key, value in header_copy.items():
        if key in emitted or key in {'type', 'dimension', 'sizes'}:
            continue
        sep = _nrrd_field_separator(str(key))
        value_text = _nrrd_header_value_text(str(key), value)
        if sep == ':=':
            lines.append(f'{str(key)}{sep}{value_text}')
        else:
            lines.append(f'{str(key)}{sep} {value_text}')

    text = '\n'.join(lines) + '\n\n'
    fh.write(text.encode('ascii', errors='ignore'))

def nrrd_layer_output_suffix(
    *,
    view_token: str,
    source: str,
    mask_kind: str,
    pass_index: int = 0,
    tile_acceptance: str = '',
    stage: str = '',
) -> str:
    """Return the single-layer NRRD filename suffix for one component layer.

 The decomposition is now one single-layer NRRD per component layer, named
 ``{Filestem}_{suffix}.seg.nrrd`` with the model name dropped (a model-ensemble holdover)."""
    source_l = str(source).strip().lower()
    mask_kind_l = str(mask_kind).strip().lower()
    stage_l = str(stage).strip().lower()
    if source_l == 'global':
        if stage_l.startswith('final_output'):
            return 'Global_final_output'
        if mask_kind_l == 'smoothing_result':
            return f'Global_smoothing_pass{int(pass_index):02d}'
        return 'Global_union_presmoothing'
    vt = _sanitize_nrrd_layer_token(view_token) or 'view'
    if source_l == 'fullframe':
        if mask_kind_l == 'bridge':
            return f'{vt}_fullframe_bridge_pass{int(pass_index):02d}'
        return f'{vt}_fullframe_yolo'
    if source_l == 'tile':
        if mask_kind_l == 'bridge':
            return f'{vt}_tile_bridge_pass{int(pass_index):02d}'
        acceptance = _sanitize_nrrd_layer_token(tile_acceptance) or 'parent_support'
        return f'{vt}_tile_yolo_{acceptance}'
    parts = [vt, source_l or 'layer', mask_kind_l or 'mask']
    if int(pass_index) > 0:
        parts.append(f'pass{int(pass_index):02d}')
    if tile_acceptance:
        parts.append(_sanitize_nrrd_layer_token(tile_acceptance))
    return '_'.join(p for p in parts if p)

def nrrd_layer_zshards_requested() -> int:
    """Requested per-global-layer z shards (0=auto, 1=disabled, >1=cap)."""
    return max(0, _env_int('YOLO_TTA_NRRD_LAYER_ZSHARDS', 0))

def nrrd_layer_zshard_min_slices() -> int:
    return max(1, _env_int('YOLO_TTA_NRRD_LAYER_ZSHARD_MIN_SLICES', 256))

_NRRD_ZSHARD_SEMAPHORE_LOCK = threading.Lock()

_NRRD_ZSHARD_SEMAPHORES: Dict[int, object] = {}

def _nrrd_zshard_capacity() -> int:
    """Process-wide concurrent band capacity (also the semaphore capacity)."""
    return max(1, int(nrrd_layer_sink_workers()))

class _WeightedNrrdZShardSemaphore:
    """Bound concurrent NRRD z-band compression with weighted process-wide permits.
    
    Ordinary independent spools consume one permit each."""

    def __init__(self, capacity: int) -> None:
        self.capacity = max(1, int(capacity))
        self.available = int(self.capacity)
        self.condition = threading.Condition()

    def acquire(self, weight: int) -> None:
        requested = max(1, int(weight))
        if requested > int(self.capacity):
            raise ValueError(
                f'NRRD z-shard reservation {requested} exceeds capacity {self.capacity}'
            )
        with self.condition:
            while int(self.available) < int(requested):
                self.condition.wait()
            self.available -= int(requested)

    def release(self, weight: int) -> None:
        returned = max(1, int(weight))
        with self.condition:
            if int(self.available) + int(returned) > int(self.capacity):
                raise RuntimeError('NRRD z-shard permit pool released too many permits')
            self.available += int(returned)
            self.condition.notify_all()

def _nrrd_zshard_semaphore() -> _WeightedNrrdZShardSemaphore:
    """Return the atomic weighted band budget for the configured sink capacity."""
    capacity = int(_nrrd_zshard_capacity())
    with _NRRD_ZSHARD_SEMAPHORE_LOCK:
        sem = _NRRD_ZSHARD_SEMAPHORES.get(int(capacity))
        if sem is None:
            sem = _WeightedNrrdZShardSemaphore(int(capacity))
            _NRRD_ZSHARD_SEMAPHORES[int(capacity)] = sem
        return sem  # type: ignore[return-value]

def _nrrd_layer_zshard_count(
    ref: 'NrrdLayerRef',
    out_t: int,
    capacity: Optional[int] = None,
) -> int:
    """Shard count resolved by the executing writer, never sink queue occupancy."""
    if str(getattr(ref, 'source', '')).strip().lower() != 'global':
        return 1
    min_slices = int(nrrd_layer_zshard_min_slices())
    by_depth = max(1, int(math.ceil(float(max(0, int(out_t))) / float(min_slices))))
    requested = int(nrrd_layer_zshards_requested())
    if requested == 1:
        return 1
    band_capacity = int(_nrrd_zshard_capacity() if capacity is None else max(1, int(capacity)))
    cap = int(band_capacity) if requested == 0 else min(int(band_capacity), requested)
    return max(1, min(int(out_t), int(by_depth), max(1, int(cap))))

def _nrrd_layer_zshard_bands(
    ref: 'NrrdLayerRef',
    out_t: int,
    shard_count: int,
) -> Tuple[List[Tuple[int, int]], Optional[List[int]]]:
    """Partition native cvol layers by indexed sparse payload bytes.

 Equal slice bands remain the conservative fallback for resampled/non-cvol layers.
 The optional second result reports each band's indexed payload weight for logging."""
    depth = max(0, int(out_t))
    count = max(1, min(int(shard_count), max(1, depth)))
    equal = [
        (int(i * depth // count), int((i + 1) * depth // count))
        for i in range(count)
    ]
    if (
        depth <= 0
        or not _nrrd_layer_ref_is_raw_bbox_store(ref)
        or int(getattr(ref, 'shape', (0, 0, 0))[0]) != int(depth)
    ):
        return equal, None
    try:
        index_path = Path(ref.path) / 'index.bin'
        index = np.fromfile(index_path, dtype=CTILE_INDEX_DTYPE, count=int(depth))
        if int(index.shape[0]) != int(depth):
            return equal, None
        weights = np.asarray(index['payload_nbytes'], dtype=np.uint64)
        total = int(np.sum(weights, dtype=np.uint64))
        if total <= 0:
            return equal, [0 for _ in equal]
        prefix = np.cumsum(weights, dtype=np.uint64)
        boundaries = [0]
        for i in range(1, count):
            target = int((int(total) * int(i) + int(count) - 1) // int(count))
            boundary = int(np.searchsorted(prefix, np.uint64(target), side='left')) + 1
            boundary = max(int(boundaries[-1]) + 1, int(boundary))
            boundary = min(int(boundary), int(depth) - int(count - i))
            boundaries.append(int(boundary))
        boundaries.append(int(depth))
        bands = [(int(boundaries[i]), int(boundaries[i + 1])) for i in range(count)]
        band_weights = [
            int(np.sum(weights[int(z0):int(z1)], dtype=np.uint64))
            for z0, z1 in bands
        ]
        return bands, band_weights
    except Exception:
        return equal, None

@runtime_telemetry_phase('nrrd.write_layer')
def write_single_layer_nrrd_from_ref(
    ref: 'NrrdLayerRef',
    output_shape: Tuple[int, int, int],
    out_path: Path,
    *,
    segment_name: Optional[str] = None,
    segment_color: Optional[Tuple[float, float, float]] = None,
    block_consumer: Optional[Callable[[int, np.ndarray], None]] = None,
    sparse_consumer: Optional[Callable[[int, int, int, np.ndarray], None]] = None,
    z_shards: Optional[int] = None,
) -> Path:
    """Write one component layer as its own single-layer 3D Slicer segmentation NRRD (X, Y, t).

 The layer is restored from its backing-store geometry directly to the final output
 geometry while streaming, reusing the same per-layer restore/stream path as the legacy
 decomposed writer. Each file holds one uint8 binary mask and is compressed
 as KAT-validated complete gzip members, preferring hardware-only QAT before
 libdeflate, ISA-L, or stdlib zlib; ``cpu`` opts out of QAT and IAA remains
 explicit-only. The Slicer segmentation fields make the file import as a Segmentation
 node; segment_name defaults to the output filename stem and segment_color to
 the deterministic palette pick for that name."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_t, out_h, out_w = (int(output_shape[0]), int(output_shape[1]), int(output_shape[2]))
    seg_name = str(segment_name) if segment_name else _slicer_segment_name_for_out_path(out_path)
    seg_color = segment_color if segment_color is not None else slicer_segment_palette_color(seg_name)
    # live-volume layers defer the segment-extent scan to THIS (background
    # sink) worker instead of blocking the producing thread with a store-encode pass.
    ref = _resolve_live_ref_extent(ref)
    raster_plan = _nrrd_raster_plan(ref, (out_t, out_h, out_w))
    stored_t, stored_h, stored_w = (int(v) for v in raster_plan.stored_shape_tyx)
    header = nrrd_slicer_header((stored_t, stored_h, stored_w))
    header['content'] = (
        'binary segmentation mask; '
        f'reference_shape_tyx=({out_t},{out_h},{out_w}); '
        f'stored_shape_tyx=({stored_t},{stored_h},{stored_w}); exported_axes=(X,Y,t)'
    )
    header.update(slicer_segmentation_header_fields(
        segment_name=seg_name,
        color_rgb=seg_color,
        extent_xyt=raster_plan.segment_extent_xyt,
    ))
    z_chunk = _nrrd_full_slice_z_chunk(1, stored_w, stored_h, stored_t)
    # resolve auto-sharding only after this sink task starts executing. Queue
    # occupancy at submit time says nothing about the child-band capacity by the time a
    # late global layer reaches the head of the sink pool. The shared semaphore remains
    # the hard cross-layer over-sharding guard.
    if z_shards is None or int(z_shards) <= 0:
        shard_count = _nrrd_layer_zshard_count(ref, int(stored_t))
    else:
        shard_count = max(
            1,
            min(int(stored_t), int(z_shards), int(_nrrd_zshard_capacity())),
        )
    bands: Optional[List[Tuple[int, int]]] = None
    band_weights: Optional[List[int]] = None
    expected_codec_request_bytes = int(stored_t) * int(stored_h) * int(stored_w)
    if shard_count > 1:
        bands, band_weights = _nrrd_layer_zshard_bands(ref, int(out_t), int(shard_count))
        expected_codec_request_bytes = min(
            max(0, int(z1) - int(z0)) * int(stored_h) * int(stored_w)
            for z0, z1 in bands
        )
    # Resolve, preflight, and KAT exactly once before a staging file/spool is opened.
    # Every z shard of this logical NRRD is pinned to this immutable codec spec.
    member_codec = _require_nrrd_member_codec(
        expected_input_bytes=int(expected_codec_request_bytes),
    )
    if shard_count <= 1:
        with _same_directory_atomic_output(out_path) as stage_path:
            with open(stage_path, 'wb') as fh:
                _write_nrrd_ascii_header(
                    fh,
                    header=header,
                    sizes=(stored_w, stored_h, stored_t),
                    dimension=3,
                    data_type='unsigned char',
                    encoding='gzip',
                )
                with _open_nrrd_payload_writer(fh, codec_spec=member_codec) as payload_writer:
                    _write_one_decomposed_nrrd_layer_payload(
                        ref,
                        (out_t, out_h, out_w),
                        payload_writer,
                        z_chunk=int(z_chunk),
                        block_consumer=block_consumer,
                        sparse_consumer=sparse_consumer,
                    )
        return out_path

    # each z band compresses into an independent anonymous memory-backed chunk.
    # A band holds ONE process-global permit only while it is actually compressing. This
    # lets two 8-band global files use a 12-lane budget concurrently and removes the
    # ordered bounded-queue backpressure that previously kept later compressors idle.
    if bands is None:
        raise RuntimeError('NRRD z-shard bands were not resolved')
    token = f'{os.getpid()}.{threading.get_ident()}'
    final_tmp = out_path.with_name(f'.{out_path.name}.{token}.assembling')
    shard_z_chunk = max(1, int(z_chunk) // int(shard_count))
    shard_fill_workers = int(nrrd_fill_workers())
    weight_note = ''
    if band_weights is not None:
        weight_note = (
            f', sparse_payload={min(band_weights) / (1024 ** 2):.1f}..'
            f'{max(band_weights) / (1024 ** 2):.1f} MiB/shard'
        )
    print(
        f'v16.0.2: {out_path.name} -> {shard_count} in-memory compressed z chunks '
        f'(min_slices={nrrd_layer_zshard_min_slices()}, z_chunk={shard_z_chunk}'
        f'{weight_note}; one global permit per active band).'
    )
    zshard_permits = _nrrd_zshard_semaphore()
    n19_started = time.perf_counter()
    n19_band_metrics: List[Optional[Tuple[float, int, int, int]]] = [None] * int(shard_count)

    def _write_band(i: int) -> object:
        z0, z1 = bands[int(i)]
        zshard_permits.acquire(1)
        spool = None
        band_started = time.perf_counter()
        try:
            # Keep the complete compressed band in an anonymous memory-backed file. No encoded
            # shard is written to the output or scratch filesystem before ordered assembly.
            spool = _open_memory_backed_encoded_chunk(
                f'{out_path.stem}.nrrd.z{int(z0):06d}-{int(z1):06d}.gz',
            )
            with _open_nrrd_payload_writer(spool, codec_spec=member_codec) as payload_writer:
                _write_one_decomposed_nrrd_layer_payload(
                    ref,
                    (out_t, out_h, out_w),
                    payload_writer,
                    z_chunk=shard_z_chunk,
                    block_consumer=block_consumer,
                    sparse_consumer=sparse_consumer,
                    z_start=int(z0),
                    z_stop=int(z1),
                    fill_workers_override=shard_fill_workers,
                )
            spool.flush()
            spool.seek(0, os.SEEK_END)
            compressed_size = int(spool.tell())
            spool.seek(0)
            n19_band_metrics[int(i)] = (
                float(time.perf_counter() - band_started),
                int(compressed_size),
                int(z0),
                int(z1),
            )
            return spool
        except BaseException:
            if spool is not None:
                try:
                    spool.close()
                except Exception:
                    pass
            raise
        finally:
            zshard_permits.release(1)

    def _copy_spool(src: object, dst: object) -> int:
        src.flush()
        dst.flush()
        src.seek(0)
        total = 0
        try:
            src_fd = int(src.fileno())
            dst_fd = int(dst.fileno())
            while True:
                copied = os.copy_file_range(src_fd, dst_fd, 64 * 1024 * 1024)
                if int(copied) <= 0:
                    break
                total += int(copied)
            return int(total)
        except Exception:
            # copy_file_range may fail after making partial progress (cross-filesystem,
            # quota, filesystem implementation). Resume at the exact byte offsets rather
            # than appending the already-copied prefix a second time.
            src.seek(int(total))
            while True:
                block = src.read(16 * 1024 * 1024)
                if not block:
                    break
                view = memoryview(block)
                written = 0
                while written < len(view):
                    step = dst.write(view[written:])
                    if step is None:
                        step = len(view) - written
                    if int(step) <= 0:
                        raise OSError('NRRD in-memory chunk copy made no forward progress')
                    written += int(step)
                total += int(written)
            return int(total)

    shard_pool = _acquire_parallel_pool(int(shard_count))
    futures: List[Future] = []
    spools: List[object] = []
    try:
        futures = [shard_pool.submit(_write_band, i) for i in range(shard_count)]
        with open(final_tmp, 'wb', buffering=0) as fh:
            _write_nrrd_ascii_header(
                fh,
                header=header,
                sizes=(stored_w, stored_h, stored_t),
                dimension=3,
                data_type='unsigned char',
                encoding='gzip',
            )
            # Wait/copy in z order. Later bands continue compressing independently in memory
            # while an earlier completed chunk streams as already-compressed bytes to disk.
            for i, fut in enumerate(futures):
                spool = fut.result()
                spools.append(spool)
                _copy_spool(spool, fh)
                spool.close()
                spools[-1] = None
        os.replace(final_tmp, out_path)
        completed_metrics = [m for m in n19_band_metrics if m is not None]
        compressed_total = sum(int(m[1]) for m in completed_metrics)
        slowest = max(completed_metrics, key=lambda m: float(m[0])) if completed_metrics else None
        slowest_note = ''
        if slowest is not None:
            slowest_note = (
                f', slowest_band=z{int(slowest[2])}:{int(slowest[3])} '
                f'{float(slowest[0]):.2f}s/{int(slowest[1]) / (1024 ** 2):.1f} MiB'
            )
        print(
            f'v16.0.2: {out_path.name} completed in '
            f'{time.perf_counter() - n19_started:.2f}s; compressed='
            f'{compressed_total / GIB:.2f} GiB{slowest_note}.'
        )
    finally:
        _settle_parallel_futures(futures)
        consumed_ids = {id(spool) for spool in spools if spool is not None}
        for fut in futures:
            if not fut.done() or fut.cancelled():
                continue
            try:
                pending_spool = fut.result()
            except BaseException:
                continue
            if pending_spool is not None and id(pending_spool) not in consumed_ids:
                try:
                    pending_spool.close()
                except Exception:
                    pass
        _release_parallel_pool(int(shard_count), shard_pool)
        for spool in spools:
            if spool is not None:
                try:
                    spool.close()
                except Exception:
                    pass
        try:
            final_tmp.unlink(missing_ok=True)
        except Exception:
            pass
    return out_path

_NRRD_GPU_MIRROR_TEE_ANNOUNCED = False

_NRRD_SPARSE_MIRROR_TEE_ANNOUNCED = False

def nrrd_gpu_mirror_tee_enabled() -> bool:
    """Derive low-quality NRRD mirror volumes on the GPU."""
    return _env_flag('YOLO_TTA_NRRD_GPU_MIRROR_TEE', True)

def nrrd_parallel_mirror_encode_enabled() -> bool:
    """Gzip-encode a layer's low-quality mirror files concurrently.

 The mirrors of one layer were encoded one after another at the tail of the layer task;
 they are independent files with separate handles and payload writers sharing
 the thread-safe member pool and zero-member cache, so they encode side by side.
 YOLO_TTA_NRRD_PARALLEL_MIRROR_ENCODE=0 restores serial encodes."""
    return _env_flag('YOLO_TTA_NRRD_PARALLEL_MIRROR_ENCODE', True)

class _GpuMirrorTee:
    """Derive downbinned mirror volumes on an exclusively leased CUDA device."""

    def __init__(
        self,
        torch_mod: object,
        device: object,
        mirror_jobs: List[Dict[str, object]],
        out_t: int,
        gpu_lease: _MainProcessGpuStageLease,
    ) -> None:
        self.torch = torch_mod
        self.device = device
        self.out_t = int(out_t)
        self.failed = False
        self.sub_batch = max(1, _env_int('YOLO_TTA_NRRD_GPU_MIRROR_TEE_SUBBATCH', 32))
        self.specs: List[Dict[str, object]] = []
        self.stream: Optional[object] = None
        self._gpu_lease: Optional[_MainProcessGpuStageLease] = gpu_lease
        self._closed = False
        try:
            with torch_mod.cuda.device(device):
                self.stream = torch_mod.cuda.Stream(device=device)
                with torch_mod.cuda.stream(self.stream):
                    for job in mirror_jobs:
                        m_t, m_h, m_w = (int(v) for v in job['shape'])  # type: ignore[misc]
                        self.specs.append({
                            'job': job,
                            'vol': torch_mod.zeros(
                                (m_t, m_h, m_w), dtype=torch_mod.uint8, device=device,
                            ),
                            'src_to_m': job['src_to_m'],
                            'm_h': int(m_h),
                            'm_w': int(m_w),
                        })
                self.stream.synchronize()
        except BaseException:
            self._release_resources()
            raise

    def _release_resources(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if self.stream is not None:
                self.stream.synchronize()
        except Exception:
            pass
        for spec in self.specs:
            spec['vol'] = None
        self.specs.clear()
        self.stream = None
        _trim_main_process_cuda_device(
            self.torch,
            self.device,
            desc='NRRD GPU mirror tee cleanup',
        )
        lease = self._gpu_lease
        self._gpu_lease = None
        if lease is not None:
            lease.release()

    def tee(self, z0: int, block: np.ndarray) -> None:
        if self.failed or self._closed:
            return
        torch = self.torch
        blk = blk_f = pooled = pooled_u8 = vol = None
        try:
            import torch.nn.functional as F  # type: ignore
            with torch.cuda.device(self.device):
                with torch.cuda.stream(self.stream):
                    z_count = int(block.shape[0])
                    for b0 in range(0, z_count, int(self.sub_batch)):
                        b1 = min(z_count, b0 + int(self.sub_batch))
                        blk = torch.from_numpy(np.ascontiguousarray(block[b0:b1])).to(self.device)
                        blk_f = blk.to(torch.float16).unsqueeze(1)
                        for spec in self.specs:
                            pooled = F.adaptive_max_pool2d(
                                blk_f, (int(spec['m_h']), int(spec['m_w'])),
                            )
                            pooled_u8 = (pooled.squeeze(1) > 0).to(torch.uint8)
                            vol = spec['vol']
                            src_to_m = spec['src_to_m']
                            for bi in range(int(b1 - b0)):
                                full_z = int(z0) + int(b0) + int(bi)
                                if full_z >= self.out_t:
                                    break
                                for mz in src_to_m[full_z]:  # type: ignore[index]
                                    torch.maximum(vol[int(mz)], pooled_u8[int(bi)], out=vol[int(mz)])
                        blk = blk_f = pooled = pooled_u8 = vol = None
        except Exception as exc:
            self.failed = True
            # Clear partially constructed per-block tensors before trimming the allocator.
            blk = blk_f = pooled = pooled_u8 = vol = None
            gc.collect()
            print(
                f'Warning: GPU mirror tee failed mid-stream ({exc}); '
                'mirrors re-encode from the layer store.'
            )
            self._release_resources()

    def discard(self) -> None:
        self._release_resources()

    def finalize_into_jobs(self) -> bool:
        if self.failed or self._closed:
            self._release_resources()
            return False
        try:
            self.stream.synchronize()
            for spec in self.specs:
                spec['job']['volume'] = spec['vol'].cpu().numpy()  # type: ignore[index]
                spec['vol'] = None
            return True
        except Exception as exc:
            print(
                f'Warning: GPU mirror tee finalize failed ({exc}); '
                'mirrors re-encode from the layer store.'
            )
            self.failed = True
            return False
        finally:
            self._release_resources()

    def __del__(self) -> None:  # pragma: no cover - final safety net
        try:
            self._release_resources()
        except Exception:
            pass

def _try_create_gpu_mirror_tee(
    mirror_jobs: List[Dict[str, object]],
    out_t: int,
) -> Optional['_GpuMirrorTee']:
    if not nrrd_gpu_mirror_tee_enabled() or not mirror_jobs:
        return None
    lease: Optional[_MainProcessGpuStageLease] = None
    try:
        import torch  # type: ignore
        if not bool(torch.cuda.is_available()):
            return None
        lease = _try_acquire_main_process_gpu_stage(torch, 'NRRD low-quality GPU mirror tee')
        if lease is None:
            _announce_main_gpu_stage_skip_once(
                'nrrd-mirror-tee-inference-busy',
                'NRRD GPU mirror tee skipped while all eligible GPUs have active/queued '
                'inference or another output-stage lease; using the CPU mirror tee.',
            )
            return None
        device = lease.torch_device(torch)
        need = sum(
            int(np.prod([int(v) for v in job['shape']]))
            for job in mirror_jobs
        )  # type: ignore[misc]
        free_bytes, _total = torch.cuda.mem_get_info(device)
        if int(free_bytes) < int(need) + 2 * GIB:
            lease.release()
            lease = None
            return None
        tee = _GpuMirrorTee(torch, device, mirror_jobs, int(out_t), lease)
        lease = None  # ownership transferred to the tee
        return tee
    except Exception as exc:
        if lease is not None:
            try:
                device = lease.torch_device(torch)  # type: ignore[name-defined]
                _trim_main_process_cuda_device(
                    torch, device, desc='failed NRRD GPU mirror tee construction',  # type: ignore[name-defined]
                )
            except Exception:
                pass
            lease.release()
        print(f'Warning: NRRD GPU mirror tee setup failed ({exc}); using the CPU mirror tee.')
        return None

def write_layer_nrrd_with_low_quality_mirrors(
    ref: 'NrrdLayerRef',
    output_shape: Tuple[int, int, int],
    out_path: Path,
    mirrors: Sequence[Tuple[Tuple[int, int, int], Path]],
    *,
    segment_name: Optional[str] = None,
    segment_color: Optional[Tuple[float, float, float]] = None,
    z_shards: Optional[int] = None,
) -> Path:
    """Write a full-quality layer and all low-quality mirrors in one source pass.
    
    Independent mirror encoders may run concurrently while sharing the streamed layer data."""
    out_t = int(output_shape[0])
    # Resolve a live layer's deferred extent once so full-quality and mirror headers agree.
    ref = _resolve_live_ref_extent(ref)
    # mirror payloads are composed in two geometry steps (backing -> full output ->
    # low quality). Compose the header extent through that same intermediate geometry;
    # mapping backing -> mirror directly can understate a positive-support boundary voxel.
    mirror_extent_ref = dataclasses_replace(
        ref,
        segment_extent_ijk=_nrrd_mapped_extent_preserve_empty(ref, output_shape),
        segment_extent_shape_tyx=tuple(int(v) for v in output_shape),
        segment_extent_source='composed_full_output_extent_for_low_quality_mirror',
    )
    # this function itself is the queued sink task, so this is execution-time—not
    # submission-time—resolution. Reuse the result for tee scheduling and the full writer.
    if z_shards is None or int(z_shards) <= 0:
        resolved_z_shards = _nrrd_layer_zshard_count(ref, int(out_t))
    else:
        resolved_z_shards = max(
            1,
            min(int(out_t), int(z_shards), int(_nrrd_zshard_capacity())),
        )
    mirror_jobs: List[Dict[str, object]] = []
    for m_shape, m_path in mirrors:
        m_t, m_h, m_w = (int(m_shape[0]), int(m_shape[1]), int(m_shape[2]))
        # Invert the mirror's t mapping once: for each full-quality output z, the mirror
        # slice(s) whose source range contains it (matches _restore_source_indices_for_output_z).
        tmp: List[List[int]] = [[] for _ in range(out_t)]
        for mz in range(m_t):
            for sz in _restore_source_indices_for_output_z(out_t, m_t, int(mz)):
                if 0 <= int(sz) < out_t:
                    tmp[int(sz)].append(int(mz))
        mirror_jobs.append({
            'shape': (m_t, m_h, m_w),
            'path': Path(m_path),
            'volume': None,  # allocated below only when the CPU tee runs
            'src_to_m': [tuple(v) for v in tmp],
            'locks': [threading.Lock() for _ in range(16)],
            'failed': False,
        })

    # raw bbox stores expose their restored output crops during the same
    # sparse member pass. Keep this tee sparse on the CPU: uploading thousands of tiny
    # crops is slower than the vectorized integral/gather resize, while uploading the old
    # dense 8.83 MiB plane would forfeit the bandwidth win. Other source types retain.
    sparse_mirror_tee = bool(
        mirror_jobs and _nrrd_layer_ref_is_raw_bbox_store(ref)
    )

    # derive the mirrors on a GPU when one is free — the per-slice resizes
    # and striped-lock ORs leave the CPU tee entirely; mirror volumes come back in one
    # small D2H after the layer streams. CPU tee (below) remains the fallback.
    gpu_tee = None if bool(sparse_mirror_tee) else _try_create_gpu_mirror_tee(mirror_jobs, out_t)
    if gpu_tee is not None:
        global _NRRD_GPU_MIRROR_TEE_ANNOUNCED
        if not _NRRD_GPU_MIRROR_TEE_ANNOUNCED:
            _NRRD_GPU_MIRROR_TEE_ANNOUNCED = True
            print(
                f'GPU low-quality mirror tee active on {gpu_tee.device} (v13.3.6 D2; '
                'YOLO_TTA_NRRD_GPU_MIRROR_TEE=0 disables).'
            )
    else:
        for job in mirror_jobs:
            m_t, m_h, m_w = (int(v) for v in job['shape'])  # type: ignore[misc]
            job['volume'] = np.zeros((m_t, m_h, m_w), dtype=np.uint8)
        if bool(sparse_mirror_tee):
            global _NRRD_SPARSE_MIRROR_TEE_ANNOUNCED
            if not _NRRD_SPARSE_MIRROR_TEE_ANNOUNCED:
                _NRRD_SPARSE_MIRROR_TEE_ANNOUNCED = True
                print(
                    'v13.3.17 (N22): crop-aware low-quality NRRD mirror tee active; '
                    'native/restored cvol layers no longer make a second dense store pass.'
                )

    # already supplies z-band concurrency. Avoid nesting another up-to-16-way CPU pool
    # inside every shard; the unsharded path retains the original slice fan-out.
    tee_workers = 1 if int(resolved_z_shards) > 1 else max(1, min(int(nrrd_fill_workers()), 16))

    def _tee(z0: int, block: np.ndarray) -> None:
        live_jobs = [j for j in mirror_jobs if not bool(j['failed'])]
        if not live_jobs:
            return
        z_count = int(block.shape[0])

        def _tee_one(zi: int) -> None:
            full_z = int(z0) + int(zi)
            if full_z >= out_t:
                return
            frame = block[int(zi)]
            for job in live_jobs:
                if bool(job['failed']):
                    continue
                try:
                    targets = job['src_to_m'][full_z]
                    if not targets:
                        continue
                    m_t, m_h, m_w = job['shape']  # type: ignore[misc]
                    resized = _resize_binary_mask_frame_to_output_shape(frame, int(m_h), int(m_w))
                    vol = job['volume']
                    locks = job['locks']
                    for mz in targets:
                        # Adjacent full-z frames can map to the same mirror slice; stripe locks
                        # keep the read-modify-write OR race-free without serializing the block.
                        with locks[int(mz) % len(locks)]:  # type: ignore[index]
                            np.bitwise_or(vol[int(mz)], resized, out=vol[int(mz)])  # type: ignore[index]
                except Exception:
                    job['failed'] = True

        if z_count > 1 and tee_workers > 1:
            _nrrd_parallel_fill_indices(
                z_count, _tee_one, requested_workers=tee_workers,
            )
        else:
            for zi in range(z_count):
                _tee_one(int(zi))

    def _tee_sparse(out_z: int, y0: int, x0: int, crop: np.ndarray) -> None:
        """Fold one already-restored output crop into every low-quality mirror."""
        if int(out_z) < 0 or int(out_z) >= int(out_t):
            return
        crop_u8 = np.asarray(crop, dtype=np.uint8)
        if crop_u8.size <= 0:
            return
        y1 = int(y0) + int(crop_u8.shape[0])
        x1 = int(x0) + int(crop_u8.shape[1])
        out_h, out_w = int(output_shape[1]), int(output_shape[2])
        for job in mirror_jobs:
            if bool(job['failed']):
                continue
            try:
                targets = job['src_to_m'][int(out_z)]
                if not targets:
                    continue
                _m_t, m_h, m_w = job['shape']  # type: ignore[misc]
                resized = _resize_sparse_binary_crop_to_output_region(
                    crop_u8,
                    source_shape=(int(out_h), int(out_w)),
                    source_bbox=(int(y0), int(x0), int(y1), int(x1)),
                    output_shape=(int(m_h), int(m_w)),
                )
                if resized is None:
                    continue
                my0, mx0, my1, mx1, mirror_crop = resized
                vol = job['volume']
                locks = job['locks']
                for mz in targets:
                    with locks[int(mz) % len(locks)]:  # type: ignore[index]
                        dst = vol[int(mz), int(my0):int(my1), int(mx0):int(mx1)]  # type: ignore[index]
                        np.bitwise_or(dst, mirror_crop, out=dst)
            except Exception:
                job['failed'] = True

    # shard consumers can arrive concurrently. CPU mirror updates use
    # per-output-z stripe locks; the GPU tee owns one stream and tensor set, so
    # serialize only its tee calls while member compression remains parallel.
    gpu_tee_call_lock = threading.Lock() if gpu_tee is not None else None

    def _consume_block(z0: int, block: np.ndarray) -> None:
        if gpu_tee is not None:
            with gpu_tee_call_lock:  # type: ignore[arg-type]
                was_ok = not gpu_tee.failed
                gpu_tee.tee(int(z0), block)
                if gpu_tee.failed and was_ok:
                    # Mid-stream GPU failure: earlier blocks never went through the CPU tee,
                    # so partial CPU teeing is useless — mark every mirror for the store
                    # re-encode fallback below.
                    for job in mirror_jobs:
                        job['failed'] = True
                    gpu_tee.discard()
            return
        _tee(int(z0), block)

    result = write_single_layer_nrrd_from_ref(
        ref, output_shape, out_path,
        segment_name=segment_name, segment_color=segment_color,
        block_consumer=(_consume_block if mirror_jobs and not sparse_mirror_tee else None),
        sparse_consumer=(_tee_sparse if sparse_mirror_tee else None),
        z_shards=int(resolved_z_shards),
    )

    # one small D2H per mirror replaces the per-block CPU resize/OR work.
    if gpu_tee is not None and not gpu_tee.finalize_into_jobs():
        for job in mirror_jobs:
            job['failed'] = True

    def _encode_mirror(job: Dict[str, object]) -> None:
        m_t, m_h, m_w = job['shape']  # type: ignore[misc]
        m_path: Path = job['path']  # type: ignore[assignment]
        try:
            if bool(job['failed']):
                raise RuntimeError('mirror tee failed during full-quality streaming')
            m_path.parent.mkdir(parents=True, exist_ok=True)
            mirror_plan = _nrrd_raster_plan(
                mirror_extent_ref,
                (int(m_t), int(m_h), int(m_w)),
            )
            stored_t, stored_h, stored_w = (int(v) for v in mirror_plan.stored_shape_tyx)
            header = nrrd_slicer_header((stored_t, stored_h, stored_w))
            header['content'] = (
                'binary segmentation mask; '
                f'reference_shape_tyx=({int(m_t)},{int(m_h)},{int(m_w)}); '
                f'stored_shape_tyx=({stored_t},{stored_h},{stored_w}); exported_axes=(X,Y,t)'
            )
            seg_name = str(segment_name) if segment_name else _slicer_segment_name_for_out_path(m_path)
            seg_color = segment_color if segment_color is not None else slicer_segment_palette_color(seg_name)
            header.update(slicer_segmentation_header_fields(
                segment_name=seg_name,
                color_rgb=seg_color,
                extent_xyt=mirror_plan.segment_extent_xyt,
            ))
            vol: Optional[np.ndarray] = job.get('volume')  # type: ignore[assignment]
            if vol is None:
                raise RuntimeError('mirror tee produced no host payload')
            step = max(1, int(_nrrd_full_slice_z_chunk(1, stored_w, stored_h, stored_t)))
            member_codec = _require_nrrd_member_codec(
                expected_input_bytes=int(stored_t) * int(stored_h) * int(stored_w),
            )
            with _same_directory_atomic_output(m_path) as stage_path:
                with open(stage_path, 'wb') as fh:
                    _write_nrrd_ascii_header(
                        fh, header=header, sizes=(stored_w, stored_h, stored_t),
                        dimension=3, data_type='unsigned char', encoding='gzip',
                    )
                    with _open_nrrd_payload_writer(fh, codec_spec=member_codec) as payload_writer:
                        for z0 in range(0, int(stored_t), step):
                            z1 = min(int(stored_t), int(z0) + step)
                            payload_writer.write(memoryview(vol[z0:z1]).cast('B'))  # type: ignore[index]
        except Exception as exc:
            print(
                f'Warning: low-quality NRRD mirror tee failed for {m_path.name} ({exc}); '
                're-encoding that mirror from the layer store.'
            )
            write_single_layer_nrrd_from_ref(
                ref, (int(m_t), int(m_h), int(m_w)), m_path,
                segment_name=segment_name, segment_color=segment_color,
            )
        finally:
            job['volume'] = None

    # the mirror files of one layer are independent — encode them side by
    # side instead of one after another at the tail of the layer task. Every task runs to
    # completion (a failing mirror re-encodes from the store, matching the serial per-mirror
    # fallback) and the first error is re-raised after all mirrors settle.
    if len(mirror_jobs) > 1 and nrrd_parallel_mirror_encode_enabled():
        pool_size = min(4, len(mirror_jobs))
        mirror_pool = _acquire_parallel_pool(int(pool_size))
        first_error: Optional[BaseException] = None
        try:
            futures = [mirror_pool.submit(_encode_mirror, job) for job in mirror_jobs]
            for fut in futures:
                try:
                    fut.result()
                except BaseException as exc:
                    if first_error is None:
                        first_error = exc
        finally:
            _release_parallel_pool(int(pool_size), mirror_pool)
        if first_error is not None:
            raise first_error
    else:
        for job in mirror_jobs:
            _encode_mirror(job)
    return result

class NrrdLayerSink:
    """Write each component layer as its own NRRD as soon as the layer becomes available."""

    def __init__(
        self,
        *,
        nrrd_dir: Path,
        stem: str,
        output_shape_tyx: Tuple[int, int, int],
        max_workers: int,
        low_quality_specs: Optional[Sequence['LowQualityDownbinSpec']] = None,
        low_quality_root: Optional[Path] = None,
    ) -> None:
        self.nrrd_dir = Path(nrrd_dir)
        self.nrrd_dir.mkdir(parents=True, exist_ok=True)
        self.stem = variant_nrrd_stem(stem)
        self.output_shape = (int(output_shape_tyx[0]), int(output_shape_tyx[1]), int(output_shape_tyx[2]))
        self.max_workers = max(1, int(max_workers))
        self._lock = threading.Lock()
        self._futures: List[Future] = []
        self._manifest: List[Dict[str, object]] = []
        self._suffix_counts: Dict[str, int] = {}
        # Slicer segment colors already assigned in this run, so two layers whose
        # suffix hashes collide still render distinctly (deterministic forward probing).
        self._segment_colors_in_use: set = set()
        # low-quality NRRDs now mirror the full-quality decomposition instead of being one
        # combined volume written at the tail. Each downbin spec gets its own
        # one-single-layer-NRRD-per-component decomposition under low_quality/<token>/nrrd/ restored
        # from the same NrrdLayerRef and submitted here as each view completes (identical scheduling).
        self.low_quality_specs: List['LowQualityDownbinSpec'] = list(low_quality_specs or [])
        self.low_quality_root = Path(low_quality_root) if low_quality_root is not None else None
        self._lq_manifests: Dict[str, List[Dict[str, object]]] = {}
        if self.low_quality_specs:
            if self.low_quality_root is None:
                raise RuntimeError(
                    'low-quality NRRD specs require a low_quality_root'
                )
            for spec in self.low_quality_specs:
                self._lq_nrrd_dir(spec).mkdir(parents=True, exist_ok=True)
        # Construct the executor only after every validating/construction-time filesystem
        # operation has succeeded. A failed mkdir can therefore never strand worker threads
        # in an object that the pipeline did not receive and cannot shut down.
        self.executor = ThreadPoolExecutor(
            max_workers=self.max_workers,
            thread_name_prefix='nrrd-layer',
        )

    def _lq_nrrd_dir(self, spec: 'LowQualityDownbinSpec') -> Path:
        if self.low_quality_root is None:
            raise RuntimeError('low-quality NRRD directory requested without a low_quality_root')
        return self.low_quality_root / str(spec.token) / 'nrrd'

    def _segment_color_for_suffix(self, unique_suffix: str) -> Tuple[float, float, float]:
        # Called with self._lock held. Deterministic palette pick keyed off the layer suffix
        # (stable across runs), probing forward past colors already used this run; beyond the
        # palette size, a golden-ratio HSV walk keeps every additional layer distinct.
        palette = _SLICER_SEGMENT_COLOR_PALETTE
        base = _stable_layer_color_index(str(unique_suffix))
        for probe in range(len(palette)):
            color = palette[(base + probe) % len(palette)]
            if color not in self._segment_colors_in_use:
                self._segment_colors_in_use.add(color)
                return color
        hue = (float(base % 4096) / 4096.0 + 0.61803398875 * float(len(self._segment_colors_in_use))) % 1.0
        r, g, b = colorsys.hsv_to_rgb(hue, 0.72, 0.88)
        color = (round(float(r), 6), round(float(g), 6), round(float(b), 6))
        self._segment_colors_in_use.add(color)
        return color

    def submit_layer(self, ref: Optional['NrrdLayerRef'], suffix: str) -> Optional[Path]:
        if ref is None:
            return None
        layer_role = str(getattr(ref, 'layer_role', 'additive_component'))
        recomposition_op = str(getattr(ref, 'recomposition_op', 'union'))
        low_quality_recomposition_op = str(
            getattr(ref, 'low_quality_recomposition_op', recomposition_op)
        )
        mirror_low_quality = bool(getattr(ref, 'mirror_low_quality', True))
        with self._lock:
            seen = int(self._suffix_counts.get(str(suffix), 0))
            self._suffix_counts[str(suffix)] = seen + 1
            unique_suffix = str(suffix) if seen == 0 else f'{suffix}_{seen + 1:02d}'
            # .seg.nrrd + Slicer segmentation header fields; the segment is named
            # after the file and colored from the deterministic per-suffix palette.
            out_path = self.nrrd_dir / f'{self.stem}_{unique_suffix}.seg.nrrd'
            segment_name = f'{self.stem}_{unique_suffix}'
            segment_color = self._segment_color_for_suffix(unique_suffix)
            manifest_entry: Dict[str, object] = {
                'filename': out_path.name,
                'suffix': unique_suffix,
                'view_name': getattr(ref, 'view_name', ''),
                'physical_view_name': getattr(ref, 'physical_view_name', ''),
                'tta_aug_id': getattr(ref, 'aug_id', ''),
                'tta_angle_deg': float(getattr(ref, 'angle_deg', 0.0)),
                'view_family': getattr(ref, 'view_family', ''),
                'source': getattr(ref, 'source', ''),
                'mask_kind': getattr(ref, 'mask_kind', ''),
                'pass_index': int(getattr(ref, 'pass_index', 0)),
                'tile_acceptance': getattr(ref, 'tile_acceptance', ''),
                'stage': getattr(ref, 'stage', ''),
                'description': getattr(ref, 'description', ''),
                'layer_role': layer_role,
                'recomposition_op': recomposition_op,
                'backing_shape_tyx': [int(v) for v in getattr(ref, 'shape', (0, 0, 0))],
                'output_shape_tyx': [int(v) for v in self.output_shape],
                'stored_shape_tyx': None,
                'empty_segment': None,
                'exported_axes': '(X, Y, t)',
                'segment_name': segment_name,
                'segment_color_rgb': [round(float(c), 6) for c in segment_color],
                # resolves the actual count inside the executing writer. Keeping the
                # policy explicit avoids recording the stale submit-time pool occupancy.
                'z_shards': None,
                'z_shards_policy': 'execution_time',
            }
            self._manifest.append(manifest_entry)
            # mirror this component layer into each low-quality downbin decomposition,
            # scheduled now (as the view completes) exactly like the full-quality layer and sharing
            # the same unique suffix so a low-quality layer maps 1:1 to its full-quality layer.
            # the mirrors are derived from the full-quality payload blocks inside ONE
            # combined write task (write_layer_nrrd_with_low_quality_mirrors) instead of each
            # re-decoding the full-resolution layer store as an independent task.
            lq_mirror_args: List[Tuple[Tuple[int, int, int], Path]] = []
            lq_manifest_entries: List[Tuple[Tuple[int, int, int], Dict[str, object]]] = []
            for spec in self.low_quality_specs if mirror_low_quality else []:
                if self.low_quality_root is None:
                    break
                lq_shape = (
                    int(spec.output_shape_t_y_x[0]),
                    int(spec.output_shape_t_y_x[1]),
                    int(spec.output_shape_t_y_x[2]),
                )
                lq_path = self._lq_nrrd_dir(spec) / f'{self.stem}_{unique_suffix}.seg.nrrd'
                lq_mirror_args.append((lq_shape, lq_path))
                lq_manifest_entry: Dict[str, object] = {
                    'filename': lq_path.name,
                    'suffix': unique_suffix,
                    'view_name': getattr(ref, 'view_name', ''),
                    'physical_view_name': getattr(ref, 'physical_view_name', ''),
                    'tta_aug_id': getattr(ref, 'aug_id', ''),
                    'tta_angle_deg': float(getattr(ref, 'angle_deg', 0.0)),
                    'view_family': getattr(ref, 'view_family', ''),
                    'source': getattr(ref, 'source', ''),
                    'mask_kind': getattr(ref, 'mask_kind', ''),
                    'pass_index': int(getattr(ref, 'pass_index', 0)),
                    'tile_acceptance': getattr(ref, 'tile_acceptance', ''),
                    'stage': getattr(ref, 'stage', ''),
                    'description': getattr(ref, 'description', ''),
                    'layer_role': layer_role,
                    'recomposition_op': low_quality_recomposition_op,
                    'full_quality_recomposition_op': recomposition_op,
                    'backing_shape_tyx': [int(v) for v in getattr(ref, 'shape', (0, 0, 0))],
                    'output_shape_tyx': [int(v) for v in lq_shape],
                    'stored_shape_tyx': None,
                    'empty_segment': None,
                    'downbin_value': str(spec.raw_value),
                    'downbin_token': str(spec.token),
                    'downbin_scale': float(spec.scale),
                    'exported_axes': '(X, Y, t)',
                    'segment_name': segment_name,
                    'segment_color_rgb': [round(float(c), 6) for c in segment_color],
                }
                self._lq_manifests.setdefault(str(spec.token), []).append(lq_manifest_entry)
                lq_manifest_entries.append((lq_shape, lq_manifest_entry))
            def _execute_layer_write() -> Path:
                # This closure begins on the sink executor, so its count reflects execution
                # time rather than submission-time queue occupancy. Record that exact value
                # and pass it through to the mirror/full writer as one immutable decision.
                resolved_ref = _resolve_live_ref_extent(ref)
                full_plan = _nrrd_raster_plan(resolved_ref, self.output_shape)
                executed_z_shards = int(_nrrd_layer_zshard_count(
                    resolved_ref, int(full_plan.stored_shape_tyx[0]),
                ))
                with self._lock:
                    manifest_entry['z_shards'] = int(executed_z_shards)
                    manifest_entry['stored_shape_tyx'] = [int(v) for v in full_plan.stored_shape_tyx]
                    manifest_entry['empty_segment'] = bool(full_plan.empty_segment)
                    composed_mirror_ref = dataclasses_replace(
                        resolved_ref,
                        segment_extent_ijk=_nrrd_mapped_extent_preserve_empty(
                            resolved_ref, self.output_shape,
                        ),
                        segment_extent_shape_tyx=tuple(int(v) for v in self.output_shape),
                        segment_extent_source='composed_full_output_extent_for_low_quality_mirror',
                    )
                    for lq_shape, lq_entry in lq_manifest_entries:
                        lq_plan = _nrrd_raster_plan(composed_mirror_ref, lq_shape)
                        lq_entry['stored_shape_tyx'] = [int(v) for v in lq_plan.stored_shape_tyx]
                        lq_entry['empty_segment'] = bool(lq_plan.empty_segment)
                if lq_mirror_args:
                    return write_layer_nrrd_with_low_quality_mirrors(
                        resolved_ref, self.output_shape, out_path, lq_mirror_args,
                        segment_name=segment_name,
                        segment_color=segment_color,
                        z_shards=int(executed_z_shards),
                    )
                return write_single_layer_nrrd_from_ref(
                    resolved_ref, self.output_shape, out_path,
                    segment_name=segment_name,
                    segment_color=segment_color,
                    z_shards=int(executed_z_shards),
                )

            fut = self.executor.submit(_execute_layer_write)
            self._futures.append(fut)
        return out_path

    def layer_count(self) -> int:
        with self._lock:
            return len(self._manifest)

    def centerline_audit_layer_count(self) -> int:
        with self._lock:
            return int(sum(
                1 for entry in self._manifest
                if str(entry.get('source', '')) == 'centerline_filter'
            ))

    def low_quality_layer_count(self) -> int:
        with self._lock:
            return int(sum(len(entries) for entries in self._lq_manifests.values()))

    def low_quality_centerline_audit_layer_count(self) -> int:
        with self._lock:
            return int(sum(
                1
                for entries in self._lq_manifests.values()
                for entry in entries
                if str(entry.get('source', '')) == 'centerline_filter'
            ))

    def progress_counts(self) -> Tuple[int, int]:
        """Return completed and submitted write-task counts without waiting."""
        with self._lock:
            futures = list(self._futures)
        return int(sum(1 for fut in futures if fut.done())), int(len(futures))

    def wait(self) -> None:
        with self._lock:
            futures = list(self._futures)
        total = int(len(futures))
        if total <= 0:
            return
        pending: set[Future] = set()
        first_error: Optional[BaseException] = None
        completed = 0
        for fut in futures:
            if not fut.done():
                pending.add(fut)
                continue
            completed += 1
            try:
                fut.result()
            except BaseException as exc:  # pragma: no cover - surfaced to main
                if first_error is None:
                    first_error = exc
        status_seconds = max(5.0, _env_float('YOLO_TTA_NRRD_WAIT_STATUS_SECONDS', 30.0))
        last_status = time.monotonic()
        print(
            f'Single-layer NRRD write status: {int(completed)}/{total} complete, '
            f'{len(pending)} pending.'
        )
        while pending:
            done, pending_remainder = wait(
                pending,
                timeout=float(status_seconds),
                return_when=FIRST_COMPLETED,
            )
            pending = set(pending_remainder)
            for fut in done:
                completed += 1
                try:
                    fut.result()
                except BaseException as exc:  # pragma: no cover - surfaced to main
                    if first_error is None:
                        first_error = exc
            now = time.monotonic()
            if done or now - float(last_status) >= float(status_seconds):
                print(
                    f'Single-layer NRRD write status: {int(completed)}/{total} complete, '
                    f'{len(pending)} pending.'
                )
                last_status = float(now)
        if first_error is not None:
            raise RuntimeError('Single-layer NRRD writing failed') from first_error

    def shutdown(self) -> None:
        self.executor.shutdown(wait=True)

    def write_manifest(self) -> Optional[Path]:
        with self._lock:
            manifest_layers = list(self._manifest)
            lq_manifests = {token: list(entries) for token, entries in self._lq_manifests.items()}
        # one manifest per low-quality downbin decomposition, mirroring the full-quality
        # sidecar so each low-quality nrrd/ folder is self-describing and recomposable on its own.
        if self.low_quality_root is not None:
            for spec in self.low_quality_specs:
                entries = lq_manifests.get(str(spec.token), [])
                if not entries:
                    continue
                lq_shape = (
                    int(spec.output_shape_t_y_x[0]),
                    int(spec.output_shape_t_y_x[1]),
                    int(spec.output_shape_t_y_x[2]),
                )
                lq_manifest_path = self._lq_nrrd_dir(spec) / f'{self.stem}_nrrd_manifest.json'
                _write_json_atomically(lq_manifest_path, {
                    'layout': 'one_single_layer_nrrd_per_component',
                    'quality': 'low_quality',
                    'downbin_value': str(spec.raw_value),
                    'downbin_token': str(spec.token),
                    'downbin_scale': float(spec.scale),
                    'exported_axes': '(X, Y, t)',
                    'output_shape_tyx': [int(v) for v in lq_shape],
                    'full_quality_output_shape_tyx': [int(v) for v in self.output_shape],
                    'layer_count': len(entries),
                    'layers': entries,
                    'notes': [
                        'Low-quality distribution: eligible component layers and complete checkpoints from the '
                        'full-quality nrrd/ folder, isotropically downbinned to this spec.',
                        'Each NRRD is one uint8 binary mask in source output geometry (X, Y, t), downbinned.',
                        'v13.2.3: each file is a 3D Slicer segmentation (.seg.nrrd) sharing its full-quality layer\'s segment name and color.',
                        'Layer suffixes match the corresponding full-quality layers. Centerline removed-component and watershed-candidate audit layers are mirrored as non-recomposable downbins (diagnostic_only and none, respectively).',
                        'Use each layer\'s recomposition_op. Only union entries marked union; select entries are complete checkpoints.',
                    ],
                })
        if not manifest_layers:
            return None
        manifest_path = self.nrrd_dir / f'{self.stem}_nrrd_manifest.json'
        manifest = {
            'layout': 'one_single_layer_nrrd_per_component',
            'exported_axes': '(X, Y, t)',
            'output_shape_tyx': [int(v) for v in self.output_shape],
            'layer_count': len(manifest_layers),
            'layers': manifest_layers,
            'notes': [
                'Each NRRD is one uint8 binary mask in source output geometry (X, Y, t).',
                'v13.2.3: each file is a 3D Slicer segmentation (.seg.nrrd) holding one segment named after the file; segment_color_rgb records the assigned Slicer color.',
                'Recomposition is role-aware: union only layers whose recomposition_op is union; select chooses a complete checkpoint; subtract_from_previous_checkpoint removes that full-quality delta; none is diagnostic only.',
                'YOLO layers are cleaned masks before interpolation bridges; bridge layers contain only voxels added by that pass.',
                'Global checkpoints, including centerline pass00_input/result and Global_final_output, are complete alternatives and must never be unioned with component or audit layers.',
                'The manifest lists only this run. In a reused output directory, similarly named files absent from this manifest are stale and must be ignored.',
            ],
        }
        _write_json_atomically(manifest_path, manifest)
        return manifest_path

_NRRD_LAYER_SINK: Optional[NrrdLayerSink] = None

def set_nrrd_layer_sink(sink: Optional[NrrdLayerSink]) -> None:
    global _NRRD_LAYER_SINK
    _NRRD_LAYER_SINK = sink

def nrrd_layer_sink() -> Optional[NrrdLayerSink]:
    return _NRRD_LAYER_SINK

def nrrd_layer_sink_workers() -> int:
    # default cap 4 -> 12. With ~72 queued files (layers x low-quality mirrors) the
    # 4-slot pool left the box near-idle through the whole write tail; most of the write volume
    # lands after inference has drained, so a wider pool is safe.
    cores = max(1, _cpu_count())
    default_workers = max(1, min(12, max(1, cores // 8)))
    return max(1, _env_int('YOLO_TTA_NRRD_LAYER_SINK_WORKERS', int(default_workers)))

def _nrrd_layer_ref_is_raw_bbox_store(ref: NrrdLayerRef) -> bool:
    return str(getattr(ref, 'storage_format', 'raw_u8')) in MASK_STORE_FORMATS or Path(ref.path).is_dir()

class _LiveArrayLayerSource:
    """Read adapter over an in-process live layer volume.

 Duck-typed for the payload streamer (shape + slicing); NOT a RawBBoxMaskStore, so the
 zero-copy direct-native branch applies when shapes match. The wrapped array is owned by
 the caller (e.g. the final union volume) and must never be closed by the sink."""

    def __init__(self, arr: np.ndarray) -> None:
        self.array = np.asarray(arr)
        self.shape = tuple(int(x) for x in self.array.shape)

    def __getitem__(self, item: object) -> np.ndarray:
        return self.array[item]

def nrrd_live_global_layer_enabled() -> bool:
    """Stream IMMUTABLE global layers straight from the live volume.

 The store-encode pass + store read-back that the final/global layers paid purely to
 hand the sink a path are skipped; the sink reads the caller's in-RAM volume directly
 (one streaming pass) and the segment extent is computed on the sink worker thread.
 YOLO_TTA_NRRD_LIVE_GLOBAL_LAYERS=0 restores the raw-bbox store path."""
    return _env_flag('YOLO_TTA_NRRD_LIVE_GLOBAL_LAYERS', True)

def _resolve_live_ref_extent(ref: NrrdLayerRef) -> NrrdLayerRef:
    """Compute a live-volume layer's deferred segment extent (idempotent).

 Live refs carry segment_extent_ijk=None so the producing thread never pays the scan;
 the first sink worker that needs the header resolves it here (one GIL-releasing
 reduction pass over the live volume) and downstream uses the returned ref."""
    if (
        getattr(ref, 'live_array', None) is None
        or _coerce_segment_extent(getattr(ref, 'segment_extent_ijk', None)) is not None
    ):
        return ref
    live_arr = np.asarray(ref.live_array)
    return dataclasses_replace(
        ref,
        segment_extent_ijk=compute_segment_extent_zyx(live_arr),
        segment_extent_shape_tyx=tuple(int(x) for x in live_arr.shape),
        segment_extent_source='live_volume_sink_scan',
    )

def _open_nrrd_layer_ref(ref: NrrdLayerRef) -> object:
    live = getattr(ref, 'live_array', None)
    if live is not None:
        return _LiveArrayLayerSource(live)  #
    if _nrrd_layer_ref_is_raw_bbox_store(ref):
        return RawBBoxMaskStore.open(
            ref.path,
            cache_payload_in_ram=True,
        )
    return np.memmap(
        ref.path,
        dtype=np.dtype(ref.dtype),
        mode='r',
        shape=tuple(int(x) for x in ref.shape),
    )

def _close_nrrd_layer_source(src: object) -> None:
    if isinstance(src, _LiveArrayLayerSource):
        return  # the live volume belongs to the caller; never close it here
    if isinstance(src, RawBBoxMaskStore):
        src.close()
        return
    close_memmap_array(src)

def _drop_nrrd_raw_store_chunks_ram_cache(src: object) -> None:
    if not isinstance(src, RawBBoxMaskStore):
        return
    # release (refcount) rather than evict — sibling full-quality /
    # low-quality writers for the same layer may still hold the entry.
    if not bool(getattr(src, '_ram_cache_ref_held', False)):
        return
    src._ram_cache_ref_held = False
    _release_raw_store_chunks_ram_cache(src.chunks_path)

def nrrd_parallel_extent_scan_enabled() -> bool:
    """Parallelize exact per-slice SegmentN_Extent reductions."""
    return _env_flag('YOLO_TTA_NRRD_PARALLEL_EXTENT_SCAN', True)

def _compute_segment_extent_zyx_serial(src: np.ndarray) -> Tuple[int, int, int, int, int, int]:
    """Return the exact Slicer segment extent through the serial fallback scan."""
    t_dim, h_dim, w_dim = (int(src.shape[0]), int(src.shape[1]), int(src.shape[2]))
    min_t, max_t = t_dim, -1
    min_y, max_y = h_dim, -1
    min_x, max_x = w_dim, -1

    for t_idx in range(t_dim):
        sl = np.asarray(src[int(t_idx)], dtype=bool)
        if not np.any(sl):
            continue
        row_has_fg = np.any(sl, axis=1)
        col_has_fg = np.any(sl, axis=0)
        ys = np.flatnonzero(row_has_fg)
        xs = np.flatnonzero(col_has_fg)
        if xs.size <= 0 or ys.size <= 0:
            continue
        min_t = min(min_t, int(t_idx))
        max_t = max(max_t, int(t_idx))
        min_y = min(min_y, int(ys[0]))
        max_y = max(max_y, int(ys[-1]))
        min_x = min(min_x, int(xs[0]))
        max_x = max(max_x, int(xs[-1]))

    if max_t < 0:
        return _nrrd_empty_segment_extent()
    return (int(min_x), int(max_x), int(min_y), int(max_y), int(min_t), int(max_t))

def compute_segment_extent_zyx(
    mask_zyx: np.ndarray,
    *,
    workers: Optional[int] = None,
) -> Tuple[int, int, int, int, int, int]:
    """Return the exact Slicer segment extent through parallel per-slice reductions."""
    src = np.asarray(mask_zyx)
    if src.ndim != 3:
        raise ValueError(f'compute_segment_extent_zyx expects a 3D (t,Y,X) layer, got {src.shape}')

    t_dim, h_dim, w_dim = (int(src.shape[0]), int(src.shape[1]), int(src.shape[2]))
    requested_workers = max(1, int(_cpu_count() if workers is None else workers))
    scan_workers = choose_slice_parallel_workers(int(requested_workers), int(t_dim))
    if (
        not nrrd_parallel_extent_scan_enabled()
        or int(t_dim) <= 1
        or int(scan_workers) <= 1
    ):
        return _compute_segment_extent_zyx_serial(src)

    # [min_y, max_y, min_x, max_x] for each t slice. Empty slices retain the
    # max<min sentinel. Each task owns one row, so there is no shared reduction race.
    slice_bounds = np.empty((int(t_dim), 4), dtype=np.int64)
    slice_bounds[:, 0] = np.int64(h_dim)
    slice_bounds[:, 1] = np.int64(-1)
    slice_bounds[:, 2] = np.int64(w_dim)
    slice_bounds[:, 3] = np.int64(-1)

    def _scan_slice(t_idx: int) -> None:
        # Preserve the serial path's truth-value conversion exactly; the pipeline stores
        # uint8 masks, but this also keeps the fallback-equivalence contract for any caller.
        sl = np.asarray(src[int(t_idx)], dtype=bool)
        if not np.any(sl):
            return
        row_has_fg = np.any(sl, axis=1)
        col_has_fg = np.any(sl, axis=0)
        ys = np.flatnonzero(row_has_fg)
        xs = np.flatnonzero(col_has_fg)
        if xs.size <= 0 or ys.size <= 0:
            return
        slice_bounds[int(t_idx), :] = (
            int(ys[0]), int(ys[-1]), int(xs[0]), int(xs[-1]),
        )

    parallel_for_indices_chunked(
        int(t_dim),
        _scan_slice,
        max_workers=int(scan_workers),
        desc='NRRD segment extent scan',
        show_progress=False,
        target_chunks_per_worker=2,
    )

    nonempty = slice_bounds[:, 1] >= 0
    nonempty_t = np.flatnonzero(nonempty)
    if nonempty_t.size <= 0:
        return _nrrd_empty_segment_extent()
    active = slice_bounds[nonempty]
    return (
        int(np.min(active[:, 2])),
        int(np.max(active[:, 3])),
        int(np.min(active[:, 0])),
        int(np.max(active[:, 1])),
        int(nonempty_t[0]),
        int(nonempty_t[-1]),
    )

def _restore_source_indices_for_output_z(in_t: int, out_t: int, out_z: int) -> List[int]:
    in_t_i = max(1, int(in_t))
    out_t_i = max(1, int(out_t))
    out_z_i = int(np.clip(int(out_z), 0, out_t_i - 1))
    if in_t_i >= out_t_i:
        src_start = int(math.floor(float(out_z_i) * float(in_t_i) / float(out_t_i)))
        src_stop = int(math.ceil(float(out_z_i + 1) * float(in_t_i) / float(out_t_i)))
        src_start = int(np.clip(src_start, 0, in_t_i - 1))
        src_stop = int(np.clip(max(src_start + 1, src_stop), 1, in_t_i))
        return [int(v) for v in range(src_start, src_stop)]

    src_z = _linear_source_index(out_z_i, out_t_i, in_t_i)
    return [int(np.clip(int(round(src_z)), 0, in_t_i - 1))]

def _resize_binary_mask_frame_to_output_shape(frame: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    frame_u8 = (np.asarray(frame, dtype=np.uint8) > 0).astype(np.uint8, copy=False)
    if int(frame_u8.shape[0]) == int(out_h) and int(frame_u8.shape[1]) == int(out_w):
        return np.ascontiguousarray(frame_u8)
    interp = cv2.INTER_AREA if (int(frame_u8.shape[0]) >= int(out_h) and int(frame_u8.shape[1]) >= int(out_w)) else cv2.INTER_NEAREST
    scaled = cv2.resize(
        np.ascontiguousarray(frame_u8 * np.uint8(255)),
        (int(out_w), int(out_h)),
        interpolation=int(interp),
    )
    return (scaled > 0).astype(np.uint8, copy=False)

@lru_cache(maxsize=32)
def _nrrd_sparse_resize_axis_map(
    in_count: int,
    out_count: int,
    area: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """Cached global-coordinate source support for every output coordinate.

 The returned inclusive-start/exclusive-stop arrays reproduce OpenCV nearest-neighbor
 sampling or the positive-support footprint of INTER_AREA. Keeping this map global is
 what lets resize a bbox independently without shifting its phase to the crop origin."""
    in_n = max(1, int(in_count))
    out_n = max(1, int(out_count))
    coords = np.arange(int(out_n), dtype=np.int64)
    if bool(area):
        starts = (coords * np.int64(in_n)) // np.int64(out_n)
        stops = (
            ((coords + np.int64(1)) * np.int64(in_n) + np.int64(out_n - 1))
            // np.int64(out_n)
        )
    else:
        starts = (coords * np.int64(in_n)) // np.int64(out_n)
        stops = starts + np.int64(1)
    starts.setflags(write=False)
    stops.setflags(write=False)
    return starts, stops

_NRRD_SPARSE_AREA_NUMBA_FAILED = False

_NRRD_SPARSE_AREA_NUMBA_ANNOUNCED = False

if _numba is not None:
    @_numba.njit(cache=True, nogil=True, inline='always')  # type: ignore[misc]
    def _numba_nrrd_continuous_integral_at(
        integral: np.ndarray,
        crop: np.ndarray,
        y_coord: float,
        x_coord: float,
    ) -> float:  # pragma: no cover - compiled implementation
        crop_h = int(crop.shape[0])
        crop_w = int(crop.shape[1])
        yy = min(float(crop_h), max(0.0, float(y_coord)))
        xx = min(float(crop_w), max(0.0, float(x_coord)))
        yy_round = round(yy)
        xx_round = round(xx)
        if abs(yy - yy_round) < 1e-12:
            yy = float(yy_round)
        if abs(xx - xx_round) < 1e-12:
            xx = float(xx_round)
        iy = int(math.floor(yy))
        ix = int(math.floor(xx))
        fy = float(yy - iy)
        fx = float(xx - ix)
        iy_cell = min(int(iy), int(crop_h - 1))
        ix_cell = min(int(ix), int(crop_w - 1))
        value = float(integral[int(iy), int(ix)])
        value += float(
            integral[int(iy), int(ix_cell + 1)]
            - integral[int(iy), int(ix_cell)]
        ) * fx
        value += float(
            integral[int(iy_cell + 1), int(ix)]
            - integral[int(iy_cell), int(ix)]
        ) * fy
        value += float(crop[int(iy_cell), int(ix_cell)]) * fy * fx
        return float(value)

    @_numba.njit(cache=True, nogil=True)  # type: ignore[misc]
    def _numba_nrrd_area_crop_resize_kernel(
        integral: np.ndarray,
        crop: np.ndarray,
        in_h: int,
        in_w: int,
        out_h: int,
        out_w: int,
        source_y0: int,
        source_x0: int,
        source_y1: int,
        source_x1: int,
        out_y0: int,
        out_x0: int,
        restored: np.ndarray,
    ) -> None:  # pragma: no cover - compiled implementation
        threshold = (
            (float(in_h) / float(out_h)) * (float(in_w) / float(out_w)) / 510.0
        )
        for local_y in range(int(restored.shape[0])):
            out_y = int(out_y0 + local_y)
            sy0 = max(
                float(out_y) * float(in_h) / float(out_h), float(source_y0),
            ) - float(source_y0)
            sy1 = min(
                float(out_y + 1) * float(in_h) / float(out_h), float(source_y1),
            ) - float(source_y0)
            for local_x in range(int(restored.shape[1])):
                out_x = int(out_x0 + local_x)
                sx0 = max(
                    float(out_x) * float(in_w) / float(out_w), float(source_x0),
                ) - float(source_x0)
                sx1 = min(
                    float(out_x + 1) * float(in_w) / float(out_w), float(source_x1),
                ) - float(source_x0)
                weighted = _numba_nrrd_continuous_integral_at(integral, crop, sy1, sx1)
                weighted -= _numba_nrrd_continuous_integral_at(integral, crop, sy0, sx1)
                weighted -= _numba_nrrd_continuous_integral_at(integral, crop, sy1, sx0)
                weighted += _numba_nrrd_continuous_integral_at(integral, crop, sy0, sx0)
                restored[int(local_y), int(local_x)] = np.uint8(
                    1 if float(weighted) >= float(threshold) else 0
                )
else:
    _numba_nrrd_continuous_integral_at = None
    _numba_nrrd_area_crop_resize_kernel = None

def _resize_sparse_binary_crop_to_output_region(
    crop: np.ndarray,
    *,
    source_shape: Tuple[int, int],
    source_bbox: Tuple[int, int, int, int],
    output_shape: Tuple[int, int],
) -> Optional[Tuple[int, int, int, int, np.ndarray]]:
    """Resize one binary source bbox in the *full frame's* coordinate system.

 Returns ``(out_y0, out_x0, out_y1, out_x1, crop)``. The output region includes every
 output pixel whose global sampling footprint can see the source bbox, including the
 one-pixel influence halo that an area downscale can create. Work is proportional to
 the affected region: OpenCV's C integral-image kernel plus vectorized native gathers
 replace a full zero-plane resize and all Python row loops."""
    global _NRRD_SPARSE_AREA_NUMBA_FAILED, _NRRD_SPARSE_AREA_NUMBA_ANNOUNCED
    in_h, in_w = (max(1, int(v)) for v in source_shape)
    out_h, out_w = (max(1, int(v)) for v in output_shape)
    y0, x0, y1, x1 = (int(v) for v in source_bbox)
    crop_arr = np.asarray(crop)
    # RawBBoxMaskStore and restored member planes are already canonical uint8 0/1.
    # Preserve their view (including a bbox-width row stride) instead of making the old
    # compare + astype copy. Non-pipeline callers retain normalization below.
    if crop_arr.dtype == np.uint8:
        crop_u8 = crop_arr
    else:
        crop_u8 = np.ascontiguousarray(crop_arr > 0, dtype=np.uint8)
    if (
        y0 < 0 or x0 < 0 or y1 > int(in_h) or x1 > int(in_w)
        or y1 <= y0 or x1 <= x0
        or tuple(int(v) for v in crop_u8.shape) != (int(y1 - y0), int(x1 - x0))
    ):
        raise ValueError(
            f'Invalid sparse resize bbox {(y0, x0, y1, x1)} / crop {crop_u8.shape} '
            f'for source {(in_h, in_w)}'
        )

    if int(in_h) == int(out_h) and int(in_w) == int(out_w):
        return int(y0), int(x0), int(y1), int(x1), np.ascontiguousarray(crop_u8)

    use_area = bool(int(in_h) >= int(out_h) and int(in_w) >= int(out_w))
    y_starts, y_stops = _nrrd_sparse_resize_axis_map(int(in_h), int(out_h), bool(use_area))
    x_starts, x_stops = _nrrd_sparse_resize_axis_map(int(in_w), int(out_w), bool(use_area))

    # Support intervals are monotonic. searchsorted finds the affected output bbox without
    # allocating an out_h/out_w boolean mask for every source crop.
    out_y0 = int(np.searchsorted(y_stops, int(y0), side='right'))
    out_y1 = int(np.searchsorted(y_starts, int(y1), side='left'))
    out_x0 = int(np.searchsorted(x_stops, int(x0), side='right'))
    out_x1 = int(np.searchsorted(x_starts, int(x1), side='left'))
    out_y0 = max(0, min(int(out_y0), int(out_h)))
    out_y1 = max(int(out_y0), min(int(out_y1), int(out_h)))
    out_x0 = max(0, min(int(out_x0), int(out_w)))
    out_x1 = max(int(out_x0), min(int(out_x1), int(out_w)))
    if int(out_y1) <= int(out_y0) or int(out_x1) <= int(out_x0):
        return None

    if not bool(use_area):
        src_y = y_starts[int(out_y0):int(out_y1)] - np.int64(y0)
        src_x = x_starts[int(out_x0):int(out_x1)] - np.int64(x0)
        restored = crop_u8[src_y[:, None], src_x[None, :]]
    else:
        # Reproduce INTER_AREA's global phase and uint8 rounding threshold without padding
        # a full source plane. cv2.integral is a compiled, GIL-releasing prefix pass over
        # the bbox; evaluating its continuous piecewise-constant integral accounts for
        # fractional first/last source pixels (a plain any would over-include tiny slivers).
        integral = cv2.integral(crop_u8, sdepth=cv2.CV_32S)
        crop_h, crop_w = (int(v) for v in crop_u8.shape)
        restored = np.empty(
            (int(out_y1 - out_y0), int(out_x1 - out_x0)), dtype=np.uint8,
        )
        compiled_ok = bool(
            _numba_nrrd_area_crop_resize_kernel is not None
            and not _NRRD_SPARSE_AREA_NUMBA_FAILED
        )
        if compiled_ok:
            try:
                _numba_nrrd_area_crop_resize_kernel(
                    integral,
                    crop_u8,
                    int(in_h), int(in_w), int(out_h), int(out_w),
                    int(y0), int(x0), int(y1), int(x1),
                    int(out_y0), int(out_x0), restored,
                )
                if not _NRRD_SPARSE_AREA_NUMBA_ANNOUNCED:
                    _NRRD_SPARSE_AREA_NUMBA_ANNOUNCED = True
                    print(
                        'v13.3.17 (N24): NRRD sparse INTER_AREA member assembly uses '
                        'the compiled no-GIL crop kernel.'
                    )
            except Exception as exc:
                _NRRD_SPARSE_AREA_NUMBA_FAILED = True
                compiled_ok = False
                print(
                    f'Warning: compiled NRRD sparse area resize unavailable ({exc}); '
                    'using vectorized OpenCV/NumPy assembly.'
                )
        if bool(compiled_ok):
            if not np.any(restored):
                return None
            return (
                int(out_y0), int(out_x0), int(out_y1), int(out_x1),
                np.ascontiguousarray(restored, dtype=np.uint8),
            )

        out_ys = np.arange(int(out_y0), int(out_y1), dtype=np.float64)
        out_xs = np.arange(int(out_x0), int(out_x1), dtype=np.float64)
        src_y0 = np.maximum(out_ys * float(in_h) / float(out_h), float(y0)) - float(y0)
        src_y1 = np.minimum((out_ys + 1.0) * float(in_h) / float(out_h), float(y1)) - float(y0)
        src_x0 = np.maximum(out_xs * float(in_w) / float(out_w), float(x0)) - float(x0)
        src_x1 = np.minimum((out_xs + 1.0) * float(in_w) / float(out_w), float(x1)) - float(x0)

        def _continuous_integral(y_coords: np.ndarray, x_coords: np.ndarray) -> np.ndarray:
            yy = np.clip(np.asarray(y_coords, dtype=np.float64), 0.0, float(crop_h))
            xx = np.clip(np.asarray(x_coords, dtype=np.float64), 0.0, float(crop_w))
            # Snap rational coordinates which should be integers; this avoids selecting
            # the preceding cell because of a 1-ulp division artifact.
            yy_round = np.rint(yy)
            xx_round = np.rint(xx)
            yy = np.where(np.abs(yy - yy_round) < 1e-12, yy_round, yy)
            xx = np.where(np.abs(xx - xx_round) < 1e-12, xx_round, xx)
            iy = np.floor(yy).astype(np.int64)
            ix = np.floor(xx).astype(np.int64)
            fy = yy - iy
            fx = xx - ix
            iy_cell = np.minimum(iy, int(crop_h - 1))
            ix_cell = np.minimum(ix, int(crop_w - 1))
            value = integral[iy[:, None], ix[None, :]].astype(np.float64)
            value += (
                integral[iy[:, None], (ix_cell + 1)[None, :]]
                - integral[iy[:, None], ix_cell[None, :]]
            ) * fx[None, :]
            value += (
                integral[(iy_cell + 1)[:, None], ix[None, :]]
                - integral[iy_cell[:, None], ix[None, :]]
            ) * fy[:, None]
            value += (
                crop_u8[iy_cell[:, None], ix_cell[None, :]]
                * fy[:, None] * fx[None, :]
            )
            return value

        # Bound float64 temporaries while leaving each batch as a few large native gathers.
        cols = max(1, int(restored.shape[1]))
        rows_per_batch = max(1, min(
            int(restored.shape[0]),
            (24 * 1024 * 1024) // max(1, int(cols) * 8 * 3),
        ))
        positive_threshold = (
            (float(in_h) / float(out_h)) * (float(in_w) / float(out_w)) / 510.0
        )
        for row0 in range(0, int(restored.shape[0]), int(rows_per_batch)):
            row1 = min(int(restored.shape[0]), int(row0) + int(rows_per_batch))
            weighted = _continuous_integral(src_y1[row0:row1], src_x1)
            weighted -= _continuous_integral(src_y0[row0:row1], src_x1)
            weighted -= _continuous_integral(src_y1[row0:row1], src_x0)
            weighted += _continuous_integral(src_y0[row0:row1], src_x0)
            restored[row0:row1] = (weighted >= float(positive_threshold)).astype(
                np.uint8, copy=False,
            )
    if not np.any(restored):
        return None
    return (
        int(out_y0), int(out_x0), int(out_y1), int(out_x1),
        np.ascontiguousarray(restored, dtype=np.uint8),
    )

@runtime_telemetry_phase('restore.read_slice')
def _read_layer_slice_in_output_shape(
    src: object,
    output_shape: Tuple[int, int, int],
    out_z: int,
) -> np.ndarray:
    """Return one output-geometry ``(Y,X)`` slice for an NRRD layer without materializing the full layer."""
    in_t, in_h, in_w = _volume_shape_tuple(src)
    out_t, out_h, out_w = (int(output_shape[0]), int(output_shape[1]), int(output_shape[2]))
    out_z_i = int(np.clip(int(out_z), 0, out_t - 1))

    if (in_t, in_h, in_w) == (out_t, out_h, out_w):
        return _read_binary_volume_slice_u8(src, out_z_i)

    source_indices = _restore_source_indices_for_output_z(in_t, out_t, out_z_i)
    if in_h == out_h and in_w == out_w:
        if len(source_indices) == 1:
            return _read_binary_volume_slice_u8(src, int(source_indices[0]))
        restored = np.zeros((out_h, out_w), dtype=np.uint8)
        for src_idx in source_indices:
            restored |= _read_binary_volume_slice_bool(src, int(src_idx)).astype(np.uint8, copy=False)
        return restored

    restored = np.zeros((out_h, out_w), dtype=np.uint8)
    for src_idx in source_indices:
        restored |= _resize_binary_mask_frame_to_output_shape(_read_binary_volume_slice_u8(src, int(src_idx)), out_h, out_w)
    return restored

def nrrd_extent_zero_skip_enabled() -> bool:
    """Payload z-ranges outside the layer's recorded segment extent are
 emitted as cached zero members/chunks without reading (or even faulting in) the source
 pages. YOLO_TTA_NRRD_EXTENT_ZERO_SKIP=0 restores full-volume streaming."""
    return _env_flag('YOLO_TTA_NRRD_EXTENT_ZERO_SKIP', True)

def _nrrd_layer_zero_skip_window(
    ref: NrrdLayerRef,
    output_shape: Tuple[int, int, int],
) -> Optional[Tuple[int, int]]:
    """Output-space t-window ``[lo, hi)`` that MAY contain foreground, from the layer extent.

 Returns None when no extent is recorded (nothing can be skipped) and ``(0, 0)`` when
 the recorded extent is the scanned-empty sentinel — every extent source
 (raw_bbox_cvol_index / raw_layer_materialization_scan / live_volume_sink_scan) records
 genuinely scanned data, so empty means the whole payload is zeros. Note the contrast
 with _slicer_segment_extent_for_output, which maps empty to FULL because Slicer needs
 a valid display extent; for zero-skipping the empty sentinel is trustworthy as-is.

 The t-axis window inverts the ACTUAL restore mappings rather than reusing the header's
 outward display scaling: for in_t >= out_t each output z reads sources
 [floor(z*in/out), ceil((z+1)*in/out)) — the floor/ceil bounds below are exact for that;
 for in_t < out_t each output z reads round(z*(in-1)/(out-1)) (endpoint-aligned
 _linear_source_index, whose slope exceeds out/in — the display scaling would UNDERSHOOT
 it), so the window inverts that line at t0-0.5 / t1+0.5. One slice of padding per side
 absorbs any residual rounding-rule differences; skipping must never reclassify a
 foreground slice as zero."""
    if not nrrd_extent_zero_skip_enabled():
        return None
    extent = _coerce_segment_extent(getattr(ref, 'segment_extent_ijk', None))
    if extent is None:
        return None
    out_t = int(output_shape[0])
    x0, x1, y0, y1, t0, t1 = (int(v) for v in extent)
    if x1 < x0 or y1 < y0 or t1 < t0:
        return (0, 0)
    in_t = int(getattr(ref, 'segment_extent_shape_tyx', (0, 0, 0))[0])
    if in_t <= 0 or out_t <= 0:
        return None
    if in_t >= out_t:
        # Downscale/identity restore: output z draws sources [floor(z*in/out), ceil((z+1)*in/out)).
        lo_f = int(math.floor(float(t0) * float(out_t) / float(in_t)))
        hi_f = int(math.ceil(float(t1 + 1) * float(out_t) / float(in_t))) - 1
    elif in_t == 1:
        # Degenerate upscale: every output z draws slice 0, which the extent says is nonzero.
        lo_f, hi_f = 0, out_t - 1
    else:
        # Upscale restore: output z draws round(z * (in-1)/(out-1)).
        slope = float(out_t - 1) / float(in_t - 1)
        lo_f = int(math.floor((float(t0) - 0.5) * slope))
        hi_f = int(math.ceil((float(t1) + 0.5) * slope))
    lo = max(0, int(lo_f) - 1)
    hi = min(int(out_t), int(hi_f) + 2)
    if hi <= lo:
        return (0, 0)
    return (int(lo), int(hi))

def _write_one_decomposed_nrrd_layer_payload(
    ref: NrrdLayerRef,
    output_shape: Tuple[int, int, int],
    payload_writer: object,
    *,
    z_chunk: int,
    pbar: Optional[object] = None,
    layer_idx: int = 0,
    block_consumer: Optional[Callable[[int, np.ndarray], None]] = None,
    sparse_consumer: Optional[Callable[[int, int, int, np.ndarray], None]] = None,
    z_start: int = 0,
    z_stop: Optional[int] = None,
    fill_workers_override: Optional[int] = None,
) -> None:
    """Stream one restored NRRD layer in bounded output-z blocks.
    
    Optional consumers observe each block, and known-zero ranges bypass source reads."""
    out_t, out_h, out_w = (int(output_shape[0]), int(output_shape[1]), int(output_shape[2]))
    z_begin = int(np.clip(int(z_start), 0, out_t))
    z_end = out_t if z_stop is None else int(np.clip(int(z_stop), z_begin, out_t))
    if z_end <= z_begin:
        return
    z_chunk_i = max(1, int(z_chunk))
    madvise_interval = nrrd_madvise_dontneed_interval()
    src: Optional[object] = None
    try:
        src = _open_nrrd_layer_ref(ref)
        _madvise_array_mmap(src, 'MADV_SEQUENTIAL')
        in_t, in_h, in_w = _volume_shape_tuple(src)
        direct_native_stream = (
            (int(in_t), int(in_h), int(in_w)) == (int(out_t), int(out_h), int(out_w))
            and not isinstance(src, RawBBoxMaskStore)
        )
        raw_store_stream = isinstance(src, RawBBoxMaskStore)
        raw_store_native_stream = bool(
            raw_store_stream
            and (int(in_t), int(in_h), int(in_w)) == (int(out_t), int(out_h), int(out_w))
        )
        # z-blocks fully outside this window are emitted as cached zero
        # members/chunks through write_zeros — no fill, no tee, no source page reads.
        zero_window = _nrrd_layer_zero_skip_window(ref, (out_t, out_h, out_w))
        can_write_zeros = callable(getattr(payload_writer, 'write_zeros', None))

        def _block_is_known_zero(z0: int, z1: int) -> bool:
            return (
                zero_window is not None
                and can_write_zeros
                and (int(z1) <= int(zero_window[0]) or int(z0) >= int(zero_window[1]))
            )

        if bool(raw_store_stream):
            # native AND restored cvol sources share one sparse path.
            # Members end only at output-slice boundaries, large assembly is vectorized
            # in native libraries, and an owned member can transfer directly to the async
            # compressor. The sparse mirror consumer observes crops during this same pass.
            member_bytes = int(np.clip(
                int(nrrd_gzip_chunk_bytes()),
                8 * 1024 * 1024,
                16 * 1024 * 1024,
            ))
            write_nonzero = getattr(payload_writer, 'write_owned_known_nonzero', None)
            if not callable(write_nonzero):
                write_nonzero = getattr(payload_writer, 'write_known_nonzero', None)
            if not callable(write_nonzero):
                write_nonzero = payload_writer.write
            write_aligned_zeros = getattr(payload_writer, 'write_aligned_zeros', None)
            if bool(raw_store_native_stream):
                members = src.iter_native_sparse_members(
                    int(z_begin), int(z_end), member_bytes=int(member_bytes),
                    sparse_consumer=sparse_consumer,
                )
            else:
                members = src.iter_restored_sparse_members(
                    (int(out_t), int(out_h), int(out_w)),
                    int(z_begin), int(z_end), member_bytes=int(member_bytes),
                    sparse_consumer=sparse_consumer,
                )
            member_z = int(z_begin)
            for raw_len, member in members:
                member_slices = int(raw_len) // max(1, int(out_h) * int(out_w))
                if member is None and can_write_zeros:
                    if callable(write_aligned_zeros):
                        write_aligned_zeros(int(raw_len))
                    else:
                        payload_writer.write_zeros(int(raw_len))
                elif member is None:
                    payload_writer.write(bytes(int(raw_len)))
                else:
                    write_nonzero(member)
                    # Preserve the generic consumer contract without the old second store
                    # pass: the aligned member is already an output-geometry dense block.
                    if block_consumer is not None:
                        block_consumer(
                            int(member_z),
                            np.asarray(member, dtype=np.uint8).reshape(
                                (int(member_slices), int(out_h), int(out_w)),
                            ),
                        )
                member_z += int(member_slices)
            if pbar is not None:
                pbar.update(int(z_end - z_begin))
            return

        if bool(direct_native_stream):
            # The memmap/live volume is already a native (t,Y,X) C-order layer: chunks are
            # views written directly — no transpose, Fortran conversion, or layer interleave.
            # Compression runs on a one-thread writer pool while this thread feeds the mirror
            # tee from the same read-only block, overlapping tee and deflate.
            slice_bytes = int(out_h) * int(out_w)
            writer_pool = _acquire_parallel_pool(1)
            pending: Optional[Future] = None
            try:
                for z0 in range(z_begin, z_end, int(z_chunk_i)):
                    z1 = min(z_end, int(z0) + int(z_chunk_i))
                    if _block_is_known_zero(int(z0), int(z1)):
                        if pending is not None:
                            pending.result()
                        pending = writer_pool.submit(payload_writer.write_zeros, int(z1 - z0) * slice_bytes)
                        if pbar is not None:
                            pbar.update(int(z1 - z0))
                        continue
                    chunk = np.asarray(src[int(z0):int(z1)], dtype=np.uint8)
                    if not chunk.flags['C_CONTIGUOUS']:
                        chunk = np.ascontiguousarray(chunk)
                    if pending is not None:
                        pending.result()
                    pending = writer_pool.submit(payload_writer.write, memoryview(chunk).cast('B'))
                    if block_consumer is not None:
                        block_consumer(int(z0), chunk)
                    if pbar is not None:
                        pbar.update(int(z1 - z0))
                    if madvise_interval > 0 and (int(z1) % int(madvise_interval) == 0):
                        # The writer must be done with these pages before dropping them.
                        pending.result()
                        pending = None
                        _madvise_array_mmap(src, 'MADV_DONTNEED')
                if pending is not None:
                    pending.result()
                    pending = None
            finally:
                if pending is not None:
                    try:
                        pending.result()
                    except Exception:
                        pass
                _release_parallel_pool(1, writer_pool)
            return

        # shared bounded-block streamer — each block is filled by a small
        # GIL-releasing thread pool (store decode memcpys / cv2 restores) and handed to a
        # one-thread writer with double buffering, so decode/restore overlaps the
        # selected validated gzip backend instead of filling the whole payload first.
        fill_workers = (
            int(nrrd_fill_workers())
            if fill_workers_override is None
            else max(1, int(fill_workers_override))
        )

        def _stream_filled_blocks(fill_one: Callable[[int, np.ndarray], None], *, madvise_src: bool) -> None:
            local_z_chunk = int(z_chunk_i)
            buffers: List[np.ndarray] = []
            for _ in range(2):
                try:
                    buffers.append(np.empty((int(local_z_chunk), int(out_h), int(out_w)), dtype=np.uint8, order='C'))
                except MemoryError:
                    break
            if not buffers:
                if int(local_z_chunk) != 1:
                    print(
                        f'Warning: NRRD payload block allocation failed for layer {int(layer_idx)}; '
                        'falling back to one output t-slice at a time.'
                    )
                local_z_chunk = 1
                buffers = [np.empty((1, int(out_h), int(out_w)), dtype=np.uint8, order='C')]

            # checkout-cached single-thread writer pool (was one build per layer).
            writer_pool = _acquire_parallel_pool(1)
            pending: Optional[Future] = None
            # buffers only cycle for blocks that are actually filled;
            # known-zero blocks go through write_zeros with no buffer at all. The counter
            # (not the z index) keeps the double-buffer alternation valid across skips:
            # at most one write is outstanding, so the buffer being refilled always
            # finished its own previous write before the intervening submit happened.
            buf_cycle = 0
            try:
                for z0 in range(z_begin, z_end, int(local_z_chunk)):
                    z1 = min(z_end, int(z0) + int(local_z_chunk))
                    z_count = int(z1 - z0)
                    if _block_is_known_zero(int(z0), int(z1)):
                        if pending is not None:
                            pending.result()
                        pending = writer_pool.submit(
                            payload_writer.write_zeros, int(z_count) * int(out_h) * int(out_w),
                        )
                        if pbar is not None:
                            pbar.update(int(z_count))
                        continue
                    if len(buffers) < 2 and pending is not None:
                        # Single-buffer fallback: the previous write must finish before refill.
                        pending.result()
                        pending = None
                    block = buffers[int(buf_cycle) % len(buffers)][:z_count, :, :]
                    buf_cycle += 1

                    if int(fill_workers) > 1 and int(z_count) > 1:
                        def _fill(zi: int, _z0: int = int(z0), _block: np.ndarray = block) -> None:
                            fill_one(int(_z0 + int(zi)), _block[int(zi)])
                        _nrrd_parallel_fill_indices(
                            int(z_count),
                            _fill,
                            requested_workers=min(int(fill_workers), int(z_count)),
                        )
                    else:
                        for zi in range(int(z_count)):
                            fill_one(int(z0 + zi), block[int(zi)])

                    if pending is not None:
                        pending.result()
                    pending = writer_pool.submit(payload_writer.write, memoryview(block).cast('B'))
                    # the mirror tee reads the block on this thread while the
                    # writer thread compresses it — both are read-only; the next refill of this
                    # buffer is still gated on pending.result two iterations out.
                    if block_consumer is not None:
                        block_consumer(int(z0), block)
                    if pbar is not None:
                        pbar.update(int(z_count))
                    if bool(madvise_src) and madvise_interval > 0 and (int(z1) % int(madvise_interval) == 0):
                        _madvise_array_mmap(src, 'MADV_DONTNEED')
                if pending is not None:
                    pending.result()
                    pending = None
            finally:
                if pending is not None:
                    try:
                        pending.result()
                    except Exception:
                        pass
                _release_parallel_pool(1, writer_pool)

        def _fill_restored(z: int, out2d: np.ndarray) -> None:
            np.copyto(out2d, np.asarray(_read_layer_slice_in_output_shape(src, output_shape, int(z)), dtype=np.uint8))

        _stream_filled_blocks(_fill_restored, madvise_src=True)
    finally:
        if src is not None:
            _madvise_array_mmap(src, 'MADV_DONTNEED')
            _close_nrrd_layer_source(src)
            _drop_nrrd_raw_store_chunks_ram_cache(src)

@dataclass(frozen=True)
class LowQualityDownbinSpec:
    raw_value: str
    token: str
    scale: float
    output_shape_t_y_x: Tuple[int, int, int]
    warning: str = ''

def _nearest_multiple_of_four(value: float) -> int:
    return max(4, int(math.floor(float(value) / 4.0 + 0.5)) * 4)

def _round_low_quality_dimension(value: float) -> int:
    """Round one isotropically scaled dimension to the nearest positive multiple of four."""
    return _nearest_multiple_of_four(max(1.0, float(value)))

def _low_quality_token(raw: str, shape_t_y_x: Tuple[int, int, int]) -> str:
    safe_raw = str(raw).strip().replace('-', 'm').replace('.', 'p').replace(',', '_')
    t_dim, h_dim, w_dim = (int(shape_t_y_x[0]), int(shape_t_y_x[1]), int(shape_t_y_x[2]))
    return f'{safe_raw}_{int(w_dim)}x{int(h_dim)}x{int(t_dim)}'

def resolve_low_quality_downbin_specs(
    downbin_values: Sequence[str] | str | None,
    low_quality_requested: bool,
    source_shape_t_y_x: Tuple[int, int, int],
) -> Tuple[List[LowQualityDownbinSpec], List[str]]:
    """Resolve isotropic low-quality downbins in native input geometry."""
    if downbin_values is None and not bool(low_quality_requested):
        return [], []

    raw_tokens = _parse_token_list(downbin_values) if downbin_values is not None else []
    if not raw_tokens:
        raw_tokens = ['1024']

    in_t, in_h, in_w = (int(source_shape_t_y_x[0]), int(source_shape_t_y_x[1]), int(source_shape_t_y_x[2]))
    max_dim = max(1, int(in_t), int(in_h), int(in_w))
    specs: List[LowQualityDownbinSpec] = []
    warnings: List[str] = []
    seen_shapes: set[Tuple[int, int, int]] = set()

    for raw in raw_tokens:
        raw_s = str(raw).strip()
        if not raw_s:
            continue
        try:
            value = float(raw_s)
        except Exception as exc:
            raise ValueError(f'--save low_quality downbin is not numeric: {raw_s!r}') from exc
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError('--save low_quality downbins must be positive finite numbers')

        warning = ''
        raw_lower = raw_s.lower()
        looks_float = ('.' in raw_s) or ('e' in raw_lower)
        if looks_float:
            if not (0.0 < float(value) < 1.0):
                raise ValueError(
                    '--save low_quality floating-point downbins must be less than 1.0; '
                    f'got {raw_s!r}'
                )
            scale = float(value)
        else:
            nearest_int = int(round(value))
            if abs(float(value) - float(nearest_int)) > 1e-6:
                raise ValueError('--save low_quality integer downbins must be whole numbers; use a fraction such as 0.5 for scale factors')
            if nearest_int <= 0:
                raise ValueError('--save low_quality integer targets must be positive')
            rounded_target = _nearest_multiple_of_four(float(nearest_int))
            if int(rounded_target) != int(nearest_int):
                warning = (
                    f'--save low_quality:{int(nearest_int)} is not a multiple of 4; '
                    f'rounded to {int(rounded_target)} for isotropic low-quality output.'
                )
                warnings.append(warning)
            scale = float(rounded_target) / float(max_dim)

        out_t = _round_low_quality_dimension(float(in_t) * float(scale))
        out_h = _round_low_quality_dimension(float(in_h) * float(scale))
        out_w = _round_low_quality_dimension(float(in_w) * float(scale))
        shape = (int(out_t), int(out_h), int(out_w))
        if shape in seen_shapes:
            continue
        seen_shapes.add(shape)
        specs.append(LowQualityDownbinSpec(
            raw_value=raw_s,
            token=_low_quality_token(raw_s, shape),
            scale=float(scale),
            output_shape_t_y_x=shape,
            warning=warning,
        ))

    return specs, warnings

def low_quality_gpu_downbin_enabled() -> bool:
    return _env_flag('YOLO_TTA_LOW_QUALITY_GPU_DOWNBIN', True)

def low_quality_gpu_downbin_chunk_slices() -> int:
    return max(1, _env_int('YOLO_TTA_LOW_QUALITY_GPU_DOWNBIN_CHUNK', 32))

_LQ_GPU_DOWNBIN_ANNOUNCED = False

def _try_gpu_downbin_volume(src_vol: np.ndarray, out_mm: np.ndarray, mode: str) -> bool:
    """Fill ``out_mm`` on an exclusively leased GPU; False selects the CPU path."""
    if not low_quality_gpu_downbin_enabled():
        return False
    # PyTorch adaptive pooling does not reproduce OpenCV INTER_AREA's fractional-bin
    # support or its uint8 rounding.  Gray distribution assets are an external output
    # contract, so keep them on the reference CPU/OpenCV path until an exact CUDA kernel
    # is available.  Binary-mask max pooling retains its existing GPU acceleration.
    if str(mode) == 'gray':
        return False
    in_t, in_h, in_w = (int(src_vol.shape[0]), int(src_vol.shape[1]), int(src_vol.shape[2]))
    out_t, out_h, out_w = (int(out_mm.shape[0]), int(out_mm.shape[1]), int(out_mm.shape[2]))
    if out_h > in_h or out_w > in_w:
        return False
    try:
        import torch  # type: ignore
        import torch.nn.functional as F  # type: ignore
        if not bool(torch.cuda.is_available()):
            return False
    except Exception:
        return False
    lease = _try_acquire_main_process_gpu_stage(torch, f'low-quality {mode} GPU downbin')
    if lease is None:
        _announce_main_gpu_stage_skip_once(
            'low-quality-downbin-inference-busy',
            'GPU low-quality downbin skipped while all eligible GPUs have active/queued '
            'inference or another output-stage lease; using the CPU resize path.',
        )
        return False
    try:
        return _try_gpu_downbin_volume_on_device(
            src_vol,
            out_mm,
            str(mode),
            torch=torch,
            F=F,
            device=lease.torch_device(torch),
        )
    finally:
        lease.release()

def _try_gpu_downbin_volume_on_device(
    src_vol: np.ndarray,
    out_mm: np.ndarray,
    mode: str,
    *,
    torch: object,
    F: object,
    device: object,
) -> bool:
    if str(mode) == 'gray':
        return False
    in_t, in_h, in_w = (int(src_vol.shape[0]), int(src_vol.shape[1]), int(src_vol.shape[2]))
    out_t, out_h, out_w = (int(out_mm.shape[0]), int(out_mm.shape[1]), int(out_mm.shape[2]))
    chunk = int(low_quality_gpu_downbin_chunk_slices())
    src_per_chunk = int(math.ceil(float(in_t) / float(max(1, out_t)) * float(chunk))) + 2
    fp_bytes = 4 if mode == 'gray' else 2
    need = (
        src_per_chunk * in_h * in_w * (1 + fp_bytes)
        + (chunk + 2) * out_h * out_w * (fp_bytes + 1)
        + GIB
    )
    try:
        free_bytes, _total = torch.cuda.mem_get_info(device)
    except Exception:
        return False
    if int(free_bytes) < int(need):
        return False

    blk = pooled = out_chunk = window = None
    rows: List[object] = []
    try:
        with torch.inference_mode():
            with torch.cuda.device(device):
                for o0 in range(0, out_t, chunk):
                    blk = pooled = out_chunk = window = None
                    rows = []
                    o1 = min(out_t, int(o0) + chunk)
                    if mode == 'gray':
                        src_pos = [
                            _linear_source_index(int(oz), int(out_t), int(in_t))
                            for oz in range(o0, o1)
                        ]
                        z0s = [int(math.floor(p)) for p in src_pos]
                        z1s = [min(in_t - 1, int(z) + 1) for z in z0s]
                        lo, hi = min(z0s), max(z1s) + 1
                        blk = torch.from_numpy(np.ascontiguousarray(src_vol[lo:hi])).to(device)
                        pooled = F.adaptive_avg_pool2d(
                            blk.to(torch.float32).unsqueeze(1), (int(out_h), int(out_w)),
                        ).squeeze(1)
                        for i in range(int(o1 - o0)):
                            alpha = float(src_pos[i] - float(z0s[i]))
                            f0 = pooled[int(z0s[i] - lo)]
                            if z1s[i] == z0s[i] or alpha <= 1e-7:
                                rows.append(f0)
                            else:
                                rows.append(torch.lerp(f0, pooled[int(z1s[i] - lo)], alpha))
                        out_chunk = torch.stack(rows).round_().clamp_(0, 255).to(torch.uint8)
                    elif mode == 'mask':
                        starts: List[int] = []
                        stops: List[int] = []
                        for oz in range(o0, o1):
                            s0 = int(math.floor(float(oz) * float(in_t) / float(out_t)))
                            s1 = int(math.ceil(float(oz + 1) * float(in_t) / float(out_t)))
                            s0 = int(np.clip(s0, 0, in_t - 1))
                            s1 = int(np.clip(max(s0 + 1, s1), 1, in_t))
                            starts.append(s0)
                            stops.append(s1)
                        lo, hi = min(starts), max(stops)
                        blk = torch.from_numpy(np.ascontiguousarray(src_vol[lo:hi])).to(device)
                        pooled = F.adaptive_max_pool2d(
                            blk.to(torch.float16).unsqueeze(1), (int(out_h), int(out_w)),
                        ).squeeze(1)
                        for i in range(int(o1 - o0)):
                            window = pooled[int(starts[i] - lo):int(stops[i] - lo)]
                            rows.append(window.amax(dim=0) if int(window.shape[0]) > 1 else window[0])
                        out_chunk = (torch.stack(rows) > 0).to(torch.uint8)
                    else:
                        return False
                    out_mm[int(o0):int(o1)] = out_chunk.cpu().numpy()
        global _LQ_GPU_DOWNBIN_ANNOUNCED
        if not _LQ_GPU_DOWNBIN_ANNOUNCED:
            _LQ_GPU_DOWNBIN_ANNOUNCED = True
            print(
                f'GPU low-quality downbin active on {device} '
                '(YOLO_TTA_LOW_QUALITY_GPU_DOWNBIN=0 disables).'
            )
        return True
    except Exception as exc:
        print(f'Warning: GPU low-quality downbin failed ({exc}); using the CPU resize path.')
        return False
    finally:
        rows.clear()
        blk = pooled = out_chunk = window = None
        _trim_main_process_cuda_device(
            torch,
            device,
            desc=f'low-quality {mode} GPU downbin cleanup',
        )

def resize_gray_volume_to_shape(
    volume_gray: np.ndarray,
    out_shape: Tuple[int, int, int],
    out_path: Path,
    *,
    workers: int = 1,
    prefer_memory: bool = True,
    reserve_bytes: int = 8 * GIB,
    desc: str = 'Resizing gray volume',
) -> np.ndarray:
    in_t, in_h, in_w = (int(volume_gray.shape[0]), int(volume_gray.shape[1]), int(volume_gray.shape[2]))
    out_t, out_h, out_w = (int(out_shape[0]), int(out_shape[1]), int(out_shape[2]))
    if (in_t, in_h, in_w) == (out_t, out_h, out_w):
        return volume_gray

    out_mm = allocate_workspace_array(
        shape=(out_t, out_h, out_w),
        dtype=np.uint8,
        path=out_path,
        desc=desc,
        prefer_memory=bool(prefer_memory),
        reserve_bytes=int(reserve_bytes),
    )
    # downbin on an idle GPU when possible; the per-slice CPU path below is
    # the unchanged fallback (it rewrites every slice, so a failed GPU pass cannot leak).
    if _try_gpu_downbin_volume(volume_gray, out_mm, 'gray'):
        flush_array(out_mm)
        return out_mm

    worker_count = choose_slice_parallel_workers(int(workers), int(out_t))
    chunk_size = choose_parallel_chunk_size(out_t, worker_count, target_chunks_per_worker=2, min_chunk_size=1)
    xy_interp = cv2.INTER_AREA if (out_h <= in_h and out_w <= in_w) else cv2.INTER_LINEAR

    def _render_target_slice(out_z: int) -> None:
        src_z = _linear_source_index(int(out_z), int(out_t), int(in_t))
        z0 = int(math.floor(src_z))
        z1 = min(in_t - 1, z0 + 1)
        alpha = float(src_z - float(z0))
        f0 = _resize_gray_slice_nearest_or_linear(volume_gray[z0], out_w, out_h, xy_interp)
        if z1 == z0 or alpha <= 1e-7:
            out_mm[int(out_z), :, :] = f0
            return
        f1 = _resize_gray_slice_nearest_or_linear(volume_gray[z1], out_w, out_h, xy_interp)
        blended = np.clip(
            np.rint((1.0 - alpha) * f0.astype(np.float32, copy=False) + alpha * f1.astype(np.float32, copy=False)),
            0.0,
            255.0,
        ).astype(np.uint8)
        out_mm[int(out_z), :, :] = blended

    parallel_for_indices_chunked(
        int(out_t),
        _render_target_slice,
        max_workers=worker_count,
        desc=desc,
        chunk_size=chunk_size,
        show_progress=True,
    )
    flush_array(out_mm)
    return out_mm

def resize_binary_mask_volume_to_shape(
    mask_u8: np.ndarray,
    out_shape: Tuple[int, int, int],
    out_path: Path,
    *,
    workers: int = 1,
    prefer_memory: bool = True,
    reserve_bytes: int = 8 * GIB,
    desc: str = 'Resizing binary mask volume',
) -> np.ndarray:
    in_t, in_h, in_w = (int(mask_u8.shape[0]), int(mask_u8.shape[1]), int(mask_u8.shape[2]))
    out_t, out_h, out_w = (int(out_shape[0]), int(out_shape[1]), int(out_shape[2]))
    if (in_t, in_h, in_w) == (out_t, out_h, out_w):
        return mask_u8

    out_mm = allocate_workspace_array(
        shape=(out_t, out_h, out_w),
        dtype=np.uint8,
        path=out_path,
        desc=desc,
        prefer_memory=bool(prefer_memory),
        reserve_bytes=int(reserve_bytes),
    )
    # downbin on an idle GPU when possible; CPU fallback below is unchanged.
    if _try_gpu_downbin_volume(mask_u8, out_mm, 'mask'):
        flush_array(out_mm)
        return out_mm

    worker_count = choose_slice_parallel_workers(int(workers), int(out_t))
    chunk_size = choose_parallel_chunk_size(out_t, worker_count, target_chunks_per_worker=2, min_chunk_size=1)

    def _resize_mask_xy(frame: np.ndarray) -> np.ndarray:
        frame_u8 = (np.asarray(frame, dtype=np.uint8) > 0).astype(np.uint8, copy=False)
        if int(frame_u8.shape[0]) == int(out_h) and int(frame_u8.shape[1]) == int(out_w):
            return np.ascontiguousarray(frame_u8)
        if int(out_h) <= int(frame_u8.shape[0]) and int(out_w) <= int(frame_u8.shape[1]):
            scaled = cv2.resize(
                np.ascontiguousarray(frame_u8 * np.uint8(255)),
                (int(out_w), int(out_h)),
                interpolation=cv2.INTER_AREA,
            )
            return (scaled > 0).astype(np.uint8, copy=False)
        scaled = cv2.resize(
            np.ascontiguousarray(frame_u8),
            (int(out_w), int(out_h)),
            interpolation=cv2.INTER_NEAREST,
        )
        return (scaled > 0).astype(np.uint8, copy=False)

    def _restore_slice(out_z: int) -> None:
        src_start = int(math.floor(float(out_z) * float(in_t) / float(out_t)))
        src_stop = int(math.ceil(float(out_z + 1) * float(in_t) / float(out_t)))
        src_start = int(np.clip(src_start, 0, in_t - 1))
        src_stop = int(np.clip(max(src_start + 1, src_stop), 1, in_t))
        restored = np.zeros((int(out_h), int(out_w)), dtype=np.uint8)
        for src_idx in range(src_start, src_stop):
            restored |= _resize_mask_xy(mask_u8[int(src_idx)])
        out_mm[int(out_z), :, :] = restored

    parallel_for_indices_chunked(
        int(out_t),
        _restore_slice,
        max_workers=worker_count,
        desc=desc,
        chunk_size=chunk_size,
        show_progress=True,
    )
    flush_array(out_mm)
    return out_mm

def x264_preset() -> str:
    """The low-quality distribution contract always uses libx264 preset slow."""
    return 'slow'

def ffmpeg_h264_rgb_writer(out_path: Path, width: int, height: int, fps: float) -> subprocess.Popen:
    return ffmpeg_rawvideo_writer(
        out_path=out_path,
        width=int(width),
        height=int(height),
        fps=float(fps),
        pix_fmt_in='rgb24',
        codec='libx264',
        pix_fmt_out='yuv420p',
        codec_args=['-preset', x264_preset()],
    )

def ffmpeg_h264_gray_writer(out_path: Path, width: int, height: int, fps: float) -> subprocess.Popen:
    return ffmpeg_rawvideo_writer(
        out_path=out_path,
        width=int(width),
        height=int(height),
        fps=float(fps),
        pix_fmt_in='gray',
        codec='libx264',
        pix_fmt_out='yuv420p',
        codec_args=['-preset', x264_preset()],
    )

def write_low_quality_overlay_video(
    volume_gray: np.ndarray,
    mask_u8: np.ndarray,
    out_path: Path,
    fps: float,
    show_progress: bool = True,
    publication_root: Optional[Path] = None,
) -> Path:
    t_dim, h_dim, w_dim = volume_gray.shape
    assert mask_u8.shape == (t_dim, h_dim, w_dim)
    stage_path, stage_dir = _atomic_publication_stage_path(
        Path(out_path), publication_root=publication_root,
    )
    try:
        proc = ffmpeg_h264_rgb_writer(stage_path, int(w_dim), int(h_dim), float(fps))
        # reused RGB buffer + bbox-restricted blend + memoryview pipe writes.
        frame_buf = np.empty((int(h_dim), int(w_dim), 3), dtype=np.uint8)
        try:
            assert proc.stdin is not None
            for t in tqdm(range(int(t_dim)), desc=f'Writing low-quality overlay ({out_path.name})', disable=not show_progress):
                frame = _gray_frame_into_rgb_buffer(volume_gray[int(t)], frame_buf)
                _overlay_blend_blue_inplace(frame, mask_u8[int(t)])
                proc.stdin.write(memoryview(frame).cast('B'))
        finally:
            close_ffmpeg_writer(proc)
        _publish_staged_file_atomically(stage_path, Path(out_path))
    finally:
        try:
            stage_path.unlink(missing_ok=True)
        except Exception:
            pass
        try:
            stage_dir.rmdir()
        except OSError:
            pass
    return out_path

def write_low_quality_binary_video(
    mask_u8: np.ndarray,
    out_path: Path,
    fps: float,
    show_progress: bool = True,
    publication_root: Optional[Path] = None,
) -> Path:
    t_dim, h_dim, w_dim = mask_u8.shape
    stage_path, stage_dir = _atomic_publication_stage_path(
        Path(out_path), publication_root=publication_root,
    )
    try:
        proc = ffmpeg_h264_gray_writer(stage_path, int(w_dim), int(h_dim), float(fps))
        # reused frame buffer + memoryview pipe writes; the mask volumes are
        # 0/1, so multiply-by-255 into the buffer replaces compare+cast temporaries.
        frame_buf = np.empty((int(h_dim), int(w_dim)), dtype=np.uint8)
        try:
            assert proc.stdin is not None
            for t in tqdm(range(int(t_dim)), desc=f'Writing low-quality binary ({out_path.name})', disable=not show_progress):
                np.multiply(np.asarray(mask_u8[int(t)]), 255, out=frame_buf)
                proc.stdin.write(memoryview(frame_buf).cast('B'))
        finally:
            close_ffmpeg_writer(proc)
        _publish_staged_file_atomically(stage_path, Path(out_path))
    finally:
        try:
            stage_path.unlink(missing_ok=True)
        except Exception:
            pass
        try:
            stage_dir.rmdir()
        except OSError:
            pass
    return out_path

def low_quality_per_bin_output_workers(requested_workers: int) -> int:
    """Number of output writer tasks launched inside one low-quality bin."""
    default_value = max(1, min(3, int(requested_workers)))
    return max(1, min(3, _env_int('YOLO_TTA_LOW_QUALITY_OUTPUT_WORKERS_PER_BIN', int(default_value))))

def save_single_low_quality_output(
    *,
    volume_gray: np.ndarray,
    mask_u8: np.ndarray,
    spec: LowQualityDownbinSpec,
    out_dir: Path,
    stem: str,
    fps: float,
    temp_dir: Path,
    workers: int = 1,
    show_progress: bool = True,
) -> Dict[str, Path]:
    """Write one low-quality downbin's overlay and binary videos concurrently."""
    low_root = Path(out_dir) / 'low_quality'
    low_root.mkdir(parents=True, exist_ok=True)
    source_t = max(1, int(mask_u8.shape[0]))
    out_t, out_h, out_w = (
        int(spec.output_shape_t_y_x[0]),
        int(spec.output_shape_t_y_x[1]),
        int(spec.output_shape_t_y_x[2]),
    )
    spec_dir = low_root / spec.token
    spec_dir.mkdir(parents=True, exist_ok=True)
    fps_lq = max(1e-6, float(fps) * float(out_t) / float(source_t))
    print(
        f'Low-quality downbin {spec.raw_value}: native (t,Y,X)=({source_t},{int(mask_u8.shape[1])},{int(mask_u8.shape[2])}) '
        f'-> ({out_t},{out_h},{out_w}); playback fps adjusted {float(fps):g} -> {fps_lq:g}; '
        f'per-bin output writer concurrency={low_quality_per_bin_output_workers(int(workers))}'
    )

    gray_lq = resize_gray_volume_to_shape(
        volume_gray,
        (out_t, out_h, out_w),
        Path(temp_dir) / 'low_quality' / spec.token / 'source.gray8.dat',
        workers=int(workers),
        prefer_memory=True,
        desc=f'Low-quality source resize {spec.token}',
    )
    mask_lq = resize_binary_mask_volume_to_shape(
        mask_u8,
        (out_t, out_h, out_w),
        Path(temp_dir) / 'low_quality' / spec.token / 'mask.u8.dat',
        workers=int(workers),
        prefer_memory=True,
        desc=f'Low-quality mask resize {spec.token}',
    )
    result_paths: Dict[str, Path] = {'low_quality_dir': low_root}
    try:
        overlay_path = spec_dir / f'{stem}_Overlay_LowQuality_{spec.token}.mp4'
        binary_path = spec_dir / f'{stem}_Binary_LowQuality_{spec.token}.mp4'

        def _write_lq_overlay() -> Path:
            return write_low_quality_overlay_video(
                gray_lq, mask_lq, overlay_path, fps_lq,
                show_progress=show_progress,
                publication_root=out_dir,
            )

        def _write_lq_binary() -> Path:
            return write_low_quality_binary_video(
                mask_lq, binary_path, fps_lq,
                show_progress=show_progress,
                publication_root=out_dir,
            )

        # the low-quality NRRD decomposition is produced per component layer by the
        # NrrdLayerSink as views complete, not here. Only the distribution videos are written below.
        writer_thunks = [_write_lq_overlay, _write_lq_binary]

        writer_count = low_quality_per_bin_output_workers(int(workers))
        if writer_count <= 1:
            for thunk in writer_thunks:
                thunk()
        else:
            with ThreadPoolExecutor(max_workers=int(writer_count), thread_name_prefix=f'lq-writer-{spec.token[:16]}') as executor:
                futures = [executor.submit(thunk) for thunk in writer_thunks]
                for fut in as_completed(futures):
                    fut.result()

        result_paths[f'low_quality_{spec.token}_overlay'] = overlay_path
        result_paths[f'low_quality_{spec.token}_binary_video'] = binary_path
        return result_paths
    finally:
        if gray_lq is not volume_gray:
            close_memmap_array(gray_lq)
        if mask_lq is not mask_u8:
            close_memmap_array(mask_lq)

def collect_low_quality_output_futures(
    executor: ThreadPoolExecutor,
    *,
    volume_gray: np.ndarray,
    mask_u8: np.ndarray,
    out_dir: Path,
    stem: str,
    fps: float,
    downbin_specs: Sequence[LowQualityDownbinSpec],
    temp_dir: Path,
    workers: int = 1,
    show_progress: bool = True,
) -> Tuple[Dict[str, Path], List[Future]]:
    """Submit each low-quality downbin as an independent background job."""
    result_paths = planned_low_quality_output_paths(
        out_dir=out_dir,
        stem=stem,
        downbin_specs=downbin_specs,
    )
    futures: List[Future] = []
    for spec in list(downbin_specs):
        futures.append(executor.submit(
            save_single_low_quality_output,
            volume_gray=volume_gray,
            mask_u8=mask_u8,
            out_dir=out_dir,
            stem=stem,
            fps=float(fps),
            spec=spec,
            temp_dir=temp_dir,
                workers=int(workers),
            show_progress=show_progress,
        ))
    return result_paths, futures

def planned_low_quality_output_paths(
    *,
    out_dir: Path,
    stem: str,
    downbin_specs: Sequence[LowQualityDownbinSpec],
) -> Dict[str, Path]:
    """Return paths for per-bin low-quality video writers."""
    low_root = out_dir / 'low_quality'
    result_paths: Dict[str, Path] = {'low_quality_dir': low_root}
    for spec in downbin_specs:
        spec_dir = low_root / spec.token
        overlay_path = spec_dir / f'{stem}_Overlay_LowQuality_{spec.token}.mp4'
        binary_path = spec_dir / f'{stem}_Binary_LowQuality_{spec.token}.mp4'
        result_paths[f'low_quality_{spec.token}_overlay'] = overlay_path
        result_paths[f'low_quality_{spec.token}_binary_video'] = binary_path
    return result_paths

@dataclass
class BackgroundOutputSubmission:
    label: str
    result_paths: Dict[str, Path]
    futures: List[Future] = field(default_factory=list)
    resources: List[object] = field(default_factory=list)

    def wait(self) -> Dict[str, Path]:
        error: Optional[BaseException] = None
        try:
            for fut in self.futures:
                fut.result()
        except BaseException as exc:  # pragma: no cover - surfaced to main
            error = exc
        finally:
            for resource in self.resources:
                close_memmap_array(resource)
        if error is not None:
            raise RuntimeError(f'Background output generation failed for {self.label}') from error
        return self.result_paths

class BackgroundOutputManager:
    def __init__(self, max_workers: int) -> None:
        self.max_workers = max(1, int(max_workers))
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers, thread_name_prefix='yolo-output')
        self.pending: List[BackgroundOutputSubmission] = []

    def submit(self, submission: BackgroundOutputSubmission) -> Dict[str, Path]:
        self.pending.append(submission)
        return submission.result_paths

    def reap_completed(self) -> None:
        remaining: List[BackgroundOutputSubmission] = []
        for submission in self.pending:
            if all(fut.done() for fut in submission.futures):
                submission.wait()
            else:
                remaining.append(submission)
        self.pending = remaining

    def wait(self) -> None:
        error: Optional[BaseException] = None
        try:
            while self.pending:
                submission = self.pending.pop(0)
                try:
                    submission.wait()
                except BaseException as exc:  # pragma: no cover - surfaced to main
                    if error is None:
                        error = exc
        finally:
            self.executor.shutdown(wait=True)
        if error is not None:
            raise error

def collect_pipeline_output_futures(
    executor: ThreadPoolExecutor,
    *,
    volume_rgb: np.memmap,
    mask_u8: np.ndarray,
    out_dir: Path,
    stem: str,
    fps: float,
    save_high_quality: bool,
    save_binary_pattern_value: Optional[str],
    save_labels_pattern_value: Optional[str],
    tag: Optional[str] = None,
    frame_workers: int = 1,
    show_progress: bool = False,
    nrrd_temp_dir: Optional[Path] = None,
) -> Tuple[Dict[str, Path], List[Future]]:
    futures: List[Future] = []
    result_paths: Dict[str, Path] = {}
    tag_suffix = f"_{tag}" if tag else ""

    if bool(save_high_quality):
        overlay_path = out_dir / f"{stem}_Overlay{tag_suffix}.mkv"
        futures.append(executor.submit(
            write_overlay_video,
            volume_rgb,
            mask_u8,
            overlay_path,
            fps,
            show_progress=show_progress,
            scratch_dir=nrrd_temp_dir,
            publication_root=out_dir,
        ))
        result_paths["overlay"] = overlay_path

    labels_pattern = _resolve_output_pattern(save_labels_pattern_value, DEFAULT_LABEL_PATTERN, out_dir, stem)
    if labels_pattern is not None:
        if tag is not None:
            labels_pattern = _tag_frame_pattern(labels_pattern, tag)
        futures.append(executor.submit(write_yolo_labels_from_pattern, mask_u8, labels_pattern, int(frame_workers), show_progress))
        result_paths["labels_dir"] = labels_pattern.parent

    binary_pattern = _resolve_output_pattern(save_binary_pattern_value, DEFAULT_BINARY_PATTERN, out_dir, stem)
    if binary_pattern is not None:
        if tag is not None:
            binary_pattern = _tag_frame_pattern(binary_pattern, tag)
        binary_video_path = out_dir / f"{stem}_Binary{tag_suffix}.mkv"
        futures.append(executor.submit(write_binary_tiff_sequence_from_pattern, mask_u8, binary_pattern, int(frame_workers), show_progress))
        futures.append(executor.submit(
            write_binary_video_from_mask_volume,
            mask_u8,
            binary_video_path,
            fps,
            show_progress=show_progress,
            scratch_dir=nrrd_temp_dir,
            publication_root=out_dir,
        ))
        result_paths["binary_tiff_dir"] = binary_pattern.parent
        result_paths["binary_video"] = binary_video_path


    return result_paths, futures

def _write_multichannel_tiff(path: Path, frame_hwc: np.ndarray, channel_count: int) -> None:
    """Write one uint8 grayscale TIFF page per model-input channel."""
    writer = getattr(cv2, 'imwritemulti', None)
    if not callable(writer):
        raise RuntimeError(
            'Saving channel-formatted images with five or more channels requires an '
            'OpenCV build that provides cv2.imwritemulti(); upgrade opencv-python.'
        )
    frame = np.ascontiguousarray(np.asarray(frame_hwc), dtype=np.uint8)
    expected_channels = int(channel_count)
    if frame.ndim != 3 or int(frame.shape[2]) != expected_channels:
        raise ValueError(
            f'Multi-page TIFF output expected HxWx{expected_channels}, got {tuple(frame.shape)}'
        )
    pages = [
        np.ascontiguousarray(frame[:, :, channel_index])
        for channel_index in range(expected_channels)
    ]
    if not bool(writer(str(path), pages)):
        raise RuntimeError(f'Failed to write multi-page TIFF: {path}')

def write_view_images(
    volume_rgb: np.ndarray,
    view: ViewInfo,
    out_dir: Path,
    stem: str,
    *,
    channel_format: ChannelFormat = DEFAULT_CHANNEL_FORMAT,
    workers: int = 1,
    show_progress: bool = True,
) -> Path:
    """Save the same channel-formatted center inputs supplied to inference."""
    fmt = resolve_channel_format(channel_format)
    view_dir = out_dir / 'images' / view.name
    view_dir.mkdir(parents=True, exist_ok=True)
    total = int(view.num_slices)
    multipage_tiff = int(fmt.channel_count) >= 5
    suffix = '.tif' if multipage_tiff else '.png'
    renderer = ChannelFormattedFrameRenderer(
        lambda source_idx: get_view_frame_by_index(volume_rgb, view, int(source_idx)),
        view,
        fmt,
    )
    final_pattern = view_dir / f'{stem}_{view.name}_%04d{suffix}'
    with _staged_frame_sequence(final_pattern) as stage_pattern:
        def _write_frame(idx: int) -> None:
            frame = np.ascontiguousarray(renderer(int(idx)), dtype=np.uint8)
            out_path = _format_frame_path(stage_pattern, int(idx) + 1)
            if multipage_tiff:
                _write_multichannel_tiff(out_path, frame, int(fmt.channel_count))
                return
            if frame.ndim == 3 and int(frame.shape[2]) == 1:
                frame = np.ascontiguousarray(frame[:, :, 0])
            if not cv2.imwrite(str(out_path), frame):
                raise RuntimeError(f'Failed to write image: {out_path}')

        parallel_for_indices(
            total,
            _write_frame,
            max_workers=choose_slice_parallel_workers(int(workers), total),
            desc=f'Writing {view.name} {fmt.token} image sequence',
            show_progress=show_progress,
        )
        _publish_staged_frame_sequence(
            stage_pattern,
            final_pattern,
            total,
            stale_extensions=('.png', '.tif', '.tiff'),
        )
    return view_dir

def write_summary_file(
    out_path: Path,
    *,
    command: str,
    input_path: Path,
    out_dir: Path,
    scratch_dir: Path,
    source_shape_x_y_t: Tuple[int, int, int],
    volume_shape: Tuple[int, int, int],
    fps: float,
    model_paths: Sequence[str],
    view_names: Sequence[str],
    view_prediction_stats: Dict[str, int],
    interpolation_stats: List[Dict[str, object]],
    enable_3d_void_fill: bool,
    gaussian_smoothing_stats: Optional[Dict[str, int | float]],
    keep_objects_stats: Optional[Dict[str, int | float]],
    voxel_volume: Optional[int],
    final_paths: Dict[str, Path],
    augmentation_workers: int,
    slice_postprocess_workers: int,
    interpolation_workers: int,
    output_workers: int,
    spec_notes: Optional[Sequence[str]] = None,
    view_prediction_labels: Optional[Dict[str, str]] = None,
) -> Path:
    lines: List[str] = []
    lines.append(f'Command: {command}')
    lines.append(f'Input: {input_path}')
    lines.append(f'Output directory: {out_dir}')
    lines.append(f'Source dimensions before cubic resizing (X, Y, t): {source_shape_x_y_t}')
    lines.append(f'Processing volume shape (t, Y, X): {volume_shape}')
    lines.append(f'FPS: {fps}')
    lines.append(f'Scratch directory: {scratch_dir}')
    lines.append('Workspace policy: in-memory first with disk fallback when the working set exceeds available RAM/swap')
    lines.append(f'3D void fill: {"enabled" if bool(enable_3d_void_fill) else "disabled"}; when enabled, it is applied once after the final global union')
    if bool(enable_3d_void_fill):
        lines.append('3D void fill background connectivity: default 6-connected; override with YOLO_TTA_VOIDFILL_CONNECTIVITY=18 or 26 if needed')
    lines.append(f'Augmentation workers: {int(augmentation_workers)}')
    lines.append(f'Slice-parallel postprocess workers: {int(slice_postprocess_workers)}')
    lines.append(f'Interpolation workers: {int(interpolation_workers)}')
    lines.append(f'Output workers: {int(output_workers)}')
    lines.append('Worker oversubscription: inference-phase pools are bounded against process-local GPU/OpenVINO reservations; tail-only CPU pools may expand after every inference worker exits.')
    if model_paths:
        lines.append('Models:')
        for model_path in model_paths:
            lines.append(f'  {str(model_path)}')
    else:
        lines.append('Models: <none>')
    lines.append(f'Views: {", ".join(view_names)}')
    if spec_notes:
        lines.append('')
        lines.append('Specification notes:')
        for note in spec_notes:
            lines.append(f'  - {note}')

    lines.append('')
    lines.append('View statistics:')
    total_prediction_count = 0
    labels = dict(view_prediction_labels or {})
    ordered_keys: List[str] = [k for k in ('transverse', 'sagittal', 'coronal', 'radial_transverse') if k in view_prediction_stats]
    tilted_keys = [k for k in view_prediction_stats.keys() if str(k).startswith('tilted_')]
    other_keys = [k for k in view_prediction_stats.keys() if k not in set(ordered_keys) and k not in set(tilted_keys)]
    for view_key in ordered_keys + sorted(tilted_keys, key=lambda k: labels.get(k, k)) + sorted(other_keys):
        label = labels.get(view_key, str(view_key).replace('_', ' ').title())
        count = int(view_prediction_stats.get(view_key, 0))
        total_prediction_count += count
        lines.append(f'  {label}: predictions={count}')
    lines.append(f'  Total prediction count: {int(total_prediction_count)}')

    if interpolation_stats:
        lines.append('')
        lines.append('Interpolation statistics (per pass):')
        pass_indices = sorted({int(s.get('pass_index', 0)) for s in interpolation_stats})
        for pass_idx in pass_indices:
            stats_this_pass = [s for s in interpolation_stats if int(s.get('pass_index', 0)) == pass_idx]
            total_objects = sum(int(s.get('num_objects', 0)) for s in stats_this_pass)
            total_endpoints = sum(int(s.get('num_endpoints', 0)) for s in stats_this_pass)
            total_candidates = sum(int(s.get('candidate_connections', 0)) for s in stats_this_pass)
            total_accepted = sum(int(s.get('accepted_connections', 0)) for s in stats_this_pass)
            total_default_bridges = sum(int(s.get('default_bridges', 0)) for s in stats_this_pass)
            total_walk_back = sum(int(s.get('walk_back_bridges', 0)) for s in stats_this_pass)
            total_skipped = sum(int(s.get('skipped_by_min_radius', 0)) for s in stats_this_pass)
            total_added_voxels = sum(int(s.get('added_voxels', 0)) for s in stats_this_pass)
            lines.append(
                f'  Pass {pass_idx}: objects={total_objects}, endpoints={total_endpoints}, '
                f'candidate_connections={total_candidates}, accepted_connections={total_accepted}, '
                f'default_bridges={total_default_bridges}, walk_back_bridges={total_walk_back}, '
                f'bridges_skipped_by_--interpolation_min_radius={total_skipped}, added_voxels={total_added_voxels}'
            )
            for s in sorted(stats_this_pass, key=lambda d: (str(d.get('model', '')), str(d.get('view', '')))):
                lines.append(
                    f"    {s.get('model', '?')}/{s.get('view', '?')}: "
                    f"objects={int(s.get('num_objects', 0))}, "
                    f"endpoints={int(s.get('num_endpoints', 0))}, "
                    f"candidate_connections={int(s.get('candidate_connections', 0))}, "
                    f"accepted_connections={int(s.get('accepted_connections', 0))}, "
                    f"default_bridges={int(s.get('default_bridges', 0))}, "
                    f"walk_back_bridges={int(s.get('walk_back_bridges', 0))}, "
                    f"bridges_skipped_by_--interpolation_min_radius={int(s.get('skipped_by_min_radius', 0))}, "
                    f"added_voxels={int(s.get('added_voxels', 0))}, "
                    f"skipped={bool(s.get('skipped', False))}"
                )

    lines.append('')
    if gaussian_smoothing_stats is not None and int(gaussian_smoothing_stats.get('enabled', 0)) > 0:
        lines.append(
            'Gaussian smoothing: enabled; '
            f"sigma={float(gaussian_smoothing_stats.get('sigma', 0.0)):g}, "
            f"passes_requested={int(gaussian_smoothing_stats.get('passes_requested', 0))}, "
            f"passes_completed={int(gaussian_smoothing_stats.get('passes_completed', 0))}"
        )
        lines.append(
            '  voxel changes after thresholding: '
            f"added={int(gaussian_smoothing_stats.get('total_added_voxels', 0))}, "
            f"removed={int(gaussian_smoothing_stats.get('total_removed_voxels', 0))}"
        )
    else:
        lines.append('Gaussian smoothing: disabled')

    if keep_objects_stats is not None:
        lines.append('')
        lines.append(
            'keep_objects: '
            f"enabled, objects={int(keep_objects_stats.get('num_objects', 0))}, "
            f"kept={int(keep_objects_stats.get('kept_objects', 0))}, "
            f"removed_objects={int(keep_objects_stats.get('removed_objects', 0))}, "
            f"removed_voxels={int(keep_objects_stats.get('removed_voxels', 0))}, "
            f"topology_slabs={int(keep_objects_stats.get('topology_slab_count', 0))}, "
            f"slab_workers={int(keep_objects_stats.get('topology_slab_workers', 0))}"
        )
        lines.append(
            '  phase_seconds: '
            f"label={float(keep_objects_stats.get('label_seconds', 0.0)):.3f}, "
            f"pair_extraction={float(keep_objects_stats.get('pair_extraction_seconds', 0.0)):.3f}, "
            f"local_union={float(keep_objects_stats.get('local_union_seconds', 0.0)):.3f}, "
            f"boundary_merge={float(keep_objects_stats.get('boundary_merge_seconds', 0.0)):.3f}, "
            f"root_expansion={float(keep_objects_stats.get('root_expansion_seconds', 0.0)):.3f}, "
            f"area_reduction={float(keep_objects_stats.get('area_reduction_seconds', 0.0)):.3f}, "
            f"decision={float(keep_objects_stats.get('decision_seconds', 0.0)):.3f}, "
            f"lut={float(keep_objects_stats.get('lut_seconds', 0.0)):.3f}, "
            f"apply={float(keep_objects_stats.get('apply_seconds', 0.0)):.3f}, "
            f"total={float(keep_objects_stats.get('total_seconds', 0.0)):.3f}"
        )
    else:
        lines.append('')
        lines.append('keep_objects: disabled')

    if voxel_volume is not None:
        lines.append('')
        lines.append(f'voxel_volume_native_input_space: {int(voxel_volume)}')

    lines.append('')
    lines.append('Final outputs:')
    for key in sorted(final_paths.keys()):
        lines.append(f'{key}: {final_paths[key]}')

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    return out_path


# Late imports keep callable-only dependency cycles import-safe.
from ._latebind import bind_late_symbols as _bind_late_symbols

_bind_late_symbols(
    __name__,
    globals(),
    {
        "backprojection": (
            "_MainProcessGpuStageLease",
            "_announce_main_gpu_stage_skip_once",
            "_trim_main_process_cuda_device",
            "_try_acquire_main_process_gpu_stage",
        ),
        "config": (
            "ChannelFormat",
            "GIB",
            "NRRD_SPACE",
            "SCRIPT_BASENAME",
            "_parse_token_list",
            "resolve_channel_format",
            "variant_nrrd_stem",
        ),
        "cuda_d1": (
            "_read_binary_volume_slice_bool",
            "_read_binary_volume_slice_u8",
            "_sanitize_nrrd_layer_token",
            "_volume_shape_tuple",
        ),
        "geometry": (
            "ChannelFormattedFrameRenderer",
            "ViewInfo",
            "get_view_frame_by_index",
        ),
        "interpolation": (
            "CTILE_INDEX_DTYPE",
            "MASK_STORE_FORMATS",
            "NrrdLayerRef",
            "NrrdRasterPlan",
            "NrrdSegmentExtent",
            "RawBBoxMaskStore",
            "_coerce_segment_extent",
            "_nrrd_empty_segment_extent",
            "_release_raw_store_chunks_ram_cache",
        ),
        "media": (
            "_linear_source_index",
            "_memory_backed_encoded_chunk_path",
            "_memory_backed_encoded_chunk_size",
            "_open_memory_backed_encoded_chunk",
            "_resize_gray_slice_nearest_or_linear",
            "_spawn_subprocess_with_retry",
            "close_ffmpeg_writer",
            "ffmpeg_ffv1_gray_writer",
            "ffmpeg_ffv1_rgb_writer",
            "ffmpeg_rawvideo_writer",
        ),
        "runtime": (
            "_acquire_parallel_pool",
            "_release_parallel_pool",
            "_settle_parallel_futures",
            "allocate_workspace_array",
            "choose_parallel_chunk_size",
            "choose_slice_parallel_workers",
            "close_memmap_array",
            "flush_array",
            "parallel_for_indices",
            "parallel_for_indices_chunked",
        ),
        "workspace": (
            "_cpu_count",
            "_env_flag",
            "_env_float",
            "_env_int",
            "available_anon_work_bytes",
        ),
    },
)
