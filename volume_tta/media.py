"""Implementation subsystem extracted from the v17.0.5 volume TTA runtime.

This physical split intentionally preserves the original numerical and scheduling
behavior. Public coordination contracts live under ``inference_backends``.
"""

from __future__ import annotations

from ._stdlib import *
import numpy as np
from ._deps import cv2, tqdm

from .config import (
    GIB,
)

def _require_bin(name: str) -> None:
    if shutil.which(name) is None:
        raise RuntimeError(f"Required executable not found on PATH: {name}")

def _spawn_subprocess_with_retry(spawn: Callable[[], object], desc: str) -> object:
    """Launch one subprocess, retrying transient OS-level spawn failures.

 Under SLURM memory pressure, fork/exec of this large process intermittently fails with
 OSError even though the binary is on PATH (observed as [Errno 14] Bad address: 'ffmpeg').
 Retry the launch up to two more times with a short wait; if it still fails, raise so the
 caller (and any VolumeReadiness consumers) fail loudly instead of continuing without
 their producer."""
    attempts = max(1, _env_int('YOLO_TTA_SUBPROCESS_SPAWN_ATTEMPTS', 3))
    wait_seconds = max(0.0, _env_float('YOLO_TTA_SUBPROCESS_SPAWN_RETRY_SECONDS', 1.0))
    last_exc: Optional[OSError] = None
    for attempt in range(1, attempts + 1):
        try:
            return spawn()
        except OSError as exc:
            # OSError covers the transient EFAULT/ENOMEM class and FileNotFoundError; a
            # process that launched but exited nonzero raises CalledProcessError instead
            # and is never retried here.
            last_exc = exc
            if attempt >= attempts:
                break
            print(
                f'Warning: {desc} failed to launch (attempt {attempt}/{attempts}: {exc}); '
                f'retrying in {wait_seconds:g}s.'
            )
            time.sleep(wait_seconds)
    raise RuntimeError(
        f'{desc} could not be launched after {attempts} attempts; aborting instead of '
        f'continuing without it: {last_exc}'
    ) from last_exc

def ffmpeg_decode_threads() -> int:
    """Return the decoder thread count for input-volume materialization.

 The earlier decode path left FFmpeg to choose its own threading. On large
 FFV1/Matroska inputs this can silently become a single-decoder bottleneck
 before the first prediction volume is even scheduled. Default to the full
 visible CPU allocation; set ``YOLO_TTA_FFMPEG_DECODE_THREADS`` to pin a
 smaller value for codecs or filesystems that prefer less parallel decode."""
    return max(1, _env_int('YOLO_TTA_FFMPEG_DECODE_THREADS', max(1, _cpu_count())))

class VolumeReadiness:
    """Slice-level readiness gate for streaming decode/cube preprocessing."""

    def __init__(self, total_slices: int, desc: str = 'volume') -> None:
        self.total_slices = max(0, int(total_slices))
        self.desc = str(desc)
        self._slice_events = [threading.Event() for _ in range(self.total_slices)]
        self._all_event = threading.Event()
        self._lock = threading.Lock()
        self._ready_count = 0
        self._exception: Optional[BaseException] = None
        if self.total_slices <= 0:
            self._all_event.set()

    def mark_slice_ready(self, idx: int) -> None:
        idx_i = int(idx)
        if idx_i < 0 or idx_i >= int(self.total_slices):
            return
        ev = self._slice_events[idx_i]
        if ev.is_set():
            return
        with self._lock:
            if not ev.is_set():
                ev.set()
                self._ready_count += 1
                if self._ready_count >= self.total_slices:
                    self._all_event.set()

    def mark_all_ready(self) -> None:
        with self._lock:
            for ev in self._slice_events:
                ev.set()
            self._ready_count = self.total_slices
            self._all_event.set()

    def mark_failed(self, exc: BaseException) -> None:
        with self._lock:
            self._mark_failed_locked(exc)

    def _mark_failed_locked(self, exc: BaseException) -> None:
        # First failure wins: never replace a recorded root cause with a later, more
        # generic one (e.g. the decode finally's returncode error after the real EOF
        # error, or the shutdown abort after a real producer failure).
        if self._exception is None:
            self._exception = exc
        for ev in self._slice_events:
            ev.set()
        self._all_event.set()

    def _raise_if_failed(self) -> None:
        exc = self._exception
        if exc is not None:
            raise RuntimeError(f'{self.desc} producer failed before required slice data was ready') from exc

    def wait_for_slice(self, idx: int) -> None:
        idx_i = int(idx)
        if idx_i < 0 or idx_i >= int(self.total_slices):
            raise IndexError(idx_i)
        self._slice_events[idx_i].wait()
        self._raise_if_failed()

    def wait_all(self) -> None:
        self._all_event.wait()
        self._raise_if_failed()

    def fail_if_incomplete(self, exc: BaseException) -> bool:
        """Mark_failed unless the volume already completed or failed; True if failed here.

 Check and mark happen under one lock hold, so a producer completing concurrently
 cannot be marked failed after the fact, and an already-recorded root-cause
 exception is never replaced by the generic abort error."""
        with self._lock:
            if self._all_event.is_set():
                return False
            self._mark_failed_locked(exc)
            return True

_VOLUME_READINESS_BY_ARRAY_ID: Dict[int, VolumeReadiness] = {}

_STREAMING_PRODUCER_ABORT = threading.Event()

def streaming_producers_aborted() -> bool:
    return _STREAMING_PRODUCER_ABORT.is_set()

def abort_streaming_producers(reason: str = 'pipeline aborted') -> None:
    _STREAMING_PRODUCER_ABORT.set()
    exc = RuntimeError(f'streaming producer aborted: {reason}')
    for readiness in list(_VOLUME_READINESS_BY_ARRAY_ID.values()):
        try:
            readiness.fail_if_incomplete(exc)
        except Exception:
            pass

def streaming_preprocess_enabled() -> bool:
    """Return True when decode/cube preprocessing may run ahead of consumers."""
    return _env_flag('YOLO_TTA_STREAMING_PREPROCESS', True)

def register_volume_readiness(arr: object, readiness: VolumeReadiness) -> None:
    _VOLUME_READINESS_BY_ARRAY_ID[id(arr)] = readiness

def volume_readiness(arr: object) -> Optional[VolumeReadiness]:
    return _VOLUME_READINESS_BY_ARRAY_ID.get(id(arr))

def wait_for_volume_slice_ready(arr: object, idx: int) -> None:
    readiness = volume_readiness(arr)
    if readiness is not None:
        readiness.wait_for_slice(int(idx))

def wait_for_volume_ready(arr: object) -> None:
    # normal consumers that explicitly wait on a lazy processing cube
    # necessarily need its bytes. The inference-tail completion check special-cases an
    # unused proxy and waits only for its decoded source, preserving true laziness.
    lazy_wait = getattr(arr, '_materialize_for_wait', None)
    if callable(lazy_wait):
        lazy_wait()
        return
    readiness = volume_readiness(arr)
    if readiness is not None:
        readiness.wait_all()

def ffprobe_info(video_path: Path) -> Dict[str, object]:
    """Return dict with width, height, fps, num_frames."""
    _require_bin("ffprobe")
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,avg_frame_rate,nb_frames",
        "-of", "json",
        str(video_path),
    ]
    p = _spawn_subprocess_with_retry(
        lambda: subprocess.run(cmd, capture_output=True, text=True, check=True),
        'ffprobe stream metadata probe',
    )
    info = json.loads(p.stdout)
    if "streams" not in info or not info["streams"]:
        raise RuntimeError(f"ffprobe: no video stream found in {video_path}")
    st = info["streams"][0]

    width = int(st.get("width"))
    height = int(st.get("height"))

    def _parse_ratio(r: str) -> float:
        if not r or r == "0/0":
            return 0.0
        num, den = r.split("/")
        den_i = int(den)
        return float(num) / float(den_i) if den_i != 0 else 0.0

    fps = _parse_ratio(str(st.get("avg_frame_rate", "0/0")))
    if fps <= 0:
        fps = _parse_ratio(str(st.get("r_frame_rate", "0/0")))
    if fps <= 0:
        fps = 30.0

    nf = st.get("nb_frames", None)
    if nf is None or str(nf).strip() == "" or str(nf) == "N/A":
        # Fast fallback: count packets without decoding
        fallback_cmd = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "v:0",
            "-count_packets",
            "-show_entries", "stream=nb_read_packets",
            "-of", "json",
            str(video_path),
        ]
        p2 = _spawn_subprocess_with_retry(
            lambda: subprocess.run(fallback_cmd, capture_output=True, text=True, check=True),
            'ffprobe packet-count probe',
        )
        info2 = json.loads(p2.stdout)
        nf = info2["streams"][0].get("nb_read_packets", None)
    if nf is None or str(nf).strip() == "" or str(nf) == "N/A":
        raise RuntimeError(
            "ffprobe could not determine frame count (nb_frames/nb_read_packets missing)."
        )
    num_frames = int(nf)
    return {"width": width, "height": height, "fps": fps, "num_frames": num_frames}

def decode_video_to_memmap_gray8(
    input_video: Path,
    out_dat: Path,
    num_frames: int,
    width: int,
    height: int,
    overwrite: bool = False,
    *,
    prefer_memory: bool = True,
    prefer_memfd: bool = False,
    reserve_bytes: int = 32 * GIB,
) -> np.ndarray:
    """Decode the input video to a contiguous ``(T, H, W)`` uint8 luma workspace."""
    _require_bin("ffmpeg")

    shape = (int(num_frames), int(height), int(width))
    reuse_existing = bool(not overwrite and out_dat.exists() and not prefer_memory and not prefer_memfd)
    arr = allocate_workspace_array(
        shape=shape,
        dtype=np.uint8,
        path=out_dat,
        desc='Decoded gray8 input volume',
        prefer_memory=bool(prefer_memory),
        prefer_memfd=bool(prefer_memfd),
        reserve_bytes=int(reserve_bytes),
        reuse_existing=bool(reuse_existing),
        initialize_zero=False,
    )
    if reuse_existing and isinstance(arr, np.memmap):
        return arr

    raw_bytes = memoryview(np.ascontiguousarray(arr) if not arr.flags['C_CONTIGUOUS'] else arr).cast('B')
    if raw_bytes.obj is not arr:
        arr = np.asarray(raw_bytes.obj).reshape(shape)
        raw_bytes = memoryview(arr).cast('B')

    frame_bytes = int(width) * int(height)
    chunk_frames = max(1, min(128, max(1, (256 * 1024 * 1024) // max(1, frame_bytes))))

    cmd = [
        "ffmpeg",
        "-v", "error",
        "-threads", str(ffmpeg_decode_threads()),
        "-i", str(input_video),
        "-f", "rawvideo",
        "-pix_fmt", "gray",
        "-vsync", "0",
        "-",
    ]
    print(f'FFmpeg gray8 decode threads: {ffmpeg_decode_threads()}')
    proc = _spawn_subprocess_with_retry(
        lambda: subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE),
        'ffmpeg gray8 decode',
    )
    assert proc.stdout is not None

    try:
        with tqdm(total=num_frames, desc='Decoding input volume (gray8)') as pbar:
            for start in range(0, num_frames, chunk_frames):
                nframes = min(chunk_frames, num_frames - start)
                need = int(nframes) * int(frame_bytes)
                offset = int(start) * int(frame_bytes)
                view = raw_bytes[offset:offset + need]
                filled = 0
                while filled < need:
                    nread = proc.stdout.readinto(view[filled:])
                    if nread is None or nread <= 0:
                        raise RuntimeError(f'Unexpected EOF while decoding frames starting at {start}/{num_frames}')
                    filled += int(nread)
                pbar.update(int(nframes))
        flush_array(arr)
    finally:
        if proc.stdout:
            proc.stdout.close()
        _, err = proc.communicate()
        if proc.returncode not in (0, None):
            msg = err.decode("utf-8", errors="ignore") if isinstance(err, (bytes, bytearray)) else str(err)
            raise RuntimeError(f"ffmpeg decode failed: {msg}")
    return arr

def decode_video_to_memmap_gray8_streaming(
    input_video: Path,
    out_dat: Path,
    num_frames: int,
    width: int,
    height: int,
    overwrite: bool = False,
    *,
    prefer_memory: bool = True,
    prefer_memfd: bool = False,
    reserve_bytes: int = 32 * GIB,
) -> np.ndarray:
    """Start ffmpeg gray8 decode in the background and return its destination array immediately."""
    _require_bin("ffmpeg")
    shape = (int(num_frames), int(height), int(width))
    reuse_existing = bool(not overwrite and out_dat.exists() and not prefer_memory and not prefer_memfd)
    arr = allocate_workspace_array(
        shape=shape,
        dtype=np.uint8,
        path=out_dat,
        desc='Decoded gray8 input volume [streaming producer]',
        prefer_memory=bool(prefer_memory),
        prefer_memfd=bool(prefer_memfd),
        reserve_bytes=int(reserve_bytes),
        reuse_existing=bool(reuse_existing),
        initialize_zero=False,
    )
    readiness = VolumeReadiness(int(num_frames), desc='streaming ffmpeg gray8 decode')
    register_volume_readiness(arr, readiness)
    if reuse_existing and isinstance(arr, np.memmap):
        readiness.mark_all_ready()
        return arr

    raw_bytes = memoryview(np.ascontiguousarray(arr) if not arr.flags['C_CONTIGUOUS'] else arr).cast('B')
    if raw_bytes.obj is not arr:
        arr = np.asarray(raw_bytes.obj).reshape(shape)
        raw_bytes = memoryview(arr).cast('B')
        register_volume_readiness(arr, readiness)

    frame_bytes = int(width) * int(height)
    chunk_frames = max(1, min(128, max(1, (256 * 1024 * 1024) // max(1, frame_bytes))))

    def _decode_worker() -> None:
        cmd = [
            "ffmpeg", "-v", "error", "-threads", str(ffmpeg_decode_threads()),
            "-i", str(input_video), "-f", "rawvideo", "-pix_fmt", "gray", "-vsync", "0", "-",
        ]
        print(f'FFmpeg gray8 streaming decode threads: {ffmpeg_decode_threads()}')
        # The launch itself must sit inside the failure path: a spawn OSError that escaped
        # this thread used to leave the VolumeReadiness unmarked, so every consumer of the
        # decoded volume blocked forever on wait_for_slice while the pipeline "continued".
        proc: Optional[subprocess.Popen] = None
        try:
            proc = _spawn_subprocess_with_retry(
                lambda: subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE),
                'ffmpeg streaming gray8 decode',
            )
            assert proc.stdout is not None
            with tqdm(total=num_frames, desc='Streaming decode input volume (gray8)') as pbar:
                for start in range(0, int(num_frames), int(chunk_frames)):
                    if streaming_producers_aborted():
                        raise RuntimeError('streaming gray8 decode aborted by pipeline shutdown')
                    nframes = min(int(chunk_frames), int(num_frames) - int(start))
                    need = int(nframes) * int(frame_bytes)
                    offset = int(start) * int(frame_bytes)
                    view = raw_bytes[offset:offset + need]
                    filled = 0
                    while filled < need:
                        nread = proc.stdout.readinto(view[filled:])
                        if nread is None or nread <= 0:
                            raise RuntimeError(f'Unexpected EOF while decoding frames starting at {start}/{num_frames}')
                        filled += int(nread)
                    for frame_idx in range(int(start), int(start) + int(nframes)):
                        readiness.mark_slice_ready(int(frame_idx))
                    pbar.update(int(nframes))
            flush_array(arr)
            readiness.mark_all_ready()
        except BaseException as exc:
            # Loud failure: waking every waiter with the root cause is what turns a silent
            # consumer hang into an immediate pipeline abort.
            readiness.mark_failed(exc)
            print(f'ERROR: streaming gray8 decode failed; failing all volume consumers: {exc}')
            # Unblock the finally's communicate: a decode we are abandoning must not be
            # waited on while it still streams the remainder of the input.
            try:
                if proc is not None and proc.poll() is None:
                    proc.kill()
            except Exception:
                pass
            raise
        finally:
            if proc is not None:
                if proc.stdout:
                    proc.stdout.close()
                _out, err = proc.communicate()
                if proc.returncode not in (0, None):
                    msg = err.decode("utf-8", errors="ignore") if isinstance(err, (bytes, bytearray)) else str(err)
                    readiness.mark_failed(RuntimeError(f"ffmpeg decode failed: {msg}"))

    # daemon=True: if main dies without reaching abort_streaming_producers,
    # interpreter shutdown must not join a producer that may have hours of decode left.
    threading.Thread(target=_decode_worker, name='streaming-gray8-decode', daemon=True).start()
    return arr

def compute_cube_resize_shape(
    t_dim: int,
    h_dim: int,
    w_dim: int,
    tolerance: float = 0.05,
) -> Tuple[int, int, int]:
    """Return the smallest processing shape within tolerance of the longest source axis.

    Axes already inside the tolerated band are preserved; only shorter axes are upsampled.
    """
    t_i = max(1, int(t_dim))
    h_i = max(1, int(h_dim))
    w_i = max(1, int(w_dim))
    longest = max(t_i, h_i, w_i)
    lower_bound = int(math.ceil(float(longest) * (1.0 - float(tolerance))))
    return (
        max(t_i, lower_bound),
        max(h_i, lower_bound),
        max(w_i, lower_bound),
    )

def processing_volume_mode() -> str:
    """Return the canonical ``cube`` or ``native`` processing geometry."""
    raw = os.environ.get('YOLO_TTA_PROCESSING_VOLUME_MODE', '').strip().lower()
    if raw in ('', 'cube'):
        return 'cube'
    if raw == 'native':
        return 'native'
    print(
        f"Warning: unsupported YOLO_TTA_PROCESSING_VOLUME_MODE={raw!r}; "
        "expected 'cube' or 'native'; using 'cube'."
    )
    return 'cube'

def should_resize_to_processing_cube(input_shape: Tuple[int, int, int], cube_shape: Tuple[int, int, int]) -> bool:
    mode = processing_volume_mode()
    if mode != 'cube':
        return False
    return tuple(int(x) for x in input_shape) != tuple(int(x) for x in cube_shape)

def _linear_source_index(out_idx: int, out_count: int, in_count: int) -> float:
    if int(out_count) <= 1 or int(in_count) <= 1:
        return 0.0
    return float(out_idx) * float(in_count - 1) / float(out_count - 1)

def _resize_gray_slice_nearest_or_linear(
    frame: np.ndarray,
    out_w: int,
    out_h: int,
    interpolation: int,
) -> np.ndarray:
    frame_arr = np.asarray(frame, dtype=np.uint8)
    if int(frame_arr.shape[0]) == int(out_h) and int(frame_arr.shape[1]) == int(out_w):
        return np.ascontiguousarray(frame_arr)
    return cv2.resize(
        np.ascontiguousarray(frame_arr),
        (int(out_w), int(out_h)),
        interpolation=int(interpolation),
    )

def _cube_t_axis_resize_backend() -> str:
    """Backend used when cubic resizing only changes the slice axis.

 ``slab`` is the default because the common 3072x3072x1930 -> approximately
 cubic case does not need XY resampling. It processes row slabs through
 OpenCV's native vertical resize and fans the slabs out across Python worker
 threads. ``slice_exact`` preserves the older endpoint-aligned per-output-slice
 interpolation path for regression testing."""
    backend = os.environ.get('YOLO_TTA_CUBE_T_RESIZE_BACKEND', 'slab').strip().lower()
    if backend not in {'slab', 'slice_exact'}:
        backend = 'slab'
    return backend

def _cube_t_axis_slab_rows(in_w: int, workers: int) -> int:
    """Choose a bounded row-slab height for T-axis-only cubic resizing."""
    env_rows = _env_int('YOLO_TTA_CUBE_T_RESIZE_SLAB_ROWS', 0)
    if env_rows > 0:
        return max(1, int(env_rows))

    # Keep OpenCV's temporary 2D image width below conservative historical limits,
    # while also keeping per-task input/output buffers comfortably bounded.
    max_cv_width = max(1, _env_int('YOLO_TTA_CUBE_T_RESIZE_MAX_CV_WIDTH', 32760))
    rows_by_width = max(1, int(max_cv_width) // max(1, int(in_w)))

    target_mib = max(16, _env_int('YOLO_TTA_CUBE_T_RESIZE_TARGET_MIB_PER_TASK', 384))
    bytes_per_row = max(1, int(in_w)) * 2  # one input byte + one output byte, approximate
    rows_by_memory = max(1, int((target_mib * 1024 * 1024) // max(1, bytes_per_row * max(1, workers))))
    return max(1, min(rows_by_width, rows_by_memory, 16))

def resize_volume_t_axis_only_gray8_slab(
    volume_gray: np.ndarray,
    out_t: int,
    out_path: Path,
    *,
    workers: int = 1,
    prefer_memory: bool = True,
    reserve_bytes: int = 32 * GIB,
) -> np.ndarray:
    """Resize only the first/T axis by processing independent row slabs.

 This is optimized for the dominant SLURM input geometry where X/Y are already
 at the target size and only the shorter frame axis is upsampled to satisfy the
 approximate-cube requirement. Each task views a ``(T, rows*X)`` slab as a 2D
 OpenCV image and resizes only its height to ``out_t``."""
    in_t, in_h, in_w = (int(volume_gray.shape[0]), int(volume_gray.shape[1]), int(volume_gray.shape[2]))
    out_t_i = int(out_t)
    if int(in_t) == out_t_i:
        return volume_gray

    out_mm = allocate_workspace_array(
        shape=(out_t_i, in_h, in_w),
        dtype=np.uint8,
        path=out_path,
        desc='v12.2.0 cubic processing volume (parallel T-axis slab resize)',
        prefer_memory=bool(prefer_memory),
        reserve_bytes=int(reserve_bytes),
        initialize_zero=False,
    )

    worker_count = choose_slice_parallel_workers(int(workers), int(in_h))
    rows_per_slab = _cube_t_axis_slab_rows(int(in_w), worker_count)
    ranges = [(int(y0), int(min(in_h, y0 + rows_per_slab))) for y0 in range(0, int(in_h), int(rows_per_slab))]
    cv_threads = max(1, _env_int('YOLO_TTA_CUBE_T_RESIZE_CV2_THREADS', 1))

    try:
        previous_cv_threads = cv2.getNumThreads()
    except Exception:
        previous_cv_threads = None
    try:
        cv2.setNumThreads(int(cv_threads))
    except Exception:
        pass

    print(
        'Cubic resize T-axis slab backend: '
        f'in=(t,Y,X)=({in_t},{in_h},{in_w}) -> out_t={out_t_i}, '
        f'slab_rows={rows_per_slab}, slab_tasks={len(ranges)}, workers={worker_count}, '
        f'cv2_threads_per_task={cv_threads}'
    )

    def _resize_slab(range_idx: int) -> None:
        y0, y1 = ranges[int(range_idx)]
        rows = int(y1 - y0)
        slab = np.ascontiguousarray(volume_gray[:, y0:y1, :], dtype=np.uint8)
        slab_2d = slab.reshape((int(in_t), int(rows) * int(in_w)))
        resized_2d = cv2.resize(
            slab_2d,
            (int(rows) * int(in_w), int(out_t_i)),
            interpolation=cv2.INTER_LINEAR,
        )
        out_mm[:, y0:y1, :] = np.ascontiguousarray(resized_2d.reshape((int(out_t_i), int(rows), int(in_w))))

    try:
        parallel_for_indices_chunked(
            len(ranges),
            _resize_slab,
            max_workers=worker_count,
            desc='Resizing orthogonal volume to v12.2.0 cube (T-axis slabs)',
            show_progress=True,
            chunk_size=1,
        )
    finally:
        if previous_cv_threads is not None:
            try:
                cv2.setNumThreads(int(previous_cv_threads))
            except Exception:
                pass

    flush_array(out_mm)
    return out_mm

def resize_volume_to_processing_cube_gray8(
    volume_gray: np.ndarray,
    out_shape: Tuple[int, int, int],
    out_path: Path,
    *,
    workers: int = 1,
    prefer_memory: bool = True,
    reserve_bytes: int = 32 * GIB,
) -> np.ndarray:
    """Resample a gray8 (t,Y,X) volume to the approximately-cubic processing shape."""
    in_t, in_h, in_w = (int(volume_gray.shape[0]), int(volume_gray.shape[1]), int(volume_gray.shape[2]))
    out_t, out_h, out_w = (int(out_shape[0]), int(out_shape[1]), int(out_shape[2]))
    if (in_t, in_h, in_w) == (out_t, out_h, out_w):
        return volume_gray

    if in_h == out_h and in_w == out_w and _cube_t_axis_resize_backend() == 'slab':
        return resize_volume_t_axis_only_gray8_slab(
            volume_gray,
            out_t,
            out_path,
            workers=int(workers),
            prefer_memory=bool(prefer_memory),
            reserve_bytes=int(reserve_bytes),
        )

    out_mm = allocate_workspace_array(
        shape=(out_t, out_h, out_w),
        dtype=np.uint8,
        path=out_path,
        desc='v12.2.0 cubic processing volume',
        prefer_memory=bool(prefer_memory),
        reserve_bytes=int(reserve_bytes),
        initialize_zero=False,
    )

    worker_count = choose_slice_parallel_workers(int(workers), int(out_t))
    chunk_size = choose_parallel_chunk_size(out_t, worker_count, target_chunks_per_worker=2, min_chunk_size=1)

    def _render_target_slice(out_z: int) -> None:
        src_z = _linear_source_index(int(out_z), int(out_t), int(in_t))
        z0 = int(math.floor(src_z))
        z1 = min(in_t - 1, z0 + 1)
        alpha = float(src_z - float(z0))

        f0 = _resize_gray_slice_nearest_or_linear(volume_gray[z0], out_w, out_h, cv2.INTER_LINEAR)
        if z1 == z0 or alpha <= 1e-7:
            out_mm[int(out_z), :, :] = f0
            return
        f1 = _resize_gray_slice_nearest_or_linear(volume_gray[z1], out_w, out_h, cv2.INTER_LINEAR)
        blended = cv2.addWeighted(
            np.ascontiguousarray(f0),
            float(1.0 - alpha),
            np.ascontiguousarray(f1),
            float(alpha),
            0.0,
        )
        out_mm[int(out_z), :, :] = blended

    parallel_for_indices_chunked(
        int(out_t),
        _render_target_slice,
        max_workers=worker_count,
        desc='Resizing orthogonal volume to v12.2.0 cube',
        chunk_size=chunk_size,
    )
    flush_array(out_mm)
    return out_mm

def resize_volume_to_processing_cube_gray8_streaming(
    volume_gray: np.ndarray,
    out_shape: Tuple[int, int, int],
    out_path: Path,
    *,
    workers: int = 1,
    prefer_memory: bool = True,
    reserve_bytes: int = 32 * GIB,
) -> np.ndarray:
    """Start cubic gray8 preprocessing in the background and return its output array immediately."""
    in_t, in_h, in_w = (int(volume_gray.shape[0]), int(volume_gray.shape[1]), int(volume_gray.shape[2]))
    out_t, out_h, out_w = (int(out_shape[0]), int(out_shape[1]), int(out_shape[2]))
    if (in_t, in_h, in_w) == (out_t, out_h, out_w):
        input_ready = volume_readiness(volume_gray)
        if input_ready is not None:
            register_volume_readiness(volume_gray, input_ready)
        return volume_gray

    out_mm = allocate_workspace_array(
        shape=(out_t, out_h, out_w), dtype=np.uint8, path=out_path,
        desc='v12.2.0 cubic processing volume [streaming producer]',
        prefer_memory=bool(prefer_memory), reserve_bytes=int(reserve_bytes), initialize_zero=False,
    )
    readiness = VolumeReadiness(int(out_t), desc='streaming cubic processing volume')
    register_volume_readiness(out_mm, readiness)
    worker_count = choose_slice_parallel_workers(int(workers), int(out_t))
    chunk_size = choose_parallel_chunk_size(out_t, worker_count, target_chunks_per_worker=2, min_chunk_size=1)

    def _render_target_slice(out_z: int) -> None:
        if streaming_producers_aborted():
            raise RuntimeError('streaming cubic resize aborted by pipeline shutdown')
        src_z = _linear_source_index(int(out_z), int(out_t), int(in_t))
        z0 = int(math.floor(src_z))
        z1 = min(in_t - 1, z0 + 1)
        alpha = float(src_z - float(z0))
        wait_for_volume_slice_ready(volume_gray, int(z0))
        f0 = _resize_gray_slice_nearest_or_linear(volume_gray[z0], out_w, out_h, cv2.INTER_LINEAR)
        if z1 == z0 or alpha <= 1e-7:
            out_mm[int(out_z), :, :] = f0
            readiness.mark_slice_ready(int(out_z))
            return
        wait_for_volume_slice_ready(volume_gray, int(z1))
        f1 = _resize_gray_slice_nearest_or_linear(volume_gray[z1], out_w, out_h, cv2.INTER_LINEAR)
        out_mm[int(out_z), :, :] = cv2.addWeighted(np.ascontiguousarray(f0), float(1.0 - alpha), np.ascontiguousarray(f1), float(alpha), 0.0)
        readiness.mark_slice_ready(int(out_z))

    def _resize_worker() -> None:
        try:
            print(
                'Streaming cubic resize producer: '
                f'in=(t,Y,X)=({in_t},{in_h},{in_w}) -> out=(t,Y,X)=({out_t},{out_h},{out_w}), '
                f'workers={int(worker_count)}, chunk_size={int(chunk_size)}'
            )
            parallel_for_indices_chunked(
                int(out_t), _render_target_slice, max_workers=worker_count,
                desc='Streaming resize orthogonal volume to v12.2.0 cube', chunk_size=chunk_size,
            )
            flush_array(out_mm)
            readiness.mark_all_ready()
        except BaseException as exc:
            readiness.mark_failed(exc)
            raise

    # daemon=True: see the streaming decode producer.
    threading.Thread(target=_resize_worker, name='streaming-cubic-resize', daemon=True).start()
    return out_mm

class LazyProcessingCube:
    """Publish processing-cube geometry while deferring host materialization.

    CUDA renderers can consume the decoded source and map the logical cube on device for
    either one or multiple GPUs. CPU and fallback consumers request one shared resize through
    filesystem markers; local array access materializes synchronously.
    """

    _is_lazy_processing_cube = True

    def __init__(
        self,
        source: np.ndarray,
        out_shape: Tuple[int, int, int],
        out_path: Path,
        *,
        workers: int,
        request_path: Path,
        ready_path: Path,
        failed_path: Path,
        streaming_backend: bool = False,
    ) -> None:
        self.source = source
        self.shape = tuple(int(v) for v in out_shape)
        self.dtype = np.dtype(np.uint8)
        self.ndim = 3
        self.nbytes = int(array_nbytes(self.shape, self.dtype))
        self.backing_path = Path(out_path)
        self.request_path = Path(request_path)
        self.ready_path = Path(ready_path)
        self.failed_path = Path(failed_path)
        self.workers = max(1, int(workers))
        self.streaming_backend = bool(streaming_backend)
        self._lock = threading.RLock()
        self._ready = threading.Event()
        self._stop = threading.Event()
        self._started = False
        self._array: Optional[np.ndarray] = None
        self._error: Optional[BaseException] = None
        self._watcher: Optional[threading.Thread] = None
        self.backing_path.parent.mkdir(parents=True, exist_ok=True)
        for marker in (self.request_path, self.ready_path, self.failed_path):
            try:
                marker.unlink(missing_ok=True)
            except Exception as exc:
                raise RuntimeError(
                    f'cannot clear stale lazy-cube marker {marker}: {exc}'
                ) from exc
            if marker.exists():
                raise RuntimeError(f'stale lazy-cube marker survived cleanup: {marker}')

    @property
    def materialized(self) -> bool:
        with self._lock:
            return self._array is not None

    @property
    def filename(self) -> str:
        # Metadata-only: exposing the planned path must not trigger construction.
        return str(self.backing_path)

    def start_request_watcher(self) -> None:
        with self._lock:
            if self._watcher is not None:
                return

            def _watch() -> None:
                while not self._stop.is_set() and not self._ready.is_set():
                    if self.request_path.exists():
                        try:
                            self.materialize(reason='GPU-worker fallback request')
                        except Exception as exc:
                            try:
                                print(f'Warning: lazy processing cube request failed ({exc}).')
                            except Exception:
                                pass
                        return
                    self._stop.wait(0.10)

            self._watcher = threading.Thread(
                target=_watch,
                name='lazy-processing-cube-request',
                daemon=True,
            )
            self._watcher.start()

    def materialize(self, *, reason: str = 'local fallback consumer') -> np.ndarray:
        owner = False
        with self._lock:
            if self._error is not None:
                raise RuntimeError('lazy processing cube construction previously failed') from self._error
            if self._array is not None:
                return self._array
            if not self._started:
                self._started = True
                owner = True
        if owner:
            built: Optional[np.ndarray] = None
            try:
                print(
                    'v13.3.17 C10: materializing deferred processing cube for '
                    f'{reason}; shape={self.shape}, bytes={self.nbytes / GIB:.2f} GiB.'
                )
                # Deferring the cube also defers its dependency on decode completion. The
                # fallback is cold and correctness-sensitive, so complete decode first and
                # use the established exact slab/slice backend once.
                wait_for_volume_ready(self.source)
                if bool(self.streaming_backend):
                    # Preserve the established streaming producer's interpolation
                    # convention for runs that selected it at startup; only its launch
                    # time changes. The owner waits here before publishing the sentinel.
                    built = resize_volume_to_processing_cube_gray8_streaming(
                        self.source,
                        self.shape,
                        self.backing_path,
                        workers=int(self.workers),
                        prefer_memory=False,
                    )
                    wait_for_volume_ready(built)
                else:
                    built = resize_volume_to_processing_cube_gray8(
                        self.source,
                        self.shape,
                        self.backing_path,
                        workers=int(self.workers),
                        prefer_memory=False,
                    )
                flush_array(built)
                # The worker-visible sentinel is the transaction commit record. Publish it
                # before exposing the local array so local and subprocess consumers cannot
                # disagree about whether the shared cube is usable.
                self.ready_path.touch()
                with self._lock:
                    self._array = built
            except BaseException as exc:
                with self._lock:
                    self._error = exc
                try:
                    self.ready_path.unlink(missing_ok=True)
                except Exception:
                    pass
                if built is not None:
                    try:
                        close_memmap_array(built)
                    except Exception:
                        pass
                try:
                    fail_tmp = self.failed_path.with_name(
                        f'.{self.failed_path.name}.{os.getpid()}.{threading.get_ident()}.tmp'
                    )
                    fail_tmp.write_text(f'{type(exc).__name__}: {exc}\n')
                    os.replace(fail_tmp, self.failed_path)
                except Exception:
                    pass
            finally:
                self._ready.set()
            if self._error is None:
                # Logging is deliberately outside the construction/publication transaction:
                # a closed stdout must not turn an already-published cube into failed state.
                try:
                    print('v13.3.17 C10: deferred processing cube complete; fallback sentinel published.')
                except Exception:
                    pass
        else:
            self._ready.wait()
        with self._lock:
            if self._error is not None:
                raise RuntimeError('lazy processing cube construction failed') from self._error
            if self._array is None:  # pragma: no cover - defensive state guard
                raise RuntimeError('lazy processing cube completed without an array')
            return self._array

    def _materialize_for_wait(self) -> None:
        self.materialize(reason='wait_for_volume_ready consumer')

    def __array__(self, dtype=None, copy=None) -> np.ndarray:
        arr = np.asarray(self.materialize())
        if dtype is not None:
            arr = np.asarray(arr, dtype=dtype)
        if copy is True:
            return np.array(arr, copy=True)
        return arr

    def __getitem__(self, key):
        return self.materialize()[key]

    def flush(self) -> None:
        with self._lock:
            arr = self._array
        if arr is not None:
            flush_array(arr)

    def close(self) -> None:
        self._stop.set()
        with self._lock:
            arr = self._array
            self._array = None
        if arr is not None:
            close_memmap_array(arr)

def restore_mask_volume_to_original_shape(
    mask_u8: np.ndarray,
    out_shape: Tuple[int, int, int],
    out_path: Path,
    *,
    workers: int = 1,
    prefer_memory: bool = True,
    reserve_bytes: int = 16 * GIB,
) -> np.ndarray:
    """Map a processing-space binary mask back to the input video's original (t,Y,X) shape."""
    in_t, in_h, in_w = (int(mask_u8.shape[0]), int(mask_u8.shape[1]), int(mask_u8.shape[2]))
    out_t, out_h, out_w = (int(out_shape[0]), int(out_shape[1]), int(out_shape[2]))
    if (in_t, in_h, in_w) == (out_t, out_h, out_w):
        return mask_u8

    out_mm = allocate_workspace_array(
        shape=(out_t, out_h, out_w),
        dtype=np.uint8,
        path=out_path,
        desc='Restored final mask in original input geometry',
        prefer_memory=bool(prefer_memory),
        reserve_bytes=int(reserve_bytes),
        initialize_zero=False,
    )
    worker_count = choose_slice_parallel_workers(int(workers), int(out_t))
    chunk_size = choose_parallel_chunk_size(out_t, worker_count, target_chunks_per_worker=2, min_chunk_size=1)

    def _resize_mask_to_output_xy(frame: np.ndarray) -> np.ndarray:
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

    def _restore_slice(out_z: int) -> None:
        if int(in_t) >= int(out_t):
            src_start = int(math.floor(float(out_z) * float(in_t) / float(out_t)))
            src_stop = int(math.ceil(float(out_z + 1) * float(in_t) / float(out_t)))
            src_start = int(np.clip(src_start, 0, in_t - 1))
            src_stop = int(np.clip(max(src_start + 1, src_stop), 1, in_t))
            restored = np.zeros((int(out_h), int(out_w)), dtype=np.uint8)
            for src_idx in range(src_start, src_stop):
                restored |= _resize_mask_to_output_xy(mask_u8[int(src_idx)])
            out_mm[int(out_z), :, :] = restored
            return

        src_z = _linear_source_index(int(out_z), int(out_t), int(in_t))
        src_idx = int(np.clip(int(round(src_z)), 0, in_t - 1))
        out_mm[int(out_z), :, :] = _resize_mask_to_output_xy(mask_u8[src_idx])

    parallel_for_indices_chunked(
        int(out_t),
        _restore_slice,
        max_workers=worker_count,
        desc='Restoring final mask to original input geometry',
        chunk_size=chunk_size,
    )
    flush_array(out_mm)
    return out_mm

def resolve_radial_azimuth_angles(
    requests: Sequence[RadialViewRequest],
    *,
    diameters: Sequence[int],
) -> List[float]:
    """Resolve each Radial request's paired spacing after its diameter is known."""
    request_list = list(requests)
    if len(diameters) != len(request_list):
        raise ValueError(
            f'internal radial diameter count mismatch: {len(diameters)} diameter(s) for '
            f'{len(request_list)} request(s)'
        )
    resolved: List[float] = []
    for request, diameter in zip(request_list, diameters):
        if request.azimuth_angle is None:
            resolved.append(radial_full_coverage_angle_deg(int(diameter)))
        else:
            resolved.append(float(request.azimuth_angle))
    return resolved

def _open_memory_backed_encoded_chunk(name: str, *, require_fileno: bool = False) -> object:
    """Return a seekable binary object whose encoded payload is retained in memory.

 Linux memfd is preferred because it is anonymous, seekable, usable by copy_file_range,
 and can be inherited by ffmpeg without creating a shard path on disk. BytesIO is a
 portable in-process fallback for writers such as NRRD that do not need a child process."""
    safe_name = re.sub(r'[^A-Za-z0-9_.-]+', '_', str(name)).strip('._') or 'encoded_chunk'
    memfd_create = getattr(os, 'memfd_create', None)
    if callable(memfd_create):
        fd: Optional[int] = None
        try:
            flags = int(getattr(os, 'MFD_CLOEXEC', 0))
            fd = int(memfd_create(safe_name[:240], flags=flags))
            return os.fdopen(fd, mode='w+b', buffering=0)
        except Exception:
            if fd is not None:
                try:
                    os.close(int(fd))
                except OSError:
                    pass
            if bool(require_fileno):
                raise
    if bool(require_fileno):
        raise RuntimeError(
            'Memory-backed FFV1 chunking requires Linux os.memfd_create; '
            'no encoded shard will be written to disk as a fallback.'
        )
    return io.BytesIO()

def _memory_backed_encoded_chunk_path(chunk: object) -> Path:
    """Expose an inherited memfd to a child process without a persistent filesystem file."""
    try:
        fd = int(chunk.fileno())
    except Exception as exc:
        raise RuntimeError('Encoded child-process chunk does not expose a file descriptor') from exc
    return Path(f'/proc/self/fd/{fd}')

def _memory_backed_encoded_chunk_size(chunk: object) -> int:
    """Return encoded size without changing the caller-visible stream position."""
    position = int(chunk.tell())
    chunk.seek(0, os.SEEK_END)
    size = int(chunk.tell())
    chunk.seek(position, os.SEEK_SET)
    return int(size)

def ffmpeg_rawvideo_writer(
    out_path: Path,
    width: int,
    height: int,
    fps: float,
    pix_fmt_in: str = "gray",
    codec: str = "ffv1",
    pix_fmt_out: Optional[str] = "gray",
    codec_args: Optional[Sequence[str]] = None,
) -> subprocess.Popen:
    """Return a Popen with stdin open for writing raw frames."""
    _require_bin("ffmpeg")
    out_path = Path(out_path)
    memfd_match = re.fullmatch(r'/proc/self/fd/(\d+)', str(out_path))
    inherited_fds: Tuple[int, ...] = ()
    if memfd_match is None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        inherited_fds = (int(memfd_match.group(1)),)
    cmd = [
        "ffmpeg",
        "-y",
        "-v", "error",
        "-f", "rawvideo",
        "-pix_fmt", pix_fmt_in,
        "-s", f"{width}x{height}",
        "-r", f"{fps}",
        "-i", "-",
        "-an",
    ]

    if str(codec) == 'ffv1':
        # Match the 30 FFV1 slices with 30 encoder threads by default; the environment
        # override remains authoritative.
        ffv1_threads = max(1, _env_int('YOLO_TTA_FFV1_THREADS', 30))
        cmd.extend(["-c:v", "ffv1", "-level", "3", "-slices", "30", "-threads", str(int(ffv1_threads))])
    else:
        cmd.extend(["-c:v", str(codec)])

    if codec_args:
        cmd.extend([str(x) for x in codec_args])

    if pix_fmt_out:
        cmd.extend(["-pix_fmt", str(pix_fmt_out)])

    if inherited_fds:
        # proc/self/fd/<N> has no filename suffix, so select the intended container explicitly.
        cmd.extend(["-f", "matroska"])
    cmd.append(str(out_path))
    proc = _spawn_subprocess_with_retry(
        lambda: subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            pass_fds=inherited_fds,
        ),
        f'ffmpeg {codec} writer ({Path(out_path).name})',
    )
    assert proc.stdin is not None
    return proc

def ffmpeg_ffv1_gray_writer(
    out_path: Path,
    width: int,
    height: int,
    fps: float,
) -> subprocess.Popen:
    """FFV1 MKV writer for single-channel temporary, prediction, and binary videos."""
    return ffmpeg_rawvideo_writer(
        out_path=out_path,
        width=int(width),
        height=int(height),
        fps=float(fps),
        pix_fmt_in='gray',
        codec='ffv1',
        pix_fmt_out='gray',
    )

def ffmpeg_ffv1_rgb_writer(
    out_path: Path,
    width: int,
    height: int,
    fps: float,
) -> subprocess.Popen:
    """FFV1 MKV writer for color presentation overlays."""
    return ffmpeg_rawvideo_writer(
        out_path=out_path,
        width=int(width),
        height=int(height),
        fps=float(fps),
        pix_fmt_in='rgb24',
        codec='ffv1',
        pix_fmt_out='yuv444p',
    )

def close_ffmpeg_writer(proc: subprocess.Popen) -> None:
    """Close an ffmpeg writer Popen safely (Python 3.12+ compatible).

 IMPORTANT:
 Calling proc.stdin.close and then proc.communicate triggers
 'ValueError: flush of closed file' on Python 3.12, because communicate
 tries to flush stdin even if it is already closed. We therefore:
 1) close stdin (if open)
 2) set proc.stdin = None
 3) call communicate to drain stdout/stderr"""
    if proc.stdin is not None and not proc.stdin.closed:
        try:
            proc.stdin.close()
        except Exception:
            pass

    # Prevent subprocess.communicate from flushing a closed stdin (Py3.12 behavior)
    proc.stdin = None  # type: ignore[attr-defined]

    _, err = proc.communicate()
    if proc.returncode not in (0, None):
        msg = err.decode("utf-8", errors="ignore") if isinstance(err, (bytes, bytearray)) else str(err)
        raise RuntimeError(f"ffmpeg write failed: {msg}")

def _path_is_relative_to(path: Path, root: Path) -> bool:
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
        return True
    except Exception:
        return False

def purge_temporary_mkv(
    video_path: Optional[Path],
    *,
    temp_dir: Path,
    keep_temp: bool,
    reason: str = '',
) -> bool:
    """Delete one temporary MKV once its last pipeline consumer has finished.

 The guard is intentionally narrow: only ``*.mkv`` files below the active scratch/temp
 directory are eligible. Final output MKVs live in the output directory and are never
 removed through this helper."""
    if bool(keep_temp) or video_path is None:
        return False

    path = Path(video_path)
    if path.suffix.lower() != '.mkv':
        return False
    if not _path_is_relative_to(path, Path(temp_dir)):
        return False
    if not path.exists():
        return False

    try:
        path.unlink(missing_ok=True)
        if reason:
            print(f'Purged temporary MKV after last use ({reason}): {path.name}')
        return True
    except Exception as exc:
        print(f'Warning: failed to purge temporary MKV {path} ({exc})')
        return False

def purge_remaining_temporary_mkvs(temp_dir: Path, *, keep_temp: bool) -> int:
    """Best-effort final sweep for temporary MKVs that survived targeted lifecycle deletion."""
    if bool(keep_temp):
        return 0
    root = Path(temp_dir)
    if not root.exists():
        return 0

    purged = 0
    for path in list(root.rglob('*.mkv')):
        if purge_temporary_mkv(path, temp_dir=root, keep_temp=False, reason='final scratch sweep'):
            purged += 1
    return int(purged)


# Late imports keep callable-only dependency cycles import-safe.
from ._latebind import bind_late_symbols as _bind_late_symbols

_bind_late_symbols(
    __name__,
    globals(),
    {
        "backprojection": (
            "radial_full_coverage_angle_deg",
        ),
        "config": (
            "GIB",
            "RadialViewRequest",
        ),
        "runtime": (
            "allocate_workspace_array",
            "array_nbytes",
            "choose_parallel_chunk_size",
            "choose_slice_parallel_workers",
            "close_memmap_array",
            "flush_array",
            "parallel_for_indices_chunked",
        ),
        "workspace": (
            "_cpu_count",
            "_env_flag",
            "_env_float",
            "_env_int",
        ),
    },
)
