"""Reserve bounded retained-payload RAM before dispatching native shell owners."""
from __future__ import annotations

import os
from pathlib import Path
from .config import GIB
from .runtime import _register_memfd_owner, memfd_workspace_enabled, workspace_anon_cap_bytes, scratch_dir_is_memory_backed
from .workspace import _env_flag, _env_float, _read_meminfo_bytes, available_anon_work_bytes


def publication_ram_headroom():
    """Use physical/cgroup headroom only; spare swap is not a RAM cache budget."""
    return max(0, min(int(_read_meminfo_bytes().get('MemAvailable', 0)),
                      int(available_anon_work_bytes())))


def retained_payload_plan(shapes, available, worker_count, publication_pending, unpack_bytes, *, cap=0, output_reserve_bytes=0):
    """Reserve future topology/union and all admitted host publication bitsets.

    Each selected layer is charged its *worst-case* packed payload for its whole
    lifetime, including layers not started yet. Independent workers cannot each
    promise themselves the same currently-free RAM. Disk fallbacks need no grant.
    """
    shapes = [tuple(map(int, s)) for s in shapes]
    if any(len(s) != 3 or min(s) <= 0 for s in shapes):
        raise ValueError('Publication memory plan requires positive 3D shapes')
    dense = max((t * h * w for t, h, w in shapes), default=0)
    words = ((dense + 31) // 32) * 4
    # Five uint8-volume equivalents reserve the final union and worst-case uint32
    # local label raster. Host bitsets cover every publication credit, plus the
    # currently computing owner on each worker. Other work keeps a fixed margin.
    reserve = 32 * GIB + 5 * dense + max(0, int(worker_count)) * (
        (max(0, int(publication_pending)) + 1) * words
        + 2 * max(1, int(publication_pending)) * max(0, int(unpack_bytes)))
    reserve += max(0, int(output_reserve_bytes))
    budget = max(0, int(available) - reserve) // 2
    if int(cap) > 0:
        budget = min(budget, int(cap))
    remaining = budget
    grants = []
    for t, h, w in shapes:
        need = t * h * ((w + 7) // 8)
        grant = need if need <= remaining else 0
        grants.append(grant)
        remaining -= grant
    return dict(reserve_bytes=reserve, budget_bytes=budget,
                reserved_bytes=budget - remaining, grants=grants)


def publication_output_reserve(sink, window_bytes, member_bytes):
    """Budget actual sink concurrency, mirror canvases and gzip input/output windows."""
    if sink is None:
        return 0
    mirrors = list(sink.low_quality_specs)
    canvases = sum(int(np_size.output_shape_t_y_x[0]) * int(np_size.output_shape_t_y_x[1]) *
                   int(np_size.output_shape_t_y_x[2]) for np_size in mirrors)
    lanes = 1 + min(4, len(mirrors))
    windows = lanes * 2 * (max(0, int(window_bytes)) + 2 * max(0, int(member_bytes)))
    dense = int(sink.output_shape[0]) * int(sink.output_shape[1]) * int(sink.output_shape[2])
    # Include a worst-case global compressed spool as well as per-writer windows.
    return int(sink.max_workers) * (canvases + windows) + dense + dense // 100 + 1024**2


def plan_native_publication_memory(tasks, *, keep_temp, worker_count, publication_pending, unpack_bytes, output_reserve_bytes=0):
    """Create parent-owned empty memfds; immutable task grants bound their growth."""
    if (keep_temp or not memfd_workspace_enabled() or scratch_dir_is_memory_backed()
            or not _env_flag('YOLO_TTA_PUBLICATION_RAM', True)
            or not _env_flag('YOLO_TTA_PACKED_OWNER_PUBLICATION', True)):
        return None
    by_path = {}
    for task in tasks:
        if (task.get('projection_contract') == 'radial_native_pull_v1'
                and task.get('result_mode') == 'd1_owner'
                and not task.get('d1_view_shadow_required')):
            by_path.setdefault(str(task['d1_store_dir']), []).append(task)
    if not by_path:
        return None
    groups = list(by_path.values())
    shapes = [group[0]['d1_output_shape'] for group in groups]
    for group, shape in zip(groups, shapes):
        owners = {(str(task['model_name']), str(task['view'].name)) for task in group}
        if len(owners) != 1 or any(task['d1_output_shape'] != shape for task in group):
            print('Native publication RAM plan declined ambiguous layer identities; retaining disk backings.', flush=True)
            return None
    cap = max(0, int(_env_float('YOLO_TTA_PUBLICATION_RAM_GIB', 0.) * GIB))
    workspace_cap = int(workspace_anon_cap_bytes())
    if workspace_cap > 0:
        cap = min(cap, workspace_cap) if cap else workspace_cap
    headroom = publication_ram_headroom()
    plan = retained_payload_plan(shapes, headroom, worker_count, publication_pending, unpack_bytes,
                                 cap=cap, output_reserve_bytes=output_reserve_bytes)
    # Small volumes can admit many more layers than a process can hold open.
    # Leave descriptors for inference IPC, readers, codecs and final output.
    try:
        import resource
        soft_limit = int(resource.getrlimit(resource.RLIMIT_NOFILE)[0])
        used_fds = len(os.listdir('/proc/self/fd'))
        descriptor_budget = 256 if soft_limit < 0 else max(0, min(256, (soft_limit-used_fds-64)//2))
    except (ImportError, OSError, ValueError):
        descriptor_budget = 0
    admitted = 0
    for index, (group, grant) in enumerate(zip(groups, plan['grants'])):
        if not grant or admitted >= descriptor_budget:
            plan['grants'][index] = 0
            continue
        fd = None
        try:
            fd = os.memfd_create('xta-packed-publication', flags=getattr(os, 'MFD_CLOEXEC', 0))
            path = Path(group[0]['d1_store_dir']).absolute() / 'chunks.bin'
            _register_memfd_owner(str(path), fd, 'planned packed source contribution')
            backing = f'/proc/{os.getpid()}/fd/{fd}'
            fd = None  # runtime registry now owns it through all downstream consumers
            for task in group:
                task['d1_memory_payload_path'] = backing
                task['d1_memory_payload_limit'] = int(grant)
                task['d1_memory_payload_reserve'] = int(plan['reserve_bytes'])
            admitted += 1
        except OSError as exc:
            plan['grants'][index] = 0
            if fd is not None:
                os.close(fd)
            print(f'Publication RAM backing unavailable; keeping disk payload: {exc}', flush=True)
    plan['reserved_bytes'] = sum(plan['grants'])
    print('Native publication RAM plan: '
          f'physical/cgroup_headroom={headroom / GIB:.1f} GiB, '
          f'future_work_reserve={plan["reserve_bytes"] / GIB:.1f} GiB, '
          f'retained_budget={plan["budget_bytes"] / GIB:.1f} GiB, '
          f'worst_case_reserved={plan["reserved_bytes"] / GIB:.1f} GiB; '
          f'{admitted}/{len(groups)} layers admitted (fd_budget={descriptor_budget}), disk spill remains available.', flush=True)
    return plan
