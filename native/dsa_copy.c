#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <linux/idxd.h>
#include <pthread.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

/*
 * This binding intentionally uses the maintained Linux idxd character-device
 * ABI rather than DML (unmaintained), DTO (may silently execute on CPU), or a
 * process-wide memcpy interceptor.  write(2) copies one or more 64-byte DSA
 * descriptors into the kernel driver, which submits them to the selected user
 * work queue.  The source/destination buffers and completion records remain
 * pinned by Python references until every submitted descriptor is terminal or
 * close(2) has synchronously released the work-queue context.
 *
 * Descriptor layout, flags, opcodes, and completion statuses come directly
 * from include/uapi/linux/idxd.h. The compile-time assertions deliberately
 * make a kernel-header ABI mismatch a build failure instead of corrupting memory.
 */

#define VT_DSA_MAX_INFLIGHT   4096
#define VT_DSA_WAIT_SECONDS   120.0

_Static_assert(sizeof(struct dsa_hw_desc) == 64, "unsupported Linux DSA descriptor ABI");
_Static_assert(sizeof(struct dsa_completion_record) == 32, "unsupported Linux DSA completion ABI");

struct vt_queue_info {
    int hardware_available;
    int device_id;
    int queue_id;
    int numa_node;
    int caller_numa_node;
    int numa_local;
    int block_on_fault;
    int prs_disable;
    uint64_t max_transfer_size;
    uint64_t queue_size;
    char path[PATH_MAX];
    char mode[32];
    char reason[256];
};

struct vt_copy_stats {
    uint64_t requested_bytes;
    uint64_t hardware_bytes;
    uint64_t submitted_descriptors;
    uint64_t descriptors;
    uint64_t batches;
    uint64_t queue_full_events;
    uint64_t partial_failures;
    uint64_t page_faults;
    uint64_t minor_page_faults;
    uint64_t major_page_faults;
    int max_inflight;
    int drained;
    int success;
    double submission_seconds;
    double wait_seconds;
    double total_seconds;
    double cpu_seconds;
    char work_queue[PATH_MAX];
    char error[512];
};

static PyObject *DsaCopyError = NULL;
static PyThread_type_lock native_lock = NULL;
static pid_t native_pid = 0;
static int last_drained = 1;
static uint64_t last_submitted = 0;
static pthread_rwlock_t fork_gate = PTHREAD_RWLOCK_INITIALIZER;
static int atfork_registered = 0;

static void
vt_atfork_prepare(void)
{
    /* Wait until every cdev capability probe/copy has closed its fd. */
    (void)pthread_rwlock_wrlock(&fork_gate);
}

static void
vt_atfork_parent(void)
{
    (void)pthread_rwlock_unlock(&fork_gate);
}

static void
vt_atfork_child(void)
{
    /* The exclusive gate proves that no active cdev fd crossed fork. The
     * inherited PyThread lock is replaced lazily with the GIL in the child. */
    native_pid = 0;
    last_drained = 1;
    last_submitted = 0;
    (void)pthread_rwlock_unlock(&fork_gate);
}

static int
vt_ensure_process_state(void)
{
    pid_t current = getpid();
    PyThread_type_lock replacement;
    if (native_lock != NULL && native_pid == current) {
        return 0;
    }
    replacement = PyThread_allocate_lock();
    if (replacement == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "failed to allocate process-local DSA native lock");
        return -1;
    }
    /* After fork, another parent thread may have held the inherited lock. It is
     * neither acquired nor freed in the child; replace it while the GIL makes
     * this PID transition single-threaded. */
    native_lock = replacement;
    native_pid = current;
    last_drained = 1;
    last_submitted = 0;
    return 0;
}

static double
vt_timespec_seconds(const struct timespec *value)
{
    return (double)value->tv_sec + (double)value->tv_nsec / 1000000000.0;
}

static double
vt_elapsed(const struct timespec *start, const struct timespec *end)
{
    return vt_timespec_seconds(end) - vt_timespec_seconds(start);
}

static void
vt_trim(char *value)
{
    size_t length;
    if (value == NULL) {
        return;
    }
    length = strlen(value);
    while (length > 0 &&
           (value[length - 1] == '\n' || value[length - 1] == '\r' ||
            value[length - 1] == ' ' || value[length - 1] == '\t')) {
        value[--length] = '\0';
    }
}

static int
vt_read_text(const char *path, char *buffer, size_t size)
{
    int fd;
    ssize_t count;
    if (size == 0) {
        return -1;
    }
    fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        return -1;
    }
    count = read(fd, buffer, size - 1);
    (void)close(fd);
    if (count < 0) {
        return -1;
    }
    buffer[(size_t)count] = '\0';
    vt_trim(buffer);
    return 0;
}

static int
vt_read_i64(const char *path, int64_t *result)
{
    char text[128];
    char *end = NULL;
    long long value;
    if (vt_read_text(path, text, sizeof(text)) != 0) {
        return -1;
    }
    errno = 0;
    value = strtoll(text, &end, 0);
    if (errno != 0 || end == text || *end != '\0') {
        return -1;
    }
    *result = (int64_t)value;
    return 0;
}

static int
vt_parse_queue_name(const char *value, int *device_id, int *queue_id)
{
    const char *name;
    int consumed = 0;
    int device = -1;
    int queue = -1;
    if (value == NULL || *value == '\0') {
        return -1;
    }
    name = strrchr(value, '/');
    name = name == NULL ? value : name + 1;
    if (sscanf(name, "wq%d.%d%n", &device, &queue, &consumed) != 2 ||
        consumed <= 0 || name[consumed] != '\0' || device < 0 || queue < 0) {
        return -1;
    }
    if (strchr(value, '/') != NULL) {
        char expected[PATH_MAX];
        if (snprintf(expected, sizeof(expected), "/dev/dsa/wq%d.%d", device, queue) < 0 ||
            strcmp(value, expected) != 0) {
            return -1;
        }
    }
    *device_id = device;
    *queue_id = queue;
    return 0;
}

static int
vt_cpu_numa_node(void)
{
    int cpu = sched_getcpu();
    char path[PATH_MAX];
    DIR *directory;
    struct dirent *entry;
    int node = -1;
    if (cpu < 0 || snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d", cpu) < 0) {
        return -1;
    }
    directory = opendir(path);
    if (directory == NULL) {
        return -1;
    }
    while ((entry = readdir(directory)) != NULL) {
        int candidate = -1;
        int consumed = 0;
        if (sscanf(entry->d_name, "node%d%n", &candidate, &consumed) == 1 &&
            consumed > 0 && entry->d_name[consumed] == '\0' && candidate >= 0) {
            node = candidate;
            break;
        }
    }
    (void)closedir(directory);
    return node;
}

static void
vt_queue_unavailable(struct vt_queue_info *info, const char *reason)
{
    info->hardware_available = 0;
    (void)snprintf(info->reason, sizeof(info->reason), "%s", reason);
}

static int
vt_inspect_queue(const char *requested, struct vt_queue_info *info)
{
    char path[PATH_MAX];
    char sysfs[PATH_MAX];
    char text[128];
    int64_t numeric;
    int probe_fd;
    struct stat status;

    memset(info, 0, sizeof(*info));
    info->device_id = -1;
    info->queue_id = -1;
    info->numa_node = -1;
    info->caller_numa_node = vt_cpu_numa_node();
    info->block_on_fault = -1;
    info->prs_disable = -1;
    if (vt_parse_queue_name(requested, &info->device_id, &info->queue_id) != 0) {
        vt_queue_unavailable(info, "work queue must be wqN.M or /dev/dsa/wqN.M");
        return -1;
    }
    if (snprintf(path, sizeof(path), "/dev/dsa/wq%d.%d", info->device_id, info->queue_id) < 0) {
        vt_queue_unavailable(info, "work-queue path is too long");
        return -1;
    }
    (void)snprintf(info->path, sizeof(info->path), "%s", path);

    if (lstat(path, &status) != 0 || !S_ISCHR(status.st_mode)) {
        vt_queue_unavailable(info, "work-queue path is not an idxd character device");
        return -1;
    }
    if (snprintf(sysfs, sizeof(sysfs), "/sys/bus/dsa/devices/wq%d.%d/state",
                 info->device_id, info->queue_id) < 0 ||
        vt_read_text(sysfs, text, sizeof(text)) != 0 || strcmp(text, "enabled") != 0) {
        vt_queue_unavailable(info, "work queue is not enabled");
        return -1;
    }
    if (snprintf(sysfs, sizeof(sysfs), "/sys/bus/dsa/devices/wq%d.%d/type",
                 info->device_id, info->queue_id) < 0 ||
        vt_read_text(sysfs, text, sizeof(text)) != 0 || strcmp(text, "user") != 0) {
        vt_queue_unavailable(info, "work queue is not a user work queue");
        return -1;
    }
    if (snprintf(sysfs, sizeof(sysfs), "/sys/bus/dsa/devices/wq%d.%d/max_transfer_size",
                 info->device_id, info->queue_id) < 0 ||
        vt_read_i64(sysfs, &numeric) != 0 || numeric <= 0) {
        vt_queue_unavailable(info, "work queue has no valid max_transfer_size");
        return -1;
    }
    info->max_transfer_size = (uint64_t)numeric;

    if (snprintf(sysfs, sizeof(sysfs), "/sys/bus/dsa/devices/wq%d.%d/size",
                 info->device_id, info->queue_id) >= 0 &&
        vt_read_i64(sysfs, &numeric) == 0 && numeric > 0) {
        info->queue_size = (uint64_t)numeric;
    } else {
        info->queue_size = 1;
    }
    if (snprintf(sysfs, sizeof(sysfs), "/sys/bus/dsa/devices/wq%d.%d/mode",
                 info->device_id, info->queue_id) >= 0 &&
        vt_read_text(sysfs, info->mode, sizeof(info->mode)) == 0) {
        /* captured for telemetry */
    } else {
        (void)snprintf(info->mode, sizeof(info->mode), "unknown");
    }
    if (snprintf(sysfs, sizeof(sysfs), "/sys/bus/dsa/devices/wq%d.%d/block_on_fault",
                 info->device_id, info->queue_id) >= 0 &&
        vt_read_i64(sysfs, &numeric) == 0) {
        info->block_on_fault = numeric != 0;
    }
    if (info->block_on_fault != 1) {
        vt_queue_unavailable(info, "initial DSA rollout requires block_on_fault=1");
        return -1;
    }
    if (snprintf(sysfs, sizeof(sysfs), "/sys/bus/dsa/devices/wq%d.%d/prs_disable",
                 info->device_id, info->queue_id) >= 0 &&
        vt_read_i64(sysfs, &numeric) == 0) {
        info->prs_disable = numeric != 0;
    }
    if (info->prs_disable == 1) {
        vt_queue_unavailable(info, "work queue prs_disable overrides block-on-fault support");
        return -1;
    }
    if (snprintf(sysfs, sizeof(sysfs), "/sys/bus/dsa/devices/dsa%d/numa_node",
                 info->device_id) < 0 || vt_read_i64(sysfs, &numeric) != 0) {
        vt_queue_unavailable(info, "unable to determine DSA device NUMA node");
        return -1;
    }
    info->numa_node = (int)numeric;
    info->numa_local = (
        info->numa_node < 0 ||
        (info->caller_numa_node >= 0 && info->caller_numa_node == info->numa_node)
    );

    probe_fd = open(path, O_WRONLY | O_CLOEXEC | O_NONBLOCK);
    if (probe_fd < 0) {
        (void)snprintf(info->reason, sizeof(info->reason),
                       "work queue cannot be opened: %s", strerror(errno));
        info->hardware_available = 0;
        return -1;
    }
    if (fstat(probe_fd, &status) != 0 || !S_ISCHR(status.st_mode)) {
        (void)close(probe_fd);
        vt_queue_unavailable(info, "opened work queue is not a character device");
        return -1;
    }
    if (close(probe_fd) != 0) {
        vt_queue_unavailable(info, "failed to close work-queue capability probe");
        return -1;
    }
    info->hardware_available = 1;
    info->reason[0] = '\0';
    return 0;
}

static int
vt_select_queue(const char *requested, struct vt_queue_info *selected)
{
    DIR *directory;
    struct dirent *entry;
    struct vt_queue_info candidate;
    int found = 0;
    if (requested != NULL && *requested != '\0') {
        return vt_inspect_queue(requested, selected);
    }
    memset(selected, 0, sizeof(*selected));
    selected->device_id = -1;
    selected->queue_id = -1;
    selected->numa_node = -1;
    selected->caller_numa_node = vt_cpu_numa_node();
    directory = opendir("/dev/dsa");
    if (directory == NULL) {
        vt_queue_unavailable(selected, "/dev/dsa is unavailable");
        return -1;
    }
    while ((entry = readdir(directory)) != NULL) {
        if (vt_inspect_queue(entry->d_name, &candidate) != 0 || !candidate.numa_local) {
            continue;
        }
        if (!found || candidate.device_id < selected->device_id ||
            (candidate.device_id == selected->device_id && candidate.queue_id < selected->queue_id)) {
            *selected = candidate;
            found = 1;
        }
    }
    (void)closedir(directory);
    if (!found) {
        vt_queue_unavailable(selected, "no accessible NUMA-local enabled DSA user work queue");
        return -1;
    }
    return 0;
}

static int
vt_dict_set_object(PyObject *dictionary, const char *key, PyObject *value)
{
    int result;
    if (value == NULL) {
        return -1;
    }
    result = PyDict_SetItemString(dictionary, key, value);
    Py_DECREF(value);
    return result;
}

static int
vt_dict_set_bool(PyObject *dictionary, const char *key, int value)
{
    return vt_dict_set_object(dictionary, key, PyBool_FromLong(value != 0));
}

static int
vt_dict_set_i64(PyObject *dictionary, const char *key, int64_t value)
{
    return vt_dict_set_object(dictionary, key, PyLong_FromLongLong((long long)value));
}

static int
vt_dict_set_u64(PyObject *dictionary, const char *key, uint64_t value)
{
    return vt_dict_set_object(dictionary, key, PyLong_FromUnsignedLongLong(value));
}

static int
vt_dict_set_double(PyObject *dictionary, const char *key, double value)
{
    return vt_dict_set_object(dictionary, key, PyFloat_FromDouble(value));
}

static int
vt_dict_set_string(PyObject *dictionary, const char *key, const char *value)
{
    return vt_dict_set_object(dictionary, key, PyUnicode_FromString(value == NULL ? "" : value));
}

static PyObject *
vt_capabilities_dict(const struct vt_queue_info *info)
{
    PyObject *result = PyDict_New();
    uint64_t max_inflight = info->queue_size;
    if (result == NULL) {
        return NULL;
    }
    if (max_inflight == 0) {
        max_inflight = 1;
    }
    if (max_inflight > VT_DSA_MAX_INFLIGHT) {
        max_inflight = VT_DSA_MAX_INFLIGHT;
    }
    if (vt_dict_set_bool(result, "hardware_available", info->hardware_available) != 0 ||
        vt_dict_set_string(result, "interface", "idxd-cdev") != 0 ||
        vt_dict_set_bool(result, "software_fallback_enabled", 0) != 0 ||
        vt_dict_set_bool(result, "drain_guaranteed", 1) != 0 ||
        vt_dict_set_string(result, "drain_contract", "idxd-cdev-per-open-release") != 0 ||
        vt_dict_set_string(result, "work_queue_type", "user") != 0 ||
        vt_dict_set_string(result, "work_queue", info->path) != 0 ||
        vt_dict_set_string(result, "work_queue_mode", info->mode) != 0 ||
        vt_dict_set_bool(result, "numa_local", info->numa_local) != 0 ||
        vt_dict_set_i64(result, "numa_node", info->numa_node) != 0 ||
        vt_dict_set_i64(result, "caller_numa_node", info->caller_numa_node) != 0 ||
        vt_dict_set_i64(result, "block_on_fault", info->block_on_fault) != 0 ||
        vt_dict_set_i64(result, "prs_disable", info->prs_disable) != 0 ||
        vt_dict_set_u64(result, "max_transfer_size", info->max_transfer_size) != 0 ||
        vt_dict_set_u64(result, "max_inflight", max_inflight) != 0 ||
        vt_dict_set_string(result, "unavailable_reason", info->reason) != 0) {
        Py_DECREF(result);
        return NULL;
    }
    return result;
}

static PyObject *
vt_dsa_capabilities(PyObject *self, PyObject *args, PyObject *kwargs)
{
    static char *keywords[] = {"work_queue", NULL};
    const char *requested = NULL;
    struct vt_queue_info info;
    (void)self;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|z:capabilities", keywords, &requested)) {
        return NULL;
    }
    if (vt_ensure_process_state() != 0) {
        return NULL;
    }
    Py_BEGIN_ALLOW_THREADS
    (void)pthread_rwlock_rdlock(&fork_gate);
    (void)vt_select_queue(requested, &info);
    (void)pthread_rwlock_unlock(&fork_gate);
    Py_END_ALLOW_THREADS
    return vt_capabilities_dict(&info);
}

static int
vt_buffers_match(const Py_buffer *src, const Py_buffer *dst, char *error, size_t error_size)
{
    int dimension;
    uintptr_t src_address;
    uintptr_t dst_address;
    uintptr_t difference;
    if (src->len <= 0 || dst->len <= 0) {
        (void)snprintf(error, error_size, "DSA copy requires a non-empty buffer");
        return -1;
    }
    if (src->len != dst->len || src->itemsize != dst->itemsize || src->ndim != dst->ndim) {
        (void)snprintf(error, error_size, "source and destination buffer metadata differ");
        return -1;
    }
    if ((src->format == NULL) != (dst->format == NULL) ||
        (src->format != NULL && strcmp(src->format, dst->format) != 0)) {
        (void)snprintf(error, error_size, "source and destination buffer formats differ");
        return -1;
    }
    for (dimension = 0; dimension < src->ndim; ++dimension) {
        if (src->shape == NULL || dst->shape == NULL || src->shape[dimension] != dst->shape[dimension]) {
            (void)snprintf(error, error_size, "source and destination buffer shapes differ");
            return -1;
        }
    }
    if (!PyBuffer_IsContiguous(src, 'C') || !PyBuffer_IsContiguous(dst, 'C')) {
        (void)snprintf(error, error_size, "DSA copy requires C-contiguous buffers");
        return -1;
    }
    src_address = (uintptr_t)src->buf;
    dst_address = (uintptr_t)dst->buf;
    difference = src_address > dst_address ? src_address - dst_address : dst_address - src_address;
    if (difference < (uintptr_t)src->len) {
        (void)snprintf(error, error_size, "source and destination byte ranges overlap");
        return -1;
    }
    return 0;
}

static int
vt_completion_status(const struct dsa_completion_record *record)
{
    return (int)DSA_COMP_STATUS(__atomic_load_n(&record->status, __ATOMIC_ACQUIRE));
}

static int
vt_status_is_page_fault(int status)
{
    return status == DSA_COMP_PAGE_FAULT_NOBOF ||
           status == DSA_COMP_PAGE_FAULT_IR ||
           status == DSA_COMP_BATCH_PAGE_FAULT ||
           status == DSA_COMP_CRA_XLAT ||
           status == DSA_COMP_PFAULT_RDBA ||
           status == DSA_COMP_TRANSLATION_FAIL;
}

static int
vt_wait_for_batch(
    const struct dsa_hw_desc *descriptors,
    struct dsa_completion_record *records,
    size_t submitted,
    struct vt_copy_stats *stats,
    int *all_terminal)
{
    struct timespec started;
    struct timespec now;
    struct timespec pause = {0, 50000};
    size_t index;
    size_t completed;
    int failure = 0;
    (void)clock_gettime(CLOCK_MONOTONIC, &started);
    for (;;) {
        completed = 0;
        for (index = 0; index < submitted; ++index) {
            if (vt_completion_status(&records[index]) != 0) {
                ++completed;
            }
        }
        if (completed == submitted) {
            *all_terminal = 1;
            break;
        }
        (void)clock_gettime(CLOCK_MONOTONIC, &now);
        if (vt_elapsed(&started, &now) >= VT_DSA_WAIT_SECONDS) {
            stats->wait_seconds += vt_elapsed(&started, &now);
            (void)snprintf(stats->error, sizeof(stats->error),
                           "timed out waiting for %zu DSA completion records", submitted);
            *all_terminal = 0;
            return -1;
        }
        (void)nanosleep(&pause, NULL);
    }
    (void)clock_gettime(CLOCK_MONOTONIC, &now);
    stats->wait_seconds += vt_elapsed(&started, &now);
    for (index = 0; index < submitted; ++index) {
        int status = vt_completion_status(&records[index]);
        if (status == DSA_COMP_SUCCESS) {
            /* A terminal success proves the whole requested transfer. Some DSA
             * generations reserve bytes_completed primarily for fault restart. */
            stats->hardware_bytes += (uint64_t)descriptors[index].xfer_size;
        } else {
            ++stats->partial_failures;
            if (records[index].fault_addr != 0 || vt_status_is_page_fault(status)) {
                ++stats->page_faults;
            }
            if (!failure) {
                (void)snprintf(stats->error, sizeof(stats->error),
                               "DSA completion failed with status 0x%02x after %u bytes",
                               status, records[index].bytes_completed);
            }
            failure = 1;
        }
    }
    return failure ? -1 : 0;
}

static int
vt_core_copy(
    const Py_buffer *src,
    const Py_buffer *dst,
    const char *work_queue,
    uint64_t requested_max_transfer,
    int max_inflight,
    struct vt_copy_stats *stats)
{
    struct vt_queue_info queue;
    struct dsa_hw_desc *descriptors = NULL;
    struct dsa_completion_record *completions = NULL;
    struct timespec total_started;
    struct timespec total_ended;
    struct timespec cpu_started;
    struct timespec cpu_ended;
    struct timespec submit_started;
    struct timespec submit_ended;
    struct rusage usage_started;
    struct rusage usage_ended;
    uint64_t max_transfer;
    uint64_t offset = 0;
    int fd = -1;
    int result = -1;
    int all_terminal = 1;
    int cdev_context_closed = 0;

    memset(stats, 0, sizeof(*stats));
    stats->requested_bytes = (uint64_t)src->len;
    stats->max_inflight = max_inflight;
    (void)snprintf(stats->work_queue, sizeof(stats->work_queue), "%s", work_queue);
    (void)clock_gettime(CLOCK_MONOTONIC, &total_started);
    (void)clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &cpu_started);
    (void)getrusage(RUSAGE_SELF, &usage_started);

    if (vt_inspect_queue(work_queue, &queue) != 0 || !queue.hardware_available) {
        (void)snprintf(stats->error, sizeof(stats->error), "DSA work queue unavailable: %s", queue.reason);
        goto settled;
    }
    if (!queue.numa_local) {
        (void)snprintf(stats->error, sizeof(stats->error),
                       "DSA work queue NUMA node %d is not local to caller node %d",
                       queue.numa_node, queue.caller_numa_node);
        goto settled;
    }
    max_transfer = requested_max_transfer;
    if (max_transfer == 0 || max_transfer > queue.max_transfer_size) {
        max_transfer = queue.max_transfer_size;
    }
    if (max_transfer > UINT32_MAX) {
        max_transfer = UINT32_MAX;
    }
    if (max_transfer == 0) {
        (void)snprintf(stats->error, sizeof(stats->error), "DSA maximum transfer size is zero");
        goto settled;
    }
    if (max_inflight <= 0 || max_inflight > VT_DSA_MAX_INFLIGHT) {
        (void)snprintf(stats->error, sizeof(stats->error), "invalid DSA max_inflight=%d", max_inflight);
        goto settled;
    }
    if (posix_memalign((void **)&descriptors, 64,
                       (size_t)max_inflight * sizeof(*descriptors)) != 0 ||
        posix_memalign((void **)&completions, 64,
                       (size_t)max_inflight * sizeof(*completions)) != 0) {
        (void)snprintf(stats->error, sizeof(stats->error), "failed to allocate aligned DSA descriptor storage");
        goto settled;
    }
    fd = open(queue.path, O_WRONLY | O_CLOEXEC);
    if (fd < 0) {
        (void)snprintf(stats->error, sizeof(stats->error),
                       "failed to open DSA work queue: %s", strerror(errno));
        goto settled;
    }

    while (offset < (uint64_t)src->len) {
        uint64_t remaining = (uint64_t)src->len - offset;
        size_t batch_count = (size_t)((remaining + max_transfer - 1) / max_transfer);
        size_t index;
        size_t batch_submitted = 0;
        int submission_failed = 0;
        if (batch_count > (size_t)max_inflight) {
            batch_count = (size_t)max_inflight;
        }
        memset(descriptors, 0, batch_count * sizeof(*descriptors));
        memset(completions, 0, batch_count * sizeof(*completions));
        for (index = 0; index < batch_count; ++index) {
            uint64_t transfer = remaining > max_transfer ? max_transfer : remaining;
            descriptors[index].flags =
                IDXD_OP_FLAG_BOF | IDXD_OP_FLAG_CRAV | IDXD_OP_FLAG_RCR;
            descriptors[index].opcode = DSA_OPCODE_MEMMOVE;
            descriptors[index].completion_addr = (uint64_t)(uintptr_t)&completions[index];
            descriptors[index].src_addr = (uint64_t)(uintptr_t)((const uint8_t *)src->buf + offset);
            descriptors[index].dst_addr = (uint64_t)(uintptr_t)((uint8_t *)dst->buf + offset);
            descriptors[index].xfer_size = (uint32_t)transfer;
            offset += transfer;
            remaining -= transfer;
        }
        all_terminal = 0;
        (void)clock_gettime(CLOCK_MONOTONIC, &submit_started);
        while (batch_submitted < batch_count) {
            size_t pending = batch_count - batch_submitted;
            ssize_t written = write(
                fd,
                &descriptors[batch_submitted],
                pending * sizeof(*descriptors)
            );
            if (written < 0 && errno == EINTR) {
                continue;
            }
            if (written < 0 && (errno == EAGAIN || errno == EBUSY)) {
                struct timespec pause = {0, 50000};
                struct timespec now;
                ++stats->queue_full_events;
                (void)clock_gettime(CLOCK_MONOTONIC, &now);
                if (vt_elapsed(&submit_started, &now) >= VT_DSA_WAIT_SECONDS) {
                    (void)snprintf(stats->error, sizeof(stats->error),
                                   "timed out submitting DSA descriptor batch");
                    submission_failed = 1;
                    break;
                }
                (void)nanosleep(&pause, NULL);
                continue;
            }
            if (written <= 0 || (written % (ssize_t)sizeof(*descriptors)) != 0) {
                (void)snprintf(stats->error, sizeof(stats->error),
                               "DSA descriptor submission failed: %s",
                               written < 0 ? strerror(errno) : "short non-descriptor write");
                submission_failed = 1;
                break;
            }
            batch_submitted += (size_t)written / sizeof(*descriptors);
            stats->submitted_descriptors += (uint64_t)written / sizeof(*descriptors);
        }
        (void)clock_gettime(CLOCK_MONOTONIC, &submit_ended);
        stats->submission_seconds += vt_elapsed(&submit_started, &submit_ended);
        if (batch_submitted > 0) {
            ++stats->batches;
            if (vt_wait_for_batch(descriptors, completions, batch_submitted, stats, &all_terminal) != 0) {
                goto settled;
            }
            stats->descriptors += batch_submitted;
        } else {
            all_terminal = 1;
        }
        if (submission_failed || batch_submitted != batch_count) {
            ++stats->partial_failures;
            goto settled;
        }
    }
    stats->success = 1;
    result = 0;

settled:
    if (fd >= 0) {
        (void)close(fd);
        fd = -1;
        cdev_context_closed = 1;
    }
    /* Linux idxd's per-open cdev release drains the PASID/shared queue or the
     * dedicated queue before releasing SVA state (see e6fd6d7e5f0f). Returning
     * from close while both Py_buffers live is therefore a drain proof; terminal
     * completion records independently prove the normal path. */
    stats->drained = stats->submitted_descriptors == 0 || all_terminal || cdev_context_closed;
    (void)getrusage(RUSAGE_SELF, &usage_ended);
    stats->minor_page_faults = usage_ended.ru_minflt >= usage_started.ru_minflt
        ? (uint64_t)(usage_ended.ru_minflt - usage_started.ru_minflt) : 0;
    stats->major_page_faults = usage_ended.ru_majflt >= usage_started.ru_majflt
        ? (uint64_t)(usage_ended.ru_majflt - usage_started.ru_majflt) : 0;
    (void)clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &cpu_ended);
    (void)clock_gettime(CLOCK_MONOTONIC, &total_ended);
    stats->cpu_seconds = vt_elapsed(&cpu_started, &cpu_ended);
    stats->total_seconds = vt_elapsed(&total_started, &total_ended);
    if (descriptors != NULL) {
        free(descriptors);
    }
    if (completions != NULL) {
        free(completions);
    }
    return result;
}

static PyObject *
vt_copy_stats_dict(const struct vt_copy_stats *stats)
{
    PyObject *result = PyDict_New();
    if (result == NULL) {
        return NULL;
    }
    if (vt_dict_set_bool(result, "drained", stats->drained) != 0 ||
        vt_dict_set_bool(result, "hardware_only", stats->success) != 0 ||
        vt_dict_set_u64(result, "software_bytes", 0) != 0 ||
        vt_dict_set_u64(result, "requested_bytes", stats->requested_bytes) != 0 ||
        vt_dict_set_u64(result, "hardware_bytes", stats->hardware_bytes) != 0 ||
        vt_dict_set_u64(result, "submitted_descriptors", stats->submitted_descriptors) != 0 ||
        vt_dict_set_u64(result, "descriptors", stats->descriptors) != 0 ||
        vt_dict_set_u64(result, "batches", stats->batches) != 0 ||
        vt_dict_set_i64(result, "max_inflight", stats->max_inflight) != 0 ||
        vt_dict_set_u64(result, "queue_full_events", stats->queue_full_events) != 0 ||
        vt_dict_set_u64(result, "partial_failures", stats->partial_failures) != 0 ||
        vt_dict_set_u64(result, "page_faults", stats->page_faults) != 0 ||
        vt_dict_set_u64(result, "minor_page_faults", stats->minor_page_faults) != 0 ||
        vt_dict_set_u64(result, "major_page_faults", stats->major_page_faults) != 0 ||
        vt_dict_set_double(result, "submission_seconds", stats->submission_seconds) != 0 ||
        vt_dict_set_double(result, "wait_seconds", stats->wait_seconds) != 0 ||
        vt_dict_set_double(result, "total_seconds", stats->total_seconds) != 0 ||
        vt_dict_set_double(result, "cpu_seconds", stats->cpu_seconds) != 0 ||
        vt_dict_set_string(result, "work_queue", stats->work_queue) != 0) {
        Py_DECREF(result);
        return NULL;
    }
    return result;
}

static PyObject *
vt_raise_copy_error(const struct vt_copy_stats *stats)
{
    PyObject *exception;
    PyObject *stats_dict = vt_copy_stats_dict(stats);
    if (stats_dict == NULL) {
        return NULL;
    }
    exception = PyObject_CallFunction(
        DsaCopyError,
        "s",
        stats->error[0] == '\0' ? "DSA hardware copy failed" : stats->error
    );
    if (exception == NULL) {
        Py_DECREF(stats_dict);
        return NULL;
    }
    if (PyObject_SetAttrString(exception, "stats", stats_dict) != 0) {
        Py_DECREF(stats_dict);
        Py_DECREF(exception);
        return NULL;
    }
    Py_DECREF(stats_dict);
    PyErr_SetObject(DsaCopyError, exception);
    Py_DECREF(exception);
    return NULL;
}

static PyObject *
vt_dsa_copy(PyObject *self, PyObject *args, PyObject *kwargs)
{
    static char *keywords[] = {
        "src", "dst", "work_queue", "max_transfer_size", "max_inflight",
        "require_hardware", NULL
    };
    PyObject *src_object;
    PyObject *dst_object;
    const char *work_queue;
    unsigned long long max_transfer_size;
    int max_inflight;
    int require_hardware = 1;
    Py_buffer src = {0};
    Py_buffer dst = {0};
    struct vt_copy_stats stats;
    int core_result;
    (void)self;

    if (!PyArg_ParseTupleAndKeywords(
            args, kwargs, "OOsKi|p:copy", keywords,
            &src_object, &dst_object, &work_queue, &max_transfer_size,
            &max_inflight, &require_hardware)) {
        return NULL;
    }
    if (!require_hardware) {
        PyErr_SetString(PyExc_ValueError, "DSA binding only supports require_hardware=True");
        return NULL;
    }
    if (vt_ensure_process_state() != 0) {
        return NULL;
    }
    if (PyObject_GetBuffer(src_object, &src, PyBUF_STRIDES | PyBUF_FORMAT) != 0) {
        return NULL;
    }
    if (PyObject_GetBuffer(dst_object, &dst,
                           PyBUF_STRIDES | PyBUF_FORMAT | PyBUF_WRITABLE) != 0) {
        PyBuffer_Release(&src);
        return NULL;
    }
    memset(&stats, 0, sizeof(stats));
    if (vt_buffers_match(&src, &dst, stats.error, sizeof(stats.error)) != 0) {
        stats.requested_bytes = src.len > 0 ? (uint64_t)src.len : 0;
        stats.drained = 1;
        PyBuffer_Release(&dst);
        PyBuffer_Release(&src);
        return vt_raise_copy_error(&stats);
    }

    Py_BEGIN_ALLOW_THREADS
    (void)pthread_rwlock_rdlock(&fork_gate);
    PyThread_acquire_lock(native_lock, WAIT_LOCK);
    last_drained = 0;
    last_submitted = 0;
    core_result = vt_core_copy(
        &src, &dst, work_queue, (uint64_t)max_transfer_size,
        max_inflight, &stats
    );
    last_drained = stats.drained;
    last_submitted = stats.submitted_descriptors;
    PyThread_release_lock(native_lock);
    (void)pthread_rwlock_unlock(&fork_gate);
    Py_END_ALLOW_THREADS

    PyBuffer_Release(&dst);
    PyBuffer_Release(&src);
    if (core_result != 0) {
        return vt_raise_copy_error(&stats);
    }
    return vt_copy_stats_dict(&stats);
}

static PyObject *
vt_dsa_drain(PyObject *self, PyObject *Py_UNUSED(args))
{
    int drained;
    uint64_t submitted;
    PyObject *result;
    (void)self;
    if (vt_ensure_process_state() != 0) {
        return NULL;
    }
    Py_BEGIN_ALLOW_THREADS
    (void)pthread_rwlock_rdlock(&fork_gate);
    PyThread_acquire_lock(native_lock, WAIT_LOCK);
    drained = last_drained;
    submitted = last_submitted;
    PyThread_release_lock(native_lock);
    (void)pthread_rwlock_unlock(&fork_gate);
    Py_END_ALLOW_THREADS
    result = PyDict_New();
    if (result == NULL) {
        return NULL;
    }
    if (vt_dict_set_bool(result, "drained", drained) != 0 ||
        vt_dict_set_u64(result, "submitted_descriptors", submitted) != 0) {
        Py_DECREF(result);
        return NULL;
    }
    return result;
}

static PyObject *
vt_dsa_close(PyObject *self, PyObject *Py_UNUSED(args))
{
    int drained;
    (void)self;
    if (vt_ensure_process_state() != 0) {
        return NULL;
    }
    Py_BEGIN_ALLOW_THREADS
    (void)pthread_rwlock_rdlock(&fork_gate);
    PyThread_acquire_lock(native_lock, WAIT_LOCK);
    drained = last_drained;
    PyThread_release_lock(native_lock);
    (void)pthread_rwlock_unlock(&fork_gate);
    Py_END_ALLOW_THREADS
    if (!drained) {
        PyErr_SetString(PyExc_RuntimeError, "DSA work queue could not be proven drained");
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyMethodDef vt_dsa_methods[] = {
    {"capabilities", (PyCFunction)vt_dsa_capabilities, METH_VARARGS | METH_KEYWORDS,
     PyDoc_STR("capabilities(*, work_queue=None) -> dict")},
    {"copy", (PyCFunction)vt_dsa_copy, METH_VARARGS | METH_KEYWORDS,
     PyDoc_STR("Submit a synchronous, hardware-only DSA MEMMOVE.")},
    {"drain", vt_dsa_drain, METH_NOARGS,
     PyDoc_STR("Prove that no submitted descriptor can still write.")},
    {"close", vt_dsa_close, METH_NOARGS,
     PyDoc_STR("Settle process-local DSA state.")},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef vt_dsa_module = {
    PyModuleDef_HEAD_INIT,
    "_dsa_copy",
    "Direct Linux idxd user-work-queue DSA copy binding.",
    -1,
    vt_dsa_methods,
    NULL,
    NULL,
    NULL,
    NULL
};

PyMODINIT_FUNC
PyInit__dsa_copy(void)
{
    PyObject *module;
    int atfork_status;
    if (!atfork_registered) {
        atfork_status = pthread_atfork(vt_atfork_prepare, vt_atfork_parent, vt_atfork_child);
        if (atfork_status != 0) {
            PyErr_Format(
                PyExc_ImportError,
                "DSA pthread_atfork registration failed: status=%d",
                atfork_status
            );
            return NULL;
        }
        atfork_registered = 1;
    }
    native_lock = PyThread_allocate_lock();
    if (native_lock == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "failed to allocate DSA native lock");
        return NULL;
    }
    native_pid = getpid();
    module = PyModule_Create(&vt_dsa_module);
    if (module == NULL) {
        PyThread_free_lock(native_lock);
        native_lock = NULL;
        return NULL;
    }
    DsaCopyError = PyErr_NewException("XTA._dsa_copy.DsaCopyError", PyExc_RuntimeError, NULL);
    if (DsaCopyError == NULL || PyModule_AddObject(module, "DsaCopyError", DsaCopyError) != 0) {
        Py_XDECREF(DsaCopyError);
        Py_DECREF(module);
        return NULL;
    }
    Py_INCREF(DsaCopyError); /* retained for vt_raise_copy_error */
    if (PyModule_AddStringConstant(module, "interface", "idxd-cdev") != 0) {
        Py_DECREF(module);
        return NULL;
    }
    return module;
}
