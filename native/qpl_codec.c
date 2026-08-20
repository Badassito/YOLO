#define _POSIX_C_SOURCE 200809L
#define PY_SSIZE_T_CLEAN
#include <Python.h>

#if !defined(__linux__)
#error "volume_tta._qpl_codec supports the Intel QPL hardware path on Linux only"
#endif

#include <qpl/qpl.h>

#include <dlfcn.h>
#include <fcntl.h>
#include <limits.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#if !defined(QPL_VERSION_MAJOR) || !defined(QPL_VERSION_MINOR)
#error "Intel QPL development headers must expose QPL_VERSION_MAJOR/MINOR"
#endif
#if QPL_VERSION_MAJOR < 1 || (QPL_VERSION_MAJOR == 1 && QPL_VERSION_MINOR < 9)
#error "volume_tta._qpl_codec requires Intel QPL 1.9 or newer"
#endif

/*
 * Optional Intel QPL companion for volume_tta.intel_compression.
 *
 * Deliberate safety properties:
 *   - qpl_path_hardware is the only execution path used;
 *   - every submitted compression is FIRST | LAST | GZIP_MODE;
 *   - enabled shared user IAX work queues are inventoried through a runtime-loaded
 *     libaccel-config, including operation and maximum-transfer eligibility;
 *   - a logical request is split into independent ordered gzip members when needed;
 *   - each member's source AND QPL safe destination bound fit the queue limit;
 *   - one QPL job is owned by each executor thread and is finalized on that thread;
 *   - inherited post-fork jobs are discarded without invoking parent native state;
 *   - fork children fail closed because QPL has no public dispatcher reset;
 *   - the GIL is released around QPL initialization, execution, and finalization.
 */

#define VT_QPL_BINDING_VERSION "1"
#define VT_QPL_GZIP_OVERHEAD 18U
#define VT_QPL_MAX_WORK_QUEUES 128U
#define VT_QPL_PATH_BUFFER 256U
#define VT_QPL_NAME_BUFFER 64U
#define VT_QPL_ERROR_BUFFER 256U
#define VT_IAA_COMPRESS_OPCODE 0x43U

typedef struct {
    char device[VT_QPL_NAME_BUFFER];
    char work_queue[VT_QPL_NAME_BUFFER];
    char path[VT_QPL_PATH_BUFFER];
    int device_id;
    int work_queue_id;
    int numa_id;
    unsigned int device_version;
    uint64_t max_transfer_bytes;
    uint32_t max_member_input_bytes;
    int block_on_fault;
    int operation_config_known;
} vt_qpl_work_queue;

typedef struct {
    int valid;
    unsigned int enabled_devices;
    unsigned int eligible_work_queues;
    unsigned int truncated_work_queues;
    unsigned int generation_mask;
    uint64_t min_transfer_bytes;
    uint64_t max_transfer_bytes;
    uint32_t min_member_input_bytes;
    uint32_t max_member_input_bytes;
    char error[VT_QPL_ERROR_BUFFER];
    vt_qpl_work_queue work_queues[VT_QPL_MAX_WORK_QUEUES];
} vt_qpl_inventory;

typedef struct {
    uint64_t logical_requests;
    uint64_t hardware_requests;
    uint64_t physical_members;
    uint64_t requested_input_bytes;
    uint64_t input_bytes;
    uint64_t output_bytes;
    uint64_t elapsed_ns;
    uint64_t failures;
    uint64_t queue_busy_events;
    uint64_t page_fault_errors;
    uint64_t sessions_created;
    uint64_t sessions_closed;
    uint64_t active_sessions;
    uint64_t preflight_calls;
    int32_t last_status;
    int32_t last_numa_id;
} vt_qpl_stats;

typedef struct {
    pid_t pid;
    void *job_buffer;
    qpl_job *job;
    uint32_t job_size;
    int initialized;
} vt_qpl_thread_state;

static pid_t g_process_id = 0;
static vt_qpl_inventory g_inventory;
static vt_qpl_stats g_stats;
static pthread_key_t g_thread_state_key;
static pthread_once_t g_thread_state_key_once = PTHREAD_ONCE_INIT;
static int g_thread_state_key_error = 0;
static int g_thread_state_key_initialized = 0;
/*
 * QPL owns a process-global hardware dispatcher.  Keep fork() from cloning it
 * while another thread is initializing or using it.  Read locks preserve the
 * normal multi-threaded submission model; the atfork prepare hook alone takes
 * the exclusive lock.
 */
static pthread_rwlock_t g_qpl_fork_lock = PTHREAD_RWLOCK_INITIALIZER;
static int g_qpl_atfork_registered = 0;
static int g_qpl_fork_write_locked = 0;
static int g_qpl_inherited_after_fork = 0;

#define VT_STAT_ADD(field, value) \
    ((void)__atomic_fetch_add(&g_stats.field, (uint64_t)(value), __ATOMIC_RELAXED))
#define VT_STAT_SUB(field, value) \
    ((void)__atomic_fetch_sub(&g_stats.field, (uint64_t)(value), __ATOMIC_RELAXED))
#define VT_STAT_LOAD(field) __atomic_load_n(&g_stats.field, __ATOMIC_RELAXED)
#define VT_STAT_EXCHANGE(field, value) \
    __atomic_exchange_n(&g_stats.field, (uint64_t)(value), __ATOMIC_RELAXED)

static uint64_t vt_monotonic_ns(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) {
        return 0U;
    }
    return ((uint64_t)ts.tv_sec * UINT64_C(1000000000)) + (uint64_t)ts.tv_nsec;
}

static void vt_qpl_atfork_prepare(void) {
    g_qpl_fork_write_locked = pthread_rwlock_wrlock(&g_qpl_fork_lock) == 0;
}

static void vt_qpl_atfork_parent(void) {
    if (g_qpl_fork_write_locked) {
        (void)pthread_rwlock_unlock(&g_qpl_fork_lock);
        g_qpl_fork_write_locked = 0;
    }
}

static void vt_qpl_atfork_child(void) {
    /*
     * QPL exposes no supported reset for its inherited process-global dispatcher,
     * work-queue mappings, or PASID-related state.  The child must fail closed and
     * use spawn/exec before IAA can be selected again.
     */
    g_process_id = 0;
    g_qpl_inherited_after_fork = 1;
    if (g_qpl_fork_write_locked) {
        (void)pthread_rwlock_unlock(&g_qpl_fork_lock);
        g_qpl_fork_write_locked = 0;
    }
}

static qpl_status vt_qpl_get_hardware_job_size(uint32_t *job_size) {
    qpl_status status;
    if (pthread_rwlock_rdlock(&g_qpl_fork_lock) != 0) {
        return QPL_STS_LIBRARY_INTERNAL_ERR;
    }
    status = qpl_get_job_size(qpl_path_hardware, job_size);
    (void)pthread_rwlock_unlock(&g_qpl_fork_lock);
    return status;
}

static qpl_status vt_qpl_init_hardware_job(qpl_job *job) {
    qpl_status status;
    if (pthread_rwlock_rdlock(&g_qpl_fork_lock) != 0) {
        return QPL_STS_LIBRARY_INTERNAL_ERR;
    }
    status = qpl_init_job(qpl_path_hardware, job);
    (void)pthread_rwlock_unlock(&g_qpl_fork_lock);
    return status;
}

static qpl_status vt_qpl_execute_hardware_job(qpl_job *job) {
    qpl_status status;
    if (pthread_rwlock_rdlock(&g_qpl_fork_lock) != 0) {
        return QPL_STS_LIBRARY_INTERNAL_ERR;
    }
    status = qpl_execute_job(job);
    (void)pthread_rwlock_unlock(&g_qpl_fork_lock);
    return status;
}

static qpl_status vt_qpl_fini_hardware_job(qpl_job *job) {
    qpl_status status;
    if (pthread_rwlock_rdlock(&g_qpl_fork_lock) != 0) {
        return QPL_STS_LIBRARY_INTERNAL_ERR;
    }
    status = qpl_fini_job(job);
    (void)pthread_rwlock_unlock(&g_qpl_fork_lock);
    return status;
}

static const char *vt_qpl_status_name(qpl_status status) {
    switch (status) {
        case QPL_STS_OK: return "QPL_STS_OK";
        case QPL_STS_BEING_PROCESSED: return "QPL_STS_BEING_PROCESSED";
        case QPL_STS_MORE_OUTPUT_NEEDED: return "QPL_STS_MORE_OUTPUT_NEEDED";
        case QPL_STS_MORE_INPUT_NEEDED: return "QPL_STS_MORE_INPUT_NEEDED";
        case QPL_STS_JOB_NOT_CONTINUABLE_ERR: return "QPL_STS_JOB_NOT_CONTINUABLE_ERR";
        case QPL_STS_QUEUES_ARE_BUSY_ERR: return "QPL_STS_QUEUES_ARE_BUSY_ERR";
        case QPL_STS_LIBRARY_INTERNAL_ERR: return "QPL_STS_LIBRARY_INTERNAL_ERR";
        case QPL_STS_JOB_NOT_SUBMITTED: return "QPL_STS_JOB_NOT_SUBMITTED";
        case QPL_STS_NOT_SUPPORTED_BY_WQ: return "QPL_STS_NOT_SUPPORTED_BY_WQ";
        case QPL_STS_NULL_PTR_ERR: return "QPL_STS_NULL_PTR_ERR";
        case QPL_STS_OPERATION_ERR: return "QPL_STS_OPERATION_ERR";
        case QPL_STS_NOT_SUPPORTED_MODE_ERR: return "QPL_STS_NOT_SUPPORTED_MODE_ERR";
        case QPL_STS_BAD_JOB_STRUCT_ERR: return "QPL_STS_BAD_JOB_STRUCT_ERR";
        case QPL_STS_PATH_ERR: return "QPL_STS_PATH_ERR";
        case QPL_STS_INVALID_PARAM_ERR: return "QPL_STS_INVALID_PARAM_ERR";
        case QPL_STS_FLAG_CONFLICT_ERR: return "QPL_STS_FLAG_CONFLICT_ERR";
        case QPL_STS_SIZE_ERR: return "QPL_STS_SIZE_ERR";
        case QPL_STS_BUFFER_OVERLAP_ERR: return "QPL_STS_BUFFER_OVERLAP_ERR";
        case QPL_STS_UNSUPPORTED_COMPRESSION_LEVEL: return "QPL_STS_UNSUPPORTED_COMPRESSION_LEVEL";
        case QPL_STS_TIMEOUT_ERR: return "QPL_STS_TIMEOUT_ERR";
        case QPL_STS_INTL_VERIFY_ERR: return "QPL_STS_INTL_VERIFY_ERR";
        case QPL_STS_INTL_PAGE_FAULT: return "QPL_STS_INTL_PAGE_FAULT";
        case QPL_STS_TRANSFER_SIZE_INVALID: return "QPL_STS_TRANSFER_SIZE_INVALID";
        case QPL_STS_INTL_TRANSLATION_PAGE_FAULT: return "QPL_STS_INTL_TRANSLATION_PAGE_FAULT";
        case QPL_STS_INTL_DRAIN_PAGE_FAULT: return "QPL_STS_INTL_DRAIN_PAGE_FAULT";
        case QPL_STS_INTL_PAGE_REQUEST_TIMEOUT: return "QPL_STS_INTL_PAGE_REQUEST_TIMEOUT";
        case QPL_STS_INTL_W_PAGE_FAULT: return "QPL_STS_INTL_W_PAGE_FAULT";
        case QPL_STS_INTL_W_TRANSLATION_PF: return "QPL_STS_INTL_W_TRANSLATION_PF";
        case QPL_STS_INIT_HW_NOT_SUPPORTED: return "QPL_STS_INIT_HW_NOT_SUPPORTED";
        case QPL_STS_INIT_LIBACCEL_NOT_FOUND: return "QPL_STS_INIT_LIBACCEL_NOT_FOUND";
        case QPL_STS_INIT_LIBACCEL_ERROR: return "QPL_STS_INIT_LIBACCEL_ERROR";
        case QPL_STS_INIT_WORK_QUEUES_NOT_AVAILABLE: return "QPL_STS_INIT_WORK_QUEUES_NOT_AVAILABLE";
        default: return "QPL_STS_UNKNOWN";
    }
}

static int vt_qpl_runtime_version_is_supported(const char *version) {
    unsigned int major = 0U;
    unsigned int minor = 0U;
    if (version == NULL || sscanf(version, "%u.%u", &major, &minor) != 2) {
        return 0;
    }
    return major > 1U || (major == 1U && minor >= 9U);
}

static int vt_qpl_status_is_page_fault(qpl_status status) {
    return status == QPL_STS_INTL_PAGE_FAULT ||
           status == QPL_STS_INTL_TRANSLATION_PAGE_FAULT ||
           status == QPL_STS_INTL_DRAIN_PAGE_FAULT ||
           status == QPL_STS_INTL_PAGE_REQUEST_TIMEOUT ||
           status == QPL_STS_INTL_W_PAGE_FAULT ||
           status == QPL_STS_INTL_W_TRANSLATION_PF;
}

static void vt_record_terminal_status(qpl_status status, int numa_id) {
    __atomic_store_n(&g_stats.last_status, (int32_t)status, __ATOMIC_RELAXED);
    __atomic_store_n(&g_stats.last_numa_id, (int32_t)numa_id, __ATOMIC_RELAXED);
    if (status == QPL_STS_QUEUES_ARE_BUSY_ERR) {
        VT_STAT_ADD(queue_busy_events, 1U);
    }
    if (vt_qpl_status_is_page_fault(status)) {
        VT_STAT_ADD(page_fault_errors, 1U);
    }
}

static PyObject *vt_raise_qpl_status(
    const char *operation,
    qpl_status status,
    int numa_id,
    uint32_t chunk_bytes,
    uint32_t output_capacity
) {
    PyErr_Format(
        PyExc_RuntimeError,
        "IAA/QPL hardware %s failed: status=%d (%s), path=qpl_path_hardware, "
        "numa_id=%d, chunk_bytes=%u, output_capacity=%u, eligible_wqs=%u, "
        "min_transfer_bytes=%llu",
        operation,
        (int)status,
        vt_qpl_status_name(status),
        numa_id,
        (unsigned int)chunk_bytes,
        (unsigned int)output_capacity,
        g_inventory.eligible_work_queues,
        (unsigned long long)g_inventory.min_transfer_bytes
    );
    return NULL;
}

static void vt_reset_process_caches_if_needed(void) {
    pid_t pid = getpid();
    if (g_process_id == pid) {
        return;
    }
    /* This runs with the GIL. A fork child must not finalize inherited QPL jobs. */
    g_process_id = pid;
    memset(&g_inventory, 0, sizeof(g_inventory));
    memset(&g_stats, 0, sizeof(g_stats));
    g_stats.last_numa_id = QPL_DEVICE_NUMA_ID_SOCKET;
}

/* Minimal libaccel-config ABI used only through dlsym; no build-time header/link needed. */
typedef struct vt_accfg_ctx vt_accfg_ctx;
typedef struct vt_accfg_device vt_accfg_device;
typedef struct vt_accfg_wq vt_accfg_wq;
typedef struct {
    uint32_t bits[8];
} vt_accfg_op_config;

typedef int (*vt_accfg_new_fn)(vt_accfg_ctx **);
typedef vt_accfg_ctx *(*vt_accfg_unref_fn)(vt_accfg_ctx *);
typedef vt_accfg_device *(*vt_accfg_device_first_fn)(vt_accfg_ctx *);
typedef vt_accfg_device *(*vt_accfg_device_next_fn)(vt_accfg_device *);
typedef int (*vt_accfg_device_int_fn)(vt_accfg_device *);
typedef unsigned int (*vt_accfg_device_uint_fn)(vt_accfg_device *);
typedef const char *(*vt_accfg_device_name_fn)(vt_accfg_device *);
typedef vt_accfg_wq *(*vt_accfg_wq_first_fn)(vt_accfg_device *);
typedef vt_accfg_wq *(*vt_accfg_wq_next_fn)(vt_accfg_wq *);
typedef int (*vt_accfg_wq_int_fn)(vt_accfg_wq *);
typedef uint64_t (*vt_accfg_wq_u64_fn)(vt_accfg_wq *);
typedef const char *(*vt_accfg_wq_name_fn)(vt_accfg_wq *);
typedef int (*vt_accfg_wq_path_fn)(vt_accfg_wq *, char *, size_t);
typedef int (*vt_accfg_wq_op_config_fn)(vt_accfg_wq *, vt_accfg_op_config *);

typedef struct {
    vt_accfg_new_fn new_context;
    vt_accfg_unref_fn unref_context;
    vt_accfg_device_first_fn device_first;
    vt_accfg_device_next_fn device_next;
    vt_accfg_device_int_fn device_type;
    vt_accfg_device_int_fn device_state;
    vt_accfg_device_int_fn device_id;
    vt_accfg_device_int_fn device_numa;
    vt_accfg_device_uint_fn device_version;
    vt_accfg_device_name_fn device_name;
    vt_accfg_wq_first_fn wq_first;
    vt_accfg_wq_next_fn wq_next;
    vt_accfg_wq_int_fn wq_state;
    vt_accfg_wq_int_fn wq_mode;
    vt_accfg_wq_int_fn wq_type;
    vt_accfg_wq_int_fn wq_id;
    vt_accfg_wq_int_fn wq_block_on_fault;
    vt_accfg_wq_u64_fn wq_max_transfer;
    vt_accfg_wq_name_fn wq_name;
    vt_accfg_wq_path_fn wq_path;
    vt_accfg_wq_op_config_fn wq_op_config;
} vt_accfg_api;

static void *vt_dlsym_required(
    void *handle,
    const char *name,
    char *error,
    size_t error_size
) {
    void *symbol;
    dlerror();
    symbol = dlsym(handle, name);
    if (symbol == NULL) {
        const char *detail = dlerror();
        snprintf(
            error,
            error_size,
            "libaccel-config is missing required %s (%s)",
            name,
            detail != NULL ? detail : "unknown dlsym error"
        );
    }
    return symbol;
}

#define VT_LOAD_ACCFG_REQUIRED(api, field, type, handle, symbol_name, error) do { \
    void *vt_symbol = vt_dlsym_required((handle), (symbol_name), (error), VT_QPL_ERROR_BUFFER); \
    if (vt_symbol == NULL) goto load_failed; \
    (api).field = (type)vt_symbol; \
} while (0)

static uint32_t vt_max_member_input_for_transfer(uint64_t transfer_bytes) {
    uint64_t high64;
    uint32_t low = 0U;
    uint32_t high;
    if (transfer_bytes <= VT_QPL_GZIP_OVERHEAD) {
        return 0U;
    }
    high64 = transfer_bytes;
    if (high64 > (uint64_t)UINT32_MAX - 35U) {
        high64 = (uint64_t)UINT32_MAX - 35U;
    }
    high = (uint32_t)high64;
    while (low < high) {
        uint32_t mid = low + (uint32_t)(((uint64_t)high - low + 1U) / 2U);
        uint32_t bound = qpl_get_safe_deflate_compression_buffer_size(mid);
        if (bound != 0U && (uint64_t)bound + VT_QPL_GZIP_OVERHEAD <= transfer_bytes) {
            low = mid;
        } else {
            high = mid - 1U;
        }
    }
    return low;
}

static int vt_scan_work_queues(vt_qpl_inventory *inventory) {
    void *handle = NULL;
    vt_accfg_api api;
    vt_accfg_ctx *context = NULL;
    vt_accfg_device *device;
    int context_status;

    memset(inventory, 0, sizeof(*inventory));
    memset(&api, 0, sizeof(api));
    inventory->min_transfer_bytes = UINT64_MAX;
    inventory->min_member_input_bytes = UINT32_MAX;

    handle = dlopen("libaccel-config.so.1", RTLD_NOW | RTLD_LOCAL);
    if (handle == NULL) {
        handle = dlopen("libaccel-config.so", RTLD_NOW | RTLD_LOCAL);
    }
    if (handle == NULL) {
        const char *detail = dlerror();
        snprintf(
            inventory->error,
            sizeof(inventory->error),
            "libaccel-config could not be loaded (%s)",
            detail != NULL ? detail : "not installed"
        );
        return -1;
    }

    VT_LOAD_ACCFG_REQUIRED(api, new_context, vt_accfg_new_fn, handle, "accfg_new", inventory->error);
    VT_LOAD_ACCFG_REQUIRED(api, unref_context, vt_accfg_unref_fn, handle, "accfg_unref", inventory->error);
    VT_LOAD_ACCFG_REQUIRED(api, device_first, vt_accfg_device_first_fn, handle, "accfg_device_get_first", inventory->error);
    VT_LOAD_ACCFG_REQUIRED(api, device_next, vt_accfg_device_next_fn, handle, "accfg_device_get_next", inventory->error);
    VT_LOAD_ACCFG_REQUIRED(api, device_type, vt_accfg_device_int_fn, handle, "accfg_device_get_type", inventory->error);
    VT_LOAD_ACCFG_REQUIRED(api, device_state, vt_accfg_device_int_fn, handle, "accfg_device_get_state", inventory->error);
    VT_LOAD_ACCFG_REQUIRED(api, device_id, vt_accfg_device_int_fn, handle, "accfg_device_get_id", inventory->error);
    VT_LOAD_ACCFG_REQUIRED(api, device_numa, vt_accfg_device_int_fn, handle, "accfg_device_get_numa_node", inventory->error);
    VT_LOAD_ACCFG_REQUIRED(api, device_version, vt_accfg_device_uint_fn, handle, "accfg_device_get_version", inventory->error);
    VT_LOAD_ACCFG_REQUIRED(api, device_name, vt_accfg_device_name_fn, handle, "accfg_device_get_devname", inventory->error);
    VT_LOAD_ACCFG_REQUIRED(api, wq_first, vt_accfg_wq_first_fn, handle, "accfg_wq_get_first", inventory->error);
    VT_LOAD_ACCFG_REQUIRED(api, wq_next, vt_accfg_wq_next_fn, handle, "accfg_wq_get_next", inventory->error);
    VT_LOAD_ACCFG_REQUIRED(api, wq_state, vt_accfg_wq_int_fn, handle, "accfg_wq_get_state", inventory->error);
    VT_LOAD_ACCFG_REQUIRED(api, wq_mode, vt_accfg_wq_int_fn, handle, "accfg_wq_get_mode", inventory->error);
    VT_LOAD_ACCFG_REQUIRED(api, wq_type, vt_accfg_wq_int_fn, handle, "accfg_wq_get_type", inventory->error);
    VT_LOAD_ACCFG_REQUIRED(api, wq_id, vt_accfg_wq_int_fn, handle, "accfg_wq_get_id", inventory->error);
    VT_LOAD_ACCFG_REQUIRED(api, wq_block_on_fault, vt_accfg_wq_int_fn, handle, "accfg_wq_get_block_on_fault", inventory->error);
    VT_LOAD_ACCFG_REQUIRED(api, wq_max_transfer, vt_accfg_wq_u64_fn, handle, "accfg_wq_get_max_transfer_size", inventory->error);
    VT_LOAD_ACCFG_REQUIRED(api, wq_name, vt_accfg_wq_name_fn, handle, "accfg_wq_get_devname", inventory->error);
    VT_LOAD_ACCFG_REQUIRED(api, wq_path, vt_accfg_wq_path_fn, handle, "accfg_wq_get_user_dev_path", inventory->error);

    /* QPL treats a missing op-config API as "all operations enabled". */
    api.wq_op_config = (vt_accfg_wq_op_config_fn)dlsym(handle, "accfg_wq_get_op_config");

    context_status = api.new_context(&context);
    if (context_status != 0 || context == NULL) {
        snprintf(
            inventory->error,
            sizeof(inventory->error),
            "accfg_new failed with status=%d",
            context_status
        );
        goto scan_failed;
    }

    for (device = api.device_first(context); device != NULL; device = api.device_next(device)) {
        vt_accfg_wq *wq;
        int device_type = api.device_type(device);
        int device_state = api.device_state(device);
        int numa_id;
        int device_id;
        unsigned int version;
        const char *device_name;

        /* libaccel-config: IAX=1, ENABLED=1. */
        if (device_type != 1 || device_state != 1) {
            continue;
        }
        inventory->enabled_devices += 1U;
        numa_id = api.device_numa(device);
        device_id = api.device_id(device);
        version = api.device_version(device);
        device_name = api.device_name(device);
        if ((version >> 8U) < 32U) {
            inventory->generation_mask |= 1U << (version >> 8U);
        }

        for (wq = api.wq_first(device); wq != NULL; wq = api.wq_next(wq)) {
            uint64_t max_transfer;
            uint32_t max_member_input;
            int op_config_known = 0;
            int compression_enabled = 1;
            int path_status;
            char path[VT_QPL_PATH_BUFFER];
            const char *wq_name;
            vt_qpl_work_queue *entry;

            /* libaccel-config: WQ ENABLED=1, SHARED=0, USER=2. */
            if (api.wq_state(wq) != 1 || api.wq_mode(wq) != 0 || api.wq_type(wq) != 2) {
                continue;
            }
            if (api.wq_op_config != NULL) {
                vt_accfg_op_config op_config;
                memset(&op_config, 0, sizeof(op_config));
                if (api.wq_op_config(wq, &op_config) == 0) {
                    unsigned int group = VT_IAA_COMPRESS_OPCODE / 32U;
                    unsigned int bit = VT_IAA_COMPRESS_OPCODE % 32U;
                    op_config_known = 1;
                    compression_enabled = (op_config.bits[group] & (1U << bit)) != 0U;
                }
            }
            if (!compression_enabled) {
                continue;
            }

            max_transfer = api.wq_max_transfer(wq);
            max_member_input = vt_max_member_input_for_transfer(max_transfer);
            if (max_transfer == 0U || max_member_input == 0U) {
                continue;
            }
            memset(path, 0, sizeof(path));
            path_status = api.wq_path(wq, path, sizeof(path) - 1U);
            if (path_status < 0 || path[0] == '\0') {
                continue;
            }
            {
                int work_queue_fd = open(path, O_RDWR | O_CLOEXEC);
                if (work_queue_fd < 0) {
                    continue;
                }
                (void)close(work_queue_fd);
            }

            if (inventory->eligible_work_queues >= VT_QPL_MAX_WORK_QUEUES) {
                inventory->truncated_work_queues += 1U;
                continue;
            }
            entry = &inventory->work_queues[inventory->eligible_work_queues++];
            memset(entry, 0, sizeof(*entry));
            wq_name = api.wq_name(wq);
            snprintf(entry->device, sizeof(entry->device), "%s", device_name != NULL ? device_name : "iax?");
            snprintf(entry->work_queue, sizeof(entry->work_queue), "%s", wq_name != NULL ? wq_name : "wq?");
            snprintf(entry->path, sizeof(entry->path), "%s", path);
            entry->device_id = device_id;
            entry->work_queue_id = api.wq_id(wq);
            entry->numa_id = numa_id;
            entry->device_version = version;
            entry->max_transfer_bytes = max_transfer;
            entry->max_member_input_bytes = max_member_input;
            entry->block_on_fault = api.wq_block_on_fault(wq) != 0;
            entry->operation_config_known = op_config_known;

            if (max_transfer < inventory->min_transfer_bytes) {
                inventory->min_transfer_bytes = max_transfer;
            }
            if (max_transfer > inventory->max_transfer_bytes) {
                inventory->max_transfer_bytes = max_transfer;
            }
            if (max_member_input < inventory->min_member_input_bytes) {
                inventory->min_member_input_bytes = max_member_input;
            }
            if (max_member_input > inventory->max_member_input_bytes) {
                inventory->max_member_input_bytes = max_member_input;
            }
        }
    }

    api.unref_context(context);
    context = NULL;
    dlclose(handle);
    handle = NULL;
    inventory->valid = 1;
    if (inventory->enabled_devices == 0U) {
        snprintf(inventory->error, sizeof(inventory->error), "no enabled Intel IAA (iax) devices were found");
        return -1;
    }
    if (inventory->eligible_work_queues == 0U) {
        snprintf(
            inventory->error,
            sizeof(inventory->error),
            "no accessible enabled shared user IAA work queue supports compression"
        );
        return -1;
    }
    inventory->error[0] = '\0';
    return 0;

load_failed:
scan_failed:
    if (context != NULL && api.unref_context != NULL) {
        api.unref_context(context);
    }
    if (handle != NULL) {
        dlclose(handle);
    }
    inventory->valid = 1;
    if (inventory->error[0] == '\0') {
        snprintf(inventory->error, sizeof(inventory->error), "IAA work-queue inventory failed");
    }
    return -1;
}

static int vt_ensure_inventory(void) {
    vt_reset_process_caches_if_needed();
    if (!g_inventory.valid) {
        (void)vt_scan_work_queues(&g_inventory);
    }
    return g_inventory.eligible_work_queues > 0U ? 0 : -1;
}

static uint32_t vt_chunk_limit_for_numa(int numa_id, unsigned int *count_out, uint64_t *transfer_out) {
    uint32_t limit = UINT32_MAX;
    uint64_t transfer = UINT64_MAX;
    unsigned int count = 0U;
    unsigned int index;
    if (vt_ensure_inventory() != 0) {
        if (count_out != NULL) *count_out = 0U;
        if (transfer_out != NULL) *transfer_out = 0U;
        return 0U;
    }
    for (index = 0U; index < g_inventory.eligible_work_queues; ++index) {
        const vt_qpl_work_queue *entry = &g_inventory.work_queues[index];
        if (numa_id >= 0 && entry->numa_id != numa_id) {
            continue;
        }
        count += 1U;
        if (entry->max_member_input_bytes < limit) {
            limit = entry->max_member_input_bytes;
        }
        if (entry->max_transfer_bytes < transfer) {
            transfer = entry->max_transfer_bytes;
        }
    }
    if (count_out != NULL) *count_out = count;
    if (transfer_out != NULL) *transfer_out = count > 0U ? transfer : 0U;
    return count > 0U ? limit : 0U;
}

static qpl_status vt_probe_hardware_context(void) {
    uint32_t job_size = 0U;
    void *buffer = NULL;
    qpl_job *job = NULL;
    qpl_status status;
    qpl_status fini_status = QPL_STS_OK;
    int initialized = 0;

    Py_BEGIN_ALLOW_THREADS
    status = vt_qpl_get_hardware_job_size(&job_size);
    Py_END_ALLOW_THREADS
    if (status != QPL_STS_OK || job_size == 0U) {
        return status != QPL_STS_OK ? status : QPL_STS_BAD_JOB_STRUCT_ERR;
    }
    buffer = malloc(job_size);
    if (buffer == NULL) {
        return QPL_STS_NO_MEM_ERR;
    }
    job = (qpl_job *)buffer;
    Py_BEGIN_ALLOW_THREADS
    status = vt_qpl_init_hardware_job(job);
    Py_END_ALLOW_THREADS
    if (status == QPL_STS_OK) {
        initialized = 1;
        if (job->data_ptr.path != qpl_path_hardware) {
            status = QPL_STS_PATH_ERR;
        }
    }
    if (initialized) {
        Py_BEGIN_ALLOW_THREADS
        fini_status = vt_qpl_fini_hardware_job(job);
        Py_END_ALLOW_THREADS
        if (status == QPL_STS_OK && fini_status != QPL_STS_OK) {
            status = fini_status;
        }
    }
    free(buffer);
    return status;
}

static qpl_status vt_finalize_thread_state(vt_qpl_thread_state *state, int finalize_native) {
    qpl_status status = QPL_STS_OK;
    if (state == NULL) {
        return status;
    }
    if (finalize_native && state->initialized && state->job != NULL) {
        status = vt_qpl_fini_hardware_job(state->job);
    }
    free(state->job_buffer);
    state->job_buffer = NULL;
    state->job = NULL;
    state->initialized = 0;
    free(state);
    return status;
}

static void vt_thread_state_destructor(void *opaque) {
    vt_qpl_thread_state *state = (vt_qpl_thread_state *)opaque;
    int finalize_native = state != NULL && state->pid == getpid();
    (void)vt_finalize_thread_state(state, finalize_native);
    if (finalize_native) {
        VT_STAT_ADD(sessions_closed, 1U);
        VT_STAT_SUB(active_sessions, 1U);
    }
}

static void vt_make_thread_state_key(void) {
    g_thread_state_key_error = pthread_key_create(&g_thread_state_key, vt_thread_state_destructor);
    if (g_thread_state_key_error == 0) {
        g_thread_state_key_initialized = 1;
    }
}

static vt_qpl_thread_state *vt_get_thread_state(qpl_status *status_out) {
    vt_qpl_thread_state *state;
    uint32_t job_size = 0U;
    qpl_status status;
    int key_status;

    *status_out = QPL_STS_OK;
    key_status = pthread_once(&g_thread_state_key_once, vt_make_thread_state_key);
    if (key_status != 0 || g_thread_state_key_error != 0) {
        PyErr_Format(
            PyExc_RuntimeError,
            "IAA/QPL pthread TLS initialization failed: status=%d",
            key_status != 0 ? key_status : g_thread_state_key_error
        );
        return NULL;
    }

    state = (vt_qpl_thread_state *)pthread_getspecific(g_thread_state_key);
    if (state != NULL && state->pid != getpid()) {
        /* Never run qpl_fini_job on a context inherited from the fork parent. */
        key_status = pthread_setspecific(g_thread_state_key, NULL);
        if (key_status != 0) {
            PyErr_Format(
                PyExc_RuntimeError,
                "IAA/QPL failed to discard inherited pthread TLS: status=%d",
                key_status
            );
            return NULL;
        }
        (void)vt_finalize_thread_state(state, 0);
        state = NULL;
    }
    if (state != NULL) {
        return state;
    }

    state = (vt_qpl_thread_state *)calloc(1U, sizeof(*state));
    if (state == NULL) {
        PyErr_NoMemory();
        return NULL;
    }
    Py_BEGIN_ALLOW_THREADS
    status = vt_qpl_get_hardware_job_size(&job_size);
    Py_END_ALLOW_THREADS
    if (status != QPL_STS_OK || job_size == 0U) {
        free(state);
        *status_out = status != QPL_STS_OK ? status : QPL_STS_BAD_JOB_STRUCT_ERR;
        return NULL;
    }
    state->job_buffer = malloc(job_size);
    if (state->job_buffer == NULL) {
        free(state);
        PyErr_NoMemory();
        return NULL;
    }
    state->pid = getpid();
    state->job_size = job_size;
    state->job = (qpl_job *)state->job_buffer;

    Py_BEGIN_ALLOW_THREADS
    status = vt_qpl_init_hardware_job(state->job);
    Py_END_ALLOW_THREADS
    if (status != QPL_STS_OK) {
        (void)vt_finalize_thread_state(state, 0);
        *status_out = status;
        return NULL;
    }
    state->initialized = 1;
    if (state->job->data_ptr.path != qpl_path_hardware) {
        Py_BEGIN_ALLOW_THREADS
        (void)vt_finalize_thread_state(state, 1);
        Py_END_ALLOW_THREADS
        *status_out = QPL_STS_PATH_ERR;
        return NULL;
    }
    key_status = pthread_setspecific(g_thread_state_key, state);
    if (key_status != 0) {
        Py_BEGIN_ALLOW_THREADS
        (void)vt_finalize_thread_state(state, 1);
        Py_END_ALLOW_THREADS
        PyErr_Format(PyExc_RuntimeError, "IAA/QPL pthread_setspecific failed: status=%d", key_status);
        return NULL;
    }
    VT_STAT_ADD(sessions_created, 1U);
    VT_STAT_ADD(active_sessions, 1U);
    return state;
}

static qpl_status vt_retire_current_thread_state(void) {
    vt_qpl_thread_state *state;
    qpl_status status = QPL_STS_OK;
    int key_status;
    if (!g_thread_state_key_initialized || g_thread_state_key_error != 0) {
        return status;
    }
    state = (vt_qpl_thread_state *)pthread_getspecific(g_thread_state_key);
    if (state == NULL) {
        return status;
    }
    key_status = pthread_setspecific(g_thread_state_key, NULL);
    if (key_status != 0) {
        return QPL_STS_LIBRARY_INTERNAL_ERR;
    }
    if (state->pid == getpid()) {
        Py_BEGIN_ALLOW_THREADS
        status = vt_finalize_thread_state(state, 1);
        Py_END_ALLOW_THREADS
        VT_STAT_ADD(sessions_closed, 1U);
        VT_STAT_SUB(active_sessions, 1U);
    } else {
        (void)vt_finalize_thread_state(state, 0);
    }
    return status;
}

static int vt_parse_numa_id(PyObject *object, int *numa_id_out) {
    long value;
    if (object == NULL || object == Py_None) {
        *numa_id_out = QPL_DEVICE_NUMA_ID_SOCKET;
        return 0;
    }
    value = PyLong_AsLong(object);
    if (value == -1 && PyErr_Occurred()) {
        return -1;
    }
    if (value < 0 || value > INT_MAX) {
        PyErr_SetString(PyExc_ValueError, "numa_id must be None or a non-negative integer");
        return -1;
    }
    *numa_id_out = (int)value;
    return 0;
}

static int vt_validate_hardware_request(int level, int require_hardware) {
    const char *library_version;
    if (!require_hardware) {
        PyErr_SetString(
            PyExc_ValueError,
            "IAA/QPL companion is hardware-only; require_hardware must remain true"
        );
        return -1;
    }
    if (g_qpl_inherited_after_fork) {
        PyErr_SetString(
            PyExc_RuntimeError,
            "IAA/QPL is unavailable in a fork child that inherited QPL state; "
            "start accelerator workers with spawn or exec"
        );
        return -1;
    }
    library_version = qpl_get_library_version();
    if (!vt_qpl_runtime_version_is_supported(library_version)) {
        PyErr_Format(
            PyExc_RuntimeError,
            "IAA/QPL runtime %s is unsupported; version 1.9.0 or newer is required",
            library_version != NULL ? library_version : "unknown"
        );
        return -1;
    }
    /* QPL 1.9 explicitly rejects qpl_high_level (level 3) on hardware path. */
    if (level != (int)qpl_default_level) {
        PyErr_Format(
            PyExc_ValueError,
            "IAA/QPL hardware compression level %d is unsupported; supported_levels=(1,)",
            level
        );
        return -1;
    }
    return 0;
}

static int vt_dict_set_owned(PyObject *dictionary, const char *key, PyObject *value) {
    int status;
    if (value == NULL) {
        return -1;
    }
    status = PyDict_SetItemString(dictionary, key, value);
    Py_DECREF(value);
    return status;
}

static PyObject *vt_qpl_capabilities(PyObject *self, PyObject *ignored) {
    PyObject *result = NULL;
    PyObject *queues = NULL;
    PyObject *levels = NULL;
    qpl_status probe_status = QPL_STS_INIT_WORK_QUEUES_NOT_AVAILABLE;
    int inventory_ok;
    int runtime_version_ok;
    int hardware_available;
    unsigned int index;
    char generation[64];
    char hardware_error[VT_QPL_ERROR_BUFFER];
    const char *library_version;
    (void)self;
    (void)ignored;

    vt_reset_process_caches_if_needed();
    library_version = g_qpl_inherited_after_fork ? QPL_VERSION : qpl_get_library_version();
    runtime_version_ok = vt_qpl_runtime_version_is_supported(library_version);
    inventory_ok = !g_qpl_inherited_after_fork && runtime_version_ok && vt_ensure_inventory() == 0;
    if (inventory_ok) {
        probe_status = vt_probe_hardware_context();
    }
    hardware_available = inventory_ok && probe_status == QPL_STS_OK;

    if (g_inventory.generation_mask == (1U << 1U)) {
        snprintf(generation, sizeof(generation), "IAA 1.x");
    } else if (g_inventory.generation_mask == (1U << 2U)) {
        snprintf(generation, sizeof(generation), "IAA 2.x");
    } else if (g_inventory.generation_mask != 0U) {
        snprintf(generation, sizeof(generation), "IAA mixed generations");
    } else {
        snprintf(generation, sizeof(generation), "unknown");
    }
    if (g_qpl_inherited_after_fork) {
        snprintf(
            hardware_error,
            sizeof(hardware_error),
            "IAA/QPL is unavailable after fork; start accelerator workers with spawn or exec"
        );
    } else if (!runtime_version_ok) {
        snprintf(
            hardware_error,
            sizeof(hardware_error),
            "Intel QPL runtime %s is unsupported; version 1.9.0 or newer is required",
            library_version != NULL ? library_version : "unknown"
        );
    } else if (!inventory_ok) {
        snprintf(hardware_error, sizeof(hardware_error), "%s", g_inventory.error);
    } else if (probe_status != QPL_STS_OK) {
        snprintf(
            hardware_error,
            sizeof(hardware_error),
            "qpl_init_job(qpl_path_hardware) failed: status=%d (%s)",
            (int)probe_status,
            vt_qpl_status_name(probe_status)
        );
    } else {
        hardware_error[0] = '\0';
    }

    result = PyDict_New();
    queues = PyList_New(0);
    levels = Py_BuildValue("(i)", (int)qpl_default_level);
    if (result == NULL || queues == NULL || levels == NULL) {
        goto failed;
    }
    for (index = 0U; index < g_inventory.eligible_work_queues; ++index) {
        const vt_qpl_work_queue *entry = &g_inventory.work_queues[index];
        PyObject *item = Py_BuildValue(
            "{s:s,s:s,s:s,s:i,s:i,s:i,s:I,s:K,s:I,s:O,s:O}",
            "device", entry->device,
            "work_queue", entry->work_queue,
            "path", entry->path,
            "device_id", entry->device_id,
            "work_queue_id", entry->work_queue_id,
            "numa_id", entry->numa_id,
            "device_version", entry->device_version,
            "max_transfer_bytes", (unsigned long long)entry->max_transfer_bytes,
            "max_member_input_bytes", entry->max_member_input_bytes,
            "block_on_fault", entry->block_on_fault ? Py_True : Py_False,
            "operation_config_known", entry->operation_config_known ? Py_True : Py_False
        );
        if (item == NULL || PyList_Append(queues, item) != 0) {
            Py_XDECREF(item);
            goto failed;
        }
        Py_DECREF(item);
    }

    if (vt_dict_set_owned(result, "backend", PyUnicode_FromString("iaa")) != 0 ||
        vt_dict_set_owned(result, "binding_version", PyUnicode_FromString(VT_QPL_BINDING_VERSION)) != 0 ||
        vt_dict_set_owned(result, "minimum_qpl_version", PyUnicode_FromString("1.9.0")) != 0 ||
        vt_dict_set_owned(result, "header_version", PyUnicode_FromString(QPL_VERSION)) != 0 ||
        vt_dict_set_owned(result, "qpl_version", PyUnicode_FromString(library_version != NULL ? library_version : "unknown")) != 0 ||
        vt_dict_set_owned(result, "library_version", PyUnicode_FromString(library_version != NULL ? library_version : "unknown")) != 0 ||
        vt_dict_set_owned(result, "runtime_version_supported", PyBool_FromLong(runtime_version_ok)) != 0 ||
        vt_dict_set_owned(result, "postfork_inherited_state", PyBool_FromLong(g_qpl_inherited_after_fork)) != 0 ||
        vt_dict_set_owned(result, "hardware_available", PyBool_FromLong(hardware_available)) != 0 ||
        vt_dict_set_owned(result, "standard_gzip", PyBool_FromLong(1)) != 0 ||
        vt_dict_set_owned(result, "software_fallback_enabled", PyBool_FromLong(0)) != 0 ||
        vt_dict_set_owned(result, "execution_path", PyUnicode_FromString("qpl_path_hardware")) != 0 ||
        vt_dict_set_owned(result, "hardware_generation", PyUnicode_FromString(generation)) != 0 ||
        vt_dict_set_owned(result, "device_identity", PyUnicode_FromString(generation)) != 0 ||
        vt_dict_set_owned(result, "device_count", PyLong_FromUnsignedLong(g_inventory.enabled_devices)) != 0 ||
        vt_dict_set_owned(result, "instance_count", PyLong_FromUnsignedLong(g_inventory.enabled_devices)) != 0 ||
        vt_dict_set_owned(result, "work_queue_count", PyLong_FromUnsignedLong(g_inventory.eligible_work_queues)) != 0 ||
        vt_dict_set_owned(result, "truncated_work_queue_count", PyLong_FromUnsignedLong(g_inventory.truncated_work_queues)) != 0 ||
        vt_dict_set_owned(result, "max_concurrency", PyLong_FromUnsignedLong(g_inventory.eligible_work_queues)) != 0 ||
        vt_dict_set_owned(result, "minimum_input_bytes", PyLong_FromLong(1L)) != 0 ||
        vt_dict_set_owned(result, "min_transfer_bytes", PyLong_FromUnsignedLongLong(g_inventory.min_transfer_bytes == UINT64_MAX ? 0U : g_inventory.min_transfer_bytes)) != 0 ||
        vt_dict_set_owned(result, "max_transfer_bytes", PyLong_FromUnsignedLongLong(g_inventory.max_transfer_bytes)) != 0 ||
        vt_dict_set_owned(result, "safe_member_input_bytes", PyLong_FromUnsignedLong(g_inventory.min_member_input_bytes == UINT32_MAX ? 0U : g_inventory.min_member_input_bytes)) != 0 ||
        vt_dict_set_owned(result, "min_member_input_bytes", PyLong_FromUnsignedLong(g_inventory.min_member_input_bytes == UINT32_MAX ? 0U : g_inventory.min_member_input_bytes)) != 0 ||
        vt_dict_set_owned(result, "max_member_input_bytes", PyLong_FromUnsignedLong(g_inventory.max_member_input_bytes)) != 0 ||
        vt_dict_set_owned(result, "gzip_overhead_bytes", PyLong_FromLong(VT_QPL_GZIP_OVERHEAD)) != 0 ||
        vt_dict_set_owned(result, "history_window_bytes", PyLong_FromLong(4096L)) != 0 ||
        vt_dict_set_owned(result, "dynamic_huffman", PyBool_FromLong(1)) != 0 ||
        vt_dict_set_owned(result, "hardware_error", PyUnicode_FromString(hardware_error)) != 0 ||
        vt_dict_set_owned(result, "unavailable_reason", PyUnicode_FromString(hardware_error)) != 0 ||
        vt_dict_set_owned(result, "probe_status", PyLong_FromLong((long)probe_status)) != 0) {
        goto failed;
    }
    if (PyDict_SetItemString(result, "supported_levels", levels) != 0 ||
        PyDict_SetItemString(result, "eligible_work_queues", queues) != 0) {
        goto failed;
    }
    Py_DECREF(levels);
    Py_DECREF(queues);
    return result;

failed:
    Py_XDECREF(levels);
    Py_XDECREF(queues);
    Py_XDECREF(result);
    return NULL;
}

static PyObject *vt_qpl_preflight_thread_state(
    PyObject *self,
    PyObject *args,
    PyObject *kwargs
) {
    static char *keywords[] = {"level", "require_hardware", "numa_id", NULL};
    int level;
    int require_hardware = 1;
    PyObject *numa_object = Py_None;
    int numa_id;
    unsigned int queue_count;
    uint32_t chunk_limit;
    qpl_status status;
    vt_qpl_thread_state *state;
    (void)self;

    if (!PyArg_ParseTupleAndKeywords(
            args,
            kwargs,
            "i|$pO:preflight_thread_state",
            keywords,
            &level,
            &require_hardware,
            &numa_object)) {
        return NULL;
    }
    vt_reset_process_caches_if_needed();
    if (vt_validate_hardware_request(level, require_hardware) != 0 ||
        vt_parse_numa_id(numa_object, &numa_id) != 0) {
        return NULL;
    }
    chunk_limit = vt_chunk_limit_for_numa(numa_id, &queue_count, NULL);
    if (chunk_limit == 0U || queue_count == 0U) {
        PyErr_Format(
            PyExc_RuntimeError,
            "IAA/QPL preflight found no eligible hardware work queue for numa_id=%d (%s)",
            numa_id,
            g_inventory.error[0] != '\0' ? g_inventory.error : "requested NUMA node unavailable"
        );
        return NULL;
    }
    state = vt_get_thread_state(&status);
    if (state == NULL) {
        if (PyErr_Occurred()) {
            return NULL;
        }
        vt_record_terminal_status(status, numa_id);
        return vt_raise_qpl_status("thread initialization", status, numa_id, 0U, 0U);
    }
    state->job->numa_id = numa_id;
    if (state->job->data_ptr.path != qpl_path_hardware) {
        VT_STAT_ADD(failures, 1U);
        vt_record_terminal_status(QPL_STS_PATH_ERR, numa_id);
        (void)vt_retire_current_thread_state();
        return vt_raise_qpl_status("thread preflight path proof", QPL_STS_PATH_ERR, numa_id, 0U, 0U);
    }
    VT_STAT_ADD(preflight_calls, 1U);
    Py_RETURN_NONE;
}

static int vt_gzip_member_shape_is_valid(const uint8_t *data, uint32_t length, uint32_t source_length) {
    uint32_t isize;
    if (length < VT_QPL_GZIP_OVERHEAD || data[0] != 0x1fU || data[1] != 0x8bU || data[2] != 0x08U) {
        return 0;
    }
    isize = (uint32_t)data[length - 4U] |
            ((uint32_t)data[length - 3U] << 8U) |
            ((uint32_t)data[length - 2U] << 16U) |
            ((uint32_t)data[length - 1U] << 24U);
    return isize == source_length;
}

static PyObject *vt_qpl_compress_gzip(PyObject *self, PyObject *args, PyObject *kwargs) {
    static char *keywords[] = {"buffer", "level", "require_hardware", "numa_id", NULL};
    PyObject *buffer_object;
    int level;
    int require_hardware = 1;
    PyObject *numa_object = Py_None;
    int numa_id;
    Py_buffer input_view;
    unsigned int queue_count;
    uint64_t transfer_limit;
    uint32_t chunk_limit;
    Py_ssize_t source_offset;
    Py_ssize_t total_bound = 0;
    Py_ssize_t output_offset = 0;
    PyObject *output = NULL;
    vt_qpl_thread_state *state;
    qpl_status status = QPL_STS_OK;
    const char *failure_operation = NULL;
    uint32_t failed_chunk = 0U;
    uint32_t failed_capacity = 0U;
    uint64_t start_ns;
    uint64_t end_ns;
    (void)self;

    memset(&input_view, 0, sizeof(input_view));
    if (!PyArg_ParseTupleAndKeywords(
            args,
            kwargs,
            "Oi|$pO:compress_gzip",
            keywords,
            &buffer_object,
            &level,
            &require_hardware,
            &numa_object)) {
        return NULL;
    }
    vt_reset_process_caches_if_needed();
    if (vt_validate_hardware_request(level, require_hardware) != 0 ||
        vt_parse_numa_id(numa_object, &numa_id) != 0) {
        return NULL;
    }
    if (PyObject_GetBuffer(buffer_object, &input_view, PyBUF_ND | PyBUF_STRIDES) != 0) {
        return NULL;
    }
    if (input_view.ndim != 1 || !PyBuffer_IsContiguous(&input_view, 'C')) {
        PyBuffer_Release(&input_view);
        PyErr_SetString(PyExc_BufferError, "IAA/QPL input must be one-dimensional and C-contiguous");
        return NULL;
    }
    if (input_view.len <= 0) {
        PyBuffer_Release(&input_view);
        PyErr_SetString(PyExc_ValueError, "IAA/QPL hardware compression requires at least one input byte");
        return NULL;
    }

    chunk_limit = vt_chunk_limit_for_numa(numa_id, &queue_count, &transfer_limit);
    if (chunk_limit == 0U || queue_count == 0U) {
        PyBuffer_Release(&input_view);
        PyErr_Format(
            PyExc_RuntimeError,
            "IAA/QPL found no eligible hardware work queue for numa_id=%d (%s)",
            numa_id,
            g_inventory.error[0] != '\0' ? g_inventory.error : "requested NUMA node unavailable"
        );
        return NULL;
    }

    for (source_offset = 0; source_offset < input_view.len;) {
        Py_ssize_t left = input_view.len - source_offset;
        uint32_t chunk = left > (Py_ssize_t)chunk_limit ? chunk_limit : (uint32_t)left;
        uint32_t bound = qpl_get_safe_deflate_compression_buffer_size(chunk);
        uint64_t capacity = (uint64_t)bound + VT_QPL_GZIP_OVERHEAD;
        if (bound == 0U || capacity > transfer_limit || capacity > UINT32_MAX ||
            total_bound > PY_SSIZE_T_MAX - (Py_ssize_t)capacity) {
            PyBuffer_Release(&input_view);
            PyErr_Format(
                PyExc_OverflowError,
                "IAA/QPL safe output bound is invalid: chunk=%u, bound=%u, transfer_limit=%llu",
                (unsigned int)chunk,
                (unsigned int)bound,
                (unsigned long long)transfer_limit
            );
            return NULL;
        }
        total_bound += (Py_ssize_t)capacity;
        source_offset += (Py_ssize_t)chunk;
    }

    output = PyBytes_FromStringAndSize(NULL, total_bound);
    if (output == NULL) {
        PyBuffer_Release(&input_view);
        return NULL;
    }
    state = vt_get_thread_state(&status);
    if (state == NULL) {
        Py_DECREF(output);
        PyBuffer_Release(&input_view);
        if (PyErr_Occurred()) {
            return NULL;
        }
        VT_STAT_ADD(failures, 1U);
        vt_record_terminal_status(status, numa_id);
        return vt_raise_qpl_status("thread initialization", status, numa_id, 0U, 0U);
    }

    VT_STAT_ADD(logical_requests, 1U);
    VT_STAT_ADD(requested_input_bytes, (uint64_t)input_view.len);
    start_ns = vt_monotonic_ns();
    for (source_offset = 0; source_offset < input_view.len;) {
        Py_ssize_t left = input_view.len - source_offset;
        uint32_t chunk = left > (Py_ssize_t)chunk_limit ? chunk_limit : (uint32_t)left;
        uint32_t deflate_bound = qpl_get_safe_deflate_compression_buffer_size(chunk);
        uint32_t capacity = deflate_bound + VT_QPL_GZIP_OVERHEAD;
        uint8_t *source = (uint8_t *)input_view.buf + source_offset;
        uint8_t *destination = (uint8_t *)PyBytes_AS_STRING(output) + output_offset;
        uint32_t produced;

        state->job->next_in_ptr = source;
        state->job->available_in = chunk;
        state->job->total_in = 0U;
        state->job->next_out_ptr = destination;
        state->job->available_out = capacity;
        state->job->total_out = 0U;
        state->job->op = qpl_op_compress;
        state->job->flags = QPL_FLAG_FIRST | QPL_FLAG_LAST | QPL_FLAG_GZIP_MODE |
                            QPL_FLAG_DYNAMIC_HUFFMAN | QPL_FLAG_OMIT_VERIFY;
        state->job->crc = 0U;
        state->job->xor_checksum = 0U;
        state->job->last_bit_offset = 0U;
        state->job->level = qpl_default_level;
        state->job->statistics_mode = qpl_compression_mode;
        state->job->huffman_table = NULL;
        state->job->dictionary = NULL;
        state->job->mini_block_size = qpl_mblk_size_none;
        state->job->idx_array = NULL;
        state->job->idx_max_size = 0U;
        state->job->idx_num_written = 0U;
        state->job->numa_id = numa_id;

        VT_STAT_ADD(hardware_requests, 1U);
        Py_BEGIN_ALLOW_THREADS
        status = vt_qpl_execute_hardware_job(state->job);
        Py_END_ALLOW_THREADS
        vt_record_terminal_status(status, numa_id);
        if (status != QPL_STS_OK) {
            failure_operation = "compression";
            failed_chunk = chunk;
            failed_capacity = capacity;
            break;
        }
        if (state->job->data_ptr.path != qpl_path_hardware) {
            status = QPL_STS_PATH_ERR;
            vt_record_terminal_status(status, numa_id);
            failure_operation = "hardware path proof";
            failed_chunk = chunk;
            failed_capacity = capacity;
            break;
        }
        if (state->job->available_out > capacity || state->job->total_out > capacity) {
            status = QPL_STS_LIBRARY_INTERNAL_ERR;
            vt_record_terminal_status(status, numa_id);
            failure_operation = "output bounds proof";
            failed_chunk = chunk;
            failed_capacity = capacity;
            break;
        }
        produced = capacity - state->job->available_out;
        if (state->job->available_in != 0U || state->job->total_in != chunk ||
            state->job->next_in_ptr != source + chunk) {
            status = QPL_STS_MORE_INPUT_NEEDED;
            vt_record_terminal_status(status, numa_id);
            failure_operation = "full input consumption proof";
            failed_chunk = chunk;
            failed_capacity = capacity;
            break;
        }
        if (produced == 0U || produced != state->job->total_out ||
            state->job->next_out_ptr != destination + produced ||
            !vt_gzip_member_shape_is_valid(destination, produced, chunk)) {
            status = QPL_STS_LIBRARY_INTERNAL_ERR;
            vt_record_terminal_status(status, numa_id);
            failure_operation = "standard gzip framing proof";
            failed_chunk = chunk;
            failed_capacity = capacity;
            break;
        }
        VT_STAT_ADD(physical_members, 1U);
        VT_STAT_ADD(input_bytes, chunk);
        VT_STAT_ADD(output_bytes, produced);
        source_offset += (Py_ssize_t)chunk;
        output_offset += (Py_ssize_t)produced;
    }
    end_ns = vt_monotonic_ns();
    if (end_ns >= start_ns) {
        VT_STAT_ADD(elapsed_ns, end_ns - start_ns);
    }
    PyBuffer_Release(&input_view);

    if (failure_operation != NULL) {
        qpl_status fini_status;
        Py_DECREF(output);
        VT_STAT_ADD(failures, 1U);
        fini_status = vt_retire_current_thread_state();
        if (fini_status != QPL_STS_OK) {
            /* Preserve the compression error; finalization remains visible in last_status. */
            vt_record_terminal_status(fini_status, numa_id);
        }
        return vt_raise_qpl_status(
            failure_operation,
            status,
            numa_id,
            failed_chunk,
            failed_capacity
        );
    }
    if (_PyBytes_Resize(&output, output_offset) != 0) {
        return NULL;
    }
    return output;
}

static PyObject *vt_qpl_stats(PyObject *self, PyObject *args, PyObject *kwargs) {
    static char *keywords[] = {"reset", NULL};
    int reset = 0;
    PyObject *result;
    uint64_t logical_requests;
    uint64_t hardware_requests;
    uint64_t physical_members;
    uint64_t requested_input_bytes;
    uint64_t input_bytes;
    uint64_t output_bytes;
    uint64_t elapsed_ns;
    uint64_t failures;
    uint64_t queue_busy_events;
    uint64_t page_fault_errors;
    uint64_t sessions_created;
    uint64_t sessions_closed;
    uint64_t active_sessions;
    uint64_t preflight_calls;
    int32_t last_status;
    int32_t last_numa_id;
    (void)self;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|$p:stats", keywords, &reset)) {
        return NULL;
    }
    vt_reset_process_caches_if_needed();
    if (reset) {
        logical_requests = VT_STAT_EXCHANGE(logical_requests, 0U);
        hardware_requests = VT_STAT_EXCHANGE(hardware_requests, 0U);
        physical_members = VT_STAT_EXCHANGE(physical_members, 0U);
        requested_input_bytes = VT_STAT_EXCHANGE(requested_input_bytes, 0U);
        input_bytes = VT_STAT_EXCHANGE(input_bytes, 0U);
        output_bytes = VT_STAT_EXCHANGE(output_bytes, 0U);
        elapsed_ns = VT_STAT_EXCHANGE(elapsed_ns, 0U);
        failures = VT_STAT_EXCHANGE(failures, 0U);
        queue_busy_events = VT_STAT_EXCHANGE(queue_busy_events, 0U);
        page_fault_errors = VT_STAT_EXCHANGE(page_fault_errors, 0U);
        sessions_created = VT_STAT_EXCHANGE(sessions_created, 0U);
        sessions_closed = VT_STAT_EXCHANGE(sessions_closed, 0U);
        active_sessions = VT_STAT_LOAD(active_sessions);
        preflight_calls = VT_STAT_EXCHANGE(preflight_calls, 0U);
    } else {
        logical_requests = VT_STAT_LOAD(logical_requests);
        hardware_requests = VT_STAT_LOAD(hardware_requests);
        physical_members = VT_STAT_LOAD(physical_members);
        requested_input_bytes = VT_STAT_LOAD(requested_input_bytes);
        input_bytes = VT_STAT_LOAD(input_bytes);
        output_bytes = VT_STAT_LOAD(output_bytes);
        elapsed_ns = VT_STAT_LOAD(elapsed_ns);
        failures = VT_STAT_LOAD(failures);
        queue_busy_events = VT_STAT_LOAD(queue_busy_events);
        page_fault_errors = VT_STAT_LOAD(page_fault_errors);
        sessions_created = VT_STAT_LOAD(sessions_created);
        sessions_closed = VT_STAT_LOAD(sessions_closed);
        active_sessions = VT_STAT_LOAD(active_sessions);
        preflight_calls = VT_STAT_LOAD(preflight_calls);
    }
    if (reset) {
        last_status = __atomic_exchange_n(
            &g_stats.last_status, (int32_t)QPL_STS_OK, __ATOMIC_RELAXED
        );
        last_numa_id = __atomic_exchange_n(
            &g_stats.last_numa_id,
            (int32_t)QPL_DEVICE_NUMA_ID_SOCKET,
            __ATOMIC_RELAXED
        );
    } else {
        last_status = __atomic_load_n(&g_stats.last_status, __ATOMIC_RELAXED);
        last_numa_id = __atomic_load_n(&g_stats.last_numa_id, __ATOMIC_RELAXED);
    }

    result = Py_BuildValue(
        "{s:s,s:s,s:K,s:K,s:K,s:K,s:K,s:K,s:K,s:K,s:K,s:K,s:K,s:K,s:K,s:K,s:K,s:i,s:s,s:i}",
        "backend", "iaa",
        "execution_path", "qpl_path_hardware",
        "logical_requests", (unsigned long long)logical_requests,
        "hardware_requests", (unsigned long long)hardware_requests,
        "software_fallback_requests", (unsigned long long)0U,
        "physical_members", (unsigned long long)physical_members,
        "requested_input_bytes", (unsigned long long)requested_input_bytes,
        "input_bytes", (unsigned long long)input_bytes,
        "output_bytes", (unsigned long long)output_bytes,
        "elapsed_ns", (unsigned long long)elapsed_ns,
        "failures", (unsigned long long)failures,
        "queue_busy_events", (unsigned long long)queue_busy_events,
        "page_fault_errors", (unsigned long long)page_fault_errors,
        "sessions_created", (unsigned long long)sessions_created,
        "sessions_closed", (unsigned long long)sessions_closed,
        "active_sessions", (unsigned long long)active_sessions,
        "preflight_calls", (unsigned long long)preflight_calls,
        "last_status", (int)last_status,
        "last_status_name", vt_qpl_status_name((qpl_status)last_status),
        "last_numa_id", (int)last_numa_id
    );
    return result;
}

static PyObject *vt_qpl_close_thread_state(PyObject *self, PyObject *ignored) {
    qpl_status status;
    (void)self;
    (void)ignored;
    vt_reset_process_caches_if_needed();
    status = vt_retire_current_thread_state();
    if (status != QPL_STS_OK) {
        VT_STAT_ADD(failures, 1U);
        vt_record_terminal_status(status, QPL_DEVICE_NUMA_ID_SOCKET);
        return vt_raise_qpl_status(
            "thread finalization",
            status,
            QPL_DEVICE_NUMA_ID_SOCKET,
            0U,
            0U
        );
    }
    Py_RETURN_NONE;
}

static PyMethodDef vt_qpl_methods[] = {
    {"capabilities", (PyCFunction)vt_qpl_capabilities, METH_NOARGS,
     PyDoc_STR("Return hardware-only Intel IAA/QPL capabilities and eligible work queues.")},
    {"compress_gzip", (PyCFunction)(void(*)(void))vt_qpl_compress_gzip,
     METH_VARARGS | METH_KEYWORDS,
     PyDoc_STR("Compress a contiguous 1-D buffer into hardware-produced gzip member(s).")},
    {"preflight_thread_state", (PyCFunction)(void(*)(void))vt_qpl_preflight_thread_state,
     METH_VARARGS | METH_KEYWORDS,
     PyDoc_STR("Initialize one executor thread's qpl_path_hardware job.")},
    {"stats", (PyCFunction)(void(*)(void))vt_qpl_stats, METH_VARARGS | METH_KEYWORDS,
     PyDoc_STR("Return process-local IAA/QPL counters; optionally reset interval counters.")},
    {"close_thread_state", (PyCFunction)vt_qpl_close_thread_state, METH_NOARGS,
     PyDoc_STR("Finalize the calling thread's QPL job.")},
    {NULL, NULL, 0, NULL}
};

static void vt_qpl_module_free(void *module) {
    vt_qpl_thread_state *state;
    (void)module;
    if (!g_thread_state_key_initialized || g_thread_state_key_error != 0) {
        return;
    }
    state = (vt_qpl_thread_state *)pthread_getspecific(g_thread_state_key);
    if (state != NULL) {
        if (pthread_setspecific(g_thread_state_key, NULL) != 0) {
            return;
        }
        (void)vt_finalize_thread_state(state, state->pid == getpid());
    }
}

static struct PyModuleDef vt_qpl_module = {
    PyModuleDef_HEAD_INIT,
    "_qpl_codec",
    "Optional Linux Intel IAA/QPL hardware-only gzip companion.",
    -1,
    vt_qpl_methods,
    NULL,
    NULL,
    NULL,
    vt_qpl_module_free
};

PyMODINIT_FUNC PyInit__qpl_codec(void) {
    PyObject *module;
    int atfork_status;
    if (!g_qpl_atfork_registered) {
        atfork_status = pthread_atfork(
            vt_qpl_atfork_prepare,
            vt_qpl_atfork_parent,
            vt_qpl_atfork_child
        );
        if (atfork_status != 0) {
            PyErr_Format(
                PyExc_ImportError,
                "IAA/QPL pthread_atfork registration failed: status=%d",
                atfork_status
            );
            return NULL;
        }
        g_qpl_atfork_registered = 1;
    }
    g_process_id = getpid();
    memset(&g_inventory, 0, sizeof(g_inventory));
    memset(&g_stats, 0, sizeof(g_stats));
    g_stats.last_numa_id = QPL_DEVICE_NUMA_ID_SOCKET;
    module = PyModule_Create(&vt_qpl_module);
    if (module == NULL) {
        return NULL;
    }
    if (PyModule_AddStringConstant(module, "BACKEND", "iaa") != 0 ||
        PyModule_AddStringConstant(module, "EXECUTION_PATH", "qpl_path_hardware") != 0 ||
        PyModule_AddStringConstant(module, "BINDING_VERSION", VT_QPL_BINDING_VERSION) != 0) {
        Py_DECREF(module);
        return NULL;
    }
    return module;
}
