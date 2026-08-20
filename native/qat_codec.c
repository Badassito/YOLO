#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#if !defined(__linux__) || !defined(__x86_64__)
#error "volume_tta._qat_codec is supported only on Linux x86_64"
#endif

#include <errno.h>
#include <limits.h>
#include <pthread.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#include <qatzip.h>

#if !defined(QATZIP_API_VERSION) || QATZIP_API_VERSION < 20500
#error "volume_tta._qat_codec requires QATzip API 2.5 or newer (QATzip 1.3.2+)"
#endif

#define VOLUME_TTA_QAT_BINDING_VERSION "17.0.8-capi1"
#define VOLUME_TTA_QAT_MIN_INPUT ((unsigned int)QZ_COMP_THRESHOLD_MINIMUM)
/* QATzip 1.3.2 can route level 9 to its software provider on QAT 1.x without
 * reliably setting qzCompressExt's SW bit.  Until upstream exposes dependable
 * generation/per-request proof, fail closed instead of advertising level 9. */
#define VOLUME_TTA_QAT_LEVEL_MAX 8

typedef struct {
    QzSession_T session;
    QzStatus_T status;
    pid_t pid;
    int level;
    int initialized;
    int setup;
    int counted;
} QatThreadState;

typedef struct {
    uint64_t logical_requests;
    uint64_t hardware_requests;
    uint64_t software_fallback_requests;
    uint64_t input_bytes;
    uint64_t output_bytes;
    uint64_t failures;
    uint64_t partial_consumption_failures;
    uint64_t timeouts;
    uint64_t buffer_errors;
    uint64_t sessions_created;
    uint64_t sessions_closed;
    uint64_t active_sessions;
    uint64_t peak_sessions;
    uint64_t elapsed_ns;
    pid_t pid;
} QatStats;

typedef struct {
    int init_rc;
    int defaults_rc;
    int setup_rc;
    int status_rc;
    signed long session_hw_status;
    int status_populated;
    QzStatus_T status;
} QatProbe;

static pthread_key_t g_session_key;
static pthread_once_t g_session_key_once = PTHREAD_ONCE_INIT;
static int g_session_key_error = 0;
static pthread_mutex_t g_stats_lock = PTHREAD_MUTEX_INITIALIZER;
static QatStats g_stats;
static PyObject *g_qatzip_error = NULL;
static int g_atfork_registered = 0;
static volatile sig_atomic_t g_forked_child = 0;

static const char *qz_status_name(int status)
{
    switch (status) {
    case QZ_OK: return "QZ_OK";
    case QZ_DUPLICATE: return "QZ_DUPLICATE";
    case QZ_FORCE_SW: return "QZ_FORCE_SW";
    case QZ_PARAMS: return "QZ_PARAMS";
    case QZ_FAIL: return "QZ_FAIL";
    case QZ_BUF_ERROR: return "QZ_BUF_ERROR";
    case QZ_DATA_ERROR: return "QZ_DATA_ERROR";
    case QZ_TIMEOUT: return "QZ_TIMEOUT";
    case QZ_INTEG: return "QZ_INTEG";
    case QZ_NO_HW: return "QZ_NO_HW";
    case QZ_NO_MDRV: return "QZ_NO_MDRV";
    case QZ_NO_INST_ATTACH: return "QZ_NO_INST_ATTACH";
    case QZ_LOW_MEM: return "QZ_LOW_MEM";
    case QZ_LOW_DEST_MEM: return "QZ_LOW_DEST_MEM";
    case QZ_UNSUPPORTED_FMT: return "QZ_UNSUPPORTED_FMT";
    case QZ_NONE: return "QZ_NONE";
    case QZ_NOSW_NO_HW: return "QZ_NOSW_NO_HW";
    case QZ_NOSW_NO_MDRV: return "QZ_NOSW_NO_MDRV";
    case QZ_NOSW_NO_INST_ATTACH: return "QZ_NOSW_NO_INST_ATTACH";
    case QZ_NOSW_LOW_MEM: return "QZ_NOSW_LOW_MEM";
    case QZ_NO_SW_AVAIL: return "QZ_NO_SW_AVAIL";
    case QZ_NOSW_UNSUPPORTED_FMT: return "QZ_NOSW_UNSUPPORTED_FMT";
    case QZ_POST_PROCESS_ERROR: return "QZ_POST_PROCESS_ERROR";
    case QZ_METADATA_OVERFLOW: return "QZ_METADATA_OVERFLOW";
    case QZ_OUT_OF_RANGE: return "QZ_OUT_OF_RANGE";
    case QZ_NOT_SUPPORTED: return "QZ_NOT_SUPPORTED";
    default: return "QZ_UNKNOWN";
    }
}

static void stats_reset_for_pid_locked(pid_t pid)
{
    if (g_stats.pid == pid) {
        return;
    }
    memset(&g_stats, 0, sizeof(g_stats));
    g_stats.pid = pid;
}

static void stats_note_session_created(void)
{
    const pid_t pid = getpid();
    pthread_mutex_lock(&g_stats_lock);
    stats_reset_for_pid_locked(pid);
    g_stats.sessions_created++;
    g_stats.active_sessions++;
    if (g_stats.active_sessions > g_stats.peak_sessions) {
        g_stats.peak_sessions = g_stats.active_sessions;
    }
    pthread_mutex_unlock(&g_stats_lock);
}

static void stats_note_session_closed(void)
{
    const pid_t pid = getpid();
    pthread_mutex_lock(&g_stats_lock);
    stats_reset_for_pid_locked(pid);
    g_stats.sessions_closed++;
    if (g_stats.active_sessions > 0) {
        g_stats.active_sessions--;
    }
    pthread_mutex_unlock(&g_stats_lock);
}

static void stats_note_request(
    int success,
    int hardware,
    int software,
    int partial,
    int timeout,
    int buffer_error,
    uint64_t input_bytes,
    uint64_t output_bytes,
    uint64_t elapsed_ns)
{
    const pid_t pid = getpid();
    pthread_mutex_lock(&g_stats_lock);
    stats_reset_for_pid_locked(pid);
    g_stats.logical_requests++;
    g_stats.hardware_requests += (uint64_t)(hardware != 0);
    g_stats.software_fallback_requests += (uint64_t)(software != 0);
    g_stats.failures += (uint64_t)(success == 0);
    g_stats.partial_consumption_failures += (uint64_t)(partial != 0);
    g_stats.timeouts += (uint64_t)(timeout != 0);
    g_stats.buffer_errors += (uint64_t)(buffer_error != 0);
    g_stats.elapsed_ns += elapsed_ns;
    if (success) {
        g_stats.input_bytes += input_bytes;
        g_stats.output_bytes += output_bytes;
    }
    pthread_mutex_unlock(&g_stats_lock);
}

/* The prepare handler makes the mutex owner in a fork child deterministic:
 * it is always the thread that called fork().  Reset process-local counters
 * before unlocking in the child so no lock or session accounting is inherited
 * from vanished parent threads. */
static void stats_atfork_prepare(void)
{
    (void)pthread_mutex_lock(&g_stats_lock);
}

static void stats_atfork_parent(void)
{
    (void)pthread_mutex_unlock(&g_stats_lock);
}

static void stats_atfork_child(void)
{
    memset(&g_stats, 0, sizeof(g_stats));
    g_stats.pid = getpid();
    /* QATzip 1.3.2 keeps process-global driver/session locks and provides no
     * public child-reset API. Never touch that inherited state in the child;
     * callers can use a spawn/exec worker to obtain a fresh QATzip process. */
    g_forked_child = 1;
    (void)pthread_mutex_unlock(&g_stats_lock);
}

static uint64_t monotonic_ns(void)
{
    struct timespec value;
    if (clock_gettime(CLOCK_MONOTONIC, &value) != 0) {
        return 0;
    }
    return (uint64_t)value.tv_sec * UINT64_C(1000000000)
        + (uint64_t)value.tv_nsec;
}

static size_t bounded_string_length(const unsigned char *value, size_t limit)
{
    size_t length = 0;
    while (length < limit && value[length] != '\0') {
        ++length;
    }
    return length;
}

static void destroy_thread_state(QatThreadState *state, int call_native)
{
    if (state == NULL) {
        return;
    }
    if (call_native && state->pid == getpid()) {
        if (state->setup) {
            (void)qzTeardownSession(&state->session);
        }
        if (state->initialized) {
            (void)qzClose(&state->session);
        }
    }
    if (state->counted && state->pid == getpid()) {
        stats_note_session_closed();
    }
    memset(state, 0, sizeof(*state));
    free(state);
}

static void session_key_destructor(void *value)
{
    destroy_thread_state((QatThreadState *)value, 1);
}

static void create_session_key(void)
{
    g_session_key_error = pthread_key_create(&g_session_key, session_key_destructor);
}

static int ensure_session_key(void)
{
    const int once_rc = pthread_once(&g_session_key_once, create_session_key);
    if (once_rc != 0 || g_session_key_error != 0) {
        PyErr_Format(
            PyExc_RuntimeError,
            "QATzip thread-local key initialization failed: pthread status=%d",
            once_rc != 0 ? once_rc : g_session_key_error);
        return -1;
    }
    return 0;
}

static int ensure_atfork_registered(void)
{
    int rc;
    if (g_atfork_registered) {
        return 0;
    }
    rc = pthread_atfork(
        stats_atfork_prepare, stats_atfork_parent, stats_atfork_child);
    if (rc != 0) {
        PyErr_Format(
            PyExc_RuntimeError,
            "QATzip fork-safety handler registration failed: pthread status=%d",
            rc);
        return -1;
    }
    g_atfork_registered = 1;
    return 0;
}

static QatThreadState *current_thread_state(void)
{
    QatThreadState *state;
    if (ensure_session_key() != 0) {
        return NULL;
    }
    state = (QatThreadState *)pthread_getspecific(g_session_key);
    if (state != NULL && state->pid != getpid()) {
        /* Never call inherited QATzip state in a fork child. QAT operations in
         * that child remain fail-closed until spawn/exec creates a clean process. */
        const int detach_rc = pthread_setspecific(g_session_key, NULL);
        if (detach_rc != 0) {
            PyErr_Format(
                PyExc_RuntimeError,
                "failed to discard inherited QATzip thread state: pthread status=%d",
                detach_rc);
            return NULL;
        }
        destroy_thread_state(state, 0);
        state = NULL;
    }
    return state;
}

static int session_proves_hardware(const QzSession_T *session)
{
    return session != NULL
        && session->internal != NULL
        && session->hw_session_stat == QZ_OK;
}

static int qz_status_was_populated(const QzStatus_T *status)
{
    if (status == NULL) {
        return 0;
    }
    return status->qat_hw_count != 0
        || status->qat_service_init != 0
        || status->qat_mem_drvr != 0
        || status->qat_instance_attach != 0
        || status->memory_alloced != 0
        || status->using_huge_pages != 0
        || status->hw_session_status != 0
        || status->algo_hw[QZ_DEFLATE] != 0
        || status->algo_sw[QZ_DEFLATE] != 0;
}

static int validate_numa_argument(PyObject *numa_id)
{
    long value;
    if (numa_id == NULL || numa_id == Py_None) {
        return 0;
    }
    value = PyLong_AsLong(numa_id);
    if (value == -1 && PyErr_Occurred()) {
        return -1;
    }
    if (value < 0) {
        return 0;
    }
    PyErr_SetString(
        PyExc_NotImplementedError,
        "QATzip's public session API does not expose deterministic instance NUMA "
        "binding; unset numa_id and use library-managed placement");
    return -1;
}

static void configure_hardware_deflate_params(
    QzSessionParamsDeflate_T *params,
    int level)
{
    params->common_params.direction = QZ_DIR_COMPRESS;
    params->common_params.comp_lvl = (unsigned int)level;
    params->common_params.comp_algorithm = QZ_DEFLATE;
    params->common_params.max_forks = 0;
    params->common_params.sw_backup = 0;
    QZ_DISABLE_SOFTWARE_BACKUP(params->common_params.sw_backup);
    QZ_DISABLE_SOFTWARE_ONLY_EXECUTION(params->common_params.sw_backup);
    params->common_params.input_sz_thrshold = VOLUME_TTA_QAT_MIN_INPUT;
    params->common_params.is_sensitive_mode = 0;
    params->data_fmt = QZ_DEFLATE_GZIP;
}

static void destroy_thread_state_with_gil(QatThreadState *state, int call_native)
{
    Py_BEGIN_ALLOW_THREADS
    destroy_thread_state(state, call_native);
    Py_END_ALLOW_THREADS
}

static QatThreadState *ensure_thread_session(int level)
{
    QatThreadState *state;
    QzSessionParamsDeflate_T params;
    int init_rc = QZ_FAIL;
    int defaults_rc = QZ_FAIL;
    int setup_rc = QZ_FAIL;
    int status_rc = QZ_FAIL;

    if (g_forked_child) {
        PyErr_SetString(
            g_qatzip_error,
            "QATzip is disabled in a post-fork child because upstream exposes no "
            "safe process-state reset; use multiprocessing spawn or exec");
        return NULL;
    }
    if (level < QZ_DEFLATE_COMP_LVL_MINIMUM
        || level > VOLUME_TTA_QAT_LEVEL_MAX) {
        PyErr_Format(
            PyExc_ValueError,
            "QATzip deflate level must be in [%d, %d], got %d",
            QZ_DEFLATE_COMP_LVL_MINIMUM,
            VOLUME_TTA_QAT_LEVEL_MAX,
            level);
        return NULL;
    }
    state = current_thread_state();
    if (state == NULL && PyErr_Occurred()) {
        return NULL;
    }
    if (state != NULL && state->level == level) {
        return state;
    }
    if (state != NULL) {
        const int detach_rc = pthread_setspecific(g_session_key, NULL);
        if (detach_rc != 0) {
            PyErr_Format(
                PyExc_RuntimeError,
                "failed to replace QATzip thread session: pthread status=%d",
                detach_rc);
            return NULL;
        }
        destroy_thread_state_with_gil(state, 1);
    }

    state = (QatThreadState *)calloc(1, sizeof(*state));
    if (state == NULL) {
        PyErr_NoMemory();
        return NULL;
    }
    state->pid = getpid();
    state->level = level;

    Py_BEGIN_ALLOW_THREADS
    init_rc = qzInit(&state->session, 0);
    if (init_rc == QZ_OK || init_rc == QZ_DUPLICATE) {
        state->initialized = 1;
        memset(&params, 0, sizeof(params));
        defaults_rc = qzGetDefaultsDeflate(&params);
        if (defaults_rc == QZ_OK) {
            configure_hardware_deflate_params(&params, level);
            setup_rc = qzSetupSessionDeflate(&state->session, &params);
            if (setup_rc == QZ_OK || setup_rc == QZ_DUPLICATE) {
                state->setup = 1;
                memset(&state->status, 0, sizeof(state->status));
                status_rc = qzGetStatus(&state->session, &state->status);
            }
        }
    }
    Py_END_ALLOW_THREADS

    if (init_rc != QZ_OK && init_rc != QZ_DUPLICATE) {
        PyErr_Format(
            g_qatzip_error,
            "QATzip qzInit failed in hardware-only mode: status=%s(%d)",
            qz_status_name(init_rc), init_rc);
        destroy_thread_state_with_gil(state, 1);
        return NULL;
    }
    if (defaults_rc != QZ_OK) {
        PyErr_Format(
            g_qatzip_error,
            "QATzip qzGetDefaultsDeflate failed: status=%s(%d)",
            qz_status_name(defaults_rc), defaults_rc);
        destroy_thread_state_with_gil(state, 1);
        return NULL;
    }
    if (setup_rc != QZ_OK && setup_rc != QZ_DUPLICATE) {
        PyErr_Format(
            g_qatzip_error,
            "QATzip qzSetupSessionDeflate failed in hardware-only mode: "
            "status=%s(%d)",
            qz_status_name(setup_rc), setup_rc);
        destroy_thread_state_with_gil(state, 1);
        return NULL;
    }
    if (!session_proves_hardware(&state->session)) {
        PyErr_Format(
            g_qatzip_error,
            "QATzip session did not prove a usable hardware deflate path: "
            "session_hw_status=%s(%ld), qzGetStatus=%s(%d), "
            "reported_hw_session_status=%s(%ld), devices=%u, "
            "instance_attached=%u, deflate_devices=%u",
            qz_status_name((int)state->session.hw_session_stat),
            state->session.hw_session_stat,
            qz_status_name(status_rc), status_rc,
            qz_status_name((int)state->status.hw_session_status),
            state->status.hw_session_status,
            (unsigned int)state->status.qat_hw_count,
            (unsigned int)state->status.qat_instance_attach,
            (unsigned int)state->status.algo_hw[QZ_DEFLATE]);
        destroy_thread_state_with_gil(state, 1);
        return NULL;
    }
    if (pthread_setspecific(g_session_key, state) != 0) {
        PyErr_SetString(PyExc_RuntimeError, "failed to retain QATzip thread session");
        destroy_thread_state_with_gil(state, 1);
        return NULL;
    }
    state->counted = 1;
    stats_note_session_created();
    return state;
}

static void probe_hardware_raw(QatProbe *probe)
{
    QatThreadState *state;
    QzSession_T temporary;
    QzSessionParamsDeflate_T params;
    int temporary_initialized = 0;
    int temporary_setup = 0;

    memset(probe, 0, sizeof(*probe));
    probe->init_rc = QZ_FAIL;
    probe->defaults_rc = QZ_FAIL;
    probe->setup_rc = QZ_FAIL;
    probe->status_rc = QZ_FAIL;
    probe->session_hw_status = QZ_NONE;
    state = (QatThreadState *)pthread_getspecific(g_session_key);
    if (state != NULL && state->pid == getpid()) {
        probe->init_rc = QZ_OK;
        probe->defaults_rc = QZ_OK;
        probe->setup_rc = QZ_OK;
        probe->session_hw_status = state->session.hw_session_stat;
        probe->status_rc = qzGetStatus(&state->session, &probe->status);
        probe->status_populated = probe->status_rc == QZ_OK
            && qz_status_was_populated(&probe->status);
        return;
    }

    memset(&temporary, 0, sizeof(temporary));
    probe->init_rc = qzInit(&temporary, 0);
    if (probe->init_rc == QZ_OK || probe->init_rc == QZ_DUPLICATE) {
        temporary_initialized = 1;
        memset(&params, 0, sizeof(params));
        probe->defaults_rc = qzGetDefaultsDeflate(&params);
        if (probe->defaults_rc == QZ_OK) {
            configure_hardware_deflate_params(
                &params, QZ_DEFLATE_COMP_LVL_MINIMUM);
            probe->setup_rc = qzSetupSessionDeflate(&temporary, &params);
            if (probe->setup_rc == QZ_OK || probe->setup_rc == QZ_DUPLICATE) {
                temporary_setup = 1;
                probe->session_hw_status = temporary.hw_session_stat;
                probe->status_rc = qzGetStatus(&temporary, &probe->status);
                probe->status_populated = probe->status_rc == QZ_OK
                    && qz_status_was_populated(&probe->status);
            }
        }
    }
    if (temporary_setup) {
        (void)qzTeardownSession(&temporary);
    }
    if (temporary_initialized) {
        (void)qzClose(&temporary);
    }
}

static void component_versions_raw(
    char *qatzip_version,
    size_t qatzip_size,
    char *driver_version,
    size_t driver_size)
{
    QzSoftwareVersionInfo_T *items = NULL;
    unsigned int count = 0;
    unsigned int allocated_count;
    unsigned int index;
    int rc;

    (void)snprintf(
        qatzip_version,
        qatzip_size,
        "api-%u.%u",
        (unsigned int)QATZIP_API_VERSION_NUM_MAJOR,
        (unsigned int)QATZIP_API_VERSION_NUM_MINOR);
    (void)snprintf(driver_version, driver_size, "unknown");
    rc = qzGetSoftwareComponentCount(&count);
    if (rc != QZ_OK || count == 0 || count > 64) {
        return;
    }
    allocated_count = count;
    items = (QzSoftwareVersionInfo_T *)calloc(count, sizeof(*items));
    if (items == NULL) {
        return;
    }
    rc = qzGetSoftwareComponentVersionList(items, &count);
    if (rc != QZ_OK) {
        free(items);
        return;
    }
    if (count > allocated_count) {
        count = allocated_count;
    }
    for (index = 0; index < count; ++index) {
        char formatted[160];
        const int name_len = (int)bounded_string_length(
            items[index].component_name, QZ_MAX_STRING_LENGTH);
        (void)snprintf(
            formatted,
            sizeof(formatted),
            "%.*s-%u.%u.%u.%u",
            name_len,
            (const char *)items[index].component_name,
            items[index].major_version,
            items[index].minor_version,
            items[index].patch_version,
            items[index].build_number);
        if (items[index].component_type == QZ_COMPONENT_QATZIP_API) {
            (void)snprintf(qatzip_version, qatzip_size, "%s", formatted);
        } else if (items[index].component_type == QZ_COMPONENT_USER_DRIVER) {
            (void)snprintf(driver_version, driver_size, "%s", formatted);
        }
    }
    free(items);
}

static int dict_set_owned(PyObject *dictionary, const char *key, PyObject *value)
{
    int rc;
    if (value == NULL) {
        return -1;
    }
    rc = PyDict_SetItemString(dictionary, key, value);
    Py_DECREF(value);
    return rc;
}

static PyObject *qat_capabilities(PyObject *self, PyObject *Py_UNUSED(args))
{
    QatProbe probe;
    char qatzip_version[160];
    char driver_version[160];
    char unavailable_reason[320];
    int available;
    int forked_child;
    unsigned int capacity;
    PyObject *result;
    (void)self;

    if (ensure_session_key() != 0) {
        return NULL;
    }
    forked_child = g_forked_child != 0;
    if (forked_child) {
        memset(&probe, 0, sizeof(probe));
        probe.init_rc = QZ_NONE;
        probe.defaults_rc = QZ_NONE;
        probe.setup_rc = QZ_NONE;
        probe.status_rc = QZ_NONE;
        probe.session_hw_status = QZ_NONE;
        (void)snprintf(
            qatzip_version, sizeof(qatzip_version), "api-%u.%u",
            (unsigned int)QATZIP_API_VERSION_NUM_MAJOR,
            (unsigned int)QATZIP_API_VERSION_NUM_MINOR);
        (void)snprintf(driver_version, sizeof(driver_version), "unknown");
    } else {
        Py_BEGIN_ALLOW_THREADS
        probe_hardware_raw(&probe);
        component_versions_raw(
            qatzip_version, sizeof(qatzip_version),
            driver_version, sizeof(driver_version));
        Py_END_ALLOW_THREADS
    }

    available = !forked_child
        && (probe.init_rc == QZ_OK || probe.init_rc == QZ_DUPLICATE)
        && probe.defaults_rc == QZ_OK
        && (probe.setup_rc == QZ_OK || probe.setup_rc == QZ_DUPLICATE)
        && probe.session_hw_status == QZ_OK;
    capacity = probe.status_populated && probe.status.qat_hw_count > 0
        ? (unsigned int)probe.status.qat_hw_count
        : 1U;
    if (available) {
        unavailable_reason[0] = '\0';
    } else if (forked_child) {
        (void)snprintf(
            unavailable_reason,
            sizeof(unavailable_reason),
            "QATzip is disabled in a post-fork child; use spawn or exec");
    } else if (probe.init_rc != QZ_OK && probe.init_rc != QZ_DUPLICATE) {
        (void)snprintf(
            unavailable_reason,
            sizeof(unavailable_reason),
            "qzInit hardware-only status=%s(%d)",
            qz_status_name(probe.init_rc), probe.init_rc);
    } else if (probe.defaults_rc != QZ_OK) {
        (void)snprintf(
            unavailable_reason,
            sizeof(unavailable_reason),
            "qzGetDefaultsDeflate status=%s(%d)",
            qz_status_name(probe.defaults_rc), probe.defaults_rc);
    } else if (probe.setup_rc != QZ_OK && probe.setup_rc != QZ_DUPLICATE) {
        (void)snprintf(
            unavailable_reason,
            sizeof(unavailable_reason),
            "qzSetupSessionDeflate hardware-only status=%s(%d)",
            qz_status_name(probe.setup_rc), probe.setup_rc);
    } else {
        (void)snprintf(
            unavailable_reason,
            sizeof(unavailable_reason),
            "hardware session proof failed: session_hw_status=%s(%ld), "
            "qzGetStatus=%s(%d), reported_devices=%u, "
            "reported_instance_attached=%u, reported_deflate_devices=%u",
            qz_status_name((int)probe.session_hw_status),
            probe.session_hw_status,
            qz_status_name(probe.status_rc), probe.status_rc,
            (unsigned int)probe.status.qat_hw_count,
            (unsigned int)probe.status.qat_instance_attach,
            (unsigned int)probe.status.algo_hw[QZ_DEFLATE]);
    }

    result = PyDict_New();
    if (result == NULL) {
        return NULL;
    }
    if (dict_set_owned(result, "hardware_available", PyBool_FromLong(available)) != 0
        || dict_set_owned(result, "standard_gzip", PyBool_FromLong(1)) != 0
        || dict_set_owned(result, "software_fallback_enabled", PyBool_FromLong(0)) != 0
        || dict_set_owned(result, "backend", PyUnicode_FromString("qat")) != 0
        || dict_set_owned(result, "binding_version", PyUnicode_FromString(VOLUME_TTA_QAT_BINDING_VERSION)) != 0
        || dict_set_owned(result, "library_version", PyUnicode_FromString(qatzip_version)) != 0
        || dict_set_owned(result, "driver_version", PyUnicode_FromString(driver_version)) != 0
        || dict_set_owned(result, "hardware_generation", PyUnicode_FromString("unknown")) != 0
        || dict_set_owned(
            result, "device_count",
            probe.status_populated && probe.status.qat_hw_count > 0
                ? PyLong_FromUnsignedLong(probe.status.qat_hw_count)
                : Py_NewRef(Py_None)) != 0
        || dict_set_owned(result, "instance_count", Py_NewRef(Py_None)) != 0
        || dict_set_owned(result, "max_concurrency", PyLong_FromUnsignedLong(capacity)) != 0
        || dict_set_owned(result, "minimum_input_bytes", PyLong_FromUnsignedLong(VOLUME_TTA_QAT_MIN_INPUT)) != 0
        || dict_set_owned(
            result, "supported_levels",
            Py_BuildValue("(iiiiiiii)", 1, 2, 3, 4, 5, 6, 7, 8)) != 0
        || dict_set_owned(result, "gzip_format", PyUnicode_FromString("QZ_DEFLATE_GZIP")) != 0
        || dict_set_owned(result, "numa_control", PyBool_FromLong(0)) != 0
        || dict_set_owned(result, "numa_policy", PyUnicode_FromString("qatzip-managed")) != 0
        || dict_set_owned(result, "physical_member_count_observable", PyBool_FromLong(0)) != 0
        || dict_set_owned(result, "queue_busy_observable", PyBool_FromLong(0)) != 0
        || dict_set_owned(result, "unavailable_reason", PyUnicode_FromString(unavailable_reason)) != 0
        || dict_set_owned(result, "qz_init_status", PyLong_FromLong(probe.init_rc)) != 0
        || dict_set_owned(result, "qz_defaults_status", PyLong_FromLong(probe.defaults_rc)) != 0
        || dict_set_owned(result, "qz_setup_status", PyLong_FromLong(probe.setup_rc)) != 0
        || dict_set_owned(result, "qz_status_status", PyLong_FromLong(probe.status_rc)) != 0
        || dict_set_owned(result, "session_hw_status", PyLong_FromLong(probe.session_hw_status)) != 0
        || dict_set_owned(result, "qz_status_populated", PyBool_FromLong(probe.status_populated)) != 0
        || dict_set_owned(
            result, "qz_hw_session_status",
            probe.status_populated
                ? PyLong_FromLong(probe.status.hw_session_status)
                : Py_NewRef(Py_None)) != 0
        || dict_set_owned(
            result, "deflate_device_count",
            probe.status_populated && probe.status.algo_hw[QZ_DEFLATE] > 0
                ? PyLong_FromUnsignedLong(probe.status.algo_hw[QZ_DEFLATE])
                : Py_NewRef(Py_None)) != 0
        || dict_set_owned(
            result, "rejected_ambiguous_levels",
            Py_BuildValue("(i)", 9)) != 0
        || dict_set_owned(
            result, "level_9_unavailable_reason",
            PyUnicode_FromString(
                "QATzip 1.3.2 cannot prove that level 9 avoided software on older QAT stacks")) != 0
        || dict_set_owned(result, "post_fork_child", PyBool_FromLong(forked_child)) != 0
        || dict_set_owned(result, "fork_child_reinitialization", PyBool_FromLong(0)) != 0) {
        Py_DECREF(result);
        return NULL;
    }
    return result;
}

static PyObject *qat_preflight_thread_state(
    PyObject *self,
    PyObject *args,
    PyObject *kwargs)
{
    static char *keywords[] = {"level", "require_hardware", "numa_id", NULL};
    int level;
    int require_hardware = 1;
    PyObject *numa_id = Py_None;
    QatThreadState *state;
    QzStatus_T status;
    int status_rc;
    (void)self;

    if (!PyArg_ParseTupleAndKeywords(
            args, kwargs, "i|$pO:preflight_thread_state", keywords,
            &level, &require_hardware, &numa_id)) {
        return NULL;
    }
    if (!require_hardware) {
        PyErr_SetString(
            PyExc_ValueError,
            "volume_tta._qat_codec is hardware-only; require_hardware must be true");
        return NULL;
    }
    if (validate_numa_argument(numa_id) != 0) {
        return NULL;
    }
    state = ensure_thread_session(level);
    if (state == NULL) {
        return NULL;
    }
    memset(&status, 0, sizeof(status));
    Py_BEGIN_ALLOW_THREADS
    status_rc = qzGetStatus(&state->session, &status);
    Py_END_ALLOW_THREADS
    if (!session_proves_hardware(&state->session)) {
        PyErr_Format(
            g_qatzip_error,
            "QATzip thread preflight lost hardware eligibility: "
            "session_hw_status=%s(%ld), qzGetStatus=%s(%d), "
            "reported_hw_session_status=%s(%ld), devices=%u, "
            "instance_attached=%u, deflate_devices=%u",
            qz_status_name((int)state->session.hw_session_stat),
            state->session.hw_session_stat,
            qz_status_name(status_rc), status_rc,
            qz_status_name((int)status.hw_session_status),
            status.hw_session_status,
            (unsigned int)status.qat_hw_count,
            (unsigned int)status.qat_instance_attach,
            (unsigned int)status.algo_hw[QZ_DEFLATE]);
        return NULL;
    }
    state->status = status;
    Py_RETURN_NONE;
}

static PyObject *qat_compress_gzip(
    PyObject *self,
    PyObject *args,
    PyObject *kwargs)
{
    static char *keywords[] = {
        "buffer", "level", "require_hardware", "numa_id", NULL
    };
    PyObject *input_object;
    PyObject *numa_id = Py_None;
    PyObject *output = NULL;
    Py_buffer input;
    const unsigned char *source_buffer;
    unsigned char *output_buffer;
    QatThreadState *state;
    QzStatus_T error_status;
    unsigned int source_length;
    unsigned int source_expected;
    unsigned int destination_length;
    unsigned int destination_capacity;
    uint64_t extended_status = 0;
    uint64_t compress_started_ns = 0;
    uint64_t compress_finished_ns = 0;
    uint64_t compress_elapsed_ns = 0;
    int level;
    int require_hardware = 1;
    int compress_rc;
    int status_rc = QZ_FAIL;
    int hardware_session;
    int software_execution;
    int timeout;
    int partial;
    int framing_invalid;
    int success;
    (void)self;

    memset(&input, 0, sizeof(input));
    memset(&error_status, 0, sizeof(error_status));
    if (!PyArg_ParseTupleAndKeywords(
            args, kwargs, "Oi|$pO:compress_gzip", keywords,
            &input_object, &level, &require_hardware, &numa_id)) {
        return NULL;
    }
    if (!require_hardware) {
        PyErr_SetString(
            PyExc_ValueError,
            "volume_tta._qat_codec is hardware-only; require_hardware must be true");
        return NULL;
    }
    if (validate_numa_argument(numa_id) != 0) {
        return NULL;
    }
    if (PyObject_GetBuffer(input_object, &input, PyBUF_CONTIG_RO) != 0) {
        return NULL;
    }
    if (input.ndim != 1) {
        PyErr_SetString(PyExc_BufferError, "QATzip input must be one-dimensional");
        goto error;
    }
    if (input.len < (Py_ssize_t)VOLUME_TTA_QAT_MIN_INPUT) {
        PyErr_Format(
            PyExc_ValueError,
            "QATzip hardware minimum input is %u bytes; received %zd",
            VOLUME_TTA_QAT_MIN_INPUT,
            input.len);
        goto error;
    }
    if ((uint64_t)input.len > (uint64_t)UINT_MAX) {
        PyErr_Format(
            PyExc_OverflowError,
            "QATzip one-shot input exceeds UINT_MAX bytes: %zd",
            input.len);
        goto error;
    }
    state = ensure_thread_session(level);
    if (state == NULL) {
        goto error;
    }
    source_expected = (unsigned int)input.len;
    source_length = source_expected;
    Py_BEGIN_ALLOW_THREADS
    destination_capacity = qzMaxCompressedLength(source_expected, &state->session);
    Py_END_ALLOW_THREADS
    if (destination_capacity < 18) {
        PyErr_Format(
            g_qatzip_error,
            "QATzip qzMaxCompressedLength returned invalid bound %u for %u input bytes",
            destination_capacity, source_expected);
        goto error;
    }
    if ((uint64_t)destination_capacity > (uint64_t)PY_SSIZE_T_MAX) {
        PyErr_SetString(PyExc_OverflowError, "QATzip output bound exceeds PY_SSIZE_T_MAX");
        goto error;
    }
    output = PyBytes_FromStringAndSize(NULL, (Py_ssize_t)destination_capacity);
    if (output == NULL) {
        goto error;
    }
    source_buffer = (const unsigned char *)input.buf;
    output_buffer = (unsigned char *)PyBytes_AS_STRING(output);
    destination_length = destination_capacity;
    Py_BEGIN_ALLOW_THREADS
    compress_started_ns = monotonic_ns();
    compress_rc = qzCompressExt(
        &state->session,
        source_buffer,
        &source_length,
        output_buffer,
        &destination_length,
        1,
        &extended_status);
    compress_finished_ns = monotonic_ns();
    Py_END_ALLOW_THREADS
    if (compress_started_ns != 0 && compress_finished_ns >= compress_started_ns) {
        compress_elapsed_ns = compress_finished_ns - compress_started_ns;
    }

    software_execution = (extended_status & QZ_SW_EXECUTION_MASK) != 0;
    hardware_session = session_proves_hardware(&state->session);
    timeout = compress_rc == QZ_TIMEOUT
        || (extended_status & QZ_TIMEOUT_MASK) != 0;
    partial = source_length != source_expected;
    framing_invalid = destination_length > destination_capacity
        || destination_length < 18
        || output_buffer[0] != 0x1f
        || output_buffer[1] != 0x8b
        || output_buffer[2] != 0x08
        || (output_buffer[3] & 0x04U) != 0;
    success = compress_rc == QZ_OK
        && hardware_session
        && !software_execution
        && !timeout
        && !partial
        && destination_length <= destination_capacity
        && !framing_invalid;
    stats_note_request(
        success,
        compress_rc == QZ_OK && hardware_session && !software_execution,
        software_execution,
        partial,
        timeout,
        compress_rc == QZ_BUF_ERROR,
        source_expected,
        destination_length,
        compress_elapsed_ns);

    if (!success) {
        Py_BEGIN_ALLOW_THREADS
        status_rc = qzGetStatus(&state->session, &error_status);
        Py_END_ALLOW_THREADS
        if (software_execution) {
            PyErr_Format(
                g_qatzip_error,
                "QATzip rejected software execution for a hardware-only request: "
                "status=%s(%d), ext_status=0x%llx, consumed=%u/%u, produced=%u, "
                "session_hw_status=%s(%ld), reported_hw_session_status=%s(%ld)",
                qz_status_name(compress_rc), compress_rc,
                (unsigned long long)extended_status,
                source_length, source_expected, destination_length,
                qz_status_name((int)state->session.hw_session_stat),
                state->session.hw_session_stat,
                qz_status_name((int)error_status.hw_session_status),
                error_status.hw_session_status);
        } else {
            PyErr_Format(
                g_qatzip_error,
                "QATzip qzCompressExt failed hardware proof: status=%s(%d), "
                "ext_status=0x%llx, consumed=%u/%u, produced=%u/%u, "
                "timeout=%d, framing_invalid=%d, session_hw_status=%s(%ld), "
                "qzGetStatus=%s(%d), reported_hw_session_status=%s(%ld), "
                "devices=%u, instance_attached=%u",
                qz_status_name(compress_rc), compress_rc,
                (unsigned long long)extended_status,
                source_length, source_expected,
                destination_length, destination_capacity,
                timeout, framing_invalid,
                qz_status_name((int)state->session.hw_session_stat),
                state->session.hw_session_stat,
                qz_status_name(status_rc), status_rc,
                qz_status_name((int)error_status.hw_session_status),
                error_status.hw_session_status,
                (unsigned int)error_status.qat_hw_count,
                (unsigned int)error_status.qat_instance_attach);
        }
        goto error;
    }
    PyBuffer_Release(&input);
    if (_PyBytes_Resize(&output, (Py_ssize_t)destination_length) != 0) {
        return NULL;
    }
    return output;

error:
    PyBuffer_Release(&input);
    Py_XDECREF(output);
    return NULL;
}

static PyObject *qat_stats(PyObject *self, PyObject *args, PyObject *kwargs)
{
    static char *keywords[] = {"reset", NULL};
    int reset = 0;
    QatStats snapshot;
    PyObject *result;
    (void)self;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|$p:stats", keywords, &reset)) {
        return NULL;
    }
    pthread_mutex_lock(&g_stats_lock);
    stats_reset_for_pid_locked(getpid());
    snapshot = g_stats;
    if (reset) {
        const uint64_t active = g_stats.active_sessions;
        memset(&g_stats, 0, sizeof(g_stats));
        g_stats.pid = getpid();
        g_stats.active_sessions = active;
        g_stats.peak_sessions = active;
    }
    pthread_mutex_unlock(&g_stats_lock);

    result = PyDict_New();
    if (result == NULL) {
        return NULL;
    }
    if (dict_set_owned(result, "logical_requests", PyLong_FromUnsignedLongLong(snapshot.logical_requests)) != 0
        || dict_set_owned(result, "hardware_requests", PyLong_FromUnsignedLongLong(snapshot.hardware_requests)) != 0
        || dict_set_owned(result, "software_fallback_requests", PyLong_FromUnsignedLongLong(snapshot.software_fallback_requests)) != 0
        || dict_set_owned(result, "input_bytes", PyLong_FromUnsignedLongLong(snapshot.input_bytes)) != 0
        || dict_set_owned(result, "output_bytes", PyLong_FromUnsignedLongLong(snapshot.output_bytes)) != 0
        || dict_set_owned(result, "failures", PyLong_FromUnsignedLongLong(snapshot.failures)) != 0
        || dict_set_owned(result, "partial_consumption_failures", PyLong_FromUnsignedLongLong(snapshot.partial_consumption_failures)) != 0
        || dict_set_owned(result, "timeouts", PyLong_FromUnsignedLongLong(snapshot.timeouts)) != 0
        || dict_set_owned(result, "buffer_errors", PyLong_FromUnsignedLongLong(snapshot.buffer_errors)) != 0
        || dict_set_owned(result, "sessions_created", PyLong_FromUnsignedLongLong(snapshot.sessions_created)) != 0
        || dict_set_owned(result, "sessions_closed", PyLong_FromUnsignedLongLong(snapshot.sessions_closed)) != 0
        || dict_set_owned(result, "session_creations", PyLong_FromUnsignedLongLong(snapshot.sessions_created)) != 0
        || dict_set_owned(result, "session_closes", PyLong_FromUnsignedLongLong(snapshot.sessions_closed)) != 0
        || dict_set_owned(result, "active_sessions", PyLong_FromUnsignedLongLong(snapshot.active_sessions)) != 0
        || dict_set_owned(result, "peak_sessions", PyLong_FromUnsignedLongLong(snapshot.peak_sessions)) != 0
        || dict_set_owned(result, "elapsed_ns", PyLong_FromUnsignedLongLong(snapshot.elapsed_ns)) != 0
        || dict_set_owned(result, "physical_members", Py_NewRef(Py_None)) != 0
        || dict_set_owned(result, "queue_busy_events", Py_NewRef(Py_None)) != 0) {
        Py_DECREF(result);
        return NULL;
    }
    return result;
}

static PyObject *qat_close_thread_state(PyObject *self, PyObject *Py_UNUSED(args))
{
    QatThreadState *state;
    int detach_rc;
    (void)self;
    if (ensure_session_key() != 0) {
        return NULL;
    }
    state = (QatThreadState *)pthread_getspecific(g_session_key);
    if (state != NULL) {
        detach_rc = pthread_setspecific(g_session_key, NULL);
        if (detach_rc != 0) {
            PyErr_Format(
                PyExc_RuntimeError,
                "failed to detach QATzip thread session: pthread status=%d",
                detach_rc);
            return NULL;
        }
        Py_BEGIN_ALLOW_THREADS
        destroy_thread_state(state, state->pid == getpid());
        Py_END_ALLOW_THREADS
    }
    Py_RETURN_NONE;
}

PyDoc_STRVAR(
    module_doc,
    "Optional Linux QATzip backend for volume_tta.\n\n"
    "The module never enables QATzip software fallback or latency-sensitive host "
    "selection. Native sessions are owned by the calling thread; admission requires "
    "a hardware-only session, and each request requires success, complete input "
    "consumption, no reported software/timeout bit, and standard gzip framing. "
    "Every native blocking operation releases the Python GIL.");

static PyMethodDef module_methods[] = {
    {
        "capabilities",
        (PyCFunction)qat_capabilities,
        METH_NOARGS,
        "Probe QATzip/QAT hardware without retaining a caller-thread session."
    },
    {
        "compress_gzip",
        (PyCFunction)(void(*)(void))qat_compress_gzip,
        METH_VARARGS | METH_KEYWORDS,
        "Compress one contiguous buffer to complete standard gzip members in hardware."
    },
    {
        "preflight_thread_state",
        (PyCFunction)(void(*)(void))qat_preflight_thread_state,
        METH_VARARGS | METH_KEYWORDS,
        "Initialize and validate the calling thread's hardware-only QATzip session."
    },
    {
        "stats",
        (PyCFunction)(void(*)(void))qat_stats,
        METH_VARARGS | METH_KEYWORDS,
        "Return process-local native request/session counters."
    },
    {
        "close_thread_state",
        (PyCFunction)qat_close_thread_state,
        METH_NOARGS,
        "Close the calling thread's QATzip session."
    },
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef module_definition = {
    PyModuleDef_HEAD_INIT,
    "_qat_codec",
    module_doc,
    -1,
    module_methods,
    NULL,
    NULL,
    NULL,
    NULL
};

PyMODINIT_FUNC PyInit__qat_codec(void)
{
    PyObject *module;
    PyObject *new_error;
    if (ensure_session_key() != 0 || ensure_atfork_registered() != 0) {
        return NULL;
    }
    module = PyModule_Create(&module_definition);
    if (module == NULL) {
        return NULL;
    }
    new_error = PyErr_NewException(
        "volume_tta._qat_codec.QATzipError", PyExc_RuntimeError, NULL);
    if (new_error == NULL) {
        Py_DECREF(module);
        return NULL;
    }
    if (PyModule_AddObjectRef(module, "QATzipError", new_error) != 0
        || PyModule_AddStringConstant(
            module, "__binding_version__", VOLUME_TTA_QAT_BINDING_VERSION) != 0) {
        Py_DECREF(new_error);
        Py_DECREF(module);
        return NULL;
    }
    Py_XDECREF(g_qatzip_error);
    g_qatzip_error = new_error;
    return module;
}
