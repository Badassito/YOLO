#include "qatzip.h"

#include <limits.h>
#include <stdlib.h>
#include <string.h>

static const char *fake_mode(void)
{
    const char *value = getenv("XTA_FAKE_QAT_MODE");
    return value == NULL ? "hardware" : value;
}

static int mode_is(const char *expected)
{
    return strcmp(fake_mode(), expected) == 0;
}

static uint32_t fake_crc32(const unsigned char *source, unsigned int length)
{
    uint32_t crc = UINT32_C(0xffffffff);
    unsigned int index;
    int bit;
    for (index = 0; index < length; ++index) {
        crc ^= source[index];
        for (bit = 0; bit < 8; ++bit) {
            const uint32_t mask = (uint32_t)-(int32_t)(crc & 1U);
            crc = (crc >> 1) ^ (UINT32_C(0xedb88320) & mask);
        }
    }
    return ~crc;
}

static void put_u16_le(unsigned char *destination, uint16_t value)
{
    destination[0] = (unsigned char)(value & 0xffU);
    destination[1] = (unsigned char)((value >> 8) & 0xffU);
}

static void put_u32_le(unsigned char *destination, uint32_t value)
{
    destination[0] = (unsigned char)(value & 0xffU);
    destination[1] = (unsigned char)((value >> 8) & 0xffU);
    destination[2] = (unsigned char)((value >> 16) & 0xffU);
    destination[3] = (unsigned char)((value >> 24) & 0xffU);
}

static unsigned int fake_gzip_bound(unsigned int source_length)
{
    const uint64_t blocks = ((uint64_t)source_length + UINT64_C(65534))
        / UINT64_C(65535);
    const uint64_t bound = UINT64_C(18) + (uint64_t)source_length
        + blocks * UINT64_C(5);
    return bound > UINT_MAX ? 0U : (unsigned int)bound;
}

static int emit_stored_gzip(
    const unsigned char *source,
    unsigned int source_length,
    unsigned char *destination,
    unsigned int *destination_length)
{
    unsigned int source_offset = 0;
    unsigned int destination_offset = 0;
    const unsigned int required = fake_gzip_bound(source_length);
    if (required == 0 || destination == NULL || destination_length == NULL
        || *destination_length < required) {
        return QZ_BUF_ERROR;
    }

    destination[destination_offset++] = 0x1f;
    destination[destination_offset++] = 0x8b;
    destination[destination_offset++] = 0x08;
    destination[destination_offset++] = 0x00;
    memset(destination + destination_offset, 0, 6);
    destination_offset += 6;

    while (source_offset < source_length) {
        const unsigned int remaining = source_length - source_offset;
        const uint16_t block_length = (uint16_t)(
            remaining > 65535U ? 65535U : remaining);
        const int final_block = source_offset + block_length == source_length;
        destination[destination_offset++] = final_block ? 0x01 : 0x00;
        put_u16_le(destination + destination_offset, block_length);
        destination_offset += 2;
        put_u16_le(
            destination + destination_offset,
            (uint16_t)~block_length);
        destination_offset += 2;
        memcpy(
            destination + destination_offset,
            source + source_offset,
            block_length);
        destination_offset += block_length;
        source_offset += block_length;
    }

    put_u32_le(
        destination + destination_offset,
        fake_crc32(source, source_length));
    destination_offset += 4;
    put_u32_le(destination + destination_offset, source_length);
    destination_offset += 4;
    *destination_length = destination_offset;
    return QZ_OK;
}

int qzInit(QzSession_T *session, unsigned char sw_backup)
{
    if (session == NULL || sw_backup != 0) {
        return QZ_PARAMS;
    }
    if (mode_is("no_hardware")) {
        session->hw_session_stat = QZ_NOSW_NO_HW;
        return QZ_NOSW_NO_HW;
    }
    return QZ_OK;
}

int qzGetDefaultsDeflate(QzSessionParamsDeflate_T *params)
{
    if (params == NULL) {
        return QZ_PARAMS;
    }
    memset(params, 0, sizeof(*params));
    params->common_params.direction = QZ_DIR_BOTH;
    params->common_params.comp_lvl = 1;
    params->common_params.comp_algorithm = QZ_DEFLATE;
    params->common_params.max_forks = 3;
    params->common_params.sw_backup = 1;
    params->common_params.hw_buff_sz = 64U * 1024U;
    params->common_params.strm_buff_sz = 64U * 1024U;
    params->common_params.input_sz_thrshold = 1024;
    params->common_params.req_cnt_thrshold = 1;
    params->common_params.wait_cnt_thrshold = 8;
    params->data_fmt = QZ_DEFLATE_GZIP_EXT;
    return mode_is("defaults_error") ? QZ_FAIL : QZ_OK;
}

int qzSetupSessionDeflate(
    QzSession_T *session,
    QzSessionParamsDeflate_T *params)
{
    if (session == NULL || params == NULL) {
        return QZ_PARAMS;
    }
    if (mode_is("setup_error")) {
        session->hw_session_stat = QZ_NOSW_NO_INST_ATTACH;
        return QZ_NOSW_NO_INST_ATTACH;
    }
    if (params->common_params.sw_backup != 0
        || params->common_params.is_sensitive_mode != 0
        || params->common_params.input_sz_thrshold != QZ_COMP_THRESHOLD_MINIMUM
        || params->common_params.direction != QZ_DIR_COMPRESS
        || params->common_params.comp_algorithm != QZ_DEFLATE
        || params->data_fmt != QZ_DEFLATE_GZIP) {
        return QZ_PARAMS;
    }
    session->internal = session;
    session->hw_session_stat = QZ_OK;
    return QZ_OK;
}

int qzGetStatus(QzSession_T *session, QzStatus_T *status)
{
    if (session == NULL || status == NULL) {
        return QZ_PARAMS;
    }
    /* Default behavior intentionally matches QATzip 1.3.2: QZ_OK without
     * populating the documented structure. */
    if (mode_is("populated_status")) {
        memset(status, 0, sizeof(*status));
        status->qat_hw_count = 2;
        status->qat_service_init = 1;
        status->qat_mem_drvr = 2;
        status->qat_instance_attach = 1;
        status->hw_session_status = session->hw_session_stat;
        status->algo_hw[QZ_DEFLATE] = 2;
    }
    return mode_is("status_error") ? QZ_FAIL : QZ_OK;
}

unsigned int qzMaxCompressedLength(
    unsigned int source_size,
    QzSession_T *session)
{
    if (mode_is("bad_bound")) {
        return 1;
    }
    return session == NULL ? 0U : fake_gzip_bound(source_size);
}

int qzCompressExt(
    QzSession_T *session,
    const unsigned char *source,
    unsigned int *source_length,
    unsigned char *destination,
    unsigned int *destination_length,
    unsigned int last,
    uint64_t *extended_status)
{
    unsigned int consumed;
    int rc;
    if (session == NULL || source == NULL || source_length == NULL
        || destination == NULL || destination_length == NULL
        || extended_status == NULL || last != 1) {
        return QZ_PARAMS;
    }
    *extended_status = 0;
    if (mode_is("timeout")) {
        *source_length = 0;
        *destination_length = 0;
        *extended_status = QZ_TIMEOUT_MASK;
        return QZ_TIMEOUT;
    }
    if (mode_is("failure")) {
        *source_length = 0;
        *destination_length = 0;
        return QZ_FAIL;
    }
    consumed = mode_is("partial") ? *source_length / 2U : *source_length;
    rc = emit_stored_gzip(source, consumed, destination, destination_length);
    if (rc != QZ_OK) {
        *source_length = 0;
        return rc;
    }
    *source_length = consumed;
    if (mode_is("software")) {
        *extended_status = QZ_SW_EXECUTION_MASK;
    } else if (mode_is("bad_framing")) {
        destination[0] = 0;
    } else if (mode_is("extended_gzip")) {
        destination[3] |= 0x04U;
    } else if (mode_is("lost_session")) {
        session->hw_session_stat = QZ_NO_INST_ATTACH;
    }
    session->total_in += consumed;
    session->total_out += *destination_length;
    return QZ_OK;
}

int qzTeardownSession(QzSession_T *session)
{
    if (session == NULL) {
        return QZ_PARAMS;
    }
    session->internal = NULL;
    return QZ_OK;
}

int qzClose(QzSession_T *session)
{
    return session == NULL ? QZ_PARAMS : QZ_OK;
}

int qzGetSoftwareComponentCount(unsigned int *count)
{
    if (count != NULL) {
        *count = 0;
    }
    return QZ_FAIL;
}

int qzGetSoftwareComponentVersionList(
    QzSoftwareVersionInfo_T *items,
    unsigned int *count)
{
    (void)items;
    (void)count;
    return QZ_FAIL;
}
