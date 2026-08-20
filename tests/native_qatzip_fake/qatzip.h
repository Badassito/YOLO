#ifndef VOLUME_TTA_TEST_FAKE_QATZIP_H
#define VOLUME_TTA_TEST_FAKE_QATZIP_H

#include <stdint.h>

#define QATZIP_API_VERSION_NUM_MAJOR 2
#define QATZIP_API_VERSION_NUM_MINOR 5
#define QATZIP_API_VERSION 20500

#define QZ_OK 0
#define QZ_DUPLICATE 1
#define QZ_FORCE_SW 2
#define QZ_PARAMS (-1)
#define QZ_FAIL (-2)
#define QZ_BUF_ERROR (-3)
#define QZ_DATA_ERROR (-4)
#define QZ_TIMEOUT (-5)
#define QZ_INTEG (-100)
#define QZ_NO_HW 11
#define QZ_NO_MDRV 12
#define QZ_NO_INST_ATTACH 13
#define QZ_LOW_MEM 14
#define QZ_LOW_DEST_MEM 15
#define QZ_UNSUPPORTED_FMT 16
#define QZ_NONE 100
#define QZ_NOSW_NO_HW (-101)
#define QZ_NOSW_NO_MDRV (-102)
#define QZ_NOSW_NO_INST_ATTACH (-103)
#define QZ_NOSW_LOW_MEM (-104)
#define QZ_NO_SW_AVAIL (-105)
#define QZ_NOSW_UNSUPPORTED_FMT (-116)
#define QZ_POST_PROCESS_ERROR (-117)
#define QZ_METADATA_OVERFLOW (-118)
#define QZ_OUT_OF_RANGE (-119)
#define QZ_NOT_SUPPORTED (-200)

#define QZ_MAX_ALGORITHMS 255
#define QZ_DEFLATE ((unsigned char)8)
#define QZ_COMP_THRESHOLD_MINIMUM 128
#define QZ_DEFLATE_COMP_LVL_MINIMUM 1
#define QZ_DEFLATE_COMP_LVL_MAXIMUM 9
#define QZ_SW_EXECUTION_MASK (1U << 4)
#define QZ_TIMEOUT_MASK (1U << 8)
#define QZ_MAX_STRING_LENGTH 64

#define QZ_DISABLE_SOFTWARE_BACKUP(value) ((value) &= (unsigned char)~1U)
#define QZ_DISABLE_SOFTWARE_ONLY_EXECUTION(value) ((value) &= (unsigned char)~2U)

typedef enum {
    QZ_DIR_COMPRESS = 0,
    QZ_DIR_DECOMPRESS = 1,
    QZ_DIR_BOTH = 2
} QzDirection_T;

typedef enum {
    QZ_DEFLATE_4B = 0,
    QZ_DEFLATE_GZIP = 1,
    QZ_DEFLATE_GZIP_EXT = 2,
    QZ_DEFLATE_RAW = 3
} QzDataFormat_T;

typedef enum {
    QZ_DYNAMIC_HDR = 0,
    QZ_STATIC_HDR = 1
} QzHuffmanHdr_T;

typedef enum {
    QZ_PERIODICAL_POLLING = 0,
    QZ_BUSY_POLLING = 1
} QzPollingMode_T;

typedef enum {
    QZ_COMPONENT_FIRMWARE = 0,
    QZ_COMPONENT_KERNEL_DRIVER,
    QZ_COMPONENT_USER_DRIVER,
    QZ_COMPONENT_QATZIP_API,
    QZ_COMPONENT_SOFTWARE_PROVIDER
} QzSoftwareComponentType_T;

typedef struct {
    QzDirection_T direction;
    unsigned int comp_lvl;
    unsigned char comp_algorithm;
    unsigned int max_forks;
    unsigned char sw_backup;
    unsigned int hw_buff_sz;
    unsigned int strm_buff_sz;
    unsigned int input_sz_thrshold;
    unsigned int req_cnt_thrshold;
    unsigned int wait_cnt_thrshold;
    QzPollingMode_T polling_mode;
    unsigned int is_sensitive_mode;
} QzSessionParamsCommon_T;

typedef struct {
    QzSessionParamsCommon_T common_params;
    QzHuffmanHdr_T huffman_hdr;
    QzDataFormat_T data_fmt;
} QzSessionParamsDeflate_T;

typedef struct {
    signed long int hw_session_stat;
    int thd_sess_stat;
    void *internal;
    unsigned long total_in;
    unsigned long total_out;
} QzSession_T;

typedef struct {
    unsigned short int qat_hw_count;
    unsigned char qat_service_init;
    unsigned char qat_mem_drvr;
    unsigned char qat_instance_attach;
    unsigned long int memory_alloced;
    unsigned char using_huge_pages;
    signed long int hw_session_status;
    unsigned char algo_sw[QZ_MAX_ALGORITHMS];
    unsigned char algo_hw[QZ_MAX_ALGORITHMS];
} QzStatus_T;

typedef struct {
    QzSoftwareComponentType_T component_type;
    unsigned char component_name[QZ_MAX_STRING_LENGTH];
    unsigned int major_version;
    unsigned int minor_version;
    unsigned int patch_version;
    unsigned int build_number;
    unsigned char reserved[52];
} QzSoftwareVersionInfo_T;

int qzInit(QzSession_T *session, unsigned char sw_backup);
int qzGetDefaultsDeflate(QzSessionParamsDeflate_T *params);
int qzSetupSessionDeflate(
    QzSession_T *session,
    QzSessionParamsDeflate_T *params);
int qzGetStatus(QzSession_T *session, QzStatus_T *status);
unsigned int qzMaxCompressedLength(
    unsigned int source_size,
    QzSession_T *session);
int qzCompressExt(
    QzSession_T *session,
    const unsigned char *source,
    unsigned int *source_length,
    unsigned char *destination,
    unsigned int *destination_length,
    unsigned int last,
    uint64_t *extended_status);
int qzTeardownSession(QzSession_T *session);
int qzClose(QzSession_T *session);
int qzGetSoftwareComponentCount(unsigned int *count);
int qzGetSoftwareComponentVersionList(
    QzSoftwareVersionInfo_T *items,
    unsigned int *count);

#endif
