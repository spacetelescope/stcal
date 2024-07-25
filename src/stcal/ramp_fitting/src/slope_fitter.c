#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>

#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <stdbool.h>
#include <string.h>

#include <numpy/arrayobject.h>
#include <numpy/npy_math.h>


/*
To build C code, make sure the setup.py file is correct and
lists all extensions, then run:

python setup.py build_ext --inplace

        or 

pip install -e .
 */

/* ========================================================================= */
/*                               TYPEDEFs                                    */
/* ------------------------------------------------------------------------- */

/* 
 * Toggle internal arrays from float to doubles.  The REAL_IS_DOUBLE is used
 * for preprocessor switches in the code.  If a developer prefers to use a
 * float for internal arrays, this macro can be set to zero to switch from
 * double to float.
 */
#define REAL_IS_DOUBLE 1
#if REAL_IS_DOUBLE
typedef double real_t;
#else
typedef float real_t;
#endif

/* for weighted or unweighted OLS */
typedef enum {
    WEIGHTED,
    UNWEIGHTED,
} weight_t;

/* ------------------------------------------------------------------------- */

/* ========================================================================= */
/*                               GLOBALS                                     */
/* ------------------------------------------------------------------------- */

/* This is mostly used for debugging, but could have other usefulness. */
static npy_intp current_integration;

/* 
 * Deals with invalid data.  This is one of the ways the python code dealt with
 * the limitations of numpy that aren't necessary in this code.  The LARGE_VARIANCE
 * variable has been removed from use in this code, but due to some strange,
 * non-flagged data that still requires the use of LARGE_VARIANCE_THRESHOLD , but
 * shouldn't.  I think that strange data should have been flagged in previous steps
 * but I don't think that has happened.
 */
const real_t LARGE_VARIANCE = 1.e8;
const real_t LARGE_VARIANCE_THRESHOLD = 1.e6;
/* ------------------------------------------------------------------------- */


/* ========================================================================= */
/*                                MACROS                                     */
/* ------------------------------------------------------------------------- */

/* Formatting to make printing more uniform. */
#define DBL "16.10f"

/* Is more general and non-type dependent. */
#define BSWAP32(X) ((((X) & 0xff000000) >> 24) | \
                    (((X) & 0x00ff0000) >> 8) |  \
                    (((X) & 0x0000ff00) << 8) |  \
                    (((X) & 0x000000ff) << 24))

/* Pointers should be set to NULL once freed. */
#define SET_FREE(X) if (X) {free(X); (X) = NULL;}

/* 
 * Wraps the clean_ramp_data function.  Ensure all allocated
 * memory gets deallocated properly for the ramp_data data
 * structure, as well as the allocation allocation for the
 * data structure itself.
 */
#define FREE_RAMP_DATA(RD) \
    if (RD) { \
        clean_ramp_data(rd); \
        free(RD); \
        (RD) = NULL; \
    }

/* 
 * Wraps the clean_pixel_ramp function.  Ensure all allocated
 * memory gets deallocated properly for the pixel_ramp data
 * structure, as well as the allocation allocation for the
 * data structure itself.
 */
#define FREE_PIXEL_RAMP(PR) \
    if (PR) { \
        clean_pixel_ramp(PR); \
        SET_FREE(PR); \
    }

/* 
 * Wraps the clean_segment_list function.  Ensure all allocated
 * memory gets deallocated properly for the segment_list data
 * structure, as well as the allocation allocation for the
 * data structure itself.
 */
#define FREE_SEGS_LIST(N, S) \
    if (S) { \
        clean_segment_list(N, S); \
        SET_FREE(S);\
    }

/* Complicated dereferencing and casting using a label. */
#define VOID_2_FLOAT(A) (*((float*)(A)))
#define VOID_2_REAL(A) (*((real_t*)(A)))
#define VOID_2_U32(A) (*((uint32_t*)(A)))
#define VOID_2_U8(A) (*((uint8_t*)(A)))

/* Print macros to include meta information about the print statement */
#define ols_base_print(F,L,...) \
    do { \
        fprintf(F, "%s - [C:%d] ", L, __LINE__); \
        fprintf(F, __VA_ARGS__); \
    } while(0)
#define dbg_ols_print(...) ols_base_print(stdout, "Debug", __VA_ARGS__)
#define err_ols_print(...) ols_base_print(stderr, "Error", __VA_ARGS__)

#define dbg_ols_print_pixel(PR) \
    printf("[C:%d] Pixel (%ld, %ld)\n", __LINE__, (PR)->row, (PR)->col)

/* ------------------------------------------------------------------------- */

/* ========================================================================= */
/*                           Data Structuress                                */
/* ------------------------------------------------------------------------- */

/*
 * Mirrors the RampData class defined in ramp_data_class.py.
 */
struct ramp_data {
    /* The dimensions for the ramp data */
    npy_intp nints;     /* The number of integrations */
    npy_intp ngroups;   /* The number of groups per integration */
    npy_intp nrows;     /* The number of rows of an image */
    npy_intp ncols;     /* The number of columns of an image */

    ssize_t cube_sz;   /* The size of an integration cube */
    ssize_t image_sz;  /* The size of an image */
    ssize_t ramp_sz;   /* The size of a pixel ramp */

    /* Functions to get the proper data. */
    float (*get_data)(PyArrayObject*, npy_intp, npy_intp, npy_intp, npy_intp);
    float (*get_err)(PyArrayObject*, npy_intp, npy_intp, npy_intp, npy_intp);
    uint32_t (*get_pixeldq)(PyArrayObject*, npy_intp, npy_intp);
    float (*get_gain)(PyArrayObject*, npy_intp, npy_intp);
    float (*get_rnoise)(PyArrayObject*, npy_intp, npy_intp);
    float (*get_zframe)(PyArrayObject*, npy_intp, npy_intp, npy_intp);
    float (*get_dcurrent)(PyArrayObject*, npy_intp, npy_intp);

    /* The 4-D arrays with dimensions (nints, ngroups, nrows, ncols) */
    PyArrayObject * data;       /* The 4-D science data */
    PyArrayObject * err;        /* The 4-D err data */
    PyArrayObject * groupdq;    /* The 4-D group DQ array */

    /* The 2-D arrays with dimensions (nrows, ncols) */
    PyArrayObject * pixeldq;    /* The 2-D pixel DQ array */
    PyArrayObject * gain;       /* The 2-D gain array */
    PyArrayObject * rnoise ;    /* The 2-D read noise array */
    PyArrayObject * dcurrent;   /* The 2-D average dark current array */
    PyArrayObject * zframe;     /* The 2-D ZEROFRAME array */

    int special1;   /* Count of segments of length 1 */
    int special2;   /* Count of segments of length 2 */

    /*
     * Group and Pixel flags: 
     * DO_NOT USE, JUMP_DET, SATURATED, NO_GAIN_VALUE, UNRELIABLE_SLOPE
     */
    uint32_t dnu, jump, sat, ngval, uslope, invalid;

    /* 
     * This is used only if the save_opt is non-zero, i.e., the option to
     * save the optional results product must be turned on.
     *
     * Optional results stuff.  The double pointer will be a pointer to a
     * cube array with dimensions (nints, nrows, ncols).  The elements of
     * the array will be the list of segments.  The max_num_segs will be
     * used by the final optional results product which will have dimensions
     * (nints, max_num_segs, nrows, ncols).
     */

    int save_opt;                   /* Save optional results value */
    int max_num_segs;               /* Max number of segments over all ramps. */
    struct simple_ll_node ** segs;  /* The segment list for each ramp. */
    real_t * pedestal;              /* The pedestal computed for each ramp. */

    /* Meta data */
    uint32_t suppress_one_group;    /* Boolean to suppress one group */
    real_t frame_time;              /* The frame time */
    real_t group_time;              /* The group time */
    int dropframes;                 /* The number of dropped frames in an integration */
    int groupgap;                   /* The group gap */
    int nframes;                    /* The number of frames */
    real_t ped_tmp;                 /* Intermediate pedestal caclulation */
    int suppress1g;                 /* Suppress one group ramps */
    real_t effintim;                /* Effective integration time */
    real_t one_group_time;          /* Time for ramps with only 0th good group */
    weight_t weight;                /* The weighting for OLS */
}; /* END: struct ramp_data */

/*
 * The ramp fit for a specific pixel.
 */
struct pixel_fit {
    real_t slope;       /* Computed slope */
    uint32_t dq;        /* Pixel DQ */
    real_t var_poisson; /* Poisson variance */
    real_t var_rnoise;  /* Read noise variance */
    real_t var_err;     /* Total variance */
}; /* END: struct pixel_fit */

/*
 * The segment information of an integration ramp is kept track of
 * using a simple linked list detailing the beginning groups and end
 * group.  The end group is NOT part of the segment.
 *
 * Note: If there is a maximum number of groups, this could be implemented as
 * an array, instead of a linked list.  Linked lists are more flexible, but are
 * require better memory management.
 */
struct simple_ll_node {
    struct simple_ll_node * flink;  /* The forward link */
    npy_intp start;                 /* The start group */
    npy_intp end;                   /* The end group */
    ssize_t length;                 /* The end group */

    /* The computed values of the segment */
    real_t slope;        /* Slope of segment */
    real_t sigslope;     /* Uncertainty in the segment slope */
    real_t var_p;        /* Poisson variance */
    real_t var_r;        /* Readnoise variance */
    real_t var_e;        /* Total variance */
    real_t yint;         /* Y-intercept */
    real_t sigyint;      /* Uncertainty in the Y-intercept */
    real_t weight;       /* Sum of weights */
}; /* END: struct simple_ll_node */

/*
 * The list of segments in an integration ramp.  The segments form the basis
 * for computation of each ramp for ramp fitting.
 */
struct segment_list {
    struct simple_ll_node * head;   /* The head node of the list */
    struct simple_ll_node * tail;   /* The tail node of the list */
    ssize_t size;                   /* The number of segments */
    npy_intp max_segment_length;    /* The max group length of a segment */
}; /* END: struct segment_list */

/*
 * For each integration, count how many groups had certain flags set.
 */
struct integ_gdq_stats {
    int cnt_sat;        /* SATURATED count */
    int cnt_dnu;        /* DO_NOT_USE count */
    int cnt_dnu_sat;    /* SATURATED | DO_NOT_USE count */
    int cnt_good;       /* GOOD count */
    int jump_det;       /* Boolean for JUMP_DET */
}; /* END: struct integ_gdq_stats */

/*
 * This contains all the information to ramp fit a specific pixel.
 */
struct pixel_ramp {
    npy_intp row;     /* The pixel row and column */
    npy_intp col;     /* The pixel row and column */
    npy_intp nints;   /* The number of integrations and groups per integration */
    npy_intp ngroups; /* The number of integrations and groups per integration */
    ssize_t ramp_sz;  /* The total size of the 2-D arrays */

    real_t * data;      /* The 2-D ramp data (nints, ngroups) */
    uint32_t * groupdq; /* The group DQ pixel array */

    uint32_t pixeldq; /* The pixel DQ pixel */
    real_t gain;      /* The pixel gain */
    real_t rnoise ;   /* The pixel read noise */
    real_t zframe;    /* The pixel ZEROFRAME */
    real_t dcurrent;  /* The pixel average dark current */

    /* Timing bool */
    uint8_t * is_zframe;    /* Boolean to use ZEROFRAME */
    uint8_t * is_0th;       /* Boolean to use zeroeth group timing*/

    /* C computed values */
    real_t median_rate;  /* The median rate of the pixel */
    real_t invvar_e_sum; /* Intermediate calculation needed for final slope */

    /* This needs to be an array for each integration */
    ssize_t max_num_segs;           /* Max number of segments in an integration */
    struct segment_list * segs;     /* Array of integration segments */

    struct integ_gdq_stats * stats; /* Array of integration GDQ stats */

    /* initialize and clean */
    struct pixel_fit rate;          /* Image information */
    struct pixel_fit * rateints;    /* Cube information */
}; /* END: struct pixel_ramp */

/*
 * Intermediate calculations for least squares.
 */
struct ols_calcs {
    real_t sumx, sumxx, sumy, sumxy, sumw;
}; /* END: struct ols_calcs */

/*
 * The rate product data structure.
 */
struct rate_product {
    int is_none;
    PyArrayObject * slope;          /* Slopes */
    PyArrayObject * dq;             /* Data quality */
    PyArrayObject * var_poisson;    /* Poisson variance */
    PyArrayObject * var_rnoise;     /* Read noise variance */
    PyArrayObject * var_err;        /* Total variance */
}; /* END: struct rate_product */

/*
 * The rateints product data structure.
 */
struct rateint_product {
    int is_none;
    PyArrayObject * slope;          /* Slopes */
    PyArrayObject * dq;             /* Data quality */
    PyArrayObject * var_poisson;    /* Poisson variance */
    PyArrayObject * var_rnoise;     /* Read noise variance */
    PyArrayObject * var_err;        /* Total variance */
}; /* END: struct rateint_product */

/*
 * The optional results product data structure.
 */
struct opt_res_product {
    PyArrayObject * slope;      /* Slope of segment */
    PyArrayObject * sigslope;   /* Uncertainty in the segment slope */

    PyArrayObject * var_p;      /* Poisson variance */
    PyArrayObject * var_r;      /* Readnoise variance */

    PyArrayObject * yint;       /* Y-intercept */
    PyArrayObject * sigyint;    /* Uncertainty in the Y-intercept */

    PyArrayObject * pedestal;   /* Pedestal */
    PyArrayObject * weights;    /* Weights */

    PyArrayObject * cr_mag;     /* Cosmic ray magnitudes */
}; /* END: struct opt_res_product */

/* ------------------------------------------------------------------------- */

/* ========================================================================= */
/*                              Prototypes                                   */
/* ------------------------------------------------------------------------- */

/* ------------------------------------------------------------------------- */
/*                            Worker Functions                               */
/* ------------------------------------------------------------------------- */
static int
add_segment_to_list(struct segment_list * segs, npy_intp start, npy_intp end);

static PyObject *
build_opt_res(struct ramp_data * rd);

static void
clean_pixel_ramp(struct pixel_ramp * pr);

static void
clean_ramp_data(struct ramp_data * pr);

static void
clean_rate_product(struct rate_product * rate_prod);

static void
clean_rateint_product(struct rateint_product * rateint_prod);

static void
clean_segment_list(npy_intp nints, struct segment_list * segs);

static int
compute_integration_segments(
    struct ramp_data * rd, struct pixel_ramp * pr, npy_intp integ);

static int
create_opt_res(struct opt_res_product * opt_res, struct ramp_data * rd);

static struct pixel_ramp *
create_pixel_ramp(struct ramp_data * rd);

static int
create_rate_product(struct rate_product * rate_prod, struct ramp_data * rd);

static int
create_rateint_product(struct rateint_product * rateint_prod, struct ramp_data * rd);

static float
get_float2(PyArrayObject * obj, npy_intp row,  npy_intp col);

static float
get_float4(
    PyArrayObject * obj, npy_intp integ, npy_intp group, npy_intp row, npy_intp col);

static float
get_float3(
    PyArrayObject * obj, npy_intp integ, npy_intp row, npy_intp col);

static uint32_t
get_uint32_2(
    PyArrayObject * obj, npy_intp row, npy_intp col);

static void
get_pixel_ramp(
    struct pixel_ramp * pr, struct ramp_data * rd, npy_intp row, npy_intp col);

static void
get_pixel_ramp_integration(
    struct pixel_ramp * pr, struct ramp_data * rd,
    npy_intp row, npy_intp col, npy_intp integ, npy_intp group, npy_intp idx);

static void
get_pixel_ramp_meta(
    struct pixel_ramp * pr, struct ramp_data * rd, npy_intp row, npy_intp col);

static void
get_pixel_ramp_zero(struct pixel_ramp * pr);

static void
get_pixel_ramp_integration_segments_and_pedestal(
    npy_intp integ, struct pixel_ramp * pr, struct ramp_data * rd);

static struct ramp_data *
get_ramp_data(PyObject * args);

static int
get_ramp_data_arrays(PyObject * Py_ramp_data, struct ramp_data * rd);

static void
get_ramp_data_meta(PyObject * Py_ramp_data, struct ramp_data * rd);

static int
get_ramp_data_parse(PyObject ** Py_ramp_data, struct ramp_data * rd, PyObject * args);

static int
get_ramp_data_new_validate(struct ramp_data * rd);

static void
get_ramp_data_dimensions(struct ramp_data * rd);

static void
get_ramp_data_getters(struct ramp_data * rd);

static int
compute_median_rate(struct ramp_data * rd, struct pixel_ramp * pr);

static int
median_rate_1ngroup(struct ramp_data * rd, struct pixel_ramp * pr);

static int
median_rate_default(struct ramp_data * rd, struct pixel_ramp * pr);

static real_t *
median_rate_get_data (
    real_t * data, npy_intp integ, struct ramp_data * rd, struct pixel_ramp * pr);

static uint8_t *
median_rate_get_dq (
    uint8_t * data, npy_intp integ, struct ramp_data * rd, struct pixel_ramp * pr);

static int
median_rate_integration(
    real_t * mrate, real_t * int_data, uint8_t * int_dq,
    struct ramp_data * rd, struct pixel_ramp * pr);

static int
median_rate_integration_sort(
    real_t * loc_integ, uint8_t * int_dq,
    struct ramp_data * rd, struct pixel_ramp * pr);

static int
median_rate_integration_sort_cmp(const void * aa, const void * bb);

static int
ols_slope_fit_pixels(
    struct ramp_data * rd, struct pixel_ramp * pr,
    struct rate_product * rate_prod, struct rateint_product * rateint_prod);

static PyObject *
package_results(
    struct rate_product * rate, struct rateint_product * rateints,
    struct ramp_data * rd);

static void
prune_segment_list(struct segment_list * segs);

static float
py_ramp_data_get_float(PyObject * rd, const char * attr);

static int
py_ramp_data_get_int(PyObject * rd, const char * attr);

static int
ramp_fit_pixel(struct ramp_data * rd, struct pixel_ramp * pr);

static int
ramp_fit_pixel_integration(
    struct ramp_data * rd, struct pixel_ramp * pr, npy_intp integ);

static int
ramp_fit_pixel_integration_fit_slope(
    struct ramp_data * rd, struct pixel_ramp * pr, npy_intp integ);

static int
ramp_fit_pixel_integration_fit_slope_seg(
    struct simple_ll_node * current,
    struct ramp_data * rd, struct pixel_ramp * pr,
    npy_intp integ, int segnum);

static int
ramp_fit_pixel_integration_fit_slope_seg_default(
    struct ramp_data * rd, struct pixel_ramp * pr, struct simple_ll_node * seg,
    npy_intp integ, int segnum);

static int
ramp_fit_pixel_integration_fit_slope_seg_len1(
    struct ramp_data * rd, struct pixel_ramp * pr, struct simple_ll_node * seg,
    npy_intp integ, int segnum);

static int
ramp_fit_pixel_integration_fit_slope_seg_len2(
    struct ramp_data * rd, struct pixel_ramp * pr, struct simple_ll_node * seg,
    npy_intp integ, int segnum);

static void
ramp_fit_pixel_integration_fit_slope_seg_default_weighted(
    struct ramp_data * rd, struct pixel_ramp * pr, struct simple_ll_node * seg,
    npy_intp integ, int segnum, real_t power);

static void
ramp_fit_pixel_integration_fit_slope_seg_default_weighted_ols(
    struct ramp_data * rd, struct pixel_ramp * pr, struct simple_ll_node * seg,
    struct ols_calcs * ols, npy_intp integ, int segnum, real_t power);

static void
ramp_fit_pixel_integration_fit_slope_seg_default_weighted_seg(
    struct ramp_data * rd, struct pixel_ramp * pr, struct simple_ll_node * seg,
    struct ols_calcs * ols, npy_intp integ, int segnum, real_t power);

static real_t
real_nan_median(real_t * arr, npy_intp len);

static int
save_opt_res(struct opt_res_product * opt_res, struct ramp_data * rd);

static int
save_ramp_fit(struct rateint_product * rateint_prod, struct rate_product * rate_prod,
    struct pixel_ramp * pr);

static int
segment_snr(
    real_t * snr, npy_intp integ, struct ramp_data * rd,
    struct pixel_ramp * pr, struct simple_ll_node * seg, int segnum);

static real_t
snr_power(real_t snr);
/* ------------------------------------------------------------------------- */

/* ------------------------------------------------------------------------- */
/*                            Debug Functions                                */
/* ------------------------------------------------------------------------- */
static void
print_real_array(char * label, real_t * arr, int len, int ret, int line);

static void
print_intp_array(npy_intp * arr, int len, int ret);

static void
print_npy_types();

static void
print_ols_calcs(struct ols_calcs * ols, npy_intp integ, int segnum, int line);

static void
print_pixel_ramp_data(struct ramp_data * rd, struct pixel_ramp * pr, int line);

static void
print_pixel_ramp_dq(struct ramp_data * rd, struct pixel_ramp * pr, int line);

static void
print_pixel_ramp_info(struct ramp_data * rd, struct pixel_ramp * pr, int line);

static void
print_pixel_ramp_stats(struct pixel_ramp * pr, int line);

static void
print_PyArrayObject_info(PyArrayObject * obj);

static void
print_ramp_data_info(struct ramp_data * rd);

static void
print_ramp_data_types(struct ramp_data * rd, int line);

static void
print_rd_type_info(struct ramp_data * rd);

static void
print_segment_list(npy_intp nints, struct segment_list * segs, int line);

static void
print_segment_list_integ(npy_intp integ, struct segment_list * segs, int line);

static void
print_segment(
    struct simple_ll_node * seg, struct ramp_data * rd, struct pixel_ramp * pr,
    npy_intp integ, int segnum, int line);

static void
print_segment_opt_res(
    struct simple_ll_node * seg, struct ramp_data * rd,
    npy_intp integ, int segnum, int line);

static void
print_stats(struct pixel_ramp * pr, npy_intp integ, int line); 

static void
print_uint8_array(uint8_t * arr, int len, int ret, int line);

static void
print_uint32_array(uint32_t * arr, int len, int ret);
/* ========================================================================= */

/* ========================================================================= */
/*                        Static Inline Functions                            */
/* ------------------------------------------------------------------------- */

/* Translate 2-D (integ, group) to a 1-D index. */
static inline npy_intp
get_ramp_index(struct ramp_data * rd, npy_intp integ, npy_intp group) {
    return rd->ngroups * integ + group;
}

/* Translate 3-D (integ, row, col) to a 1-D index. */
static inline npy_intp
get_cube_index(struct ramp_data * rd, npy_intp integ, npy_intp row, npy_intp col) {
    return rd->image_sz * integ + rd->ncols * row + col;
}

/* Print a line delimiter for visual separation.  Used for debugging. */
static inline void
print_delim() {
    int k;
    const char c = '-';
    for (k=0; k<80; ++k) {
        printf("%c", c);
    }
    printf("\n");
}

/* Print a line delimiter for visual separation.  Used for debugging. */
static inline void
print_delim_char(char c, int len) {
    int k;
    for (k=0; k<len; ++k) {
        printf("%c", c);
    }
    printf("\n");
}

/* Used for debugging to determine if a pixel is in a list */
static inline int
is_pix_in_list(struct pixel_ramp * pr)
{
    // (1014, 422),
    const int len = 1;
    npy_intp rows[len];
    npy_intp cols[len];
    int k;

    rows[0] = 1014;
    cols[0] = 422;

    for (k=0; k<len; ++k) {
        if (pr->row==rows[k] && pr->col==cols[k]) {
            return 1;
        }
    }
    return 0;
}

/* ------------------------------------------------------------------------- */
/*                            Module Functions                                */
/* ------------------------------------------------------------------------- */

/*
 * This is the entry point into the C extension for ramp fitting.  It gets the
 * ramp meta data and arrays from the python RampData class, along with the
 * gain, read noise, weighting, and save optional results value flag.
 *
 * It create the output classes to be returned from ramp fitting.
 *
 * Fits each ramp, then saves the results in the output classes.
 *
 * MAIN
 */
static PyObject *
ols_slope_fitter(
        PyObject * module,  /* The ramp fitting module for the C extension. */
        PyObject * args)    /* The arguments for the C extension */
{
    PyObject * result = Py_None;
    struct ramp_data * rd = NULL;
    struct pixel_ramp * pr = NULL;
    struct rate_product rate_prod = {0};
    struct rateint_product rateint_prod = {0};

    /* Allocate, fill, and validate ramp data */
    rd = get_ramp_data(args);
    if (NULL == rd) { 
        goto ERROR;
    }

    /* Prepare output products */
    if (create_rate_product(&rate_prod, rd) ||
        create_rateint_product(&rateint_prod, rd))
    {
        goto ERROR;
    }

    /* Prepare the pixel ramp data structure */
    pr = create_pixel_ramp(rd);
    if (NULL==pr) {
        goto ERROR;
    }

    /* Fit ramps for each pixel */
    if (ols_slope_fit_pixels(rd, pr, &rate_prod, &rateint_prod)) {
        goto ERROR;
    }

    /* Package up results to be returned */
    result = package_results(&rate_prod, &rateint_prod, rd);
    if ((NULL==result) || (Py_None==(PyObject*)result)) {
        goto ERROR;
    }

    goto CLEANUP;
ERROR:
    Py_XDECREF(result);

    /* Clean up errors */
    clean_rate_product(&rate_prod);
    clean_rateint_product(&rateint_prod);

    /* Return (None, None, None) */
    result = Py_BuildValue("(NNN)", Py_None, Py_None, Py_None);

CLEANUP:
    FREE_RAMP_DATA(rd);
    FREE_PIXEL_RAMP(pr);

    return result;
}

/* ------------------------------------------------------------------------- */

/* ========================================================================= */
/*                         Prototypes Definitions                            */
/* ------------------------------------------------------------------------- */

/* ------------------------------------------------------------------------- */
/*                            Worker Functions                               */
/* ------------------------------------------------------------------------- */

/*
 * Add a segment to the segment list for the ramp.  A linked list is used to
 * keep track of the segments for a ramp.
 */
static int
add_segment_to_list(
        struct segment_list * segs, /* The list to add the segment to. */
        npy_intp start,             /* The start, inclusive, of the segment. */
        npy_intp end)               /* The end, non-inclusive, of the segment. */
{
    struct simple_ll_node * seg = NULL;
    const char * msg = "Couldn't allocate memory for segment.";

    /* Ignore length 1 segments if longer segments exist */
    if ((1==(end-start)) && (segs->max_segment_length > 1)) {
        return 0;
    }

    /* Make sure memory allocation worked */
    seg = (struct simple_ll_node*)calloc(1, sizeof(*seg));
    if (NULL==seg) {
        PyErr_SetString(PyExc_MemoryError, msg);
        err_ols_print("%s\n", msg);
        return 1;
    }

    /* Populate new segment information. */
    seg->start = start;
    seg->end = end;
    seg->length = end - start;
    seg->flink = NULL;

    /* Add segment to list as the tail */
    if (NULL == segs->head) {
        segs->head = seg;
        segs->size = 1;
    } else {
        segs->tail->flink = seg;
        segs->size++;
    }
    segs->tail = seg;

    /* Is the new segment length the longest segment length? */
    if (seg->length > segs->max_segment_length) {
        segs->max_segment_length = seg->length;
    }

    return 0;
}

/*
 * Build the optional results class to be returned from ramp fitting.
 */
static PyObject *
build_opt_res(
        struct ramp_data * rd) /* The ramp fitting data */
{
    struct opt_res_product opt_res = {0};
    PyObject * opt_res_info = Py_None;

    /* Make PyObjectArray's */
    if (create_opt_res(&opt_res, rd)) {
        return Py_None;
    }

    /* Copy data from rd->segs to these arrays */
    save_opt_res(&opt_res, rd);

    /* Package arrays into output tuple */
    opt_res_info = Py_BuildValue("NNNNNNNNN",
        opt_res.slope, opt_res.sigslope, opt_res.var_p, opt_res.var_r,
        opt_res.yint, opt_res.sigyint, opt_res.pedestal, opt_res.weights,
        opt_res.cr_mag);

    return opt_res_info;
}

/*
 * Clean up all allocated memory for a pixel ramp, except the allocated memory
 * for the data structure itself.
 */
static void
clean_pixel_ramp(
        struct pixel_ramp * pr)  /* Ramp fitting data for a pixel. */
{
    if (NULL == pr) {
        return;  /* Nothing to do */
    }

    /* Free all internal arrays */
    SET_FREE(pr->data);
    SET_FREE(pr->groupdq);
    SET_FREE(pr->rateints);
    SET_FREE(pr->stats);
    SET_FREE(pr->is_zframe);
    SET_FREE(pr->is_0th);

    /* Clean up the allocated memory for the linked lists. */
    FREE_SEGS_LIST(pr->nints, pr->segs);
}

/* Cleans up the ramp data structure */
static void
clean_ramp_data(
        struct ramp_data * rd) /* The ramp fitting data structure */
{
    npy_intp idx;
    struct simple_ll_node * current;
    struct simple_ll_node * next;

    if (NULL == rd->segs) {
        return; /* Nothing to do. */
    }

    /*
     *  For each pixel, check to see if there is any allocated
     *  memory for the linked list of ramp segments and free them.
     */
    for (idx=0; idx < rd->cube_sz; ++idx) {
        current = rd->segs[idx];
        if (current) {
            next = current->flink;
            SET_FREE(current);
            current = next;
        }
    }
    SET_FREE(rd->segs);
}

/*
 * Cleans up the rate producte data structure.
 */
static void
clean_rate_product(
        struct rate_product * rate_prod)  /* Rate product data structure */
{
    /* Free all arrays */
    Py_XDECREF(rate_prod->slope);
    Py_XDECREF(rate_prod->dq);
    Py_XDECREF(rate_prod->var_poisson);
    Py_XDECREF(rate_prod->var_rnoise);
    Py_XDECREF(rate_prod->var_err);

    /* Zero out any memory */
    memset(rate_prod, 0, sizeof(*rate_prod));

    /* Ensure the return value for the rate product is NoneType. */
    rate_prod->is_none = 1;

    return;
}

/*
 * Cleans up the rate producte data structure.
 */
static void
clean_rateint_product(
        struct rateint_product * rateint_prod)  /* Rateints product data structure */
{
    /* Free all arrays */
    Py_XDECREF(rateint_prod->slope);
    Py_XDECREF(rateint_prod->dq);
    Py_XDECREF(rateint_prod->var_poisson);
    Py_XDECREF(rateint_prod->var_rnoise);
    Py_XDECREF(rateint_prod->var_err);

    /* Zero out any memory */
    memset(rateint_prod, 0, sizeof(*rateint_prod));

    /* Ensure the return value for the rate product is NoneType. */
    rateint_prod->is_none = 1;

    return;
}

/*
 * Clean any allocated memory in a segment list.  This is implemented
 * as linked lists, so walk the list and free each node in the list.
 */
static void
clean_segment_list(
        npy_intp nints,             /* The number of integrations */
        struct segment_list * segs) /* The list of segments for the integration */
{
    npy_intp integ;
    struct simple_ll_node * current = NULL;
    struct simple_ll_node * next = NULL;

    /* 
     * Clean each list for each integration.   Each integration for
     * each pixel is segmented.  For each integration, there is a
     * linked list of segments, so walk the linked lists and free
     * each node in each list.
     */
    for (integ=0; integ < nints; ++integ) {
        current = segs[integ].head;
        while (current) {
            next = current->flink;
            memset(current, 0, sizeof(*current));
            SET_FREE(current);
            current = next;
        }

        /* Zero the memory for the integration list structure. */
        memset(&(segs[integ]), 0, sizeof(segs[integ]));
    }
}

/*
 * For the current integration ramp, compute all segments.
 * Save the segments in a linked list.
 */
static int
compute_integration_segments(
        struct ramp_data * rd,  /* Ramp fitting data */
        struct pixel_ramp * pr, /* Pixel ramp fitting data */
        npy_intp integ)         /* Current integration */
{
    int ret = 0;
    uint32_t * groupdq = pr->groupdq + integ * pr->ngroups;
    npy_intp idx, start, end;
    int in_seg=0;

    /* If the whole integration is saturated, then no valid slope. */
    if (groupdq[0] & rd->sat) {
        pr->rateints[integ].dq |= rd->dnu;
        pr->rateints[integ].dq |= rd->sat;
        pr->rateints[integ].slope = NAN;
        return 0;
    }

    /* Find all flagged groups and segment based on those flags. */
    for (idx=0; idx < pr->ngroups; ++idx) {
        if (0 == groupdq[idx]) {
            if (!in_seg) {
                /* A new segment is detected */
                if (idx > 0 && groupdq[idx-1]==rd->jump) {
                    /* Include jumps as first group of next group */
                    start = idx - 1;
                } else {
                    start = idx;
                }
                in_seg = 1;
            }
        } else {
            if (in_seg) {
                /* The end of a segment is detected. */
                end = idx;
                if (add_segment_to_list(&(pr->segs[integ]), start, end)) {
                    return 1;
                }
                in_seg = 0;
            }
        }
    }
    /* The last segment of the integration is at the end of the integration */
    if (in_seg) {
        end = idx;
        if (add_segment_to_list(&(pr->segs[integ]), start, end)) {
            return 1;
        }
    }

    /*
     * If any segment has more than one group, all one group ramps are
     * discarded.  If the longest segment has length one, then only
     * the first first one group segment is used and all subsequent
     * one group segments are discarded.
     */
    prune_segment_list(&(pr->segs[integ]));

    return ret;
}

/*
 * Create the optional results class to be returned from ramp fitting.
 */
static int
create_opt_res(
        struct opt_res_product * opt_res, /* The optional results product */
        struct ramp_data * rd)            /* The ramp fitting data */
{
    const npy_intp nd = 4;
    npy_intp dims[nd];
    const npy_intp pnd = 3;
    npy_intp pdims[pnd];
    const int fortran = 0;  /* Want C order */
    const char * msg = "Couldn't allocate memory for opt_res products.";

    dims[0] = rd->nints;
    dims[1] = rd->max_num_segs;
    dims[2] = rd->nrows;
    dims[3] = rd->ncols;

    /* Note fortran = 0 */
    opt_res->slope = (PyArrayObject*)PyArray_EMPTY(nd, dims, NPY_FLOAT, fortran);
    if (!opt_res->slope) {
        goto FAILED_ALLOC;
    }

    opt_res->sigslope = (PyArrayObject*)PyArray_EMPTY(nd, dims, NPY_FLOAT, fortran);
    if (!opt_res->sigslope) {
        goto FAILED_ALLOC;
    }

    opt_res->var_p = (PyArrayObject*)PyArray_EMPTY(nd, dims, NPY_FLOAT, fortran);
    if (!opt_res->var_p) {
        goto FAILED_ALLOC;
    }

    opt_res->var_r = (PyArrayObject*)PyArray_EMPTY(nd, dims, NPY_FLOAT, fortran);
    if (!opt_res->var_r) {
        goto FAILED_ALLOC;
    }

    opt_res->yint = (PyArrayObject*)PyArray_EMPTY(nd, dims, NPY_FLOAT, fortran);
    if (!opt_res->yint) {
        goto FAILED_ALLOC;
    }

    opt_res->sigyint = (PyArrayObject*)PyArray_EMPTY(nd, dims, NPY_FLOAT, fortran);
    if (!opt_res->sigyint) {
        goto FAILED_ALLOC;
    }

    opt_res->weights = (PyArrayObject*)PyArray_EMPTY(nd, dims, NPY_FLOAT, fortran);
    if (!opt_res->weights) {
        goto FAILED_ALLOC;
    }

    pdims[0] = rd->nints;
    pdims[1] = rd->nrows;
    pdims[2] = rd->ncols;
    opt_res->pedestal = (PyArrayObject*)PyArray_EMPTY(pnd, pdims, NPY_FLOAT, fortran);
    if (!opt_res->pedestal) {
        goto FAILED_ALLOC;
    }

    /* XXX */
    //->cr_mag = (PyArrayObject*)PyArray_EMPTY(nd, dims, NPY_FLOAT, fortran);
    opt_res->cr_mag = (PyArrayObject*)Py_None;

    return 0;

FAILED_ALLOC:
    Py_XDECREF(opt_res->slope);
    Py_XDECREF(opt_res->sigslope);
    Py_XDECREF(opt_res->var_p);
    Py_XDECREF(opt_res->var_r);
    Py_XDECREF(opt_res->yint);
    Py_XDECREF(opt_res->sigyint);
    Py_XDECREF(opt_res->pedestal);
    Py_XDECREF(opt_res->weights);

    return 1;
}

/*
 * Allocate the pixel ramp data structure.  This data structure will be reused
 * for each pixel in the exposure.
 */
static struct pixel_ramp *
create_pixel_ramp(
        struct ramp_data * rd) /* The ramp fitting data */
{
    struct pixel_ramp * pr = (struct pixel_ramp*)calloc(1, sizeof(*rd));
    char msg[256] = {0};

    /* Make sure memory allocation worked */
    if (NULL==pr) {
        snprintf(msg, 255, "Couldn't allocate memory for pixel ramp data structure.");
        PyErr_SetString(PyExc_MemoryError, (const char*)msg);
        err_ols_print("%s\n", msg);
        goto END;
    }

    pr->nints = rd->nints;
    pr->ngroups = rd->ngroups;
    pr->ramp_sz = rd->ramp_sz;

    /* Allocate array data */
    pr->data = (real_t*)calloc(pr->ramp_sz, sizeof(pr->data[0]));
    pr->groupdq = (uint32_t*)calloc(pr->ramp_sz, sizeof(pr->groupdq[0]));

    /* This is an array of integrations for the fit for each integration */
    pr->rateints = (struct pixel_fit*)calloc(pr->nints, sizeof(pr->rateints[0]));
    pr->stats = (struct integ_gdq_stats*)calloc(pr->nints, sizeof(pr->stats[0]));
    pr->segs = (struct segment_list*)calloc(pr->nints, sizeof(pr->segs[0]));

    pr->is_zframe = calloc(pr->nints, sizeof(pr->is_zframe[0]));
    pr->is_0th = calloc(pr->nints, sizeof(pr->is_0th[0]));

    if ((NULL==pr->data) || (NULL==pr->groupdq) ||  (NULL==pr->rateints) ||
        (NULL==pr->segs) || (NULL==pr->stats) || (NULL==pr->is_zframe) ||
        (NULL==pr->is_0th)) 
    {
        snprintf(msg, 255, "Couldn't allocate memory for pixel ramp data structure.");
        PyErr_SetString(PyExc_MemoryError, (const char*)msg);
        err_ols_print("%s\n", msg);
        FREE_PIXEL_RAMP(pr);
        goto END;
    }

END:
    return pr;
}

/*
 * Set up the ndarrays for the output rate product.
 */
static int
create_rate_product(
        struct rate_product * rate, /* The rate product */
        struct ramp_data * rd)      /* The ramp fitting data */
{
    const npy_intp nd = 2;
    npy_intp dims[nd];
    const int fortran = 0;
    const char * msg = "Couldn't allocate memory for rate products.";

    dims[0] = rd->nrows;
    dims[1] = rd->ncols;

    rate->slope = (PyArrayObject*)PyArray_EMPTY(nd, dims, NPY_FLOAT, fortran);
    if (Py_None==(PyObject*)rate->slope) {
        goto FAILED_ALLOC;
    }

    rate->dq = (PyArrayObject*)PyArray_EMPTY(nd, dims, NPY_UINT32, fortran);
    if (Py_None==(PyObject*)rate->dq) {
        goto FAILED_ALLOC;
    }

    rate->var_poisson = (PyArrayObject*)PyArray_EMPTY(nd, dims, NPY_FLOAT, fortran);
    if (Py_None==(PyObject*)rate->var_poisson) {
        goto FAILED_ALLOC;
    }

    rate->var_rnoise = (PyArrayObject*)PyArray_EMPTY(nd, dims, NPY_FLOAT, fortran);
    if (Py_None==(PyObject*)rate->var_rnoise) {
        goto FAILED_ALLOC;
    }

    rate->var_err = (PyArrayObject*)PyArray_EMPTY(nd, dims, NPY_FLOAT, fortran);
    if (Py_None==(PyObject*)rate->var_err) {
        goto FAILED_ALLOC;
    }

    return 0;

FAILED_ALLOC:
    Py_XDECREF(rate->slope);
    Py_XDECREF(rate->dq);
    Py_XDECREF(rate->var_poisson);
    Py_XDECREF(rate->var_rnoise);
    Py_XDECREF(rate->var_err);

    return 1;
}

/*
 * Set up the ndarrays for the output rateint product.
 */
static int
create_rateint_product(
        struct rateint_product * rateint, /* The rateints product */
        struct ramp_data * rd)            /* The ramp fitting data */
{
    const npy_intp nd = 3;
    npy_intp dims[nd];
    const int fortran = 0;
    const char * msg = "Couldn't allocate memory for rateint products.";

    dims[0] = rd->nints;
    dims[1] = rd->nrows;
    dims[2] = rd->ncols;

    rateint->slope = (PyArrayObject*)PyArray_EMPTY(nd, dims, NPY_FLOAT, fortran);
    if (Py_None==(PyObject*)rateint->slope) {
        goto FAILED_ALLOC;
    }

    rateint->dq = (PyArrayObject*)PyArray_EMPTY(nd, dims, NPY_UINT32, fortran);
    if (Py_None==(PyObject*)rateint->dq) {
        goto FAILED_ALLOC;
    }

    rateint->var_poisson = (PyArrayObject*)PyArray_EMPTY(nd, dims, NPY_FLOAT, fortran);
    if (Py_None==(PyObject*)rateint->var_poisson) {
        goto FAILED_ALLOC;
    }

    rateint->var_rnoise = (PyArrayObject*)PyArray_EMPTY(nd, dims, NPY_FLOAT, fortran);
    if (Py_None==(PyObject*)rateint->var_rnoise) {
        goto FAILED_ALLOC;
    }

    rateint->var_err = (PyArrayObject*)PyArray_EMPTY(nd, dims, NPY_FLOAT, fortran);
    if (Py_None==(PyObject*)rateint->var_err) {
        goto FAILED_ALLOC;
    }

    return 0;

FAILED_ALLOC:
    Py_XDECREF(rateint->slope);
    Py_XDECREF(rateint->dq);
    Py_XDECREF(rateint->var_poisson);
    Py_XDECREF(rateint->var_rnoise);
    Py_XDECREF(rateint->var_err);

    return 1;
}

/*
 * Compute the median of a sorted array that accounts (ignores) the
 * NaN's at the end of the array.
 */
static real_t
real_nan_median(
        real_t * arr, /* Array in which to find the median */
        npy_intp len) /* Length of array */
{
    real_t med = -1.;
    npy_intp nan_idx = 0, med_idx;

    /* Find first NaN.  The median will be only of the non-NaN data. */
    while(nan_idx < len && !isnan(arr[nan_idx])) {
        nan_idx++;
    }

    /* Some special cases */
    switch(nan_idx) {
        case 0:
            return NAN;
        case 1:
            return arr[0];
        case 2:
            return ((arr[0] + arr[1]) / 2.);
        default:
            break;
    }

    /* The array is sufficiently long enough now the math can work */
    med_idx = nan_idx >> 1;
    if (nan_idx & 1) {
        med = arr[med_idx];
    } else {
        med = (arr[med_idx] + arr[med_idx-1]) / 2.;
    }

    return med;
}

/* Get a float from a 2-D NDARRAY */
static float
get_float2(
        PyArrayObject * obj,    /* Object from which to get float */
        npy_intp row,           /* Row index into object */
        npy_intp col)           /* Column index into object */
{
    float ans;

    ans = VOID_2_FLOAT(PyArray_GETPTR2(obj, row, col));

    return ans;
}

/* Get a float from a 4-D NDARRAY */
static float
get_float4(
    PyArrayObject * obj,    /* Object from which to get float */
    npy_intp integ,         /* Integration index into object */
    npy_intp group,         /* Group index into object */
    npy_intp row,           /* Row index into object */
    npy_intp col)           /* Column index into object */
{
    float ans;

    ans = VOID_2_FLOAT(PyArray_GETPTR4(obj, integ, group, row, col));

    return ans;
}

/* Get a float from a 3-D NDARRAY. */
static float
get_float3(
    PyArrayObject * obj,    /* Object from which to get float */
    npy_intp integ,         /* Integration index into object */
    npy_intp row,           /* Row index into object */
    npy_intp col)           /* Column index into object */
{
    float ans;

    ans = VOID_2_FLOAT(PyArray_GETPTR3(obj, integ, row, col));

    return ans;
}

/* Get a uint32_t from a 2-D NDARRAY. */
static uint32_t
get_uint32_2(
    PyArrayObject * obj,    /* Object from which to get float */
    npy_intp row,           /* Row index into object */
    npy_intp col)           /* Column index into object */
{
    return VOID_2_U32(PyArray_GETPTR2(obj, row, col));
}

/*
 * From the ramp data structure get all the information needed for a pixel to
 * fit a ramp for that pixel.  The ramp data structure points to PyObjects for
 * ndarrays.  The data from these arrays are retrieved from this data structure
 * and put in simple arrays, indexed by nints and ngroups.  The internal
 * structure for each pixel is now a simple array of length nints*ngroups and
 * is nothing more than each integration of length ngroups concatenated together,
 * in order from the 0th integration to the last integration.
 *
 * Integration level flag data is also computed, as well as setting flags
 * to use the 0th frame timing, rather than group time, or use the ZEROFRAME.
 */
static void
get_pixel_ramp(
        struct pixel_ramp * pr, /* Pixel ramp data */
        struct ramp_data * rd,  /* Ramp data */
        npy_intp row,           /* Pixel row */
        npy_intp col)           /* Pixel column */
{
    npy_intp integ, group;
    ssize_t idx = 0, integ_idx;
    real_t zframe;

    get_pixel_ramp_zero(pr);
    get_pixel_ramp_meta(pr, rd, row, col);

    /* Get array data */
    for (integ = 0; integ < pr->nints; ++integ) {
        current_integration = integ;
        memset(&(pr->stats[integ]), 0, sizeof(pr->stats[integ]));
        integ_idx = idx;
        for (group = 0; group < pr->ngroups; ++group) {
            get_pixel_ramp_integration(pr, rd, row, col, integ, group, idx);
            idx++;
        }
        /* Check for 0th group and ZEROFRAME */
        if (!rd->suppress1g) {
            if ((1==pr->stats[integ].cnt_good) && (0==pr->groupdq[integ_idx])) {
                pr->is_0th[integ] = 1;
            } else if ((0==pr->stats[integ].cnt_good) &&
                       ((PyObject*)rd->zframe) != Py_None) {
                zframe = (real_t)rd->get_zframe(rd->zframe, integ, row, col);
                if (0. != zframe ) {
                    pr->data[integ_idx] = zframe;
                    pr->groupdq[integ_idx] = 0;
                    pr->stats[integ].cnt_good = 1;
                    pr->stats[integ].cnt_dnu_sat--;
                    if (pr->ngroups == pr->stats[integ].cnt_sat) {
                        pr->stats[integ].cnt_sat--;
                    }
                    if (pr->ngroups == pr->stats[integ].cnt_dnu) {
                        pr->stats[integ].cnt_dnu--;
                    }
                    pr->is_zframe[integ] = 1;
                }
            }
        }

        if (pr->stats[integ].jump_det) {
            pr->rateints[integ].dq |= rd->jump;
            pr->rate.dq |= rd->jump;
        }
    }
}

/*
 * For a pixel, get the current integration and group information.
 */
static void
get_pixel_ramp_integration(
        struct pixel_ramp * pr, /* Pixel ramp data */
        struct ramp_data * rd,  /* Ramp data */
        npy_intp row,           /* Pixel row index */
        npy_intp col,           /* Pixel column index */
        npy_intp integ,         /* Current integration */
        npy_intp group,         /* Current group */
        npy_intp idx)           /* Index into object */
{
    /* For a single byte, no endianness handling necessary. */
    pr->groupdq[idx] = VOID_2_U8(PyArray_GETPTR4(
        rd->groupdq, integ, group, row, col));

    /* Compute group DQ statistics */
    if (pr->groupdq[idx] & rd->jump) {
        pr->stats[integ].jump_det = 1;
    }
    if (0==pr->groupdq[idx]) {
        pr->stats[integ].cnt_good++;
    } else if (pr->groupdq[idx] & rd->dnu) {
        pr->stats[integ].cnt_dnu++;
    }
    if ((pr->groupdq[idx] & rd->dnu) || (pr->groupdq[idx] & rd->sat)) {
        pr->stats[integ].cnt_dnu_sat++;
    }

    /* Just make saturated groups NaN now. */
    if (pr->groupdq[idx] & rd->sat) {
        pr->data[idx] = NAN;
        pr->stats[integ].cnt_sat++;
    } else {
        /* Use endianness handling functions. */
        pr->data[idx] = (real_t) rd->get_data(rd->data, integ, group, row, col);
    }
}

/*
 * Get the meta data for a pixel.
 */
static void
get_pixel_ramp_meta(
        struct pixel_ramp * pr, /* Pixel ramp data */
        struct ramp_data * rd,  /* Ramp data */
        npy_intp row,           /* Pixel row */
        npy_intp col)           /* Pixel column */
{
    /* Get pixel and dimension data */
    pr->row = row;
    pr->col = col;
    npy_intp integ;

    pr->pixeldq = rd->get_pixeldq(rd->pixeldq, row, col);

    pr->gain = (real_t)rd->get_gain(rd->gain, row, col);
    if (pr->gain <= 0. || isnan(pr->gain)) {
        pr->pixeldq |= (rd->dnu | rd->ngval);
    }
    for (integ=0; integ<rd->nints; ++integ) {
        pr->rateints[integ].dq = pr->pixeldq;
    }
    pr->rnoise = (real_t) rd->get_rnoise(rd->rnoise, row, col);
    pr->dcurrent = (real_t) rd->get_dcurrent(rd->dcurrent, row, col);
    pr->rate.dq = pr->pixeldq;
}

/*
 * Clean the pixel ramp data structure in preparation for data
 * for the next pixel.
 */
static void
get_pixel_ramp_zero(
        struct pixel_ramp * pr) /* Pixel ramp data */
{
    pr->pixeldq = 0.;
    pr->gain = 0.;
    pr->rnoise = 0.;

    /* Zero out flags */
    memset(pr->is_zframe, 0, pr->nints * sizeof(pr->is_zframe[0]));
    memset(pr->is_0th , 0, pr->nints * sizeof(pr->is_0th[0]));

    /* C computed values */
    pr->median_rate = 0.;  /* The median rate of the pixel */
    pr->invvar_e_sum = 0.; /* Intermediate calculation needed for final slope */

    memset(pr->rateints, 0, pr->nints * sizeof(pr->rateints[0]));
    memset(&(pr->rate), 0, sizeof(pr->rate));
}

/*
 * Compute the pedestal for an integration segment.
 */
static void
get_pixel_ramp_integration_segments_and_pedestal(
        npy_intp integ,             /* The current integration */
        struct pixel_ramp * pr,     /* The pixel ramp data */
        struct ramp_data * rd)      /* The ramp data */
{
    npy_intp idx, idx_pr;
    real_t fframe, int_slope;

    /* Add list to ramp data structure */
    idx = get_cube_index(rd, integ, pr->row, pr->col);
    rd->segs[idx] = pr->segs[integ].head;
    if (pr->segs[integ].size > rd->max_num_segs) {
        rd->max_num_segs = pr->segs[integ].size;
    }

    /* Remove list from pixel ramp data structure */
    pr->segs[integ].head = NULL;
    pr->segs[integ].tail = NULL;

    /* Get pedestal */
    if (pr->rateints[integ].dq & rd->sat) {
        rd->pedestal[idx] = 0.;
        return;
    }

    idx_pr = get_ramp_index(rd, integ, 0);
    fframe = pr->data[idx_pr];
    int_slope = pr->rateints[integ].slope;

    // tmp = ((rd->nframes + 1) / 2. + rd->dropframes) / (rd->nframes + rd->groupgap);
    rd->pedestal[idx] = fframe - int_slope * rd->ped_tmp;

    if (isnan(rd->pedestal[idx])) {
        rd->pedestal[idx] = 0.;
    }
}

/*
 * This function takes in the args object to parse it, validate the input
 * arguments, then fill out a ramp data structure to be used for ramp fitting.
 */
static struct ramp_data *
get_ramp_data(
        PyObject * args)    /* The C extension module arguments */
{
    struct ramp_data * rd = calloc(1, sizeof(*rd));  /* Allocate memory */
    PyObject * Py_ramp_data;
    char * msg = "Couldn't allocate memory for ramp data structure.";

    /* Make sure memory allocation worked */
    if (NULL==rd) {
        PyErr_SetString(PyExc_MemoryError, msg);
        err_ols_print("%s\n", msg);
        return NULL;
    }
    
    if (get_ramp_data_parse(&Py_ramp_data, rd, args)) {
        FREE_RAMP_DATA(rd);
        return NULL;
    }

    if (get_ramp_data_arrays(Py_ramp_data, rd)) {
        FREE_RAMP_DATA(rd);
        return NULL;
    }

    /* Void function */
    get_ramp_data_meta(Py_ramp_data, rd);

    /* One time computations. */
    rd->effintim = (rd->nframes + rd->groupgap) * rd->frame_time;
    rd->one_group_time = ((float)rd->nframes + 1.) * (float)rd->frame_time / 2.;

    /* Allocate optional results arrays. */
    if (rd->save_opt) {
        rd->max_num_segs = -1;
        rd->segs = (struct simple_ll_node **)calloc(rd->cube_sz, sizeof(rd->segs[0]));
        rd->pedestal = (real_t*)calloc(rd->cube_sz, sizeof(rd->pedestal[0]));

        if ((NULL==rd->segs) || (NULL==rd->pedestal)){
            PyErr_SetString(PyExc_MemoryError, msg);
            err_ols_print("%s\n", msg);
            FREE_RAMP_DATA(rd);
            return NULL;
        }
    }

    return rd;
}

/*
 * Get the numpy arrays from the ramp_data class defined in ramp_fit_class.
 * Also, validate the types for each array and get endianness functions.
 */
static int
get_ramp_data_arrays(
        PyObject * Py_ramp_data, /* The inputted RampData */
        struct ramp_data * rd)   /* The ramp data */
{
    /* Get numpy arrays */
    rd->data = (PyArrayObject*)PyObject_GetAttrString(Py_ramp_data, "data");
    rd->groupdq = (PyArrayObject*)PyObject_GetAttrString(Py_ramp_data, "groupdq");
    rd->err = (PyArrayObject*)PyObject_GetAttrString(Py_ramp_data, "err");
    rd->pixeldq = (PyArrayObject*)PyObject_GetAttrString(Py_ramp_data, "pixeldq");
    rd->zframe = (PyArrayObject*)PyObject_GetAttrString(Py_ramp_data, "zeroframe");
    rd->dcurrent = (PyArrayObject*)PyObject_GetAttrString(Py_ramp_data, "average_dark_current");

    /* Validate numpy array types */
    if (get_ramp_data_new_validate(rd)) {
        FREE_RAMP_DATA(rd);
        return 1;
    }

    /* Check endianness of the arrays, as well as the dimensions. */
    get_ramp_data_getters(rd);
    get_ramp_data_dimensions(rd);

    return 0;
}

/*
 * Get the meta data from the ramp_data class defined in ramp_fit_class.
 * Also, validate the types for each array and get endianness functions.
 */
static void
get_ramp_data_meta(
        PyObject * Py_ramp_data, /* The RampData class */
        struct ramp_data * rd)   /* The ramp data */
{
    /* Get integer meta data */
    rd->groupgap = py_ramp_data_get_int(Py_ramp_data, "groupgap");
    rd->nframes = py_ramp_data_get_int(Py_ramp_data, "nframes");
    rd->suppress1g = py_ramp_data_get_int(Py_ramp_data, "suppress_one_group_ramps");
    rd->dropframes = py_ramp_data_get_int(Py_ramp_data, "drop_frames1");

    rd->ped_tmp = ((rd->nframes + 1) / 2. + rd->dropframes) / (rd->nframes + rd->groupgap);

    /* Get flag values */
    rd->dnu = py_ramp_data_get_int(Py_ramp_data, "flags_do_not_use");
    rd->jump = py_ramp_data_get_int(Py_ramp_data, "flags_jump_det");
    rd->sat = py_ramp_data_get_int(Py_ramp_data, "flags_saturated");
    rd->ngval = py_ramp_data_get_int(Py_ramp_data, "flags_no_gain_val");
    rd->uslope = py_ramp_data_get_int(Py_ramp_data, "flags_unreliable_slope");
    rd->invalid = rd->dnu | rd->sat;

    /* Get float meta data */
    rd->group_time = (real_t)py_ramp_data_get_float(Py_ramp_data, "group_time");
    rd->frame_time = (real_t)py_ramp_data_get_float(Py_ramp_data, "frame_time");
}

/*
 * Parse the arguments for the entry point function into this module.
 */
static int
get_ramp_data_parse(
        PyObject ** Py_ramp_data, /* The RampData class */
        struct ramp_data * rd,    /* The ramp data */
        PyObject * args)          /* The C extension module arguments */
{
    char * weight = NULL;
    const char * optimal = "optimal";
    char * msg = NULL;

    if (!PyArg_ParseTuple(args, "OOOsI:get_ramp_data",
                Py_ramp_data, &(rd->gain), &(rd->rnoise),
                &weight, &rd->save_opt)) {
        msg = "Parsing arguments failed.";
        PyErr_SetString(PyExc_ValueError, msg);
        err_ols_print("%s\n", msg);
        return 1;
    }

    if (!strcmp(weight, optimal)) {
        rd->weight = WEIGHTED;
    //} else if (!strcmp(weight, unweighted)) {
    //    rd->weight = UNWEIGHTED;
    } else {
        msg = "Bad value for weighting.";
        PyErr_SetString(PyExc_ValueError, msg);
        err_ols_print("%s\n", msg);
        return 1;
    }

    return 0;
}

/*
 * Validate the numpy arrays inputted from the ramp_data class have
 * the correct data types.
 */
static int
get_ramp_data_new_validate(
        struct ramp_data * rd) /* the ramp data */
{
    char * msg = NULL;

    /* Validate the types for each of the ndarrays */
    if (!(
            (NPY_FLOAT==PyArray_TYPE(rd->data)) &&
            (NPY_FLOAT==PyArray_TYPE(rd->err)) &&
            (NPY_UBYTE == PyArray_TYPE(rd->groupdq)) &&
            (NPY_UINT32 == PyArray_TYPE(rd->pixeldq)) &&
            (NPY_FLOAT==PyArray_TYPE(rd->dcurrent)) &&
            (NPY_FLOAT==PyArray_TYPE(rd->gain)) &&
            (NPY_FLOAT==PyArray_TYPE(rd->rnoise))
         )) {
        msg = "Bad type array for pass ndarrays to C.";
        PyErr_SetString(PyExc_TypeError, msg);
        err_ols_print("%s\n", msg);
        return 1;
    }

    /* ZEROFRAME could be NoneType, so needed a separate check */
    if ((((PyObject*)rd->zframe) != Py_None) && (NPY_FLOAT != PyArray_TYPE(rd->zframe))) {
        msg = "Bad type array ZEROFRAME.";
        PyErr_SetString(PyExc_TypeError, msg);
        err_ols_print("%s\n", msg);
        return 1;
    }

    return 0;
}

/*
 * Get dimensional information about the data.
 */
static void
get_ramp_data_dimensions(
        struct ramp_data * rd) /* The ramp data */
{
    npy_intp * dims;

    /* Unpack the data dimensions */
    dims = PyArray_DIMS(rd->data);
    rd->nints = dims[0];
    rd->ngroups = dims[1];
    rd->nrows = dims[2];
    rd->ncols = dims[3];

    rd->cube_sz = rd->ncols * rd->nrows * rd->ngroups;
    rd->image_sz = rd->ncols * rd->nrows;
    rd->ramp_sz = rd->nints * rd->ngroups;
}

/*
 * Set getter functions based on type and dimensions.
 */
static void
get_ramp_data_getters(
        struct ramp_data * rd) /* The ramp data */
{
    rd->get_data = get_float4;
    rd->get_err = get_float4;

    rd->get_pixeldq = get_uint32_2;

    rd->get_dcurrent = get_float2;
    rd->get_gain = get_float2;
    rd->get_rnoise = get_float2;

    rd->get_zframe = get_float3;
}

/*
 * Compute the median rate for a pixel ramp.
 * MEDIAN RATE
 */
static int
compute_median_rate(
        struct ramp_data * rd,  /* The ramp data */
        struct pixel_ramp * pr) /* The pixel ramp data */
{
    if (1 == rd->ngroups) {
        return median_rate_1ngroup(rd, pr);
    }
    return median_rate_default(rd, pr);
}

/*
 * Compute the 1 group special case median.
 */
static int
median_rate_1ngroup(
        struct ramp_data * rd,  /* The ramp data */
        struct pixel_ramp * pr) /* The pixel ramp data */
{
    npy_intp idx, integ;
    real_t accum_mrate = 0.;
    real_t timing = rd->one_group_time;

    for (integ=0; integ < rd->nints; integ++) {
        idx = get_ramp_index(rd, integ, 0);
        accum_mrate += (pr->data[idx] / timing);
    }
    pr->median_rate = accum_mrate / (real_t)rd->nints;

    return 0;
}

/*
 * Compute the median rate of a pixel ramp.
 */
static int
median_rate_default(
        struct ramp_data * rd,  /* The ramp data */
        struct pixel_ramp * pr) /* The pixel ramp data */
{
    int ret = 0;
    real_t * int_data = (real_t*)calloc(pr->ngroups, sizeof(*int_data));
    uint8_t * int_dq = (uint8_t*)calloc(pr->ngroups, sizeof(*int_dq));
    npy_intp integ, start_idx;
    real_t mrate = 0., accum_mrate = 0.;
    const char * msg = "Couldn't allocate memory for median rates.";

    /* Make sure memory allocation worked */
    if (NULL==int_data || NULL==int_dq) {
        PyErr_SetString(PyExc_MemoryError, msg);
        err_ols_print("%s\n", msg);
        ret = 1;
        goto END;
    }

    // print_delim();
    // dbg_ols_print("Pixel (%ld, %ld)\n", pr->row, pr->col);
    /* Compute the median rate for  the pixel. */
    for (integ = 0; integ < pr->nints; ++integ) {
        current_integration = integ;

        if (pr->is_0th[integ]) {
            // dbg_ols_print("col %ld, is_0th\n", pr->col);
            /* Special case of only good 0th group */
            start_idx = get_ramp_index(rd, integ, 0);
            mrate = pr->data[start_idx] / rd->one_group_time;
        } else if (pr->is_zframe[integ]) {
            // dbg_ols_print("col %ld, is_zframe\n", pr->col);
            /* Special case of using ZERFRAME data */
            start_idx = get_ramp_index(rd, integ, 0);
            mrate = pr->data[start_idx] / rd->frame_time;
        } else {
            // dbg_ols_print("col %ld, is_default\n", pr->col);
            /* Get the data and DQ flags for this integration. */
            int_data = median_rate_get_data(int_data, integ, rd, pr);
            int_dq = median_rate_get_dq(int_dq, integ, rd, pr);

            /* Compute the median rate for the integration. */
            if (median_rate_integration(&mrate, int_data, int_dq, rd, pr)) {
                goto END;
            }
        }
        if (isnan(mrate)) {
            mrate = 0.;
        }
        accum_mrate += mrate;
    }
    
    /* The pixel median rate is the average of the integration median rates. */
    pr->median_rate = accum_mrate /= (float)pr->nints;

END:
    SET_FREE(int_data);
    SET_FREE(int_dq);
    return ret;
}

/*
 * Get integration data to compute the median rate for an integration.
 */
static real_t *
median_rate_get_data (
        real_t * data,          /* Integration data */
        npy_intp integ,         /* The integration number */
        struct ramp_data * rd,  /* The ramp data */
        struct pixel_ramp * pr) /* The pixel ramp data */
{
    npy_intp start_idx = get_ramp_index(rd, integ, 0);

    memcpy(data, pr->data + start_idx, pr->ngroups * sizeof(pr->data[0]));

    return data;
}

/*
 * Get integration DQ to compute the median rate for an integration.
 */
static uint8_t *
median_rate_get_dq (
        uint8_t * data,         /* Integration data quality */
        npy_intp integ,         /* The integration number */
        struct ramp_data * rd,  /* The ramp data */
        struct pixel_ramp * pr) /* The pixel ramp data */
{
    npy_intp group, idx = get_ramp_index(rd, integ, 0);

    for (group=0; group<pr->ngroups; ++group) {
        idx = get_ramp_index(rd, integ, group);
        data[group] = pr->groupdq[idx];
    }

    return data;
}

/*
 * For an integration, create a local copy of the data.
 * Set flagged groups to NaN.
 * Sort the modified data.
 * Using the sorted modified data, find the median value.
 */
static int
median_rate_integration(
        real_t * mrate,         /* The NaN median rate */
        real_t * int_data,      /* The integration data */
        uint8_t * int_dq,       /* The integration data quality */
        struct ramp_data * rd,  /* The ramp data */
        struct pixel_ramp * pr) /* The pixel ramp data */
{
    int ret = 0;
    real_t * loc_integ = (real_t*)calloc(pr->ngroups, sizeof(*loc_integ));
    const char * msg = "Couldn't allocate memory for integration median rate.";
    npy_intp k, loc_integ_len;
    int nan_cnt;

    /* Make sure memory allocation worked */
    if (NULL==loc_integ) {
        PyErr_SetString(PyExc_MemoryError, msg);
        err_ols_print("%s\n", msg);
        ret = 1;
        goto END;
    }

    /* Create a local copy because it will be modiified */
    for (k=0; k<pr->ngroups; ++k) {
        if (int_dq[k] & rd->dnu) {
            loc_integ[k] = NAN;
            continue;
        }
        loc_integ[k] = int_data[k] / rd->group_time;
    }

    /* Sort first differences with NaN's based on DQ flags */
    nan_cnt = median_rate_integration_sort(loc_integ, int_dq, rd, pr);

    /* 
     * Get the NaN median using the sorted first differences.  Note that the
     * first differences has a length ngroups-1.
     */
    if (1 == pr->ngroups) {
        *mrate = loc_integ[0];
    } else {
        loc_integ_len = pr->ngroups - 1;
        *mrate = real_nan_median(loc_integ, loc_integ_len);
    }

END:
    SET_FREE(loc_integ);
    return ret;
}

/*
 * For an integration, create a local copy of the data.
 * Set flagged groups to NaN.
 * Sort the modified data.
 */
static int
median_rate_integration_sort(
        real_t * loc_integ,     /* Local copy of integration data */
        uint8_t * int_dq,       /* The integration data quality */
        struct ramp_data * rd,  /* The ramp data */
        struct pixel_ramp * pr) /* The pixel ramp data */
{
    npy_intp k, ngroups = pr->ngroups;
    real_t loc0 = loc_integ[0];
    int nan_cnt = 0, all_nan = 1;

    /* Compute first differences */
    if (1 == ngroups ) {
        return nan_cnt;
    } else {
        for (k=0; k<ngroups-1; ++k) {
            if (rd->jump & int_dq[k+1]) {
                /* NaN out jumps */
                loc_integ[k] = NAN;
            } else {
                loc_integ[k] = loc_integ[k+1] - loc_integ[k];
            }
            if (isnan(loc_integ[k])) {
                nan_cnt++;
            } else {
                all_nan = 0;
            }
        }
    }

    if (all_nan && !isnan(loc0)) {
        loc_integ[0] = loc0;
    }

    /* XXX */
    // print_real_array("Pre-sort: ", loc_integ, ngroups-1, 1, __LINE__);
    /* NaN sort first differences */
    qsort(loc_integ, ngroups-1, sizeof(loc_integ[0]), median_rate_integration_sort_cmp);

    return nan_cnt;
}

/* The comparison function for qsort with NaN's */
static int
median_rate_integration_sort_cmp(
        const void * aa, /* First comparison element */
        const void * bb) /* Second comparison element */
{
    real_t a = VOID_2_REAL(aa);
    real_t b = VOID_2_REAL(bb);
    int ans = 0;

    /* Sort low to high, where NaN is high */
    if (isnan(b)) {
        if (isnan(a)) {
            ans = 0;        /* a == b */
        } else {
            ans = -1;       /* a < b */
        }
    } else if (isnan(a)) {
        ans = 1;            /* a > b */
    } else {
        ans = (a < b) ? -1 : 1;
    }

    return ans;
}

/*
 * Fit slope for each pixel.
 */
static int
ols_slope_fit_pixels(
        struct ramp_data * rd,                  /* The ramp data */
        struct pixel_ramp * pr,                 /* The pixel ramp data */
        struct rate_product * rate_prod,        /* The rate product */
        struct rateint_product * rateint_prod)  /* The rateints product */
{
    npy_intp row, col;

    for (row = 0; row < rd->nrows; ++row) {
        for (col = 0; col < rd->ncols; ++col) {
            get_pixel_ramp(pr, rd, row, col);

            /* Compute ramp fitting */
            if (ramp_fit_pixel(rd, pr)) {
                return 1;
            }

            /* Save fitted pixel data for output packaging */
            if (save_ramp_fit(rateint_prod, rate_prod, pr)) {
                return 1;
            }
        }
    }

    return 0;
}

/*
 * For debugging, print the type values and the array types.
 */
static void
print_ramp_data_types(
        struct ramp_data * rd,  /* The ramp data */
        int line)               /* Calling line number */
{
    printf("[%s:%d]\n", __FILE__, line);
    printf("NPY_DOUBLE = %d\n", NPY_DOUBLE);
    printf("NPY_FLOAT = %d\n", NPY_FLOAT);
    printf("NPY_UBYTE = %d\n", NPY_UBYTE);
    printf("NPY_UINT322 = %d\n", NPY_UINT32);
    printf("PyArray_TYPE(rd->data))    = %d\n", PyArray_TYPE(rd->data));
    printf("PyArray_TYPE(rd->err))     = %d\n", PyArray_TYPE(rd->err));
    printf("PyArray_TYPE(rd->groupdq)) = %d\n", PyArray_TYPE(rd->groupdq));
    printf("PyArray_TYPE(rd->pixeldq)) = %d\n", PyArray_TYPE(rd->pixeldq));
    printf("\n");
    printf("PyArray_TYPE(rd->gain))   = %d\n", PyArray_TYPE(rd->gain));
    printf("PyArray_TYPE(rd->rnoise)) = %d\n", PyArray_TYPE(rd->rnoise));
}

/*
 * Prepare the output products for return from C extension.
 */
static PyObject *
package_results(
        struct rate_product * rate,         /* The rate product */
        struct rateint_product * rateints,  /* The rateints product */
        struct ramp_data * rd)              /* The ramp data */
{
    PyObject * image_info = Py_None;
    PyObject * cube_info = Py_None;
    PyObject * opt_res = Py_None;
    PyObject * result = Py_None;

    image_info = Py_BuildValue("(NNNNN)", 
        rate->slope, rate->dq, rate->var_poisson, rate->var_rnoise, rate->var_err);
    if (!image_info) {
        goto FAILED_ALLOC;
    }

    cube_info = Py_BuildValue("(NNNNN)", 
        rateints->slope, rateints->dq, rateints->var_poisson, rateints->var_rnoise, rateints->var_err);
    if (!cube_info) {
        goto FAILED_ALLOC;
    }

    if (rd->save_opt) {
        opt_res = build_opt_res(rd);
        if (!opt_res) {
            goto FAILED_ALLOC;
        }
    }

    result = Py_BuildValue("(NNN)", image_info, cube_info, opt_res);

    return result;

FAILED_ALLOC:
    Py_XDECREF(image_info);
    Py_XDECREF(cube_info);
    Py_XDECREF(opt_res);

    return NULL;
}

/*
 * For debugging print the type values of ramp data
 * arrays and the expected type values.
 */
static void
print_rd_type_info(struct ramp_data * rd) { /* The ramp data */
    print_delim();
    print_npy_types();
    dbg_ols_print("data = %d (%d)\n", PyArray_TYPE(rd->data), NPY_FLOAT);
    dbg_ols_print("err  = %d (%d)\n", PyArray_TYPE(rd->err), NPY_FLOAT);
    dbg_ols_print("gdq  = %d (%d)\n", PyArray_TYPE(rd->groupdq), NPY_UBYTE);
    dbg_ols_print("pdq  = %d (%d)\n", PyArray_TYPE(rd->pixeldq), NPY_UINT32);
    dbg_ols_print("dcur = %d (%d)\n", PyArray_TYPE(rd->dcurrent), NPY_FLOAT);
    dbg_ols_print("gain = %d (%d)\n", PyArray_TYPE(rd->gain), NPY_FLOAT);
    dbg_ols_print("rn   = %d (%d)\n", PyArray_TYPE(rd->rnoise), NPY_FLOAT);
    print_delim();
}

/*
 * Segments of length one get removed if there is a segment
 * longer than one group.
 */
static void
prune_segment_list(
        struct segment_list * segs)  /* Linked list of segments */
{
    struct simple_ll_node * seg = NULL;
    struct simple_ll_node * prev = NULL;
    struct simple_ll_node * next = NULL;

    /*
     * Nothing to do if one or fewer segments are in list or the max segment
     * length is 1.
     */
    if (segs->size < 2) {
        return;
    }

    /* If max segment length is 1, then there should only be one segment. */
    if (segs->max_segment_length < 2) {
        seg = segs->head;
        prev = seg->flink;

        while (prev) {
            next = prev->flink;
            SET_FREE(prev);
            prev = next;
        }

        seg->flink = NULL;
        seg->length = 1;
        return;
    }

    /* Remove segments of size 1, since the max_segment length is greater than 1 */
    seg = segs->head;
    while (seg) {
        next = seg->flink;
        if (1==seg->length) {
            /* Remove segment from list */
            if (seg == segs->head) {
                segs->head = seg->flink;
            } else {
                prev->flink = seg->flink;
            }
            SET_FREE(seg);
            segs->size--;
        } else {
            /* Save previous known segment still in list */
            prev = seg;
        }
        seg = next;
    }
}

/*
 * Get a float value from an attribute of the ramp_data class defined in
 * ramp_fit_class.
 */
static float
py_ramp_data_get_float(
        PyObject * rd,      /* The RampData class */
        const char * attr)  /* The attribute to get from the class */
{
    PyObject * Obj;
    float val;

    Obj = PyObject_GetAttrString(rd, attr);
    val = (float)PyFloat_AsDouble(Obj);

    return val;
}

/*
 * Get a integer value from an attribute of the ramp_data class defined in
 * ramp_fit_class.
 */
static int
py_ramp_data_get_int(
        PyObject * rd,      /* The RampData class */
        const char * attr)  /* The attribute to get from the class */
{
    PyObject * Obj;
    int val;

    Obj = PyObject_GetAttrString(rd, attr);
    val = (int)PyLong_AsLong(Obj);

    return val;
}

#define DBG_RATE_INFO do { \
    dbg_ols_print("(%ld, %ld) median rate = %f\n", pr->row, pr->col, pr->median_rate); \
    dbg_ols_print("Rate slope: %f\n", pr->rate.slope); \
    dbg_ols_print("Rate DQ: %f\n", pr->rate.dq); \
    dbg_ols_print("Rate var_p: %f\n", pr->rate.var_poisson); \
    dbg_ols_print("Rate var_r: %f\n\n", pr->rate.var_rnoise); \
} while(0)

/*
 * Ramp fit a pixel ramp.
 * PIXEL RAMP
 */
static int
ramp_fit_pixel(
        struct ramp_data * rd,  /* The ramp data */
        struct pixel_ramp * pr) /* The pixel ramp data */
{
    int ret = 0;
    npy_intp integ;
    int sat_cnt = 0, dnu_cnt = 0;

    /* Ramp fitting depends on the averaged median rate for each integration */
    if (compute_median_rate(rd, pr)) {
        ret = 1;
        goto END;
    }
#if 0
    print_delim();
    dbg_ols_print("Pixel (%ld, %ld)\n", pr->row, pr->col);
    dbg_ols_print("Median Rate = %.10f\n", pr->median_rate);
    print_delim();
#endif

    /* Clean up any thing from the last pixel ramp */
    clean_segment_list(pr->nints, pr->segs);

    /* Compute the ramp fit per each integration. */
    for (integ=0; integ < pr->nints; ++integ) {
        current_integration = integ;

        if (ramp_fit_pixel_integration(rd, pr, integ)) {
            ret = 1;
            goto END;
        }
        
        if (pr->rateints[integ].dq & rd->dnu) {
            dnu_cnt++;
            pr->rateints[integ].slope = NAN;
        }
        if (pr->rateints[integ].dq & rd->sat) {
            sat_cnt++;
            pr->rateints[integ].slope = NAN;
        }

        if (rd->save_opt) {
            get_pixel_ramp_integration_segments_and_pedestal(integ, pr, rd);
        }
    }

    if (rd->nints == dnu_cnt) {
        pr->rate.dq |= rd->dnu;
    }
    if (rd->nints == sat_cnt) {
        pr->rate.dq |= rd->sat;
    }

    if (pr->rate.var_poisson > 0.) {
        pr->rate.var_poisson = 1. / pr->rate.var_poisson;
    }
    if ((pr->rate.var_poisson >= LARGE_VARIANCE_THRESHOLD)
         || (pr->rate.var_poisson < 0.)){
        pr->rate.var_poisson = 0.;
    }
    if (pr->rate.var_rnoise > 0.) {
        pr->rate.var_rnoise = 1. / pr->rate.var_rnoise;
    }
    if (pr->rate.var_rnoise >= LARGE_VARIANCE_THRESHOLD) {
        pr->rate.var_rnoise = 0.;
    }
    pr->rate.var_err = sqrt(pr->rate.var_poisson + pr->rate.var_rnoise);

    if (pr->rate.dq & rd->invalid) {
        pr->rate.slope = NAN;
        pr->rate.var_poisson = 0.;
        pr->rate.var_rnoise = 0.;
        pr->rate.var_err = 0.;
    }

    if (!isnan(pr->rate.slope)) {
        pr->rate.slope = pr->rate.slope / pr->invvar_e_sum;
    }

    // DBG_RATE_INFO;  /* XXX */

END:
    return ret;
}

/*
 * Compute the ramp fit for a specific integratio.
 */
static int
ramp_fit_pixel_integration(
        struct ramp_data * rd,  /* The ramp data */
        struct pixel_ramp * pr, /* The pixel ramp data */
        npy_intp integ)         /* The integration number */
{
    int ret = 0;

    if (compute_integration_segments(rd, pr, integ)) {
        ret = 1;
        goto END;
    }

    if (rd->ngroups == pr->stats[integ].cnt_dnu_sat) {
        pr->rateints[integ].dq |= rd->dnu;
        if (rd->ngroups == pr->stats[integ].cnt_sat) {
            pr->rateints[integ].dq |= rd->sat;
        }
        return 0;
    } 

    ramp_fit_pixel_integration_fit_slope(rd, pr, integ);

END:
    return ret;
}

#define DBG_SEG_ID do {\
    dbg_ols_print("   *** [Integ: %ld] (%ld, %ld) Seg: %d, Length: %ld, Start: %ld, End: %ld\n", \
        integ, pr->row, pr->col, segcnt, current->length, current->start, current->end); \
} while(0)

#define DBG_INTEG_INFO do {\
    dbg_ols_print("Integ %ld slope: %.10f\n", integ, pr->rateints[integ].slope); \
    dbg_ols_print("Integ %ld dq: %.10f\n", integ, pr->rateints[integ].dq); \
    dbg_ols_print("Integ %ld var_p: %.10f\n", integ, pr->rateints[integ].var_poisson); \
    dbg_ols_print("Integ %ld var_r: %.10f\n\n", integ, pr->rateints[integ].var_rnoise); \
} while(0)

#define DBG_DEFAULT_SEG do {\
    dbg_ols_print("current->slope = %.10f\n", current->slope); \
    dbg_ols_print("current->var_p = %.10f\n", current->var_p); \
    dbg_ols_print("current->var_r = %.10f\n", current->var_r); \
    dbg_ols_print("current->var_e = %.10f\n", current->var_e); \
} while(0)

/*
 * Fit a slope to a pixel integration.
 */
static int
ramp_fit_pixel_integration_fit_slope(
        struct ramp_data * rd,  /* The ramp data */
        struct pixel_ramp * pr, /* The pixel ramp data */
        npy_intp integ)         /* The integration number */
{
    int ret = 0;
    int segcnt = 0;
    struct simple_ll_node * current = NULL;
    real_t invvar_r=0., invvar_p=0., invvar_e=0., slope_i_num=0., var_err;

    if (pr->segs[integ].size == 0) {
        return ret;
    }

    /* Fit slope to each segment. */
    for (current = pr->segs[integ].head; current; current = current->flink) {
        segcnt++;

        // DBG_SEG_ID;  /* XXX */

        ret = ramp_fit_pixel_integration_fit_slope_seg(
                    current, rd, pr, integ, segcnt);
        if (-1 == ret) {
            continue;
        }

        // DBG_DEFAULT_SEG; /* XXX */

        invvar_r += (1. / current->var_r);
	if (current->var_p > 0.) {
            invvar_p += (1. / current->var_p);
        }

        invvar_e += (1. / current->var_e);
        slope_i_num += (current->slope / current->var_e);
    }

    /* Get rateints computations */
    if (invvar_p > 0.) {
        pr->rateints[integ].var_poisson = 1. / invvar_p;
    }

    if (pr->rateints[integ].var_poisson >= LARGE_VARIANCE_THRESHOLD) {
        pr->rateints[integ].var_poisson = 0.;
    }

    pr->rateints[integ].var_rnoise = 1. / invvar_r;
    if (pr->rateints[integ].var_rnoise >= LARGE_VARIANCE_THRESHOLD) {
        pr->rateints[integ].var_rnoise = 0.;
    }

    if (pr->rateints[integ].dq & rd->invalid) {
        pr->rateints[integ].slope = NAN;
        pr->rateints[integ].var_poisson = 0.;
        pr->rateints[integ].var_rnoise = 0.;
        pr->rateints[integ].var_err = 0.;
    } else {
        var_err = 1. / invvar_e;

        pr->rateints[integ].slope = slope_i_num * var_err;
        if (var_err > LARGE_VARIANCE_THRESHOLD) {
            pr->rateints[integ].var_err = 0.;
        } else {
            pr->rateints[integ].var_err = sqrt(var_err);
        }
    }

    // DBG_INTEG_INFO;  /* XXX */

    /* Get rate pre-computations */
    if (invvar_p > 0.) {
        pr->rate.var_poisson += invvar_p;
    }
    pr->rate.var_rnoise += invvar_r;
    pr->invvar_e_sum += invvar_e;
    pr->rate.slope += slope_i_num;

    return ret;
}

/*
 * Fit a slope to an integration segment.
 */
static int
ramp_fit_pixel_integration_fit_slope_seg(
        struct simple_ll_node * current, /* The current segment */
        struct ramp_data * rd,           /* The ramp data */
        struct pixel_ramp * pr,          /* The pixel ramp data */
        npy_intp integ,                  /* The integration number */
        int segnum)                      /* The segment number */
{
    // dbg_ols_print("[%ld] segnum = %d, length = %ld\n", integ, segnum, current->length);
    if (1 == current->length) {
        // dbg_ols_print("(%ld, %ld) Segment %d has length 1\n", pr->row, pr->col, segnum);
        rd->special1++;
        return ramp_fit_pixel_integration_fit_slope_seg_len1(
                rd, pr, current, integ, segnum);
    } else if (2 == current->length) {
        // dbg_ols_print("(%ld, %ld) Segment %d has length 2\n", pr->row, pr->col, segnum);
        rd->special2++;
        return ramp_fit_pixel_integration_fit_slope_seg_len2(
                    rd, pr, current, integ, segnum);
    }
    // dbg_ols_print("(%ld, %ld) Segment %d has length >2\n", pr->row, pr->col, segnum);

    return ramp_fit_pixel_integration_fit_slope_seg_default(
                    rd, pr, current, integ, segnum);
}

/*
 * The default computation for a segment.
 */
static int
ramp_fit_pixel_integration_fit_slope_seg_default(
        struct ramp_data * rd,      /* The ramp data */
        struct pixel_ramp * pr,     /* The pixel ramp data */
        struct simple_ll_node * seg,/* The integration segment */
        npy_intp integ,             /* The integration number */
        int segnum)                 /* Teh segment number */
{
    int ret = 0;
    real_t snr, power;

    if (segment_snr(&snr, integ, rd, pr, seg, segnum)) {
        return 1;
    }
    power = snr_power(snr);
    if (WEIGHTED == rd->weight) {
        ramp_fit_pixel_integration_fit_slope_seg_default_weighted(
            rd, pr, seg, integ, segnum, power);
    } else {
        err_ols_print("Only 'optimal' weighting is allowed for OLS.");
        return 1;
    }

    return ret;
}

/*
 * Fit slope for the special case of an integration
 * segment of length 1.
 */
static int
ramp_fit_pixel_integration_fit_slope_seg_len1(
        struct ramp_data * rd,      /* The ramp data */
        struct pixel_ramp * pr,     /* The pixel ramp data */
        struct simple_ll_node * seg,/* The ingtegration segment */
        npy_intp integ,             /* The integration number */
        int segnum)                 /* The segment integration */
{
    npy_intp idx;
    real_t timing = rd->group_time;
    real_t pden, rnum, rden, tmp;

    /* Check for special cases */
    if (!rd->suppress1g) {
        if (pr->is_0th[integ]) {
            timing = rd->one_group_time;
        } else if (pr->is_zframe[integ]) {
            timing = rd->frame_time;
        }
    }

    idx = get_ramp_index(rd, integ, seg->start);

    seg->slope = pr->data[idx] / timing;

    /* Segment Poisson variance */
    pden = (timing * pr->gain);
    if (pr->median_rate > 0.) {
        seg->var_p = (pr->median_rate + pr->dcurrent) / pden;
    } else {
        seg->var_p = (pr->dcurrent)/pden;
    }

    /* Segment read noise variance */
    rnum = pr->rnoise / timing;
    rnum = 12. * rnum * rnum;
    rden = 6.;  /* seglen * seglen * seglen - seglen; where siglen = 2 */
    rden = rden * pr->gain * pr->gain;
    seg->var_r = rnum / rden;
    seg->var_e = seg->var_p + seg->var_r;

    if (rd->save_opt) {
        tmp = 1. / seg->var_e;
        seg->weight = tmp * tmp;
    }

    return 0;
}

/*
 * Fit slope for the special case of an integration
 * segment of length 2.
 */
static int
ramp_fit_pixel_integration_fit_slope_seg_len2(
        struct ramp_data * rd,      /* The ramp data */
        struct pixel_ramp * pr,     /* The pixel ramp data */
        struct simple_ll_node * seg,/* The integration segment */
        npy_intp integ,             /* The integration number */
        int segnum)                 /* The segment number */
{
    npy_intp idx;
    real_t data_diff, _2nd_read, data0, data1, rnum, rden, pden;
    real_t sqrt2 = 1.41421356;  /* The square root of 2 */
    real_t tmp, wt;

    // dbg_ols_print("   *** Seg %d, Length: %ld (%ld, %ld) ***\n",
    //         segnum, seg->length, seg->start, seg->end);

    /* Special case of 2 group segment */
    idx = get_ramp_index(rd, integ, seg->start);
    data0 = pr->data[idx];
    data1 = pr->data[idx+1];
    data_diff = pr->data[idx+1] - pr->data[idx];
    seg->slope = data_diff / rd->group_time;

    /* Segment Poisson variance */
    pden = (rd->group_time * pr->gain);
    if (pr->median_rate > 0.) {
        seg->var_p = (pr->median_rate + pr->dcurrent) / pden;
    } else {
        seg->var_p = (pr->dcurrent)/pden;
    }

    /* Segment read noise variance */
    rnum = pr->rnoise / rd->group_time;
    rnum = 12. * rnum * rnum;
    rden = 6.; // seglen * seglen * seglen - seglen; where siglen = 2
    rden = rden * pr->gain * pr->gain;
    seg->var_r = rnum / rden;

    /* Segment total variance */
    // seg->var_e = 2. * pr->rnoise * pr->rnoise;  /* XXX Is this right? */
    seg->var_e = seg->var_p + seg->var_r;
#if 0
    if (is_pix_in_list(pr)) {
        tmp = sqrt2 * pr->rnoise;
        dbg_ols_print("rnoise     = %.10f\n", pr->rnoise);
        dbg_ols_print("seg->var_s = %.10f\n", tmp);
        dbg_ols_print("seg->var_p = %.10f\n", seg->var_p);
        dbg_ols_print("seg->var_r = %.10f\n", seg->var_r);
        dbg_ols_print("seg->var_e = %.10f\n", seg->var_e);
    }
#endif

    if (rd->save_opt) {
        seg->sigslope = sqrt2 * pr->rnoise;
        _2nd_read = (real_t)seg->start + 1.;
        seg->yint = data1 * (1. - _2nd_read) + data0 * _2nd_read;
        seg->sigyint = seg->sigslope;

        /* WEIGHTS */
        tmp = (seg->var_p + seg->var_r);
        wt = 1. / tmp;
        wt *= wt;
        seg->weight = wt;
    }

    return 0;
}

/*
 * Compute the optimally weighted OLS fit for a segment.
 */
static void
ramp_fit_pixel_integration_fit_slope_seg_default_weighted(
        struct ramp_data * rd,          /* The ramp data */
        struct pixel_ramp * pr,         /* The pixel ramp data */
        struct simple_ll_node * seg,    /* The integration ramp */
        npy_intp integ,                 /* The integration number */
        int segnum,                     /* The segment number */
        real_t power)                   /* The power of the segment */
{
    struct ols_calcs ols = {0};
    
    /* Make sure the initial values are zero */
    memset(&ols, 0, sizeof(ols));
    ramp_fit_pixel_integration_fit_slope_seg_default_weighted_ols(
        rd, pr, seg, &ols, integ, segnum, power);

    /* From weighted OLS variables fit segment. */
    ramp_fit_pixel_integration_fit_slope_seg_default_weighted_seg(
        rd, pr, seg, &ols, integ, segnum, power);
}

/*
 * Compute the intermediate values for the optimally weighted
 * OLS fit for a segment.
 */
static void
ramp_fit_pixel_integration_fit_slope_seg_default_weighted_ols(
        struct ramp_data * rd,          /* The ramp data */
        struct pixel_ramp * pr,         /* The pixel ramp data */
        struct simple_ll_node * seg,    /* The intgration segment */
        struct ols_calcs * ols,         /* Intermediate calculations */
        npy_intp integ,                 /* The integration number */
        int segnum,                     /* The segment number */
        real_t power)                   /* The power of the segment */
{
    npy_intp idx, group;
    real_t mid, weight, invrn2, invmid, data, xval, xwt;
    
    /* Find midpoint for weight computation */
    mid = (real_t)(seg->length - 1) / 2.;
    invmid = 1. / mid;
    invrn2 = 1. / (pr->rnoise * pr->rnoise);

    idx = get_ramp_index(rd, integ, seg->start);
    for (group=0; group < seg->length; ++group) {
        /* Compute the optimal weight (is 0 based). */
        xval = (real_t)group;
        weight = fabs((xval - mid) * invmid);
        weight = powf(weight, power) * invrn2;

        /* Adjust xval to the actual group number in the ramp. */
        xval += (real_t)seg->start;

        data = pr->data[idx + group];
        data = (isnan(data)) ? 0. : data;
        xwt = xval * weight;

        /* Weighted OLS values */
        ols->sumw += weight;
        ols->sumx += xwt;
        ols->sumxx += (xval * xwt);
        ols->sumy += (data * weight);
        ols->sumxy += (data * xwt);
    }
}

#define DBG_OLS_CALCS do { \
    dbg_ols_print("sumx = %f\n", sumx); \
    dbg_ols_print("sumxx = %f\n", sumxx); \
    dbg_ols_print("sumy = %f\n", sumy); \
    dbg_ols_print("sumxy = %f\n", sumxy); \
    dbg_ols_print("sumw = %f\n", sumw); \
    dbg_ols_print("num = %f\n", num); \
    dbg_ols_print("den = %f\n", den); \
    dbg_ols_print("slope = %f\n", slope); \
} while(0)


/*
 * From the intermediate values compute the optimally weighted
 * OLS fit for a segment.
 */
static void
ramp_fit_pixel_integration_fit_slope_seg_default_weighted_seg(
        struct ramp_data * rd,      /* The ramp data */
        struct pixel_ramp * pr,     /* The pixel ramp data */
        struct simple_ll_node * seg,/* The integration segment */
        struct ols_calcs * ols,     /* Intermediate calculations */
        npy_intp integ,             /* The integration number */
        int segnum,                 /* The segment number */
        real_t power)               /* The power of the segment */
{
    real_t slope, num, den, invden, rnum=0., rden=0., pden=0., seglen;
    real_t sumx=ols->sumx, sumxx=ols->sumxx, sumy=ols->sumy,
          sumxy=ols->sumxy, sumw=ols->sumw;

    den = (sumw * sumxx - sumx * sumx);
    num = (sumw * sumxy - sumx * sumy);
    invden = 1. / den;

    /* Segment slope and uncertainty */
    slope = num * invden;
    seg->slope = slope / rd->group_time;
    seg->sigslope = sqrt(sumw * invden);

    // DBG_OLS_CALCS;

    /* Segment Y-intercept and uncertainty */
    seg->yint = (sumxx * sumy - sumx * sumxy) * invden;
    seg->sigyint = sqrt(sumxx * invden);

    seglen = (float)seg->length;

    /* Segment Poisson variance */
    pden = (rd->group_time * pr->gain * (seglen - 1.));
    if (pr->median_rate > 0.) {
        seg->var_p = (pr->median_rate + pr->dcurrent) / pden;
    } else {
        seg->var_p = (pr->dcurrent) / pden;
    }

    /* Segment read noise variance */
    if ((pr->gain <= 0.) || (isnan(pr->gain))) {
        seg->var_r = 0.;
    } else {
        rnum = pr->rnoise / rd->group_time;
        rnum = 12. * rnum * rnum;
        rden = seglen * seglen * seglen - seglen;

        rden = rden * pr->gain * pr->gain;
        seg->var_r = rnum / rden;
    }

    /* Segment total variance */
    seg->var_e = seg->var_p + seg->var_r;

    if (rd->save_opt) {
        seg->weight = 1. / seg->var_e;
        seg->weight *= seg->weight;
    }
}

/*
 * Save off the optional results calculations in the
 * optional results product.
 */
static int
save_opt_res(
        struct opt_res_product * opt_res, /* The optional results product */
        struct ramp_data * rd)            /* The ramp data */
{
    void * ptr = NULL;
    npy_intp integ, segnum, row, col, idx;
    struct simple_ll_node * current;
    struct simple_ll_node * next;
#if REAL_IS_DOUBLE
    float float_tmp;
#endif

    //dbg_ols_print(" **** %s ****\n", __FUNCTION__);

    /*
       XXX Possibly use a temporary float value to convert the doubles
           in the ramp_data to floats to be put into the opt_res.
     */

    for (integ=0; integ < rd->nints; integ++) {
        for (row=0; row < rd->nrows; row++) {
            for (col=0; col < rd->ncols; col++) {
                idx = get_cube_index(rd, integ, row, col);

                ptr = PyArray_GETPTR3(opt_res->pedestal, integ, row, col);
#if REAL_IS_DOUBLE
                float_tmp = (float) rd->pedestal[idx];
                memcpy(ptr, &(float_tmp), sizeof(float_tmp));
#else
                memcpy(ptr, &(rd->pedestal[idx]), sizeof(rd->pedestal[idx]));
#endif

                segnum = 0;
                current = rd->segs[idx];
                while(current) {
                    next = current->flink;
                    //print_segment_opt_res(current, rd, integ, segnum, __LINE__);

                    ptr = PyArray_GETPTR4(opt_res->slope, integ, segnum, row, col);
#if REAL_IS_DOUBLE
                    float_tmp = (float) current->slope;
                    memcpy(ptr, &(float_tmp), sizeof(float_tmp));
#else
                    memcpy(ptr, &(current->slope), sizeof(current->slope));
#endif

                    ptr = PyArray_GETPTR4(opt_res->sigslope, integ, segnum, row, col);
#if REAL_IS_DOUBLE
                    float_tmp = (float) current->sigslope;
                    memcpy(ptr, &(float_tmp), sizeof(float_tmp));
#else
                    memcpy(ptr, &(current->sigslope), sizeof(current->sigslope));
#endif

                    ptr = PyArray_GETPTR4(opt_res->var_p, integ, segnum, row, col);
#if REAL_IS_DOUBLE
                    float_tmp = (float) current->var_p;
                    memcpy(ptr, &(float_tmp), sizeof(float_tmp));
#else
                    memcpy(ptr, &(current->var_p), sizeof(current->var_p));
#endif

                    ptr = PyArray_GETPTR4(opt_res->var_r, integ, segnum, row, col);
#if REAL_IS_DOUBLE
                    float_tmp = (float) current->var_r;
                    memcpy(ptr, &(float_tmp), sizeof(float_tmp));
#else
                    memcpy(ptr, &(current->var_r), sizeof(current->var_r));
#endif

                    ptr = PyArray_GETPTR4(opt_res->yint, integ, segnum, row, col);
#if REAL_IS_DOUBLE
                    float_tmp = (float) current->yint;
                    memcpy(ptr, &(float_tmp), sizeof(float_tmp));
#else
                    memcpy(ptr, &(current->yint), sizeof(current->yint));
#endif

                    ptr = PyArray_GETPTR4(opt_res->sigyint, integ, segnum, row, col);
#if REAL_IS_DOUBLE
                    float_tmp = (float) current->sigyint;
                    memcpy(ptr, &(float_tmp), sizeof(float_tmp));
#else
                    memcpy(ptr, &(current->sigyint), sizeof(current->sigyint));
#endif

                    ptr = PyArray_GETPTR4(opt_res->weights, integ, segnum, row, col);
#if REAL_IS_DOUBLE
                    float_tmp = (float) current->weight;
                    memcpy(ptr, &(float_tmp), sizeof(float_tmp));
#else
                    memcpy(ptr, &(current->weight), sizeof(current->weight));
#endif

                    current = next;
                    segnum++;
                } /* Segment list loop */
            } /* Column loop */
        } /* Row loop */
    } /* Integration loop */

    return 0;
}

/*
 * Save off the ramp fit computations to the output products.
 */
static int
save_ramp_fit(
        struct rateint_product * rateint_prod,  /* The rateints product */
        struct rate_product * rate_prod,        /* The rate product */
        struct pixel_ramp * pr)                 /* The pixel ramp data */
{
    void * ptr = NULL;
    npy_intp integ;
#if REAL_IS_DOUBLE
    float float_tmp;
#endif

    /* Get rate product information for the pixel */
    ptr = PyArray_GETPTR2(rate_prod->slope, pr->row, pr->col);
#if REAL_IS_DOUBLE
    float_tmp = (float) pr->rate.slope;
    memcpy(ptr, &(float_tmp), sizeof(float_tmp));
#else
    memcpy(ptr, &(pr->rate.slope), sizeof(pr->rate.slope));
#endif

    ptr = PyArray_GETPTR2(rate_prod->dq, pr->row, pr->col);
    memcpy(ptr, &(pr->rate.dq), sizeof(pr->rate.dq));

    ptr = PyArray_GETPTR2(rate_prod->var_poisson, pr->row, pr->col);
#if REAL_IS_DOUBLE
    float_tmp = (float) pr->rate.var_poisson;
    memcpy(ptr, &(float_tmp), sizeof(float_tmp));
#else
    memcpy(ptr, &(pr->rate.var_poisson), sizeof(pr->rate.var_poisson));
#endif

    ptr = PyArray_GETPTR2(rate_prod->var_rnoise, pr->row, pr->col);
#if REAL_IS_DOUBLE
    float_tmp = (float) pr->rate.var_rnoise;
    memcpy(ptr, &(float_tmp), sizeof(float_tmp));
#else
    memcpy(ptr, &(pr->rate.var_rnoise), sizeof(pr->rate.var_rnoise));
#endif

    ptr = PyArray_GETPTR2(rate_prod->var_err, pr->row, pr->col);
#if REAL_IS_DOUBLE
    float_tmp = (float) pr->rate.var_err;
    memcpy(ptr, &(float_tmp), sizeof(float_tmp));
#else
    memcpy(ptr, &(pr->rate.var_err), sizeof(pr->rate.var_err));
#endif

    /* Get rateints product information for the pixel */
    for (integ=0; integ < pr->nints; integ++) {
        ptr = PyArray_GETPTR3(rateint_prod->slope, integ, pr->row, pr->col);
#if REAL_IS_DOUBLE
        float_tmp = (float) pr->rateints[integ].slope;
        memcpy(ptr, &(float_tmp), sizeof(float_tmp));
#else
        memcpy(ptr, &(pr->rateints[integ].slope), sizeof(pr->rateints[integ].slope));
#endif

        ptr = PyArray_GETPTR3(rateint_prod->dq, integ, pr->row, pr->col);
        memcpy(ptr, &(pr->rateints[integ].dq), sizeof(pr->rateints[integ].dq));

        ptr = PyArray_GETPTR3(rateint_prod->var_poisson, integ, pr->row, pr->col);
#if REAL_IS_DOUBLE
        float_tmp = (float) pr->rateints[integ].var_poisson;
        memcpy(ptr, &(float_tmp), sizeof(float_tmp));
#else
        memcpy(ptr, &(pr->rateints[integ].var_poisson), sizeof(pr->rateints[integ].var_poisson));
#endif

        ptr = PyArray_GETPTR3(rateint_prod->var_rnoise, integ, pr->row, pr->col);
#if REAL_IS_DOUBLE
        float_tmp = (float) pr->rateints[integ].var_rnoise;
        memcpy(ptr, &(float_tmp), sizeof(float_tmp));
#else
        memcpy(ptr, &(pr->rateints[integ].var_rnoise), sizeof(pr->rateints[integ].var_rnoise));
#endif

        ptr = PyArray_GETPTR3(rateint_prod->var_err, integ, pr->row, pr->col);
#if REAL_IS_DOUBLE
        float_tmp = (float) pr->rateints[integ].var_err;
        memcpy(ptr, &(float_tmp), sizeof(float_tmp));
#else
        memcpy(ptr, &(pr->rateints[integ].var_err), sizeof(pr->rateints[integ].var_err));
#endif
    }

    return 0;
}

/* Compute the signal to noise ratio of the segment. */
static int
segment_snr(
        real_t * snr,               /* The signal to noise ratio for a segment */
        npy_intp integ,             /* The intergration number */
        struct ramp_data * rd,      /* The ramp data */
        struct pixel_ramp * pr,     /* The pixel ramp data */
        struct simple_ll_node * seg,/* The integration segment */
        int segnum)                 /* The segment number */
{
    npy_intp idx_s, idx_e;
    real_t data, num, den, S, start, end, sqrt_den = 0.;

    idx_s = get_ramp_index(rd, integ, seg->start);
    idx_e = idx_s + seg->length - 1;
    end = pr->data[idx_e];
    start = pr->data[idx_s];
    data = end - start;
    den = pr->rnoise * pr->rnoise + data * pr->gain;

    if ((den <= 0.) || (pr->gain == 0.)) {
        *snr = 0.;
    } else {
        num = data * pr->gain;
        sqrt_den = sqrt(den);
        S = num / sqrt_den;
        *snr = (S < 0.) ? 0. : S;
    }

    return 0;
}

/* Compute the weighting power based on the SNR. */
static real_t
snr_power(
        real_t snr)  /* The signal to noise ratio of a segment */
{
    if (snr < 5.) {
        return  0.;
    } else if (snr < 10.) {
        return  0.4;
    } else if (snr < 20.) {
        return  1.;
    } else if (snr < 50.) {
        return  3.;
    } else if (snr < 100.) {
        return  6.;
    }
    return 10.;
}
/* ------------------------------------------------------------------------- */

/* ------------------------------------------------------------------------- */
/*                            Debug Functions                                */
/* ------------------------------------------------------------------------- */

/*
 * This prints some of the ramp_data information.  This function is primarily
 * used for debugging and development purposes.
 */
static void
print_ramp_data_info(
        struct ramp_data * rd)
{
    printf("    Data\n");
    printf("Dims = [%ld, %ld, %ld, %ld]\n", rd->nints, rd->ngroups, rd->nrows, rd->ncols);

    printf("\n    Meta Data\n");
    printf("Frame Time: %f\n", rd->frame_time);
    printf("Group Time: %f\n", rd->group_time);
    printf("Group Gap: %d\n", rd->groupgap);
    printf("NFrames: %d\n", rd->nframes);

    printf("\n    Flags\n");
    printf("DO_NOT_USE:  %08x\n", rd->dnu);
    printf("JUMP_DET:    %08x\n", rd->jump);
    printf("SATURATED:   %08x\n", rd->sat);
    printf("NO_GAIN_VAL: %08x\n", rd->ngval);
    printf("UNRELIABLE:  %08x\n", rd->uslope);
}

static void
print_segment_list(
        npy_intp nints,
        struct segment_list * segs,
        int line)
{
    npy_intp integ;
    struct simple_ll_node * current;
    struct simple_ll_node * next;
    const char * indent = "    ";

    print_delim();
    printf("[%d] Ingegration Segments\n", line);
    for (integ=0; integ<nints; ++integ) {
        current = segs[integ].head;
        printf("%sSegments of Integration [%ld] has %ld segment(s), with max length %ld.\n",
            indent, integ, segs[integ].size, segs[integ].max_segment_length);
        while(current) {
            next = current->flink;
            printf("%s%s(%ld, %ld) - %ld\n",
                indent, indent, current->start, current->end, current->length);
            current = next;
        }
    }
    print_delim();
}

static void
print_segment_list_integ(
        npy_intp integ,
        struct segment_list * segs,
        int line)
{
    struct simple_ll_node * current;

    print_delim();
    dbg_ols_print("[%d] Integration %ld has %zd segments\n", line, integ, segs[integ].size);
    for (current=segs[integ].head; current; current=current->flink) {
        dbg_ols_print("    Start = %ld, End = %ld\n", current->start, current->end);
    }
    print_delim();
}


static void
print_segment(
        struct simple_ll_node * seg,
        struct ramp_data * rd,
        struct pixel_ramp * pr,
        npy_intp integ,
        int segnum,
        int line)
{
    npy_intp idx, group;

    print_delim();
    if (line > 0) {
        printf("[%d] - ", line);
    }
    printf("Integration %ld, segment %d, has length %ld.\n", integ, segnum, seg->length);
    
    idx = get_ramp_index(rd, integ, seg->start);
    printf("Science Data\n[%" DBL, pr->data[idx]);
    for (group = seg->start + 1; group < seg->end; ++group) {
        idx = get_ramp_index(rd, integ, group);
        printf(", %" DBL, pr->data[idx]);
    }
    printf("]\n");
    
    idx = get_ramp_index(rd, integ, seg->start);
    printf("Group DQ\n[%02x", pr->groupdq[idx]);
    for (group = seg->start + 1; group < seg->end; ++group) {
        idx = get_ramp_index(rd, integ, group);
        printf(", %02x", pr->groupdq[idx]);
    }
    printf("]\n");
    print_delim();
}

static void
print_segment_opt_res(
        struct simple_ll_node * seg,
        struct ramp_data * rd,
        npy_intp integ,
        int segnum,
        int line)
{
        print_delim();
        printf("[%d] Integration: %ld, Segment: %d\n", line, integ, segnum);

        printf("slope    = %f\n", seg->slope);
        printf(" **slope = %f (divide by group time)\n", seg->slope / rd->group_time);
        printf("sigslope = %f\n", seg->sigslope);

        printf("yint     = %f\n", seg->yint);
        printf("sigyint  = %f\n", seg->sigyint);

        printf("var_p    = %f\n", seg->var_p);
        printf("var_r    = %f\n", seg->var_r);

        printf("yint     = %f\n", seg->yint);
        printf("sigyint  = %f\n", seg->sigyint);

        printf("weight   = %f\n", seg->weight);

        print_delim();
}

static void
print_stats(struct pixel_ramp * pr, npy_intp integ, int line) {
    print_delim();
    printf("[%d] GDQ stats for integration %ld\n", line, integ);
    dbg_ols_print("    cnt_sat = %d\n", pr->stats[integ].cnt_sat);
    dbg_ols_print("    cnt_dnu = %d\n", pr->stats[integ].cnt_dnu);
    dbg_ols_print("    cnt_dnu_sat = %d\n", pr->stats[integ].cnt_dnu_sat);
    dbg_ols_print("    cnt_good = %d\n", pr->stats[integ].cnt_good);
    dbg_ols_print("    jump_det = %d\n", pr->stats[integ].jump_det);
    print_delim();
}

static void
print_uint8_array(uint8_t * arr, int len, int ret, int line) {
    int k;

    if (line>0) {
        printf("[Line %d] ", line);
    }

    if (len < 1) {
        printf("[void]");
        return;
    }
    printf("[%02x", arr[0]);
    for (k=1; k<len; ++k) {
        printf(", %02x", arr[k]);
    }
    printf("]");
    if (ret) {
        printf("\n");
    }
    return;
}

static void
print_uint32_array(uint32_t * arr, int len, int ret) {
    int k;

    if (len < 1) {
        printf("[void]");
        return;
    }
    printf("[%d", arr[0]);
    for (k=1; k<len; ++k) {
        printf(", %d", arr[k]);
    }
    printf("]");
    if (ret) {
        printf("\n");
    }
    return;
}

static void
print_ols_calcs(struct ols_calcs * ols, npy_intp integ, int segnum, int line)
{
    print_delim();
    printf("*** [%d] Segment %d, Integration %ld ***\n", line, segnum, integ);
    dbg_ols_print("    sumx  = %.12f\n", ols->sumx);
    dbg_ols_print("    sumxx = %.12f\n", ols->sumxx);
    dbg_ols_print("    sumy  = %.12f\n", ols->sumy);
    dbg_ols_print("    sumxy = %.12f\n", ols->sumxy);
    dbg_ols_print("    sumw  = %.12f\n", ols->sumw);
    print_delim();
}

static void
print_pixel_ramp_data(
        struct ramp_data * rd,
        struct pixel_ramp * pr,
        int line) {
    npy_intp integ, group;
    ssize_t idx;

    if (line > 0) {
        printf("Line: %d - \n", line);
    }
    for (integ = 0; integ < pr->nints; ++integ) {
        idx = get_ramp_index(rd, integ, 0);
        printf("[%ld] [%" DBL, integ, pr->data[idx]);
        for (group = 1; group < pr->ngroups; ++group) {
            idx = get_ramp_index(rd, integ, group);
            printf(", %" DBL, pr->data[idx]);
        }
        printf("]\n");
    }
}
static void
print_pixel_ramp_dq(
        struct ramp_data * rd,
        struct pixel_ramp * pr,
        int line) {
    npy_intp integ, group;
    ssize_t idx;

    if (line > 0) {
        printf("Line: %d - \n", line);
    }
    for (integ = 0; integ < pr->nints; ++integ) {
        idx = get_ramp_index(rd, integ, 0);
        printf("[%ld] (%ld, %p) [%02x", integ, idx, pr->groupdq + idx, pr->groupdq[idx]);
        for (group = 1; group < pr->ngroups; ++group) {
            idx = get_ramp_index(rd, integ, group);
            printf(", %02x", pr->groupdq[idx]);
        }
        printf("]\n");
    }
}

static void
print_pixel_ramp_info(struct ramp_data * rd, struct pixel_ramp * pr, int line) {
    print_delim();
    printf("[%s, %d] Pixel (%ld, %ld)\n", __FUNCTION__, line, pr->row, pr->col);
    printf("Data:\n");
    print_pixel_ramp_data(rd, pr, -1);
    printf("DQ:\n");
    print_pixel_ramp_dq(rd, pr, -1);

    print_delim();
}

static void
print_real_array(char * label, real_t * arr, int len, int ret, int line) {
    int k;

    if (line>0) {
        printf("[Line %d] ", line);
    }

    if (NULL != label) {
        printf("%s - ", label);
    }

    if (len < 1) {
        printf("[void]");
        return;
    }
    printf("[%f", arr[0]);
    for (k=1; k<len; ++k) {
        printf(", %.10f", arr[k]);
    }
    printf("]");
    if (ret) {
        printf("\n");
    }
    return;
}

/*
 * This prints an integer array.  If the 'ret' value is non-zero,
 * then print a character after the array.
 */
static void
print_intp_array(npy_intp * arr, int len, int ret) {
    int k;

    if (len < 1) {
        printf("[void]");
        return;
    }
    printf("[%ld", arr[0]);
    for (k=1; k<len; ++k) {
        printf(", %ld", arr[k]);
    }
    printf("]");
    if (ret) {
        printf("\n");
    }
    return;
}

static void
print_pixel_ramp_stats(struct pixel_ramp * pr, int line)
{
    npy_intp integ;

    print_delim();
    printf("[%d] Pixel ramp stats: \n", line);
    for (integ=0; integ<pr->nints; ++integ) {
        print_stats(pr, integ, __LINE__);
        printf("\n");
    }
    print_delim();
}

/*
 * Print some information about a PyArrayObject.  This function is primarily
 * used for debugging and development.
 */
static void
print_PyArrayObject_info(PyArrayObject * obj) {
    int ndims = -1, flags = 0;
    npy_intp * dims = NULL;
    npy_intp * strides = NULL;

    ndims = PyArray_NDIM(obj);
    dims = PyArray_DIMS(obj);
    strides = PyArray_STRIDES(obj);
    flags = PyArray_FLAGS(obj);

    printf("The 'obj' array array has %d dimensions: ", ndims);
    print_intp_array(dims, ndims, 1);
    printf("Strides: ");
    print_intp_array(strides, ndims, 1);

    printf("flags:\n");
    if (NPY_ARRAY_C_CONTIGUOUS & flags ) {
        printf("    NPY_ARRAY_C_CONTIGUOUS\n"); 
    }
    if (NPY_ARRAY_F_CONTIGUOUS & flags ) {
        printf("    NPY_ARRAY_F_CONTIGUOUS\n");
    }
    if (NPY_ARRAY_OWNDATA & flags ) {
        printf("    NPY_ARRAY_OWNDATA\n");
    }
    if (NPY_ARRAY_ALIGNED & flags ) {
        printf("    NPY_ARRAY_ALIGNED\n");
    }
    if (NPY_ARRAY_WRITEABLE & flags ) {
        printf("    NPY_ARRAY_WRITEABLE\n");
    }
    if (NPY_ARRAY_WRITEBACKIFCOPY & flags ) {
        printf("    NPY_ARRAY_WRITEBACKIFCOPY\n");
    }
}

/*
 * Print the values of NPY_{types}.
 */
static void
print_npy_types() {
    printf("NPY_BOOL = %d\n",NPY_BOOL);
    printf("NPY_BYTE = %d\n",NPY_BYTE);
    printf("NPY_UBYTE = %d\n",NPY_UBYTE);
    printf("NPY_SHORT = %d\n",NPY_SHORT);
    printf("NPY_INT = %d\n",NPY_INT);
    printf("NPY_UINT = %d\n",NPY_UINT);

    printf("NPY_FLOAT = %d\n",NPY_FLOAT);
    printf("NPY_DOUBLE = %d\n",NPY_DOUBLE);

    printf("NPY_VOID = %d\n",NPY_VOID);
    printf("NPY_NTYPES_LEGACY = %d\n",NPY_NTYPES_LEGACY);
    printf("NPY_NOTYPE = %d\n",NPY_NOTYPE);
    /*
    NPY_SHORT
    NPY_USHORT
    NPY_INT
    NPY_UINT
    NPY_LONG
    NPY_ULONG
    NPY_LONGLONG
    NPY_ULONGLONG
    NPY_FLOAT
    NPY_DOUBLE
    NPY_LONGDOUBLE
    NPY_CFLOAT
    NPY_CDOUBLE
    NPY_CLONGDOUBLE
    NPY_OBJECT
    NPY_STRING
    NPY_UNICODE
     */
}
/* ========================================================================= */


/* ========================================================================= */
/*                            Python Module API                              */
/* ------------------------------------------------------------------------- */
static PyMethodDef
ols_slope_fitter_methods[] =
{
    {
        "ols_slope_fitter",
	    ols_slope_fitter,
	    METH_VARARGS,
        "Compute the slope and variances using ramp fitting OLS.",
    },
    {NULL, NULL, 0, NULL}               /* Sentinel */
};


static struct PyModuleDef
moduledef = {
    PyModuleDef_HEAD_INIT,
    "slope_fitter",                     /* m_name */
    "Computes slopes and variances",    /* m_doc */
    -1,                                 /* m_size */
    ols_slope_fitter_methods,           /* m_methods */
    NULL,                               /* m_reload */
    NULL,                               /* m_traverse */
    NULL,                               /* m_clear */
    NULL,                               /* m_free */
};

PyMODINIT_FUNC
PyInit_slope_fitter(void)
{
    PyObject* m;
    m = PyModule_Create(&moduledef);
    import_array();
    return m;
}
