#include <arm_neon.h>

// vec_mul_f64: dst[i] = a[i] * b[i] (elementwise multiply, float64)
// Used for Hadamard sign flips: dst[i] *= signs[i]
void vec_mul_f64(double *dst, double *a, double *b, long *len)
{
    int n = (int)*len;
    int aligned = n - (n % 2);

    int i = 0;
    for (; i + 8 <= aligned; i += 8) {
        float64x2_t a0 = vld1q_f64(a + i);
        float64x2_t b0 = vld1q_f64(b + i);
        float64x2_t a1 = vld1q_f64(a + i + 2);
        float64x2_t b1 = vld1q_f64(b + i + 2);
        float64x2_t a2 = vld1q_f64(a + i + 4);
        float64x2_t b2 = vld1q_f64(b + i + 4);
        float64x2_t a3 = vld1q_f64(a + i + 6);
        float64x2_t b3 = vld1q_f64(b + i + 6);
        vst1q_f64(dst + i,     vmulq_f64(a0, b0));
        vst1q_f64(dst + i + 2, vmulq_f64(a1, b1));
        vst1q_f64(dst + i + 4, vmulq_f64(a2, b2));
        vst1q_f64(dst + i + 6, vmulq_f64(a3, b3));
    }
    for (; i < aligned; i += 2) {
        float64x2_t va = vld1q_f64(a + i);
        float64x2_t vb = vld1q_f64(b + i);
        vst1q_f64(dst + i, vmulq_f64(va, vb));
    }
    for (; i < n; i++)
        dst[i] = a[i] * b[i];
}

// vec_scale_f64: dst[i] = a[i] * scalar
// Used for NormalizeTo: dst[i] = src[i] * invNorm
void vec_scale_f64(double *dst, double *a, double *scalar, long *len)
{
    int n = (int)*len;
    int aligned = n - (n % 2);
    float64x2_t s = vdupq_n_f64(*scalar);

    int i = 0;
    for (; i + 8 <= aligned; i += 8) {
        float64x2_t a0 = vld1q_f64(a + i);
        float64x2_t a1 = vld1q_f64(a + i + 2);
        float64x2_t a2 = vld1q_f64(a + i + 4);
        float64x2_t a3 = vld1q_f64(a + i + 6);
        vst1q_f64(dst + i,     vmulq_f64(a0, s));
        vst1q_f64(dst + i + 2, vmulq_f64(a1, s));
        vst1q_f64(dst + i + 4, vmulq_f64(a2, s));
        vst1q_f64(dst + i + 6, vmulq_f64(a3, s));
    }
    for (; i < aligned; i += 2) {
        float64x2_t va = vld1q_f64(a + i);
        vst1q_f64(dst + i, vmulq_f64(va, s));
    }
    for (; i < n; i++)
        dst[i] = a[i] * (*scalar);
}

// dot_f64: dot product of float64 slices
void dot_f64(double *a, double *b, double *res, long *len)
{
    int n = (int)*len;
    int aligned = n - (n % 2);

    float64x2_t acc0 = vdupq_n_f64(0.0);
    float64x2_t acc1 = vdupq_n_f64(0.0);
    float64x2_t acc2 = vdupq_n_f64(0.0);
    float64x2_t acc3 = vdupq_n_f64(0.0);

    int i = 0;
    for (; i + 8 <= aligned; i += 8) {
        float64x2_t a0 = vld1q_f64(a + i);
        float64x2_t b0 = vld1q_f64(b + i);
        float64x2_t a1 = vld1q_f64(a + i + 2);
        float64x2_t b1 = vld1q_f64(b + i + 2);
        float64x2_t a2 = vld1q_f64(a + i + 4);
        float64x2_t b2 = vld1q_f64(b + i + 4);
        float64x2_t a3 = vld1q_f64(a + i + 6);
        float64x2_t b3 = vld1q_f64(b + i + 6);
        acc0 = vfmaq_f64(acc0, a0, b0);
        acc1 = vfmaq_f64(acc1, a1, b1);
        acc2 = vfmaq_f64(acc2, a2, b2);
        acc3 = vfmaq_f64(acc3, a3, b3);
    }
    for (; i < aligned; i += 2) {
        float64x2_t va = vld1q_f64(a + i);
        float64x2_t vb = vld1q_f64(b + i);
        acc0 = vfmaq_f64(acc0, va, vb);
    }

    acc0 = vaddq_f64(acc0, acc1);
    acc2 = vaddq_f64(acc2, acc3);
    acc0 = vaddq_f64(acc0, acc2);
    double sum = vgetq_lane_f64(acc0, 0) + vgetq_lane_f64(acc0, 1);

    for (int j = aligned; j < n; j++)
        sum += a[j] * b[j];

    *res = sum;
}
