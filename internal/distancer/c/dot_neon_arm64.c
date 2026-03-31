#include <arm_neon.h>

// dot_neon computes the dot product of two float32 slices using NEON FMA.
// Requires len >= 16. Caller handles smaller sizes with pure Go fallback.
void dot_neon(float *a, float *b, float *res, long *len)
{
    int n = (int)*len;
    int aligned = n - (n % 4);

    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    float32x4_t acc2 = vdupq_n_f32(0.0f);
    float32x4_t acc3 = vdupq_n_f32(0.0f);

    int i = 0;
    // Process 16 floats per iteration (4 NEON registers x 4 lanes).
    for (; i + 16 <= aligned; i += 16) {
        float32x4x4_t va = vld1q_f32_x4(a + i);
        float32x4x4_t vb = vld1q_f32_x4(b + i);
        acc0 = vfmaq_f32(acc0, va.val[0], vb.val[0]);
        acc1 = vfmaq_f32(acc1, va.val[1], vb.val[1]);
        acc2 = vfmaq_f32(acc2, va.val[2], vb.val[2]);
        acc3 = vfmaq_f32(acc3, va.val[3], vb.val[3]);
    }
    // Handle remaining 4-float chunks.
    for (; i < aligned; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        acc0 = vfmaq_f32(acc0, va, vb);
    }

    // Horizontal reduction.
    acc0 = vaddq_f32(acc0, acc1);
    acc2 = vaddq_f32(acc2, acc3);
    acc0 = vaddq_f32(acc0, acc2);
    float sum = vaddvq_f32(acc0);

    // Scalar tail.
    for (int j = aligned; j < n; j++)
        sum += a[j] * b[j];

    *res = sum;
}

// neg_dot_neon computes the negative dot product (for HNSW distance).
void neg_dot_neon(float *a, float *b, float *res, long *len)
{
    dot_neon(a, b, res, len);
    *res = -(*res);
}

// l2_neon computes L2 squared distance: sum((a[i]-b[i])^2).
void l2_neon(float *a, float *b, float *res, long *len)
{
    int n = (int)*len;
    int aligned = n - (n % 4);

    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    float32x4_t acc2 = vdupq_n_f32(0.0f);
    float32x4_t acc3 = vdupq_n_f32(0.0f);

    int i = 0;
    for (; i + 16 <= aligned; i += 16) {
        float32x4x4_t va = vld1q_f32_x4(a + i);
        float32x4x4_t vb = vld1q_f32_x4(b + i);
        float32x4_t d0 = vsubq_f32(va.val[0], vb.val[0]);
        float32x4_t d1 = vsubq_f32(va.val[1], vb.val[1]);
        float32x4_t d2 = vsubq_f32(va.val[2], vb.val[2]);
        float32x4_t d3 = vsubq_f32(va.val[3], vb.val[3]);
        acc0 = vfmaq_f32(acc0, d0, d0);
        acc1 = vfmaq_f32(acc1, d1, d1);
        acc2 = vfmaq_f32(acc2, d2, d2);
        acc3 = vfmaq_f32(acc3, d3, d3);
    }
    for (; i < aligned; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        float32x4_t d = vsubq_f32(va, vb);
        acc0 = vfmaq_f32(acc0, d, d);
    }

    acc0 = vaddq_f32(acc0, acc1);
    acc2 = vaddq_f32(acc2, acc3);
    acc0 = vaddq_f32(acc0, acc2);
    float sum = vaddvq_f32(acc0);

    for (int j = aligned; j < n; j++) {
        float d = a[j] - b[j];
        sum += d * d;
    }

    *res = sum;
}
