#include <arm_neon.h>

// dot_prefetch_neon: same as dot_neon but prefetches the next pair of vectors
// while computing the current dot product. Useful for sequential scans
// (brute-force search, HNSW neighbor expansion).
//
// next_a and next_b point to the NEXT vectors to be processed.
// Set to NULL on the last iteration.
void dot_prefetch_neon(float *a, float *b, float *res, long *len,
                       float *next_a, float *next_b)
{
    // Prefetch next vectors into L1 data cache.
    if (next_a) __builtin_prefetch(next_a, 0, 3);  // PRFM PLDL1KEEP
    if (next_b) __builtin_prefetch(next_b, 0, 3);

    int n = (int)*len;
    int aligned = n - (n % 4);

    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    float32x4_t acc2 = vdupq_n_f32(0.0f);
    float32x4_t acc3 = vdupq_n_f32(0.0f);

    int i = 0;
    for (; i + 16 <= aligned; i += 16) {
        // Prefetch further ahead in the next vectors.
        if (next_a && i < 128) __builtin_prefetch(next_a + i + 64, 0, 3);
        if (next_b && i < 128) __builtin_prefetch(next_b + i + 64, 0, 3);

        float32x4x4_t va = vld1q_f32_x4(a + i);
        float32x4x4_t vb = vld1q_f32_x4(b + i);
        acc0 = vfmaq_f32(acc0, va.val[0], vb.val[0]);
        acc1 = vfmaq_f32(acc1, va.val[1], vb.val[1]);
        acc2 = vfmaq_f32(acc2, va.val[2], vb.val[2]);
        acc3 = vfmaq_f32(acc3, va.val[3], vb.val[3]);
    }
    for (; i < aligned; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        acc0 = vfmaq_f32(acc0, va, vb);
    }

    acc0 = vaddq_f32(acc0, acc1);
    acc2 = vaddq_f32(acc2, acc3);
    acc0 = vaddq_f32(acc0, acc2);
    float sum = vaddvq_f32(acc0);

    for (int j = aligned; j < n; j++)
        sum += a[j] * b[j];

    *res = sum;
}
