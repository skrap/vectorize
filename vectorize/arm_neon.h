#include <arm_neon.h>

namespace vectorize { namespace arch {
    typedef float32x4_t float32x4_t;
    float32x4_t vec_set(float fValue) { return vdupq_n_f32(fValue); }
    float32x4_t vec_load(const float* fVec) { return vld1q_f32(fVec); }
    void vec_store(float* fTarget, float32x4_t fVec) { vst1q_f32(fTarget, fVec); }
    
    float32x4_t vec_add(float32x4_t x, float32x4_t y) { return vaddq_f32(x, y); }
    float32x4_t vec_sub(float32x4_t x, float32x4_t y) { return vsubq_f32(x, y); }
    float32x4_t vec_mul(float32x4_t x, float32x4_t y) { return vmulq_f32(x, y); }
    float32x4_t vec_min(float32x4_t x, float32x4_t y) { return vminq_f32(x, y); }
    float32x4_t vec_max(float32x4_t x, float32x4_t y) { return vmaxq_f32(x, y); }
    
    float32x4_t vec_abs(float32x4_t x) { return vabsq_f32(x); }
    float32x4_t vec_sqrt(float32x4_t x) {
        // There is no neon sqrt on armv7, so we need to use VFP to do this
        // calculation.
        float f4[4];
        vst1q_f32(f4,x);
        f4[0] = __builtin_sqrtf(f4[0]);
        f4[1] = __builtin_sqrtf(f4[1]);
        f4[2] = __builtin_sqrtf(f4[2]);
        f4[3] = __builtin_sqrtf(f4[3]);
        return vld1q_f32(f4);
    }
}}