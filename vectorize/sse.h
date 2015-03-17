#include <xmmintrin.h>

namespace vectorize { namespace arch {
    typedef __m128 float32x4_t;
    
    float32x4_t vec_set(float fValue) { return _mm_set1_ps(fValue); }
    float32x4_t vec_load(const float* fVec) { return _mm_load_ps(fVec); }
    void vec_store(float* fTarget, float32x4_t fVec) { _mm_store_ps(fTarget, fVec); }
    
    float32x4_t vec_add(float32x4_t x, float32x4_t y) { return _mm_add_ps(x, y); }
    float32x4_t vec_sub(float32x4_t x, float32x4_t y) { return _mm_sub_ps(x, y); }
    float32x4_t vec_mul(float32x4_t x, float32x4_t y) { return _mm_mul_ps(x, y); }
    float32x4_t vec_min(float32x4_t x, float32x4_t y) { return _mm_min_ps(x, y); }
    float32x4_t vec_max(float32x4_t x, float32x4_t y) { return _mm_max_ps(x, y); }
    
    float32x4_t vec_abs(float32x4_t x) {
        // There's no abs instruction until AVX-512, so we simulate it.
        return _mm_andnot_ps(_mm_set1_ps(-0.0f), x);
    }
    float32x4_t vec_sqrt(float32x4_t x) { return _mm_sqrt_ps(x); }
}}