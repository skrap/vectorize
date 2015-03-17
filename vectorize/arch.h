#if defined(__arm__)
#include <vectorize/arm_neon.h>

#elif defined(__SSE__)
#include <vectorize/sse.h>

#else
#error Unsupported architecture
#endif