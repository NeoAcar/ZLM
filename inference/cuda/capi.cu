// C interface to zlm::Decoder for Python ctypes.
// Build: nvcc -O3 -std=c++17 -arch=compute_80 -code=sm_86,compute_80 -shared -Xcompiler -fPIC \
//          inference/cuda/capi.cu -o outputs/native/libzlm_decode.so
// Exceptions are not translated: an invalid call terminates the process with the decoder's message.
#include "decode.cuh"
#include <cstdint>

extern "C" {
void* zlm_create(const char* weights, int capacity, int yarn) {
    return new zlm::Decoder(weights, capacity, yarn != 0);
}
void zlm_destroy(void* decoder) { delete static_cast<zlm::Decoder*>(decoder); }
int zlm_vocab_size() { return zlm::V; }
int zlm_length(void* decoder) { return static_cast<zlm::Decoder*>(decoder)->length; }
// Positions at and after length are overwritten by later forwards, so rewinding reuses the cached prefix.
void zlm_set_length(void* decoder, int length) { static_cast<zlm::Decoder*>(decoder)->length = length; }
// Feeds one token. With a non-null destination, copies the BF16 vocabulary logits there and waits for them.
void zlm_forward(void* decoder, int token, std::uint16_t* logits) {
    auto* self = static_cast<zlm::Decoder*>(decoder);
    zlm::B* output = self->forward(token, logits != nullptr);
    if (logits) {
        zlm::check(cudaMemcpyAsync(logits, output, sizeof(zlm::B) * zlm::V, cudaMemcpyDeviceToHost, self->cuda_stream()));
        zlm::check(cudaStreamSynchronize(self->cuda_stream()));
    }
}
}
