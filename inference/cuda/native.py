"""ctypes wrapper around the native BF16 decoder (inference/cuda/capi.cu)."""
import ctypes
from pathlib import Path

import numpy as np


class NativeDecoder:
    def __init__(self, library, weights, capacity, yarn):
        lib = ctypes.CDLL(str(library))
        lib.zlm_create.restype = ctypes.c_void_p
        lib.zlm_create.argtypes = (ctypes.c_char_p, ctypes.c_int, ctypes.c_int)
        lib.zlm_destroy.argtypes = (ctypes.c_void_p,)
        lib.zlm_length.restype = ctypes.c_int
        lib.zlm_length.argtypes = (ctypes.c_void_p,)
        lib.zlm_set_length.argtypes = (ctypes.c_void_p, ctypes.c_int)
        lib.zlm_forward.argtypes = (ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p)
        self.lib, self.capacity = lib, capacity
        self.vocab_size = lib.zlm_vocab_size()
        self.handle = lib.zlm_create(str(Path(weights)).encode(), capacity, int(yarn))
        self.logits = np.empty(self.vocab_size, dtype=np.uint16)

    @property
    def length(self):
        return self.lib.zlm_length(self.handle)

    @length.setter
    def length(self, value):
        self.lib.zlm_set_length(self.handle, value)

    def feed(self, token):
        """Append one token to the cache without computing logits."""
        self.lib.zlm_forward(self.handle, token, None)

    def step(self, token):
        """Append one token and return the next-token logits as FP32."""
        self.lib.zlm_forward(self.handle, token, self.logits.ctypes.data)
        return (self.logits.astype(np.uint32) << 16).view(np.float32)

    def close(self):
        self.lib.zlm_destroy(self.handle)
