import math
import multiprocessing as mp
import struct
import wave
from functools import partial
from itertools import accumulate

import numpy as np
import scipy.signal
from bitarray import bitarray
from tqdm import tqdm

from encoder import (
    CHUNK_SIZE,
    DIVISOR,
    SAMPLES_LEN,
    BIT_DEPTH,
    CHUNKS_LEN,
    DEGREES,
    CODE_LENS,
    NEXT_DEPTH,
    POLYNOMIALS,
)

PROCESSES = 16


def rice_golomb_decode(code: str, divisor: int) -> list[int]:
    bin_len = int(math.floor(math.log2(divisor)))
    inv_divisor = 2 ** (bin_len + 1) - divisor

    i = 0
    numbers = []
    while i < len(code):
        quotient = next(j for j in range(len(code) - i) if code[i + j] == '0')
        i += quotient + 1

        reminder = int(code[i:i + bin_len], base=2)
        if reminder >= inv_divisor:
            bin_len += 1
            reminder = int(code[i:i + bin_len], base=2) - inv_divisor
        i += bin_len

        dividend = quotient * divisor + reminder
        numbers.append(dividend)
    return numbers


def positive_demapping(array: np.ndarray, array_is_copy=False) -> np.ndarray:
    even = array % 2 == 0
    array[even] = array[even] // 2
    array[~even] = -(array[~even] + 1) // 2
    if array_is_copy:
        return array


def get_chunk(degree_warmup_code: (int, slice, slice), warmups_: tuple, bitbuffer_: bitarray):
    degree, warmup_slice, code_slice = degree_warmup_code
    if degree == 0:
        return np.frombuffer(bitbuffer_[code_slice].tobytes(), dtype=BIT_DEPTH.TYPE)
    else:
        numbers = np.array(rice_golomb_decode(bitbuffer_[code_slice].to01(), divisor=DIVISOR))
        return scipy.signal.deconvolve(
            np.concatenate([
                warmups_[warmup_slice],
                positive_demapping(array=numbers, array_is_copy=True),
                np.zeros(warmup_slice.stop - warmup_slice.start, dtype=NEXT_DEPTH.TYPE)
            ], dtype=NEXT_DEPTH.TYPE),
            POLYNOMIALS[degree],
        )[0].astype(BIT_DEPTH.TYPE)


def slice_by_part_sizes(sizes: list[int]) -> list[slice]:
    cum_lens = [0] + list(accumulate(sizes))
    slices = [slice(a, b) for a, b in zip(cum_lens, cum_lens[1:])]
    return slices


if __name__ == '__main__':
    with open('compressed.flc', 'rb') as f:
        bytes_buffer = f.read()

    return_dict = mp.Manager().dict()

    samples_len = struct.unpack(f'={1}{SAMPLES_LEN.FORMAT}', bytes_buffer[:SAMPLES_LEN.BYTES])[0]
    bytes_buffer = bytes_buffer[SAMPLES_LEN.BYTES:]
    chunks_len = struct.unpack(f'={1}{CHUNKS_LEN.FORMAT}', bytes_buffer[:CHUNKS_LEN.BYTES])[0]
    bytes_buffer = bytes_buffer[CHUNKS_LEN.BYTES:]
    degrees = struct.unpack(f'={chunks_len}{DEGREES.FORMAT}', bytes_buffer[:chunks_len * DEGREES.BYTES])
    bytes_buffer = bytes_buffer[chunks_len * DEGREES.BYTES:]
    code_lens = struct.unpack(f'={chunks_len}{CODE_LENS.FORMAT}', bytes_buffer[:chunks_len * CODE_LENS.BYTES])
    bytes_buffer = bytes_buffer[chunks_len * CODE_LENS.BYTES:]
    warmups = struct.unpack(f'={sum(degrees)}{NEXT_DEPTH.FORMAT}', bytes_buffer[:sum(degrees) * NEXT_DEPTH.BYTES])
    bytes_buffer = bytes_buffer[sum(degrees) * NEXT_DEPTH.BYTES:]

    bitbuffer = bitarray()
    bitbuffer.frombytes(bytes_buffer)

    code_slices = slice_by_part_sizes(code_lens)
    warmup_slices = slice_by_part_sizes(degrees)

    chunks = list(mp.Pool(PROCESSES).map(
        func=partial(get_chunk, warmups_=warmups, bitbuffer_=bitbuffer),
        iterable=tqdm(list(zip(degrees, warmup_slices, code_slices))),
        chunksize=1,
    ))
    assert all(size == CHUNK_SIZE for size in map(len, chunks))

    waveform = np.hstack(chunks)
    assert all(waveform[samples_len:] == 0)
    waveform = waveform[:samples_len]

    wavefile = wave.open('audio.wav', 'r')
    original_waveform = np.frombuffer(wavefile.readframes(wavefile.getnframes()), BIT_DEPTH.TYPE)

    assert np.array_equal(waveform, original_waveform)
