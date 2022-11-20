import math
import multiprocessing as mp
import struct
import wave
from typing import NamedTuple

import numpy as np
import scipy.signal
from bitarray import bitarray
from tqdm import tqdm

NUMPY_FORMAT_MAP = {
    np.int8: 'b',
    np.int16: 'h',
    np.int32: 'l',
    np.int64: 'q',
    np.uint8: 'B',
    np.uint16: 'H',
    np.uint32: 'L',
    np.uint64: 'Q',
}


def get_type_tuple(np_type):
    class TypeTuple(NamedTuple):
        TYPE: type
        MAX: int
        BYTES: int
        FORMAT: str

    return TypeTuple(
        TYPE=np_type,
        MAX=np.iinfo(np_type).max,
        BYTES=np.iinfo(np_type).bits // 8,
        FORMAT=NUMPY_FORMAT_MAP[np_type]
    )


#
PROCESSES = 16
CHUNK_SIZE = 4095
MAX_DEGREE = 5  # magic number
DIVISOR = 2 ** 6  # empiric magic number
SAMPLES_LEN = get_type_tuple(np.uint64)
BIT_DEPTH = get_type_tuple(np.int16)
CHUNKS_LEN = get_type_tuple(np.uint64)
DEGREES = get_type_tuple(np.uint8)
CODE_LENS = get_type_tuple(np.uint16)
NEXT_DEPTH = get_type_tuple(np.sctypes['int'][np.sctypes['int'].index(BIT_DEPTH.TYPE) + 1])
POLYNOMIALS = [[1]]
for i in range(MAX_DEGREE):
    POLYNOMIALS.append(list(np.array(POLYNOMIALS[-1] + [0]) - np.array([0] + POLYNOMIALS[-1])))
#

assert MAX_DEGREE <= DEGREES.MAX, 'Decrease MAX_DEGREE or enlarge DEGREES'
assert np.issubdtype(BIT_DEPTH.TYPE, np.signedinteger), 'BIT_DEPTH must be signed'
assert CODE_LENS.MAX >= CHUNK_SIZE * np.iinfo(BIT_DEPTH.TYPE).bits, 'Decrease CHUNK_SIZE or enlarge CODE_LENS'


# numbers, divisor = N, M
def rice_golomb_encode(numbers: list[int], divisor: int) -> str:
    bin_len = int(math.floor(math.log2(divisor)))
    inv_divisor = 2 ** (bin_len + 1) - divisor

    code = ''
    for dividend in numbers:
        quotient = dividend // divisor
        code += '1' * quotient

        code += '0'

        reminder = dividend % divisor
        if reminder < inv_divisor:
            code += f'{reminder:0{bin_len}b}'
        else:
            bin_len += 1
            code += f'{reminder + inv_divisor:0{bin_len}b}'
    return code


def positive_mapping(array: np.ndarray, array_is_copy=False) -> np.ndarray:
    positive = array >= 0
    array[positive] = 2 * array[positive]
    array[~positive] = -2 * array[~positive] - 1
    if array_is_copy:
        return array


def get_degree_warmup_code(chunk: np.ndarray) -> (int, list, bitarray):
    array = bitarray()
    array.frombytes(chunk.tobytes())

    candidates = [array.to01()]
    candidates.extend([
        rice_golomb_encode(
            positive_mapping(
                array=scipy.signal.convolve(chunk, poly, mode='valid'),
                array_is_copy=True,
            ),
            divisor=DIVISOR,
        )
        for poly in POLYNOMIALS[1:]
    ])

    best_degree = np.argmin(list(map(len, candidates)))
    if best_degree > 0:
        warmup = scipy.signal.convolve(chunk[:best_degree], POLYNOMIALS[best_degree], mode='full')[:best_degree]
        warmup = warmup.tolist()
    else:
        warmup = []
    code = bitarray(candidates[best_degree])
    return best_degree, warmup, code


if __name__ == '__main__':
    wavefile = wave.open('audio.wav', 'r')
    waveform = np.frombuffer(wavefile.readframes(wavefile.getnframes()), BIT_DEPTH.TYPE)

    waveform_padded = np.pad(waveform, (0, int(CHUNK_SIZE * np.ceil(len(waveform) / CHUNK_SIZE)) - len(waveform)))

    chunks = np.split(waveform_padded, len(waveform_padded) // CHUNK_SIZE)

    pool = mp.Pool(PROCESSES)
    degrees, warmups, codes = zip(*pool.map(get_degree_warmup_code, tqdm(chunks), chunksize=1))

    bytes_buffer = struct.pack(
        '='  # prevent alignment
        f'{1}{SAMPLES_LEN.FORMAT}'
        f'{1}{CHUNKS_LEN.FORMAT}'
        f'{len(degrees)}{DEGREES.FORMAT}'
        f'{len(codes)}{CODE_LENS.FORMAT}'
        f'{sum(degrees)}{NEXT_DEPTH.FORMAT}',
        len(waveform),
        len(chunks),
        *degrees,
        *map(len, codes),
        *(w for lst in filter(bool, warmups) for w in lst),
    )

    bits_buffer = bitarray()
    bits_buffer.frombytes(bytes_buffer)
    bits_buffer = sum(codes, bits_buffer)

    with open('compressed.flc', 'wb') as f:
        f.write(bits_buffer)
