import cmath
import math


def next_power_of_two(n):
    """Return the smallest power of two >= n.

    The radix-2 Cooley-Tukey algorithm requires the input length to be an exact
    power of two. Callers use this helper to zero-pad real data to a suitable
    length before transforming it.
    """
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def _bit_reverse_indices(n):
    """Yield indices reordered so that the i-th element sits at its
    bit-reversed position for a length-n array.

    This avoids a separate permutation step in the recursive Cooley-Tukey
    implementation by directly feeding the indices in the order the recursion
    would otherwise produce.
    """
    bits = n.bit_length() - 1
    for i in range(n):
        rev = 0
        x = i
        for _ in range(bits):
            rev = (rev << 1) | (x & 1)
            x >>= 1
        yield rev


def _fft(a, inverse=False):
    """Iterative radix-2 Cooley-Tukey FFT for inputs whose length is a power
    of two.

    The iterative form is used instead of the textbook recursive one because
    Python's per-call overhead is high; the in-place butterfly loop keeps the
    transformation on input sizes that are already powers of two fast enough
    for the convolution use cases this library targets.
    """
    n = len(a)
    if n == 0:
        return []
    if n & (n - 1) != 0:
        raise ValueError("input length must be a power of two, got %r" % n)

    a = [complex(x) for x in a]
    for i, j in enumerate(_bit_reverse_indices(n)):
        if j > i:
            a[i], a[j] = a[j], a[i]

    sign = 1.0 if inverse else -1.0
    scale = 1.0 / n if inverse else 1.0

    half = 1
    while half < n:
        full = half << 1
        # cmath.exp keeps everything in the standard library; it is the
        # primitive root of unity for the current butterfly layer.
        w_step = cmath.exp(sign * 1j * math.pi / half)
        for start in range(0, n, full):
            w = 1 + 0j
            for k in range(half):
                u = a[start + k]
                v = a[start + k + half] * w
                a[start + k] = u + v
                a[start + k + half] = u - v
                w *= w_step
        half = full

    if inverse:
        a = [z * scale for z in a]
    return a


def fft(a):
    """Forward discrete Fourier transform of a sequence of complex numbers.

    Requires len(a) to be a power of two (including the empty list and the
    length-one case). Returns a list of complex numbers of the same length.
    """
    return _fft(a, inverse=False)


def ifft(a):
    """Inverse discrete Fourier transform, the matching inverse of fft.

    Applies the conjugate twiddle factors and divides by n. The same power-of-
    two length restriction applies.
    """
    return _fft(a, inverse=True)


def convolve(a, b):
    """Linear convolution of two real-or-complex sequences.

    The result has length len(a) + len(b) - 1. Both inputs are zero-padded to
    the next power of two that is at least that length, transformed, multiplied
    pointwise, and inverted. The empty-convolution cases (one or both inputs
    empty) return an empty list rather than raising.
    """
    if not a or not b:
        return []
    target = len(a) + len(b) - 1
    n = next_power_of_two(target)
    A = _fft(list(a) + [0] * (n - len(a)))
    B = _fft(list(b) + [0] * (n - len(b)))
    C = [A[i] * B[i] for i in range(n)]
    c = _fft(C, inverse=True)
    # Truncate the zero-padding region. Coefficients beyond target are
    # numerical residue from the inverse transform of padded data.
    return [c[i].real if abs(c[i].imag) < 1e-9 else c[i] for i in range(target)]


def multiply_polys(a, b):
    """Coefficient-wise polynomial multiplication via FFT.

    Inputs are lists of real coefficients in ascending order of degree.
    Returns a list of real coefficients. Because FFT involves floating
    point, results are rounded to the nearest integer when the imaginary part
    is negligible and the real part is within 1e-9 of one; this is the standard
    pragmatic handling for polynomial convolution where inputs are exact
    integers.
    """
    if not a or not b:
        return []
    c = convolve(a, b)
    out = []
    for z in c:
        r = z.real if isinstance(z, complex) else z
        if abs(round(r) - r) < 1e-9:
            out.append(int(round(r)))
        else:
            out.append(r)
    return out
