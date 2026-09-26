# fast_fourier_transform

Radix-2 Cooley-Tukey FFT over complex numbers, built only on the Python
standard library `cmath`. Provides `fft`, `ifft`, `convolve`, and
`multiply_polys` for length-power-of-two transforms and linear convolution of
real or complex sequences.

## Usage

```python
from fast_fourier_transform import fft, ifft, convolve, multiply_polys

spectrum = fft([1, 2, 3, 4])         # length-4 forward transform, list of complex
original = ifft(spectrum)            # inverse, back to near-identity

line   = convolve([1, 1], [1, 1])    # [1, 2, 1]
coeffs = multiply_polys([1, 1], [2, 0, 1])  # [2, 2, 1, 1]
```

`fft` and `ifft` require `len(input)` to be a power of two (0, 1, 2, 4, ...).
`convolve` and `multiply_polys` zero-pad internally and accept any non-empty
length; either input empty returns `[]`.

## Why this exists

The library is a small, dependency-free building block for signal convolution
and polynomial multiplication when pulling in NumPy is not worth it or not
possible. The trade-off is speed: an iterative in-place butterfly loop in pure
Python is fine for transforms up to a few thousand points, but it cannot
compete with a compiled FFT for large arrays. What you get in exchange is a
zero-dependency module that reads in one file and runs anywhere CPython does.

## Edge cases worth knowing

- Non-power-of-two input to `fft`/`ifft` raises `ValueError` rather than
  silently padding. Zero-pad yourself or call `convolve`, which pads.
- FFT introduces floating-point noise. `multiply_polys` rounds coefficients to
  the nearest integer when they are within `1e-9` of one; pass `convolve`
  directly if you want the raw floating-point output.
- `convolve` of two empty or one-empty inputs returns `[]`, matching the
  degenerate-product interpretation rather than raising.
