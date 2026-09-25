import math
import unittest

from fast_fourier_transform import fft, ifft, convolve, multiply_polys, next_power_of_two


class TestNextPowerOfTwo(unittest.TestCase):
    def test_zero_and_one(self):
        self.assertEqual(next_power_of_two(0), 1)
        self.assertEqual(next_power_of_two(1), 1)

    def test_exact_powers_stay(self):
        for k in range(1, 11):
            self.assertEqual(next_power_of_two(1 << k), 1 << k)

    def test_rounds_up_between_powers(self):
        self.assertEqual(next_power_of_two(5), 8)
        self.assertEqual(next_power_of_two(9), 16)


class TestFFTLengthRestriction(unittest.TestCase):
    def test_non_power_of_two_raises(self):
        with self.assertRaises(ValueError):
            fft([1, 2, 3])


class TestRoundTrip(unittest.TestCase):
    def _assert_close(self, got, want, tol=1e-9):
        self.assertEqual(len(got), len(want))
        for g, w in zip(got, want):
            self.assertLessEqual(abs(complex(g) - complex(w)), tol)

    def test_empty(self):
        self.assertEqual(fft([]), [])
        self.assertEqual(ifft([]), [])

    def test_singleton(self):
        self._assert_close(fft([5]), [5])
        self._assert_close(ifft([5]), [5])

    def test_roundtrip_power_of_two(self):
        data = [complex(i, i + 1) for i in range(8)]
        self._assert_close(ifft(fft(data)), data)


class TestKnownTransforms(unittest.TestCase):
    def test_dc_signal(self):
        # A constant signal transforms to a single DC bin.
        out = fft([1 + 0j] * 4)
        self.assertEqual(len(out), 4)
        self.assertLessEqual(abs(out[0] - 4), 1e-9)
        for k in range(1, 4):
            self.assertLessEqual(abs(out[k]), 1e-9)

    def test_impulse(self):
        # An impulse at index 0 spreads uniformly across all bins.
        out = fft([1 + 0j] + [0 + 0j] * 3)
        for z in out:
            self.assertLessEqual(abs(z - 1), 1e-9)

    def test_ifft_dc(self):
        out = ifft([4 + 0j, 0 + 0j, 0 + 0j, 0 + 0j])
        for z in out:
            self.assertLessEqual(abs(z - 1), 1e-9)


class TestConvolution(unittest.TestCase):
    def _assert_close(self, got, want, tol=1e-9):
        self.assertEqual(len(got), len(want))
        for g, w in zip(got, want):
            self.assertLessEqual(abs(complex(g) - complex(w)), tol)

    def test_empty_either_side(self):
        self.assertEqual(convolve([], [1, 2, 3]), [])
        self.assertEqual(convolve([1, 2, 3], []), [])
        self.assertEqual(convolve([], []), [])

    def test_simple_box(self):
        # [1,1] * [1,1] = [1,2,1]
        c = convolve([1, 1], [1, 1])
        self._assert_close(c, [1, 2, 1])

    def test_unequal_lengths(self):
        # [1,2] * [1,2,3] = [1,4,7,6]
        c = convolve([1, 2], [1, 2, 3])
        self._assert_close(c, [1, 4, 7, 6])

    def test_complex_inputs(self):
        a = [1 + 1j, 2 + 0j]
        b = [0 + 1j, 1 + 0j]
        # By hand: [-1+1j, 1+3j, 2]
        c = convolve(a, b)
        self._assert_close(c, [-1 + 1j, 1 + 3j, 2 + 0j])


class TestMultiplyPolys(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(multiply_polys([], [1, 2]), [])
        self.assertEqual(multiply_polys([1, 2], []), [])

    def test_integer_coeffs(self):
        # (x + 1)(x^2 + 2) = x^3 + x^2 + 2x + 2
        self.assertEqual(multiply_polys([1, 1], [2, 0, 1]), [2, 2, 1, 1])

    def test_large_coeffs_stay_integers(self):
        # Products up to a few thousand round cleanly.
        self.assertEqual(multiply_polys([10, 20], [30, 40]), [300, 1000, 800])

    def test_real_coeffs_when_not_integer(self):
        # Non-integer inputs should survive as floats, not be rounded.
        out = multiply_polys([0.5, 0.5], [0.5, 0.5])
        self.assertEqual(len(out), 3)
        self.assertAlmostEqual(out[0], 0.25, places=9)
        self.assertAlmostEqual(out[1], 0.5, places=9)
        self.assertAlmostEqual(out[2], 0.25, places=9)


if __name__ == "__main__":
    unittest.main()
