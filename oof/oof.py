import click
import numpy as np
from numpy.fft import fft, ifft
from scipy.special import jv as besselj
import nibabel as nib


EPSILON = 1e-12


class OOF:
    def __init__(self, input_path=None):
        self.nifti = None
        self.array = None
        self.radii = None

        self.spacing = 1, 1, 1

        if input_path is not None:
            self.nifti = nib.load(str(input_path))
            self.array = self.nifti.get_data()
            self.spacing = self.get_spacing()
            self.radii = self.get_radii()

        self.sigma = min(self.spacing)
        self.num_radii = 6

        self.response_type = 0
        self.use_absolute = True
        self.normalization_type = 1

    def get_spacing(self):
        return self.nifti.header.get_zooms()

    def get_radii(self):
        return np.arange(1, self.num_radii + 1) * min(self.spacing)

    def check_normalization(self, radii):
        if min(radii) < self.sigma and self.normalization_type > 0:
            print('Sigma must be >= minimum range to enable the advanced'
                  ' normalization. The current setting falls back to'
                  ' normalization_type = 0 because of the undersize sigma.')
            self.normalization_type = 0

    def compute_oof(self, array, radii):
        array = array.astype(np.double)
        shape = array.shape
        output = np.zeros(shape)
        self.check_normalization(radii)
        imgfft = fft(array)
        x, y, z, sphere_radius = get_min_sphere_radius(shape, self.spacing)

        for radius in radii:
            print(f'Computing radius {radius:.3f}...')
            circle = circle_length(radius)
            ν = 1.5
            z = circle * EPSILON
            bessel = besselj(ν, z) / EPSILON**(3 / 2)
            base = radius / np.sqrt(2 * radius * self.sigma - self.sigma**2)
            exponent = self.normalization_type
            volume = get_sphere_volume(radius)
            normalization = volume / bessel / radius**2 * base**exponent

            exponent = - self.sigma**2 * 2 * np.pi**2 * sphere_radius**2
            num = normalization * np.exp(exponent)
            den = sphere_radius**(3/2)
            besselj_buffer = num / den

            cs = circle * sphere_radius
            a = np.sin(cs) / cs - np.cos(cs)
            b = np.sqrt(1 / (np.pi**2 * radius * sphere_radius))
            besselj_buffer = besselj_buffer * a * b * imgfft

            outputfeature_11 = np.real(ifft(x * x * besselj_buffer))
            outputfeature_12 = np.real(ifft(x * y * besselj_buffer))
            outputfeature_13 = np.real(ifft(x * z * besselj_buffer))
            outputfeature_22 = np.real(ifft(y * y * besselj_buffer))
            outputfeature_23 = np.real(ifft(y * z * besselj_buffer))
            outputfeature_33 = np.real(ifft(z * z * besselj_buffer))

            eigenvalues = eigenvalue_field33(
                outputfeature_11,
                outputfeature_12,
                outputfeature_13,
                outputfeature_22,
                outputfeature_23,
                outputfeature_33
            )
            lambda_1, lambda_2, lambda_3 = eigenvalues

            maxe = np.copy(lambda_1)
            mine = np.copy(lambda_1)
            mide = maxe + lambda_2 + lambda_3

            if self.use_absolute:
                maxe[np.abs(lambda_2) > np.abs(maxe)] = lambda_2[np.abs(lambda_2) > np.abs(maxe)]
                mine[np.abs(lambda_2) < np.abs(mine)] = lambda_2[np.abs(lambda_2) < np.abs(mine)]

                maxe[np.abs(lambda_3) > np.abs(maxe)] = lambda_3[np.abs(lambda_3) > np.abs(maxe)]
                mine[np.abs(lambda_3) < np.abs(mine)] = lambda_3[np.abs(lambda_3) < np.abs(mine)]
            else:
                maxe[lambda_2 > np.abs(maxe)] = lambda_2[lambda_2 > np.abs(maxe)]
                mine[lambda_2 < np.abs(mine)] = lambda_2[lambda_2 < np.abs(mine)]

                maxe[lambda_3 > np.abs(maxe)] = lambda_3[lambda_3 > np.abs(maxe)]
                mine[lambda_3 < np.abs(mine)] = lambda_3[lambda_3 < np.abs(mine)]

            mide -= maxe + mine

            if self.response_type == 0:
                tmpfeature = maxe
            elif self.response_type == 1:
                tmpfeature = maxe + mide
            elif self.response_type == 2:
                tmpfeature = np.sqrt(np.maximum(0, maxe * mide))
            elif self.response_type == 3:
                tmpfeature = np.sqrt(
                    np.maximum(0, maxe * mide) * np.maximum(0, mide))
            elif self.response_type == 4:
                tmpfeature = np.maximum(0, maxe)
            elif self.response_type == 5:
                tmpfeature = np.maximum(0, maxe + mide)

            stronger_response = np.abs(tmpfeature) > np.abs(output)
            output[stronger_response] = tmpfeature[stronger_response]
        return output

    def run(self, output_path):
        oof = self.compute_oof(self.array)
        output_nii = nib.Nifti1Image(oof, self.nifti.affine)
        output_nii.header['sform_code'] = 0
        output_nii.header['qform_code'] = 1
        output_nii.to_filename(str(output_path))


def get_min_sphere_radius(shape, spacing):
    x, y, z = ifft_shifted_coordinates_matrix(shape)
    si, sj, sk = shape
    pi, pj, pk = spacing
    x /= si * pi
    y /= sj * pj
    z /= sk * pk
    sphere_radius = np.sqrt(x**2 + y**2 + z**2) + EPSILON
    return x, y, z, sphere_radius


def get_sphere_volume(radius):
    return 4 / 3 * np.pi * radius**3


def circle_length(radius):
    return 2 * np.pi * radius


def ifft_shifted_coordinates_matrix(shape):
    shape = np.array(shape)
    dimensions = len(shape)
    p = shape // 2
    result = []

    for i in range(dimensions):
        x = np.arange(p[i], shape[i])
        y = np.arange(p[i])
        a = np.concatenate((x, y)) - p[i]
        reshapepara = np.ones(dimensions, np.uint16)
        reshapepara[i] = shape[i]
        A = np.reshape(a, reshapepara)
        repmatpara = np.copy(shape)
        repmatpara[i] = 1
        coords = np.tile(A, repmatpara).astype(float)
        result.append(coords)
    return result


def freq_op(freq, marginwidth):
    result = freq[marginwidth[0]:-1 - marginwidth[0],
                  marginwidth[1]:-1 - marginwidth[1],
                  marginwidth[2]:-1 - marginwidth[2]]
    return result


def eigenvalue_field33(a11, a12, a13, a22, a23, a33, epsilon=1e-50):
    """
    Calculate the eigenvalues of massive 3x3 real symmetric matrices.
    Computation is based on matrix operation and GPU computation is
    supported.

    Syntax:
    λ1, λ2, λ3 = eigenvaluefield33(a11, a12, a13, a22, a23, a33)
    a11, a12, a13, a22, a23 and a33 specify the symmetric 3x3 real symmetric
    matrices as:
    [[a11, a12, a13],
     [a12, a22, a13],
     [a13, a23, a33]]
    These six inputs must have the same size. They can be 2D, 3D or any
    dimension. The outputs eigenvalue1, eigenvalue2 and eigenvalue3 will
    follow the size and dimension of these inputs. Owing to the use of
    trigonometric functions, the inputs must be double to maintain the
    accuracy.

    eigenvalue1, eigenvalue2 and eigenvalue3 are the unordered resultant
    eigenvalues. They are solved using the cubic equation solver, see
    http://en.wikipedia.org/wiki/Eigenvalue_algorithm

    The peak memory consumption of the method is about 1.5 times of the total
    of all inputs, in addition to the original inputs.

    Author: Max W.K. Law
    Email:  max.w.k.law@gmail.com
    Page:   http://www.cse.ust.hk/~maxlawwk/

    Python implementation by:
    Fernando Perez-Garcia
    fernando.perezgarcia.17@ucl.ac.uk
    """
    a11 = a11.astype(np.double)
    a12 = a12.astype(np.double)
    a13 = a13.astype(np.double)
    a22 = a22.astype(np.double)
    a23 = a23.astype(np.double)
    a33 = a33.astype(np.double)

    b = a11 + epsilon
    d = a22 + epsilon
    j = a33 + epsilon

    c = - (a12**2 + a13**2 + a23**2 - b * d - d * j - j * b)
    mul1 = a23**2 * b + a12**2 * j + a13**2 * d
    mul2 = a13 * a12 * a23
    d = - (b * d * j - mul1 + 2 * mul2)
    b = - a11 - a22 - a33 - epsilon - epsilon - epsilon
    d += (2 * b**3 - 9 * b * c) / 27
    c *= -1
    c += b**2 / 3
    c **= 3
    c /= 27
    np.maximum(0, c, out=c)
    np.sqrt(c, out=c)
    j = c**(1 / 3)
    c += c == 0
    d *= - 1 / 2 / c
    np.clip(d, -1, 1, out=d)
    d = np.real(np.arccos(d) / 3)
    c = j * np.cos(d)
    d = j * np.sqrt(3) * np.sin(d)
    b *= - 1 / 3
    j = - c - d + b
    d += b - c
    b += 2 * c

    lambda_1 = b.astype(np.single)
    lambda_2 = j.astype(np.single)
    lambda_3 = d.astype(np.single)
    return lambda_1, lambda_2, lambda_3


@click.command()
@click.argument('input-path', type=click.Path(exists=True))
@click.argument('output-path', type=click.Path())
def main(input_path, output_path):
    OOF(input_path).run(output_path)
