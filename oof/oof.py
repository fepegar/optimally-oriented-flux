import numpy as np
from numpy import pi as π
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

        self.σ = min(self.spacing)
        self.num_radii = 6

        self.response_type = 0
        self.use_absolute = True
        self.normalization_type = 1



    def set_stuff(self):
        pass


    def get_spacing(self):
        return self.nifti.header.get_zooms()


    def get_radii(self):
        return np.arange(1, self.num_radii + 1) * min(self.spacing)


    def check_normalization(self):
        if min(self.radii) < self.σ and self.normalization_type > 0:
            print('Sigma must be >= minimum range to enable the advanced'
                  ' normalization. The current setting falls back to'
                  ' normalization_type = 0 because of the undersize sigma.')
            self.normalization_type = 0


    def compute_oof(self, array):
        array = array.astype(np.double)
        shape = array.shape
        output = np.zeros(shape)
        self.check_normalization()
        imgfft = fft(array)
        x, y, z, sphere_radius = get_min_sphere_radius(shape, self.spacing)

        for radius in self.radii:
            print(f'Computing radius {radius:.3f}...')
            circle = circle_length(radius)
            ν = 1.5
            z = circle * EPSILON
            bessel = besselj(ν, z) / EPSILON**(3 / 2)
            base = radius / np.sqrt(2 * radius * self.σ - self.σ**2)
            exponent = self.normalization_type
            volume = get_sphere_volume(radius)
            normalization = volume / bessel / radius**2 * base**exponent

            exponent = - self.σ**2 * 2 * π**2 * sphere_radius**2
            num = normalization * np.exp(exponent)
            den = sphere_radius**(3/2)
            besselj_buffer = num / den

            cs = circle * sphere_radius
            a = np.sin(cs) / cs - np.cos(cs)
            b = np.sqrt(1 / (π**2 * radius * sphere_radius))
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
            λ1, λ2, λ3 = eigenvalues

            maxe = λ1
            mine = λ1
            mide = maxe + λ2 + λ3

            if self.use_absolute:
                maxe[np.abs(λ2) > np.abs(maxe)] = λ2[np.abs(λ2) > np.abs(maxe)]
                mine[np.abs(λ2) < np.abs(mine)] = λ2[np.abs(λ2) < np.abs(mine)]

                maxe[np.abs(λ3) > np.abs(maxe)] = λ3[np.abs(λ3) > np.abs(maxe)]
                mine[np.abs(λ3) < np.abs(mine)] = λ3[np.abs(λ3) < np.abs(mine)]
            else:
                maxe[λ2 > np.abs(maxe)] = λ2[λ2 > np.abs(maxe)]
                mine[λ2 < np.abs(mine)] = λ2[λ2 < np.abs(mine)]

                maxe[λ3 > np.abs(maxe)] = λ3[λ3 > np.abs(maxe)]
                mine[λ3 < np.abs(mine)] = λ3[λ3 < np.abs(mine)]

            mide = mide - maxe - mine

            if self.response_type == 0:
                tmpfeature = maxe
            elif self.response_type == 1:
                tmpfeature = maxe + mide
            elif self.response_type == 2:
                tmpfeature = np.sqrt(np.maximum(0, maxe * mide))
            elif self.response_type == 3:
                max1 = np.maximum(0, maxe * mide)
                max2 = np.maximum(0, mide)
                tmpfeature = np.sqrt(max1 * max2)
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
    return 4 / 3 * π * radius**3


def circle_length(radius):
    return 2 * π * radius


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


def eigenvalue_field33(a11, a12, a13, a22, a23, a33):
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
    ε = 1e-50

    a11 = a11.astype(np.double)
    a12 = a12.astype(np.double)
    a13 = a13.astype(np.double)
    a22 = a22.astype(np.double)
    a23 = a23.astype(np.double)
    a33 = a33.astype(np.double)

    b = a11 + ε
    d = a22 + ε
    j = a33 + ε

    c = - (a12**2 + a13**2 + a23**2 - b * d - d * j - j * b)
    mul1 = a23**2 * b + a12**2 * j + a13**2 * d
    mul2 = a13 * a12 * a23
    d = - (b * d * j - mul1 + 2 * mul2)
    b = - a11 - a22 - a33 - ε - ε - ε
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

    λ1 = b.astype(np.single)
    λ2 = j.astype(np.single)
    λ3 = d.astype(np.single)
    return λ1, λ2, λ3
