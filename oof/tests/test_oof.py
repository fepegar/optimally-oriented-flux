from pathlib import Path
from scipy.io import loadmat
from numpy.testing import assert_array_equal

from ..oof import eigenvalue_field33, ifft_shifted_coordinates_matrix, OOF

fixtures_dir = Path(__file__).parent / 'fixtures'
fixture_path = fixtures_dir / 'oof.mat'
fixture_dict = loadmat(str(fixture_path))

def test_eigenvalue_field33():
    b, j, d = eigenvalue_field33(
        fixture_dict['a11'],
        fixture_dict['a12'],
        fixture_dict['a13'],
        fixture_dict['a22'],
        fixture_dict['a23'],
        fixture_dict['a33'],
    )
    assert_array_equal(b, fixture_dict['b'])
    assert_array_equal(j, fixture_dict['j'])
    assert_array_equal(d, fixture_dict['d'])


def test_ifft_shifted_coordinates_matrix():
    fixture_shape = fixture_dict['shape'].squeeze()
    x, y, z = ifft_shifted_coordinates_matrix(fixture_shape)
    assert_array_equal(x, fixture_dict['x'])
    assert_array_equal(y, fixture_dict['y'])
    assert_array_equal(z, fixture_dict['z'])


def test_oof():
    array = fixture_dict['image']
    radii = fixture_dict['radii'].squeeze()
    response = OOF().compute_oof(array, radii)
    assert_array_equal(response, fixture_dict['response'])
