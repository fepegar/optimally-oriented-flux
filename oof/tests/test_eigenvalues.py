from pathlib import Path
from scipy.io import loadmat
from numpy.testing import assert_array_equal

from ..oof import eigenvalue_field33

fixtures_dir = Path(__file__).parent / 'fixtures'

def test_eigenvalue_field33():
    fixture_path = fixtures_dir / 'eigen.mat'
    fixture_dict = loadmat(str(fixture_path))

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


if __name__ == '__main__':
    test_eigenvalue_field33()