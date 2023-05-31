import numpy as np

from hpobench.util.openml_data_manager import OpenMLHoldoutDataManager


def test_convert_nan_values_in_cat_columns():
    x = np.array([[1,  np.nan,  3,      4],
                  [5,       6,  7,      8],
                  [np.nan, 10, 11, np.nan]])

    is_cat = [True, True, False, False]

    x, _, _, categories = OpenMLHoldoutDataManager.replace_nans_in_cat_columns(x, x, x, is_cat)

    solution = np.array([[1., 5.,  3., 4.],
                         [5., 6.,  7., 8.],
                         [0., 10., 11., np.nan]])

    solution_cat = np.array([[1., 5., 0.],
                             [5., 6., 10.]])

    assert np.array_equiv(x[:, :3], solution[:, :3])  # unfortunately np.nan != np.nan :)
    assert np.isnan(x[2, 3])

    cats = np.array(categories).flatten()
    cats.sort()
    solution_cat = solution_cat.flatten()
    solution_cat.sort()
    assert np.array_equal(cats, solution_cat)
