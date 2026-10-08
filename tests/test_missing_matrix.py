import numpy as np
import pandas as pd
from mgdiagnose.process import missing_matrix


def test_missing_matrix():
    df = pd.DataFrame({'a': [1.5, np.nan], 'a_l': [np.nan, 0.], 'age': [30, np.nan]})
    out = missing_matrix(df, cols=['a', 'a_l'])
    assert out['a'].tolist() == [1, 0]
    assert out['a_l'].tolist() == [0, 1]
    assert out['age'].isna().tolist() == [False, True]  # non-selected columns untouched
    assert df['a'].iloc[0] == 1.5  # input not mutated
