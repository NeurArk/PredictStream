import time
import pandas as pd
import numpy as np
from utils import eda


def test_correlation_matrix_performance():
    df = pd.DataFrame(np.random.rand(1000, 5), columns=list("abcde"))
    start = time.perf_counter()
    corr = eda.correlation_matrix(df)
    elapsed = time.perf_counter() - start
    assert corr.shape == (5, 5)
    assert elapsed < 1.0
