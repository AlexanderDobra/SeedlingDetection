from unittest import TestCase
from src.models.train_model import *
import pandas as pd

from src.analysis.analysis_functions import *

class ModelFixingTest(TestCase):

    def test_basic(self):
        df = pd.DataFrame([[1,2,3,4], [5,6,7,8]])
        df.iloc[0,0] = pd.NA
        df.iloc[1,3] = pd.NA
        index = pd.MultiIndex.from_tuples([("m-m", "a-a"), ("m_m", "b"), ("m_m", "c_c"), ("m_m", "a_a")], names=["top","bottom"])
        df.columns = index
        new_df = repair_mixed_metrics(df)
        correct_df = df.drop([("m-m", "a-a")], axis=1).copy()
        correct_df.iloc[1,2] = 5
        assert(new_df.equals(correct_df))
