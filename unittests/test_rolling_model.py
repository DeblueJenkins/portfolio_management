import pandas as pd
import numpy as np
import unittest
from models.rolling import RollingModel

np.random.seed(0)

errors = np.array([ 0.12289683, -0.56525535, -0.10821304, -0.44839389, -0.20697223,
                    0.1986956 , -0.03395258, -0.72627173,  0.81032686,  0.41036279,
                    0.46273105,  0.02809171,  0.29942835, -0.00559426,  0.20548669,
                    0.36168938,  0.18147049, -0.47258361, -0.07218761, -0.54776078,
                    -0.25021506,  0.25592731,  0.22598451,  0.22311497,  0.02840598,
                    -0.09495401, -0.21295083,  0.07687242, -0.37955009,  0.17175185,
                    -0.07152377, -0.10218275,  0.24030577,  0.12292843, -0.23696529,
                    -0.89913696,  0.75753778,  0.5103201 , -0.40834006,  0.17470522,
                    -0.04658747, -0.00383562, -0.18711325, -0.21434599,  0.6539228 ,
                    0.01257596, -0.84794925,  0.10748592, -0.13855179,  0.4814305 ])

y = np.array([ 1.68738505,  0.32348991,  0.90207069,  2.1642259 ,  1.79089069,
               -1.05394518,  0.87342112, -0.2280245 , -0.17988615,  0.33393121,
               0.06737628,  1.37760621,  0.68437043,  0.04500772,  0.36719594,
               0.25700703,  1.41741178, -0.28182556,  0.23640041, -0.93076303,
               -2.62965711,  0.5769513 ,  0.7877689 , -0.81883232,  2.19308733,
               -1.53103297, -0.03090878, -0.26385115,  1.45611192,  1.39269147,
               0.07828013,  0.30149522, -0.96445304, -2.05746376, -0.42457944,
               0.07968167,  1.15362339,  1.12571255, -0.46399411, -0.37897005,
               -1.12522026, -1.49668523, -1.78293749,  1.8741081 , -0.58631948,
               -0.5147416 , -1.32946266,  0.70082306, -1.69056514, -0.28940758,
               -0.97213386,  0.3102352 , -0.58747243, -1.25729948, -0.10484952,
               0.35166457, -0.01015007,  0.2258046 , -0.71098939, -0.43940846])

class TestRollingModel(unittest.TestCase):

    def setUp(self):
        date_range = pd.date_range(start='20140101', end='20140301', freq='D')
        self.T = len(date_range)

        data = pd.DataFrame({
            'Date': date_range,
            'Asset1': np.random.normal(size=self.T),
            'Asset2': np.random.normal(size=self.T),
            'Asset3': np.random.normal(size=self.T),
            'Asset4': np.random.normal(size=self.T),
        })
        self.model = RollingModel(data, rolling_window=10)

    def test_estimate(self):
        config = {
            'PCA': True,
            'n_components': 2,
            'OLS': True
        }
        RIC = 'Asset1'
        self.model.estimate(config, RIC)

        # Check if the model's attributes are set correctly after estimation
        assert [np.isclose(i,j) for i,j in zip(self.model.y, y)]
        self.assertListEqual(self.model.size_of_rolling_windows, np.repeat(10, 50).tolist())
        self.assertEqual(len(self.model.singular_values), self.T)
        self.assertEqual(len(self.model.eig_vals), self.T)
        self.assertEqual(len(self.model.eig_vecs), self.T)
        self.assertListEqual(len(self.model.errors), self.T)
        assert [np.isclose(i,j) for i,j in zip(self.model.errors, errors)]
        self.assertAlmostEqual(self.model.r_sqr, 0.8900675304468947)  # AssertAlmostEqual for floating-point comparison

if __name__ == '__main__':
    unittest.main()
