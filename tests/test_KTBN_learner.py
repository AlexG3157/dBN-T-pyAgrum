import unittest
import pandas as pd
import numpy as np
from pyAgrum.lib.discretizer import Discretizer

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ktbn import Learner

class TestLearner(unittest.TestCase):

    def setUp(self):

        self.df1 = pd.DataFrame(
            {'A': np.ones(20), 
             'B': range(20), 
             'C' : np.ones(20)*2,
             'D' : range(50,70)
             })
        self.df2 = pd.DataFrame(
            {'A': np.ones(20)*5, 
             'B': np.ones(20), 
             'C' : np.ones(20)*48,
             'D' : range(20)
             })

        self.dfs = [self.df1, self.df2]
        self.discretizer = Discretizer()
        self.delimiter = '_'
        self.k = 4  
        self.learner = Learner.Learner(self.dfs, self.discretizer, self.delimiter, self.k)

    def test_init_learner_valid(self):
        # Test that a Learner initializes correctly 
        self.assertEqual(['A', 'C'], self.learner.atemporal_vars)
        self.assertEqual(['B', 'D'], self.learner.temporal_vars)

    def test_get_delimiter(self):
        self.assertEqual(self.learner.get_delimiter(), self.delimiter)

    def test_verify_timeslice_valid(self):
        # Verify that a valid time slice order does not raise an error.
        try:
            Learner.verify_timeslice(1, 2)
        except ValueError:
            self.fail("verify_timeslice raised ValueError unexpectedly for valid input.")

    def test_verify_timeslice_invalid(self):
        # Verify that an invalid time slice (arc from future to past) raises ValueError.
        with self.assertRaises(ValueError):
            Learner.verify_timeslice(2, 1)

    def test_is_atemporal(self):

        self.assertTrue(Learner.is_atemporal(self.dfs, 'A'))
        self.assertFalse(Learner.is_atemporal(self.dfs, 'B'))

    def test_create_sequences(self):

        lags, first = Learner.create_sequences(self.dfs, self.k, self.delimiter, 
                                               self.learner.temporal_vars, self.learner.atemporal_vars)
        
        # Check that the atemporal variables are in the sequences. 
        self.assertIn('A', lags.columns)
        self.assertIn('A', first.columns)
        self.assertIn('C', lags.columns)
        self.assertIn('C', first.columns)

        # Check that all of the lagged columns are in the sequences and that no atemporal variable
        # was treated as temporal.
        for i in range(self.k):
            
            self.assertIn(f'B{self.delimiter}{i}', lags.columns)
            self.assertIn(f'B{self.delimiter}{i}', first.columns)
            self.assertIn(f'D{self.delimiter}{i}', lags.columns)
            self.assertIn(f'D{self.delimiter}{i}', first.columns)
            
            self.assertNotIn(f'A{self.delimiter}{i}', lags.columns)
            self.assertNotIn(f'A{self.delimiter}{i}', first.columns)
            self.assertNotIn(f'C{self.delimiter}{i}', lags.columns)
            self.assertNotIn(f'C{self.delimiter}{i}', first.columns)

        # Ensure that the length of the first df is 2 (only two trajectories were provided).
        self.assertEqual(len(first), 2)

    def test_learn_ktbn_structure(self):
        ktbn = self.learner.learn_ktbn()

        # Check that the KTBN contains temporal variables from all time slices.
        for col in self.learner.temporal_vars:
            for i in range(self.k):
                self.assertIn(f"{col}{self.delimiter}{i}", ktbn.names())
        
        # Check that the KTBN contains atemporal variables and that they were not treated as temporal.
        for col in self.learner.atemporal_vars:
            self.assertIn(col, ktbn.names())
            for i in range(self.k):
                self.assertFalse(f'{col}{self.delimiter}{i}' in ktbn.names())


