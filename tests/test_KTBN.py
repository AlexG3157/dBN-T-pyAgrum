import unittest
import os
import tempfile
import pyagrum as gum
import numpy as np
from typing import Tuple, List, Set

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../ktbn')))

from KTBN import KTBN

class TestKTBN(unittest.TestCase):
    """
    Test cases for the KTBN class.
    """

    def setUp(self):
        """
        Set up a simple KTBN for testing.
        """
        self.k = 2
        self.delimiter = '#'
        self.ktbn = KTBN(k=self.k, delimiter=self.delimiter)
        
        # Add variables
        self.ktbn.addVariable("A", temporal=True)  # Temporal variable
        self.ktbn.addVariable("B", temporal=True)  # Temporal variable
        self.ktbn.addVariable("C", temporal=False)  # Atemporal variable
        
        # Add arcs
        self.ktbn.addArc(("A", 0), ("B", 0))  # Arc from A_0 to B_0
        self.ktbn.addArc(("A", 0), ("A", 1))  # Arc from A_0 to A_1
        self.ktbn.addArc(("C", -1), ("B", 1))  # Arc from C to B_1

    def test_init(self):
        """
        Test that a KTBN initializes correctly.
        """
        ktbn = KTBN(k=3, delimiter='_')
        self.assertEqual(3, ktbn.get_k())
        self.assertEqual('_', ktbn._delimiter)
        self.assertEqual(set(), ktbn._temporal_variables)
        self.assertEqual(set(), ktbn._atemporal_variables)
        self.assertIsInstance(ktbn._bn, gum.BayesNet)

    def test_set_get_k(self):
        """
        Test setting and getting the number of time slices.
        """
        self.ktbn.set_k(4)
        self.assertEqual(4, self.ktbn.get_k())

    def test_add_variable_temporal(self):
        """
        Test adding a temporal variable.
        """
        ktbn = KTBN(k=2, delimiter='#')
        ktbn.addVariable("X", temporal=True)
        
        # Check that X is in temporal variables
        self.assertIn("X", ktbn._temporal_variables)
        
        # Check that X#0, X#1, X#2 are in the BN
        self.assertIn("X#0", ktbn._bn.names())
        self.assertIn("X#1", ktbn._bn.names())

    def test_add_variable_atemporal(self):
        """
        Test adding an atemporal variable.
        """
        ktbn = KTBN(k=2, delimiter='#')
        ktbn.addVariable("Y", temporal=False)
        
        # Check that Y is in atemporal variables
        self.assertIn("Y", ktbn._atemporal_variables)
        
        # Check that Y is in the BN
        self.assertIn("Y", ktbn._bn.names())

    def test_add_arc_valid(self):
        """
        Test adding valid arcs.
        """
        ktbn = KTBN(k=2, delimiter='#')
        ktbn.addVariable("X", temporal=True)
        ktbn.addVariable("Y", temporal=True)
        ktbn.addVariable("Z", temporal=False)
        
        # Add arcs and check they exist in the BN
        ktbn.addArc(("X", 0), ("Y", 0))
        self.assertTrue(ktbn._bn.existsArc("X#0", "Y#0"))
        
        ktbn.addArc(("X", 0), ("X", 1))
        self.assertTrue(ktbn._bn.existsArc("X#0", "X#1"))
        
        ktbn.addArc(("Z", -1), ("Y", 1))
        self.assertTrue(ktbn._bn.existsArc("Z", "Y#1"))

    def test_add_arc_invalid(self):
        """
        Test adding invalid arcs.
        """
        ktbn = KTBN(k=2, delimiter='#')
        ktbn.addVariable("X", temporal=True)
        ktbn.addVariable("Y", temporal=True)
        ktbn.addVariable("Z", temporal=False)
        ktbn.addVariable("W", temporal=False)
        
        # Test adding arc between two atemporal variables
        with self.assertRaises(ValueError):
            ktbn.addArc(("Z", -1), ("W", -1))
        
        # Test adding arc with non-existent variable
        with self.assertRaises(ValueError):
            ktbn.addArc(("Q", 0), ("X", 0))
        
        # Test adding arc with invalid time slice
        with self.assertRaises(ValueError):
            ktbn.addArc(("X", 3), ("Y", 0))

    def test_unroll(self):
        """
        Test unrolling the KTBN.
        """
        # Unroll for 2 additional time slices
        unrolled_bn = self.ktbn.unroll(2)
        
        # Check that the unrolled BN contains the expected variables
        self.assertIn("A#0", unrolled_bn.names())
        self.assertIn("A#1", unrolled_bn.names())
        self.assertIn("A#2", unrolled_bn.names())
        self.assertIn("A#3", unrolled_bn.names())
        
        
        self.assertIn("B#0", unrolled_bn.names())
        self.assertIn("B#1", unrolled_bn.names())
        self.assertIn("B#2", unrolled_bn.names())
        self.assertIn("B#3", unrolled_bn.names())
        #Check that no extra time-slices are added
        self.assertNotIn("B#4", unrolled_bn.names())
        self.assertNotIn("A#4", unrolled_bn.names())
        
        self.assertIn("C", unrolled_bn.names())
        
        # Check that the unrolled BN contains the expected arcs
        self.assertTrue(unrolled_bn.existsArc("A#0", "B#0"))
        self.assertTrue(unrolled_bn.existsArc("A#0", "A#1"))
        self.assertTrue(unrolled_bn.existsArc("C", "B#1"))
        
        # Check that the unrolled BN contains the expected additional arcs
        self.assertTrue(unrolled_bn.existsArc("A#1", "A#2"))
        self.assertTrue(unrolled_bn.existsArc("A#2", "A#3"))
        
        self.assertTrue(unrolled_bn.existsArc("C", "B#2"))
        self.assertTrue(unrolled_bn.existsArc("C", "B#3"))

    def test_cpt(self):
        """
        Test accessing CPTs.
        """
        # Get CPT for a temporal variable
        cpt_a0 = self.ktbn.cpt("A", 0)
        self.assertIsInstance(cpt_a0, gum.Tensor)
        
        # Get CPT for an atemporal variable
        cpt_c = self.ktbn.cpt("C", -1)
        self.assertIsInstance(cpt_c, gum.Tensor)

    def test_to_bn(self):
        """
        Test converting KTBN to BayesNet.
        """
        bn = self.ktbn.to_bn()
        self.assertIsInstance(bn, gum.BayesNet)
        
        # Check that the BN contains the expected variables
        self.assertIn("A#0", bn.names())
        self.assertIn("A#1", bn.names())
        self.assertIn("B#0", bn.names())
        self.assertIn("B#1", bn.names())
        self.assertIn("C", bn.names())
        
        # Check that the BN contains the expected arcs
        self.assertTrue(bn.existsArc("A#0", "B#0"))
        self.assertTrue(bn.existsArc("A#0", "A#1"))
        self.assertTrue(bn.existsArc("C", "B#1"))

    def test_save_load(self):
        """
        Test saving and loading a KTBN.
        """
        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix='.bifxml', delete=False) as tmp:
            tmp_path = tmp.name
        
        
        # Save the KTBN
        self.ktbn.save(tmp_path)
            
        # Load the KTBN
        loaded_ktbn = KTBN.load(tmp_path)
            
        # Check that the loaded KTBN has the same properties
        self.assertEqual(self.ktbn.get_k(), loaded_ktbn.get_k())
        self.assertEqual(self.ktbn._delimiter, loaded_ktbn._delimiter)
        self.assertEqual(self.ktbn._temporal_variables, loaded_ktbn._temporal_variables)
        self.assertEqual(self.ktbn._atemporal_variables, loaded_ktbn._atemporal_variables)
            
        # Check that the loaded KTBN has the same variables
        for var in self.ktbn._bn.names():
            self.assertIn(var, loaded_ktbn._bn.names())
            
        # Check that the loaded KTBN has the same arcs
        for arc in self.ktbn._bn.arcs():
            tail, head = arc
            tail_name = self.ktbn._bn.variable(tail).name()
            head_name = self.ktbn._bn.variable(head).name()
            self.assertTrue(loaded_ktbn._bn.existsArc(tail_name, head_name))
        
        os.unlink(tmp_path)

    def test_from_bn(self):
        """
        Test creating a KTBN from a BayesNet.
        """
        # Create a BayesNet
        bn = gum.BayesNet()
        
        # Add variables
        id_a0 = bn.add(gum.LabelizedVariable("A#0", "A at time 0", 2))
        id_a1 = bn.add(gum.LabelizedVariable("A#1", "A at time 1", 2))
        id_b0 = bn.add(gum.LabelizedVariable("B#0", "B at time 0", 2))
        id_c = bn.add(gum.LabelizedVariable("C", "C", 2))
        
        # Add arcs
        bn.addArc(id_a0, id_b0)
        bn.addArc(id_a0, id_a1)
        bn.addArc(id_c, id_b0)
        
        # Create KTBN from BN
        ktbn = KTBN.from_bn(bn, delimiter='#')
        
        # Check that the KTBN has the correct properties
        self.assertEqual(2, ktbn.get_k())
        self.assertEqual('#', ktbn._delimiter)
        self.assertEqual({"A", "B"}, ktbn._temporal_variables)
        self.assertEqual({"C"}, ktbn._atemporal_variables)
        
        # Check that the KTBN has the correct variables
        self.assertIn("A#0", ktbn._bn.names())
        self.assertIn("A#1", ktbn._bn.names())
        self.assertIn("B#0", ktbn._bn.names())
        self.assertIn("C", ktbn._bn.names())
        
        # Check that the KTBN has the correct arcs
        self.assertTrue(ktbn._bn.existsArc("A#0", "B#0"))
        self.assertTrue(ktbn._bn.existsArc("A#0", "A#1"))
        self.assertTrue(ktbn._bn.existsArc("C", "B#0"))

if __name__ == '__main__':
    unittest.main()
