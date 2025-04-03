import unittest
import pyAgrum as gum
from src.DynamicBayesNet import DynamicBayesNet, dTensor
from src.notebook import unrollKTBN, getPosterior

class TestDynamicBayesNet(unittest.TestCase):

    def setUp(self):
        self.dBN = DynamicBayesNet(3)
        self.varA = gum.LabelizedVariable("A", "A", 2)
        self.varB = gum.LabelizedVariable("B", "B", 2)
        self.dBN.add(self.varA)
        self.dBN.add(self.varB)

    def test_variable_addition(self):
        self.assertIn("A", self.dBN.variables)
        self.assertIn("B", self.dBN.variables)
        for t in range(3):
            self.assertTrue(self.dBN.kTBN.exists(self.dBN._userToCodeName("A", t)))

    def test_id_from_name(self):
        var_id = self.dBN.idFromName(("A", 1))
        self.assertEqual(self.dBN.kTBN.variable(var_id).name(), "A#1")

    def test_add_arc_and_arcs_output(self):
        self.dBN.addArc(("A", 0), ("B", 1))
        arcs = self.dBN.arcs()
        self.assertTrue(any("('A', 0) -> ('B', 1)" in arc for arc in arcs))

    def test_erase_arc(self):
        self.dBN.addArc(("A", 0), ("B", 1))
        self.dBN.eraseArc(("A", 0), ("B", 1))
        self.assertNotIn(("A#0", "B#1"), [(self.dBN.kTBN.variable(t).name(), self.dBN.kTBN.variable(h).name()) for t, h in self.dBN.kTBN.arcs()])

    def test_erase_variable(self):
        self.dBN.erase("B")
        self.assertNotIn("B", self.dBN.variables)
        for t in range(3):
            self.assertFalse(self.dBN.kTBN.exists(f"B#{t}"))

    def test_addFast(self):
        self.dBN.addFast("C{on|off}")
        self.assertIn("C", self.dBN.variables)
        for t in range(3):
            self.assertTrue(self.dBN.kTBN.exists(f"C#{t}"))

    def test_cpt_set_and_get(self):
        self.dBN.addArc(("A", 0), ("B", 1))
        cpt = self.dBN.cpt(("B", 1))
        cpt[{("A", 0): 1}] = 0.75
        self.assertAlmostEqual(cpt[{("A", 0): 1}][0], 0.75)

    def test_generateCPTs(self):
        self.dBN.generateCPTs()
        cpt = self.dBN.cpt(("A", 1))
        self.assertIsInstance(cpt, dTensor)
    
    def test_unroll_ktbn_preserves_structure_and_cpt(self):
        dBN = DynamicBayesNet(3)
        a = gum.LabelizedVariable("A", "A", 2)
        b = gum.LabelizedVariable("B", "B", 2)
        dBN.add(a)
        dBN.add(b)
        dBN.addArc(("A", 0), ("A", 1))
        dBN.addArc(("B", 0), ("B", 1))
        dBN.addArc(("A", 1), ("B", 2))
        dBN.addArc(("B", 1), ("B", 2))
        dBN.cpt(('B', 2)).fillWith([0.3333, 0.7777, 0.6, 0.4, 0.5, 0.5, 0.2, 0.8])

        unrolled = unrollKTBN(dBN, 5)

        # Check structure
        self.assertTrue(unrolled.exists("A#4"))
        self.assertTrue(unrolled.exists("B#4"))

        # Check CPT propagation
        val = unrolled.cpt("B#4")[{"B#3": 1}]
        self.assertAlmostEqual(unrolled.cpt("B#4")[{"B#3": 1}].all(), unrolled.cpt("B#2")[{"B#1": 1}].all())

    def test_tensor_fill_and_readback(self):
        dBN = DynamicBayesNet(3)
        b = gum.LabelizedVariable("B", "B", 2)
        dBN.add(b)
        dBN.cpt(('B', 2)).fillWith([0.1, 0.9])

        self.assertAlmostEqual(dBN.cpt(('B', 2))[0], 0.1, delta=1e-3)
        self.assertAlmostEqual(dBN.cpt(('B', 2))[1], 0.9, delta=1e-3)

    # def test_showCPT_string_format(self):
    #     dBN = DynamicBayesNet(2)
    #     a = gum.LabelizedVariable("A", "A", 2)
    #     dBN.add(a)
    #     dBN.generateCPTs()

    #     from notebook import showCPT
    #     cpt = showCPT(dBN, ('A', 1))
    #     self.assertIn("('A'", str(cpt))  # formatted user name check

    def test_complex_unroll_structure(self):
        dbn = DynamicBayesNet(2)
        a = gum.LabelizedVariable("a", "a", 6)
        b = gum.LabelizedVariable("b", "b", 6)
        c = gum.LabelizedVariable("c", "c", 6)
        d = gum.LabelizedVariable("d", "d", 3)
        for v in [a, b, c, d]:
            dbn.add(v)
        dbn.addArc(("c", 0), ("d", 0))
        dbn.addArc(("c", 0), ("a", 1))
        dbn.addArc(("c", 0), ("c", 1))
        dbn.addArc(("d", 0), ("c", 1))
        dbn.addArc(("d", 0), ("d", 1))
        dbn.addArc(("a", 0), ("a", 1))
        dbn.addArc(("a", 0), ("d", 1))
        dbn.addArc(("a", 0), ("b", 1))
        dbn.addArc(("a", 0), ("b", 0))
        dbn.addArc(("a", 1), ("c", 1))
        dbn.addArc(("b", 0), ("b", 1))
        dbn.generateCPTs()

        unrolled = unrollKTBN(dbn, 5)
        self.assertTrue(unrolled.exists("c#4"))
        self.assertTrue(unrolled.exists("d#3"))
        self.assertGreaterEqual(len(unrolled.arcs()), 11)


    # def test_getPosterior_output(self):
    #     from notebook import getPosterior
    
    #     dbn = DynamicBayesNet(3)
    #     a = gum.LabelizedVariable("A", "A", 2)
    #     b = gum.LabelizedVariable("B", "B", 2)
    #     dbn.add(a)
    #     dbn.add(b)
    #     dbn.addArc(("A", 0), ("B", 1))
    #     dbn.generateCPTs()
    
    #     bn = unrollKTBN(dbn, 3)
    #     # We test that this function returns a matplotlib Figure object
    #     hist = getPosterior(bn, evs={}, target=("B", 1))
    #     self.assertTrue(hasattr(hist, "get_axes") or "Figure" in type(hist).__name__)


if __name__ == "__main__":
    unittest.main()
