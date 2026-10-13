import unittest
from phase3.template_system.entities import ENTITY_POOLS, assign_entities
from phase3.template_system.selector import select_template
from phase3.template_system.templates import get_templates

class TestTemplateSystem(unittest.TestCase):
    
    def test_entity_pools(self):
        for lang in ["hi", "ne"]:
            pool = ENTITY_POOLS[lang]
            train = pool["train"]
            val = pool["val"]
            test = pool["test"]
            
            self.assertEqual(len(train), 40)
            self.assertEqual(len(val), 10)
            self.assertEqual(len(test), 10)
            
            # Check disjoint
            self.assertTrue(set(train).isdisjoint(set(val)))
            self.assertTrue(set(train).isdisjoint(set(test)))
            self.assertTrue(set(val).isdisjoint(set(test)))
            
            # Check unique names total 60
            all_names = train + val + test
            self.assertEqual(len(all_names), 60)
            self.assertEqual(len(set(all_names)), 60)
            
    def test_assign_entities_deterministic(self):
        ids = ["E01", "E02", "E03"]
        m1 = assign_entities("hi", "train", ids, 42)
        m2 = assign_entities("hi", "train", ids, 42)
        m3 = assign_entities("hi", "train", ids, 43)
        
        self.assertEqual(m1, m2)
        self.assertNotEqual(m1, m3) # Very likely different
        
    def test_template_counts(self):
        hi_t = get_templates("hi")
        ne_t = get_templates("ne")
        self.assertEqual(len(hi_t), 28)
        self.assertEqual(len(ne_t), 28)
        
    def test_selector_balancing(self):
        # We test that exactly 250 of each template are assigned across 1000 records
        counts = {0: 0, 1: 0, 2: 0, 3: 0}
        
        # Test direct_comparison (family 1, IDs 000001 to 001000)
        family_templates = [t for t in get_templates("hi") if t.task_family == "direct_comparison"]
        
        for i in range(1, 1001):
            record_id = f"{i:06d}"
            t = select_template("direct_comparison", "hi", record_id)
            idx = family_templates.index(t)
            counts[idx] += 1
            
        for i in range(4):
            self.assertEqual(counts[i], 250)
            
    def test_selector_deterministic(self):
        t1 = select_template("direct_comparison", "hi", "000005")
        t2 = select_template("direct_comparison", "hi", "000005")
        self.assertEqual(t1.template_id, t2.template_id)
        
if __name__ == "__main__":
    unittest.main()
