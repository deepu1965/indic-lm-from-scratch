import sys
sys.path.append(".")
import unittest
from unittest.mock import patch
from phase3.template_system.validator import validate_linguistic_record, LinguisticValidationError, is_distinct_entity
from phase3.template_system.renderer import render_example
from phase3.template_system.templates import get_templates
from phase3.template_system.lexicon import get_lexicon
from phase3.scripts.dataset.audit_dataset import independent_semantic_audit, TEMPLATE_CONTRACT

class TestAuditSemantics(unittest.TestCase):
    def setUp(self):
        self.base_record = {
            "id": "000001",
            "seed": 42,
            "split": "train",
            "task_family": "direct_comparison",
            "attribute": "height",
            "representation": "numerical",
            "language": "hi",
            "values": {"E01": 150, "E02": 180},
            "facts": [{"subject": "E01", "relation": "less_than", "object": "E02"}],
            "question": {"type": "compare", "entities": ["E01", "E02"]},
            "ground_truth": {"relation": "less_than", "answer_entity": "E02"},
            "entities": [{"id": "E01"}, {"id": "E02"}]
        }
        
    def _test_and_assert(self, r, lang="hi"):
        rendered = render_example(r, lang)
        self.assertTrue(validate_linguistic_record(rendered))
        
        # Explicit direct assertions
        tgt = rendered["target_text"]
        mapping = rendered["entity_mapping"]
        lex = get_lexicon(lang)["attributes"][r["attribute"]]
        adj_greater = lex["adj_greater"]
        adj_less = lex["adj_less"]
        q_type = r["question"]["type"]
        tid = r.get("template_id")
        
        # Determine expected direction directly from the audited Phase 3.2 template contract
        key = (lang, r.get("task_family"), tid)
        expected_dir = TEMPLATE_CONTRACT.get(key)
        
        if q_type == "equal" or (q_type == "compare" and r["ground_truth"]["relation"] == "equal"):
            self.assertTrue(any(w in tgt for w in ["बराबर", "समान", "जत्तिकै"]))
            for eid in r["question"]["entities"]:
                self.assertTrue(is_distinct_entity(mapping[eid], tgt))
            self.assertNotIn(adj_greater, tgt)
            self.assertNotIn(adj_less, tgt)
        elif q_type == "largest":
            ans_eid = r["ground_truth"]["answer_entity"]
            self.assertTrue(is_distinct_entity(mapping[ans_eid], tgt))
            self.assertIn(adj_greater, tgt)
            self.assertNotIn(adj_less, tgt)
        elif q_type == "smallest":
            ans_eid = r["ground_truth"]["answer_entity"]
            self.assertTrue(is_distinct_entity(mapping[ans_eid], tgt))
            self.assertIn(adj_less, tgt)
            self.assertNotIn(adj_greater, tgt)
        elif expected_dir == "greater":
            ans_eid = r["ground_truth"]["answer_entity"]
            self.assertTrue(is_distinct_entity(mapping[ans_eid], tgt))
            self.assertIn(adj_greater, tgt)
            self.assertNotIn(adj_less, tgt)
            for eid, name in mapping.items():
                if eid != ans_eid:
                    self.assertFalse(is_distinct_entity(name, tgt))
        elif expected_dir == "less":
            ans_eid = r["ground_truth"]["answer_entity"]
            less_eid = [e for e in r["question"]["entities"] if e != ans_eid][0]
            self.assertTrue(is_distinct_entity(mapping[less_eid], tgt))
            self.assertIn(adj_less, tgt)
            self.assertNotIn(adj_greater, tgt)
            for eid, name in mapping.items():
                if eid != less_eid:
                    self.assertFalse(is_distinct_entity(name, tgt))
                    
        return rendered

    @patch('phase3.template_system.validator.select_template')
    @patch('phase3.template_system.renderer.select_template')
    def test_A_greater_question(self, mock_render_select, mock_val_select):
        for lang, t_id in [("hi", "HI_DC_01"), ("ne", "NE_DC_01")]:
            mock_tpl = [t for t in get_templates(lang) if t.template_id == t_id][0]
            mock_render_select.return_value = mock_tpl
            mock_val_select.return_value = mock_tpl
            r = dict(self.base_record)
            r["language"] = lang
            r["template_id"] = t_id
            self._test_and_assert(r, lang)
            
    @patch('phase3.template_system.validator.select_template')
    @patch('phase3.template_system.renderer.select_template')
    def test_B_less_question(self, mock_render_select, mock_val_select):
        for lang, t_id in [("hi", "HI_DC_03"), ("ne", "NE_DC_03")]:
            mock_tpl = [t for t in get_templates(lang) if t.template_id == t_id][0]
            mock_render_select.return_value = mock_tpl
            mock_val_select.return_value = mock_tpl
            r = dict(self.base_record)
            r["language"] = lang
            r["template_id"] = t_id
            self._test_and_assert(r, lang)

    @patch('phase3.template_system.validator.select_template')
    @patch('phase3.template_system.renderer.select_template')
    def test_C_reversed_greater_question(self, mock_render_select, mock_val_select):
        for lang, t_id in [("hi", "HI_RR_01"), ("ne", "NE_RR_01")]:
            mock_tpl = [t for t in get_templates(lang) if t.template_id == t_id][0]
            mock_render_select.return_value = mock_tpl
            mock_val_select.return_value = mock_tpl
            r = dict(self.base_record)
            r["language"] = lang
            r["template_id"] = t_id
            r["task_family"] = "reversed_relation"
            r["question"]["entities"] = ["E02", "E01"]
            self._test_and_assert(r, lang)
            
    @patch('phase3.template_system.validator.select_template')
    @patch('phase3.template_system.renderer.select_template')
    def test_D_reversed_less_question(self, mock_render_select, mock_val_select):
        for lang, t_id in [("hi", "HI_RR_03"), ("ne", "NE_RR_03")]:
            mock_tpl = [t for t in get_templates(lang) if t.template_id == t_id][0]
            mock_render_select.return_value = mock_tpl
            mock_val_select.return_value = mock_tpl
            r = dict(self.base_record)
            r["language"] = lang
            r["template_id"] = t_id
            r["task_family"] = "reversed_relation"
            r["question"]["entities"] = ["E02", "E01"]
            self._test_and_assert(r, lang)

    @patch('phase3.template_system.validator.select_template')
    @patch('phase3.template_system.renderer.select_template')
    def test_E_equality(self, mock_render_select, mock_val_select):
        for lang, t_id in [("hi", "HI_ME_01"), ("ne", "NE_ME_01")]:
            mock_tpl = [t for t in get_templates(lang) if t.template_id == t_id][0]
            mock_render_select.return_value = mock_tpl
            mock_val_select.return_value = mock_tpl
            r = dict(self.base_record)
            r["task_family"] = "mixed_equality"
            r["language"] = lang
            r["template_id"] = t_id
            r["ground_truth"] = {"relation": "equal", "answer_entity": None}
            self._test_and_assert(r, lang)

    @patch('phase3.template_system.validator.select_template')
    @patch('phase3.template_system.renderer.select_template')
    def test_F_largest(self, mock_render_select, mock_val_select):
        for lang, t_id in [("hi", "HI_LS_01"), ("ne", "NE_LS_01")]:
            mock_tpl = [t for t in get_templates(lang) if t.template_id == t_id][0]
            mock_render_select.return_value = mock_tpl
            mock_val_select.return_value = mock_tpl
            r = dict(self.base_record)
            r["language"] = lang
            r["template_id"] = t_id
            r["task_family"] = "largest_smallest"
            r["question"] = {"type": "largest", "entities": ["E01", "E02", "E03"]}
            r["ground_truth"] = {"answer_entity": "E03"}
            r["entities"].append({"id": "E03"})
            r["facts"].append({"subject": "E03", "relation": "greater_than", "object": "E02"}) 
            r["values"]["E03"] = 200
            self._test_and_assert(r, lang)
            
    @patch('phase3.template_system.validator.select_template')
    @patch('phase3.template_system.renderer.select_template')
    def test_G_smallest(self, mock_render_select, mock_val_select):
        for lang, t_id in [("hi", "HI_LS_03"), ("ne", "NE_LS_03")]:
            mock_tpl = [t for t in get_templates(lang) if t.template_id == t_id][0]
            mock_render_select.return_value = mock_tpl
            mock_val_select.return_value = mock_tpl
            r = dict(self.base_record)
            r["language"] = lang
            r["template_id"] = t_id
            r["task_family"] = "largest_smallest"
            r["question"] = {"type": "smallest", "entities": ["E01", "E02", "E03"]}
            r["ground_truth"] = {"answer_entity": "E01"}
            r["entities"].append({"id": "E03"})
            r["facts"].append({"subject": "E03", "relation": "greater_than", "object": "E02"}) 
            r["values"]["E03"] = 200
            self._test_and_assert(r, lang)

    @patch('phase3.template_system.validator.select_template')
    @patch('phase3.template_system.renderer.select_template')
    def test_independent_semantic_audit_negatives(self, mock_render_select, mock_val_select):
        # Test 1: Question asks GREATER, but target mathematically equivalent answers LESS
        mock_tpl = [t for t in get_templates("hi") if t.template_id == "HI_DC_01"][0]
        mock_render_select.return_value = mock_tpl
        mock_val_select.return_value = mock_tpl
        rendered = render_example(self.base_record, "hi")
        
        # Manually alter target to answer for the lesser entity ("B is less")
        bad1 = dict(rendered)
        bad1["target_text"] = f"{rendered['entity_mapping']['E01']} की ऊँचाई कम है।"
        ok, msg = independent_semantic_audit(bad1)
        self.assertFalse(ok)
        expected_name_greater = rendered['entity_mapping']['E02']
        self.assertIn(f"Target missing expected entity '{expected_name_greater}' for greater question", msg)
        
        # Test 2: Question asks LESS, but target mathematically equivalent answers GREATER
        mock_tpl_less = [t for t in get_templates("hi") if t.template_id == "HI_DC_03"][0]
        mock_render_select.return_value = mock_tpl_less
        mock_val_select.return_value = mock_tpl_less
        rendered_less = render_example(self.base_record, "hi")
        
        # Manually alter target to answer for the greater entity ("A is greater")
        bad2 = dict(rendered_less)
        bad2["target_text"] = f"{rendered_less['entity_mapping']['E02']} की ऊँचाई अधिक है।"
        ok, msg = independent_semantic_audit(bad2)
        self.assertFalse(ok)
        expected_name_less = rendered_less['entity_mapping']['E01']
        self.assertIn(f"Target missing expected entity '{expected_name_less}' for less question", msg)

    def test_contract_coverage(self):
        expected_keys = {
            (t.language, t.task_family, t.template_id)
            for lang in ["hi", "ne"]
            for t in get_templates(lang)
            if t.task_family != "largest_smallest"
        }
        
        # 1. Exact expected keys equality
        self.assertEqual(set(TEMPLATE_CONTRACT.keys()), expected_keys)
        
        # 2. Exactly 48 compare templates expected
        self.assertEqual(len(expected_keys), 48)
        self.assertEqual(len(TEMPLATE_CONTRACT), 48)
        
        # 3. Every contract value is exactly "greater" or "less"
        for val in TEMPLATE_CONTRACT.values():
            self.assertIn(val, ["greater", "less"])
            
        # 4. largest_smallest has no entries
        ls_keys = [k for k in TEMPLATE_CONTRACT.keys() if k[1] == "largest_smallest"]
        self.assertEqual(len(ls_keys), 0)
        
        # Python dicts cannot have duplicate keys, but we also assert exact len equality
        
    def test_contract_consistency(self):
        # Verifies the static contract agrees with the actual declared question_direction of every compare template.
        from phase3.scripts.dataset.audit_dataset import validate_template_contract_consistency
        validate_template_contract_consistency()

    def test_contract_consistency_negative(self):
        # Add a test that deliberately creates a mismatch between templates.py and the static contract
        from phase3.scripts.dataset.audit_dataset import validate_template_contract_consistency
        t = [tpl for tpl in get_templates("hi") if tpl.template_id == "HI_DC_01"][0]
        original_dir = t.question_direction
        
        try:
            # Create a deliberate mismatch
            t.question_direction = "less"
            
            # The consistency test logic should now fail
            with self.assertRaises(AssertionError):
                validate_template_contract_consistency()
                
        finally:
            t.question_direction = original_dir
            
    def test_unmocked_e2e_smoke(self):
        # One unmocked end-to-end smoke test that exercises the real selector, renderer, validator, and semantic audit.
        from phase3.template_system.selector import select_template
        from phase3.scripts.dataset.audit_dataset import check_numerical_preservation
        
        for lang in ["hi", "ne"]:
            r = dict(self.base_record)
            r["language"] = lang
            
            # Selector assigns a template
            tpl = select_template(r["task_family"], lang, r["id"])
            r["template_id"] = tpl.template_id
            
            # Render text (this also maps entities)
            rendered_record = render_example(r, lang)
            
            # Linguistic validation
            self.assertTrue(validate_linguistic_record(rendered_record))
            
            # Semantic Audit
            ok, msg = independent_semantic_audit(rendered_record)
            self.assertTrue(ok, msg)
            
            # Numerical Preservation Audit
            ok, msg = check_numerical_preservation(rendered_record)
            self.assertTrue(ok, msg)
                
    @patch('phase3.template_system.validator.select_template')
    @patch('phase3.template_system.renderer.select_template')
    def test_contract_mismatch_detection(self, mock_render_select, mock_val_select):
        # Deliberately modify a template so its rendered question conflicts with the contract
        mock_tpl = [t for t in get_templates("hi") if t.template_id == "HI_DC_01"][0]
        original_dir = mock_tpl.question_direction
        try:
            # Contract says HI_DC_01 is "greater"
            # We manually alter the template object to generate a "less" question
            mock_tpl.question_direction = "less" 
            mock_render_select.return_value = mock_tpl
            mock_val_select.return_value = mock_tpl
            
            rendered = render_example(self.base_record, "hi")
            
            ok, msg = independent_semantic_audit(rendered)
            self.assertFalse(ok)
            self.assertIn("implies direction less but template contract expects greater", msg)
        finally:
            mock_tpl.question_direction = original_dir

if __name__ == "__main__":
    unittest.main()

