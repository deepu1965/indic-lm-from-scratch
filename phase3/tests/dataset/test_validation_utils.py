import json
import unittest
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from phase3.scripts.dataset.validation_utils import validate_entity_preservation, validate_metadata_integrity, full_acceptance_validation, validate_pilot_ids

class TestPilotValidation(unittest.TestCase):
    def setUp(self):
        self.original = {
            "id": "0001",
            "task_family": "direct_comparison",
            "difficulty": "easy",
            "attribute": "height",
            "representation": "numerical",
            "entities": [{"id": "E01"}, {"id": "E02"}],
            "facts": [{"subject": "E01", "relation": "less_than", "object": "E02"}],
            "question": {"type": "compare", "entities": ["E01", "E02"]},
            "ground_truth": {"relation": "less_than", "answer_entity": "E02"},
            "reasoning_path": ["E01", "E02"],
            "relation_pattern_id": "P01",
            "seed": 123,
            "split": "train",
            "values": {"E01": 150, "E02": 180},
            "template_id": "HI_DC_01",
            "language": "hi",
            "entity_mapping": {"E01": "मनीषा", "E02": "गीता"},
            "input_text": "मनीषा की ऊँचाई 150 सेमी है। गीता की ऊँचाई 180 सेमी है। दोनों में से किसकी ऊँचाई अधिक है?",
            "target_text": "गीता की ऊँचाई अधिक है।"
        }
        import copy
        self.generated = copy.deepcopy(self.original)
        self.generated["input_text"] = "मनीषा की ऊँचाई 150 सेमी है। गीता 180 सेमी की है। किसकी ऊँचाई अधिक है?"
        
    def test_A_entity_mapping_changed(self):
        self.generated["entity_mapping"]["E01"] = "नितिन"
        ok, msg = validate_entity_preservation(self.original, self.generated)
        self.assertFalse(ok)
        self.assertIn("Entity mapping changed", msg)

    def test_B_entity_removed(self):
        self.generated["input_text"] = "गीता 180 सेमी की है। किसकी ऊँचाई अधिक है?"
        ok, msg = validate_entity_preservation(self.original, self.generated)
        self.assertFalse(ok)
        self.assertIn("removed from generated input", msg)
        
    def test_C_new_entity_introduced(self):
        self.generated["input_text"] = "मनीषा की ऊँचाई 150 सेमी है। गीता 180 सेमी की है। नितिन भी है। किसकी ऊँचाई अधिक है?"
        ok, msg = validate_entity_preservation(self.original, self.generated)
        self.assertFalse(ok)
        self.assertIn("Unexpected entity", msg)

    def test_D_entity_substituted(self):
        self.generated["input_text"] = "नितिन की ऊँचाई 150 सेमी है। गीता 180 सेमी की है। किसकी ऊँचाई अधिक है?"
        ok, msg = validate_entity_preservation(self.original, self.generated)
        self.assertFalse(ok) # Should fail either because removed or unexpected
        
    def test_E_numerical_value_changed(self):
        self.generated["input_text"] = "मनीषा की ऊँचाई 155 सेमी है। गीता 180 सेमी की है। किसकी ऊँचाई अधिक है?"
        ok, msg = full_acceptance_validation(self.original, self.generated)
        self.assertFalse(ok)
        self.assertTrue("Numerical Audit Failed" in msg or "Linguistic Validation" in msg)

    def test_F_numerical_value_swapped(self):
        self.generated["input_text"] = "मनीषा की ऊँचाई 180 सेमी है। गीता 150 सेमी की है। किसकी ऊँचाई अधिक है?"
        ok, msg = full_acceptance_validation(self.original, self.generated)
        self.assertFalse(ok)
        self.assertIn("Numerical Audit Failed", msg)
        
    def test_G_wrong_answer(self):
        self.generated["target_text"] = "मनीषा की ऊँचाई अधिक है।"
        ok, msg = full_acceptance_validation(self.original, self.generated)
        self.assertFalse(ok)
        self.assertTrue("Semantic Audit Failed" in msg or "removed from generated target" in msg)
        
    def test_H_wrong_question_direction(self):
        self.generated["input_text"] = "मनीषा की ऊँचाई 150 सेमी है। गीता 180 सेमी की है। दोनों में से किसकी ऊँचाई कम है?"
        ok, msg = full_acceptance_validation(self.original, self.generated)
        self.assertFalse(ok)
        self.assertIn("Semantic Audit Failed", msg)
        
    def test_I_metadata_changed(self):
        self.generated["difficulty"] = "hard"
        ok, msg = validate_metadata_integrity(self.original, self.generated)
        self.assertFalse(ok)
        
    def test_J_missing_pilot_id(self):
        orig_list = [self.original, {"id": "0002"}]
        gen_list = [self.generated]
        ok, msg = validate_pilot_ids(orig_list, gen_list)
        self.assertFalse(ok)
        
    def test_K_extra_pilot_id(self):
        orig_list = [self.original]
        gen_list = [self.generated, {"id": "0002"}]
        ok, msg = validate_pilot_ids(orig_list, gen_list)
        self.assertFalse(ok)
        
    def test_L_duplicate_pilot_id(self):
        orig_list = [self.original]
        gen_list = [self.generated, self.generated]
        ok, msg = validate_pilot_ids(orig_list, gen_list)
        self.assertFalse(ok)
        
    def test_M_valid_paraphrase(self):
        self.generated["input_text"] = "मनीषा की ऊँचाई 150 सेमी और गीता की ऊँचाई 180 सेमी है। इन दोनों में से किसकी ऊँचाई अधिक है?"
        self.generated["target_text"] = "गीता की ऊँचाई अधिक है।"
        ok, msg = full_acceptance_validation(self.original, self.generated)
        self.assertTrue(ok, msg)

if __name__ == '__main__':
    unittest.main()
