import unittest
import json
from phase3.scripts.evaluation.evaluate_reasoning import evaluate_record

# Real records from the training set
REC_DC = {"id": "000001", "task_family": "direct_comparison", "difficulty": "easy", "attribute": "height", "representation": "relational", "entities": [{"id": "E01"}, {"id": "E02"}], "facts": [{"subject": "E01", "relation": "less_than", "object": "E02"}], "question": {"type": "compare", "entities": ["E01", "E02"]}, "ground_truth": {"relation": "less_than", "answer_entity": "E02"}, "reasoning_path": ["E01", "E02"], "relation_pattern_id": "P02", "seed": 478163327, "split": "train", "template_id": "HI_DC_01", "language": "hi", "entity_mapping": {"E01": "आरव", "E02": "स्वाति"}, "input_text": "आरव की ऊँचाई स्वाति से कम है। आरव और स्वाति में से किसकी ऊँचाई अधिक है?", "target_text": "स्वाति की ऊँचाई अधिक है।"}

REC_RR = {"id": "005001", "task_family": "reversed_relation", "difficulty": "medium", "attribute": "price", "representation": "relational", "entities": [{"id": "E01"}, {"id": "E02"}, {"id": "E03"}], "facts": [{"subject": "E02", "relation": "equal", "object": "E03"}, {"subject": "E02", "relation": "less_than", "object": "E01"}], "question": {"type": "compare", "entities": ["E03", "E01"]}, "ground_truth": {"relation": "less_than", "answer_entity": "E01"}, "reasoning_path": ["E03", "E02", "E01"], "relation_pattern_id": "P06", "seed": 599310825, "split": "train", "template_id": "HI_RR_02", "language": "hi", "entity_mapping": {"E01": "अजय", "E02": "अंकित", "E03": "पूजा"}, "input_text": "अंकित और पूजा की कीमत बराबर है, और साथ ही अंकित की कीमत अजय से कम है। बताइए, पूजा और अजय में किसकी कीमत अधिक है?", "target_text": "अजय की कीमत अधिक है।"}

REC_THT = {"id": "003001", "task_family": "two_hop_transitivity", "difficulty": "hard", "attribute": "price", "representation": "relational", "entities": [{"id": "E01"}, {"id": "E02"}, {"id": "E03"}], "facts": [{"subject": "E02", "relation": "greater_than", "object": "E01"}, {"subject": "E02", "relation": "less_than", "object": "E03"}], "question": {"type": "compare", "entities": ["E01", "E03"]}, "ground_truth": {"relation": "less_than", "answer_entity": "E03"}, "reasoning_path": ["E01", "E02", "E03"], "relation_pattern_id": "P05", "seed": 1051802512, "split": "train", "template_id": "HI_THT_04", "language": "hi", "entity_mapping": {"E01": "सुनीता", "E02": "राजेश", "E03": "राहुल"}, "input_text": "दी गई जानकारी यह है कि राजेश की कीमत सुनीता से अधिक है, राजेश की कीमत राहुल से कम है। सुनीता और राहुल की तुलना करने पर किसकी कीमत कम है?", "target_text": "सुनीता की कीमत कम है।"}

REC_LS = {"id": "001001", "task_family": "largest_smallest", "difficulty": "easy", "attribute": "quantity", "representation": "numerical", "entities": [{"id": "E01"}, {"id": "E02"}, {"id": "E03"}], "facts": [{"subject": "E01", "relation": "greater_than", "object": "E02"}, {"subject": "E02", "relation": "greater_than", "object": "E03"}], "question": {"type": "largest", "entities": ["E01", "E02", "E03"]}, "ground_truth": {"answer_entity": "E01"}, "reasoning_path": ["E01", "E02", "E03"], "relation_pattern_id": "P04", "seed": 107420369, "split": "train", "values": {"E01": 801, "E02": 755, "E03": 740}, "template_id": "HI_LS_02", "language": "hi", "entity_mapping": {"E01": "अंकित", "E02": "अनिता", "E03": "निधि"}, "input_text": "हमें निम्नलिखित जानकारी दी गई है: अंकित की मात्रा 801 इकाई है, अनिता की मात्रा 755 इकाई है, निधि की मात्रा 740 इकाई है। अंकित और अनिता में से किसकी मात्रा अधिक है?", "target_text": "अंकित की मात्रा सबसे अधिक है।"}

REC_EQ = {"id": "000008", "task_family": "direct_comparison", "difficulty": "easy", "attribute": "price", "representation": "relational", "entities": [{"id": "E01"}, {"id": "E02"}], "facts": [{"subject": "E01", "relation": "equal", "object": "E02"}], "question": {"type": "compare", "entities": ["E01", "E02"]}, "ground_truth": {"relation": "equal"}, "reasoning_path": ["E01", "E02"], "relation_pattern_id": "P03", "seed": 18505557360, "split": "val", "template_id": "HI_DC_03", "language": "hi", "entity_mapping": {"E01": "विनय", "E02": "विवेक"}, "input_text": "दी गई जानकारी के अनुसार, विनय और विवेक की कीमत बराबर है। क्या विनय और विवेक की कीमत एक समान है?", "target_text": "विनय और विवेक की कीमत बराबर है।"}

class TestEvaluateReasoning(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        from tokenizers import Tokenizer
        cls.tokenizer = Tokenizer.from_file("hindi/tokenizer/models/bpe_v10000.json")
        
    def _mock_gen(self, text):
        return self.tokenizer.decode(self.tokenizer.encode(text).ids)
    
    def test_correct_greater_than(self):
        res = evaluate_record(REC_DC, self._mock_gen(REC_DC["target_text"]), self.tokenizer)
        self.assertEqual(res["predicted_answer"], "E02")
        self.assertEqual(res["independently_resolved_answer"], "E02")
        self.assertTrue(res["reasoning_correct"])
        self.assertTrue(res["exact_match"])

    def test_correct_less_than(self):
        res = evaluate_record(REC_THT, self._mock_gen(REC_THT["target_text"]), self.tokenizer)
        self.assertEqual(res["predicted_answer"], "E01")
        self.assertEqual(res["independently_resolved_answer"], "E01")
        self.assertTrue(res["reasoning_correct"])
        self.assertTrue(res["exact_match"])
        
    def test_reversed_relation(self):
        res = evaluate_record(REC_RR, self._mock_gen(REC_RR["target_text"]), self.tokenizer)
        self.assertEqual(res["predicted_answer"], "E01")
        self.assertTrue(res["reasoning_correct"])
        
    def test_largest_smallest(self):
        res = evaluate_record(REC_LS, self._mock_gen(REC_LS["target_text"]), self.tokenizer)
        self.assertEqual(res["predicted_answer"], "E01")
        self.assertTrue(res["reasoning_correct"])
        
    def test_equality_case_correct_not_exact(self):
        # Equality reasoning correct but exact match false
        res = evaluate_record(REC_EQ, self._mock_gen("विनय और विवेक की मात्रा समान है।"), self.tokenizer)
        self.assertEqual(res["predicted_answer"], "EQUAL_CASE")
        self.assertIsNone(res["independently_resolved_answer"])
        self.assertTrue(res["reasoning_correct"])
        self.assertFalse(res["exact_match"])

    def test_equality_case_wrong_entities(self):
        # Equality reasoning incorrect (wrong entities)
        res = evaluate_record(REC_EQ, self._mock_gen("अंकित और राजेश की मात्रा बराबर है।"), self.tokenizer)
        self.assertEqual(res["predicted_answer"], "EQUAL_CASE")
        self.assertIsNone(res["independently_resolved_answer"])
        self.assertFalse(res["reasoning_correct"])
        self.assertFalse(res["exact_match"])

    def test_equality_case_missing_keyword(self):
        # Correct entities but missing equality relation keyword
        res = evaluate_record(REC_EQ, self._mock_gen("विनय और विवेक की मात्रा अलग है।"), self.tokenizer)
        self.assertEqual(res["predicted_answer"], "EQUAL_CASE")
        self.assertIsNone(res["independently_resolved_answer"])
        self.assertFalse(res["reasoning_correct"])
        self.assertFalse(res["exact_match"])

    def test_equality_case_ambiguous(self):
        # Ambigous or only one entity
        res = evaluate_record(REC_EQ, self._mock_gen("विनय की मात्रा बराबर है।"), self.tokenizer)
        self.assertEqual(res["predicted_answer"], "EQUAL_CASE")
        self.assertIsNone(res["independently_resolved_answer"])
        self.assertFalse(res["reasoning_correct"])
        self.assertFalse(res["exact_match"])

    def test_single_entity_answer(self):
        # Correctly identify a single entity when the answer clearly names it
        res = evaluate_record(REC_DC, self._mock_gen("स्वाति की ऊँचाई अधिक है।"), self.tokenizer)
        self.assertEqual(res["predicted_answer"], "E02")
        self.assertTrue(res["reasoning_correct"])

    def test_ambiguous_two_entity_answer(self):
        # Both entities in generated text!
        gen_text = self._mock_gen("आरव और स्वाति की ऊँचाई अधिक है।")
        res = evaluate_record(REC_DC, gen_text, self.tokenizer)
        self.assertEqual(res["predicted_answer"], "AMBIGUOUS")
        self.assertFalse(res["reasoning_correct"])
        self.assertFalse(res["exact_match"])
        
    def test_no_recognizable_entity(self):
        # No recognizable entity
        gen_text = self._mock_gen("किसी की ऊँचाई अधिक नहीं है।")
        res = evaluate_record(REC_DC, gen_text, self.tokenizer)
        self.assertIsNone(res["predicted_answer"])
        self.assertFalse(res["reasoning_correct"])
        self.assertFalse(res["exact_match"])

    def test_exact_match_normalization(self):
        # Added trailing spaces
        res = evaluate_record(REC_DC, self._mock_gen("  स्वाति की ऊँचाई अधिक है।  "), self.tokenizer)
        self.assertEqual(res["predicted_answer"], "E02")
        self.assertTrue(res["reasoning_correct"])
        self.assertTrue(res["exact_match"])
        
    def test_ignore_ground_truth_metadata(self):
        # A known case where ground_truth.answer_entity differs from actual natural-language answer
        discrepancy_record = {
          "id": "000041",
          "language": "hi",
          "task_family": "direct_comparison",
          "difficulty": "easy",
          "representation": "relational",
          "facts": [
            {
              "subject": "E01",
              "relation": "greater_than",
              "object": "E02"
            }
          ],
          "question": {
            "type": "compare",
            "entities": ["E01", "E02"]
          },
          "ground_truth": {
            "relation": "greater_than",
            "answer_entity": "E01" # Hardcoded to E01 even though question asks for 'less'
          },
          "entity_mapping": {
            "E01": "करण",
            "E02": "सिमरन"
          },
          "input_text": "दी गई जानकारी के अनुसार, करण की मात्रा सिमरन से अधिक है। इनमें से किसकी मात्रा कम है: करण या सिमरन?",
          "target_text": "सिमरन की मात्रा कम है।",
          "template_id": "HI_DC_03"
        }
        # The generated text is correct based on the logic!
        res = evaluate_record(discrepancy_record, self._mock_gen(discrepancy_record["target_text"]), self.tokenizer)
        
        # The independently resolved answer SHOULD be E02 (since the template asks for less)
        self.assertEqual(res["independently_resolved_answer"], "E02")
        self.assertEqual(res["predicted_answer"], "E02")
        self.assertTrue(res["reasoning_correct"])
        self.assertTrue(res["exact_match"])

if __name__ == '__main__':
    unittest.main()
