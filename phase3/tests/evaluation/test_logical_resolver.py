import unittest
from phase3.scripts.evaluation.logical_resolver import resolve_expected_answer

class TestLogicalResolver(unittest.TestCase):
    
    def test_direct_comparison_greater_question(self):
        # HI_DC_01 asks "किसकी ... अधिक है?" -> greater
        record = {
            "id": "000001",
            "task_family": "direct_comparison",
            "facts": [{"subject": "E01", "relation": "less_than", "object": "E02"}],
            "question": {"type": "compare", "entities": ["E01", "E02"]},
            "ground_truth": {"relation": "less_than", "answer_entity": "E02"},
            "template_id": "HI_DC_01",
            "language": "hi"
        }
        res = resolve_expected_answer(record)
        self.assertEqual(res["expected_answer_entity"], "E02")
        self.assertEqual(res["question_direction"], "greater")

    def test_direct_comparison_less_question_discrepancy(self):
        # Discrepancy example ID 000003
        # E01 > E02. Question asks whose age is less.
        # ground_truth.answer_entity is E01 (wrong). The true answer is E02.
        record = {
            "id": "000003",
            "task_family": "direct_comparison",
            "facts": [{"subject": "E01", "relation": "greater_than", "object": "E02"}],
            "question": {"type": "compare", "entities": ["E01", "E02"]},
            "ground_truth": {"relation": "greater_than", "answer_entity": "E01"},
            "template_id": "HI_DC_03", # HI_DC_03 q_dir is "greater" ??
            "language": "hi"
        }
        # Wait, HI_DC_03 was q_dir="greater" according to the code:
        # q_dir = "greater" if i % 2 == 0 else "less". i=2 -> index 2 -> 3rd template -> %2==0 -> greater?
        # Let's see _build_hi_templates:
        # for i in range(4): tid = f"HI_DC_0{i+1}"
        # i=0 (01): q_dir="greater"
        # i=1 (02): q_dir="greater" (overridden in direct_comparison if i==1)
        # i=2 (03): q_dir="less" (overridden if i==2)
        # i=3 (04): q_dir="greater"
        # So HI_DC_03 has q_dir = "less".
        res = resolve_expected_answer(record)
        self.assertEqual(res["expected_answer_entity"], "E02")
        self.assertEqual(res["question_direction"], "less")

    def test_reversed_relation(self):
        record = {
            "id": "rev01",
            "task_family": "reversed_relation",
            "facts": [{"subject": "E01", "relation": "greater_than", "object": "E02"}],
            "question": {"type": "compare", "entities": ["E02", "E01"]},
            "template_id": "HI_RR_01", # q_dir = "greater"
            "language": "hi"
        }
        res = resolve_expected_answer(record)
        self.assertEqual(res["expected_answer_entity"], "E01")
        
    def test_largest(self):
        record = {
            "id": "large01",
            "task_family": "largest_smallest",
            "facts": [
                {"subject": "E01", "relation": "greater_than", "object": "E02"},
                {"subject": "E01", "relation": "greater_than", "object": "E03"}
            ],
            "question": {"type": "largest", "entities": ["E01", "E02", "E03"]},
            "template_id": "HI_LS_01",
            "language": "hi"
        }
        res = resolve_expected_answer(record)
        self.assertEqual(res["expected_answer_entity"], "E01")
        
    def test_smallest(self):
        record = {
            "id": "small01",
            "task_family": "largest_smallest",
            "facts": [
                {"subject": "E01", "relation": "less_than", "object": "E02"},
                {"subject": "E01", "relation": "less_than", "object": "E03"}
            ],
            "question": {"type": "smallest", "entities": ["E01", "E02", "E03"]},
            "template_id": "HI_LS_01",
            "language": "hi"
        }
        res = resolve_expected_answer(record)
        self.assertEqual(res["expected_answer_entity"], "E01")

    def test_equality(self):
        record = {
            "id": "eq01",
            "task_family": "direct_comparison",
            "facts": [{"subject": "E01", "relation": "equal", "object": "E02"}],
            "question": {"type": "compare", "entities": ["E01", "E02"]},
            "template_id": "HI_DC_01",
            "language": "hi"
        }
        res = resolve_expected_answer(record)
        self.assertIsNone(res["expected_answer_entity"])
        self.assertEqual(res["expected_relation"], "equal")
        
    def test_two_hop_transitivity(self):
        # E02 > E01, E02 < E03 => E01 < E02 < E03
        # Compare E01 and E03, less. Answer = E01.
        record = {
            "id": "003001",
            "task_family": "two_hop_transitivity",
            "facts": [
                {"subject": "E02", "relation": "greater_than", "object": "E01"},
                {"subject": "E02", "relation": "less_than", "object": "E03"}
            ],
            "question": {"type": "compare", "entities": ["E01", "E03"]},
            "template_id": "HI_THT_04", # i=3 (index 3). For THT, i=3 -> q_dir="greater" usually? Wait, let's see templates.py
            # For two_hop_transitivity, i=3 -> q_fmt compare... no q_dir override, so it defaults to "less" (since 3%2 != 0).
            "language": "hi"
        }
        res = resolve_expected_answer(record)
        # E01 < E03. Question asks for "less". Answer is E01.
        self.assertEqual(res["expected_answer_entity"], "E01")

    def test_three_hop_transitivity(self):
        # E01 > E02, E02 > E03, E03 > E04 => E01 > E04
        record = {
            "id": "three01",
            "task_family": "three_hop_transitivity",
            "facts": [
                {"subject": "E01", "relation": "greater_than", "object": "E02"},
                {"subject": "E02", "relation": "greater_than", "object": "E03"},
                {"subject": "E03", "relation": "greater_than", "object": "E04"}
            ],
            "question": {"type": "compare", "entities": ["E04", "E01"]},
            "template_id": "HI_THT_01", # i=0, q_dir="greater"
            "language": "hi"
        }
        res = resolve_expected_answer(record)
        # E04 < E01. Question asks for "greater". Answer is E01.
        self.assertEqual(res["expected_answer_entity"], "E01")

    def test_mixed_equality(self):
        # E01 = E02, E02 > E03 => E01 > E03
        record = {
            "id": "mix01",
            "task_family": "mixed_equality",
            "facts": [
                {"subject": "E01", "relation": "equal", "object": "E02"},
                {"subject": "E02", "relation": "greater_than", "object": "E03"}
            ],
            "question": {"type": "compare", "entities": ["E03", "E01"]},
            "template_id": "HI_ME_01", # i=0 -> q_dir="greater"
            "language": "hi"
        }
        res = resolve_expected_answer(record)
        # E03 < E01. Question asks for "greater". Answer is E01.
        self.assertEqual(res["expected_answer_entity"], "E01")


    def test_target_text_extraction(self):
        # We test that get_target_answer_entity extracts from target_text
        # and ignores ground_truth.answer_entity.
        import sys
        # We need to import get_target_answer_entity from phase3.scripts.evaluation.audit_ground_truth
        from phase3.scripts.evaluation.audit_ground_truth import get_target_answer_entity
        
        record = {
            "id": "target_extract_01",
            "task_family": "direct_comparison",
            "question": {"type": "compare", "entities": ["E01", "E02"]},
            "ground_truth": {"relation": "greater_than", "answer_entity": "E01"}, # Intentional fake ground truth
            "entity_mapping": {"E01": "Karan", "E02": "Simran"},
            "target_text": "Simran is the right answer." # "Simran" is E02
        }
        ans = get_target_answer_entity(record)
        self.assertEqual(ans, "E02")

if __name__ == "__main__":
    unittest.main()
