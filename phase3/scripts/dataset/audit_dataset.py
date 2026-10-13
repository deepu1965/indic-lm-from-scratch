import json
import sys
from collections import defaultdict
import hashlib
import re

sys.path.append(".")
from phase3.template_system.validator import validate_linguistic_record, LinguisticValidationError, is_distinct_entity
from phase3.template_system.entities import ENTITY_POOLS
from phase3.template_system.lexicon import get_lexicon

# EXPLICIT PHASE 3.2 SEMANTIC DIRECTION CONTRACT
# This explicitly defines whether a given compare template expects the GREATER or LESSER entity answer.
# Note: General comparison templates (which neutrally ask to "compare" or find the "difference") 
# are explicitly contracted in Phase 3.2 to request the GREATER entity.
# Largest/smallest templates are excluded as they are verified differently.
TEMPLATE_CONTRACT = {
    ('hi', 'direct_comparison', 'HI_DC_01'): "greater",
    ('hi', 'direct_comparison', 'HI_DC_02'): "greater",
    ('hi', 'direct_comparison', 'HI_DC_03'): "less",
    ('hi', 'direct_comparison', 'HI_DC_04'): "greater",
    ('hi', 'general_comparison', 'HI_GC_01'): "greater",
    ('hi', 'general_comparison', 'HI_GC_02'): "greater",
    ('hi', 'general_comparison', 'HI_GC_03'): "greater",
    ('hi', 'general_comparison', 'HI_GC_04'): "greater",
    ('hi', 'two_hop_transitivity', 'HI_THT_01'): "greater",
    ('hi', 'two_hop_transitivity', 'HI_THT_02'): "greater",
    ('hi', 'two_hop_transitivity', 'HI_THT_03'): "less",
    ('hi', 'two_hop_transitivity', 'HI_THT_04'): "less",
    ('hi', 'three_hop_transitivity', 'HI_THT_01'): "greater",
    ('hi', 'three_hop_transitivity', 'HI_THT_02'): "greater",
    ('hi', 'three_hop_transitivity', 'HI_THT_03'): "less",
    ('hi', 'three_hop_transitivity', 'HI_THT_04'): "greater",
    ('hi', 'reversed_relation', 'HI_RR_01'): "greater",
    ('hi', 'reversed_relation', 'HI_RR_02'): "greater",
    ('hi', 'reversed_relation', 'HI_RR_03'): "less",
    ('hi', 'reversed_relation', 'HI_RR_04'): "greater",
    ('hi', 'mixed_equality', 'HI_ME_01'): "greater",
    ('hi', 'mixed_equality', 'HI_ME_02'): "greater",
    ('hi', 'mixed_equality', 'HI_ME_03'): "less",
    ('hi', 'mixed_equality', 'HI_ME_04'): "greater",
    ('ne', 'direct_comparison', 'NE_DC_01'): "greater",
    ('ne', 'direct_comparison', 'NE_DC_02'): "greater",
    ('ne', 'direct_comparison', 'NE_DC_03'): "less",
    ('ne', 'direct_comparison', 'NE_DC_04'): "greater",
    ('ne', 'general_comparison', 'NE_GC_01'): "greater",
    ('ne', 'general_comparison', 'NE_GC_02'): "greater",
    ('ne', 'general_comparison', 'NE_GC_03'): "greater",
    ('ne', 'general_comparison', 'NE_GC_04'): "greater",
    ('ne', 'two_hop_transitivity', 'NE_THT_01'): "greater",
    ('ne', 'two_hop_transitivity', 'NE_THT_02'): "greater",
    ('ne', 'two_hop_transitivity', 'NE_THT_03'): "less",
    ('ne', 'two_hop_transitivity', 'NE_THT_04'): "less",
    ('ne', 'three_hop_transitivity', 'NE_THT_01'): "greater",
    ('ne', 'three_hop_transitivity', 'NE_THT_02'): "greater",
    ('ne', 'three_hop_transitivity', 'NE_THT_03'): "less",
    ('ne', 'three_hop_transitivity', 'NE_THT_04'): "greater",
    ('ne', 'reversed_relation', 'NE_RR_01'): "greater",
    ('ne', 'reversed_relation', 'NE_RR_02'): "greater",
    ('ne', 'reversed_relation', 'NE_RR_03'): "less",
    ('ne', 'reversed_relation', 'NE_RR_04'): "greater",
    ('ne', 'mixed_equality', 'NE_ME_01'): "greater",
    ('ne', 'mixed_equality', 'NE_ME_02'): "greater",
    ('ne', 'mixed_equality', 'NE_ME_03'): "less",
    ('ne', 'mixed_equality', 'NE_ME_04'): "greater",
}

def validate_template_contract_consistency():
    from phase3.template_system.templates import get_templates
    
    expected_keys = set()
    
    for lang in ["hi", "ne"]:
        for t in get_templates(lang):
            if t.task_family == "largest_smallest":
                continue
                
            key = (t.language, t.task_family, t.template_id)
            expected_keys.add(key)
            
            template_dir = t.question_direction
            if template_dir == "general":
                template_dir = "greater"
                
            if key not in TEMPLATE_CONTRACT:
                raise AssertionError(f"Missing template {key} in TEMPLATE_CONTRACT")
                
            if TEMPLATE_CONTRACT[key] != template_dir:
                raise AssertionError(f"Mismatch for {key}: contract says {TEMPLATE_CONTRACT[key]} but template says {template_dir}")
                
    if set(TEMPLATE_CONTRACT.keys()) != expected_keys:
        unexpected = set(TEMPLATE_CONTRACT.keys()) - expected_keys
        raise AssertionError(f"Unexpected entries in TEMPLATE_CONTRACT: {unexpected}")

def independent_semantic_audit(r):
    inp = r["input_text"]
    tgt = r["target_text"]
    mapping = r["entity_mapping"]
    q_type = r["question"]["type"]
    ans_eid = r["ground_truth"].get("answer_entity")
    rel = r["ground_truth"].get("relation")
    
    lex = get_lexicon(r["language"])["attributes"][r["attribute"]]
    adj_greater = lex["adj_greater"]
    adj_less = lex["adj_less"]
    
    # 1. Equality
    if rel == "equal" or q_type == "equal":
        # must have equality word
        if not any(w in tgt for w in ["बराबर", "समान", "जत्तिकै"]):
            return False, "Equality keyword missing in target"
        for eid in r["question"]["entities"]:
            if not is_distinct_entity(mapping[eid], tgt):
                return False, f"Missing {mapping[eid]} in equality target"
        if adj_greater in tgt or adj_less in tgt:
            return False, "Contains greater/less adj in equality target"
        return True, ""
        
    # 2. Largest / Smallest
    if q_type == "largest":
        if not is_distinct_entity(mapping[ans_eid], tgt): return False, "Missing largest entity in target"
        if adj_greater not in tgt: return False, "Missing greater adj in largest target"
        if adj_less in tgt: return False, "Contains less adj in largest target"
        return True, ""
    
    if q_type == "smallest":
        # Phase 3.1 smallest query: ans_eid is the smallest entity.
        if not is_distinct_entity(mapping[ans_eid], tgt): return False, "Missing smallest entity in target"
        if adj_less not in tgt: return False, "Missing less adj in smallest target"
        if adj_greater in tgt: return False, "Contains greater adj in smallest target"
        return True, ""
        
    # 3. Compare (direct, reversed, hops)
    if q_type == "compare":
        # 1. Establish expected direction from explicit Phase 3.2 contract
        tid = r.get("template_id")
        key = (r["language"], r["task_family"], tid)
        if not tid or key not in TEMPLATE_CONTRACT:
            return False, f"Unknown or missing template_id: {tid} for family {r['task_family']}"
            
        contract_dir = TEMPLATE_CONTRACT[key]
        
        # 2. Verify that the generated question text actually matches this contracted direction
        sentences = [s.strip() for s in inp.replace("?", "।").split("।") if s.strip()]
        question_sentence = sentences[-1] if sentences else inp
        
        asks_greater = is_distinct_entity(adj_greater, question_sentence)
        asks_less = is_distinct_entity(adj_less, question_sentence)
        asks_general = any(w in question_sentence for w in ["तुलना", "अंतर", "फरक", "स्थितिको"])
        
        if asks_greater and not asks_less:
            actual_q_dir = "greater"
        elif asks_less and not asks_greater:
            actual_q_dir = "less"
        elif asks_general:
            # General-comparison templates are explicitly contracted in Phase 3.2
            # to request the greater entity. This is a template contract rule, not
            # an inferred mathematical relation.
            actual_q_dir = "greater"
        else:
            return False, f"Cannot unambiguously verify question direction from '{question_sentence}'"
            
        if actual_q_dir != contract_dir:
            return False, f"Question text '{question_sentence}' implies direction {actual_q_dir} but template contract expects {contract_dir}"
            
        # 3. Establish expected answer based on contract direction and Phase 3.1 ground truth
        greater_eid = ans_eid
        q_entities = r["question"]["entities"]
        lesser_eid = q_entities[0] if q_entities[1] == greater_eid else q_entities[1]
        
        if contract_dir == "greater":
            expected_eid = greater_eid
            expected_adj = adj_greater
            forbidden_adj = adj_less
        else:
            expected_eid = lesser_eid
            expected_adj = adj_less
            forbidden_adj = adj_greater
            
        expected_name = mapping[expected_eid]
        
        # Verify target strictly matches the expected semantics
        if not is_distinct_entity(expected_name, tgt):
            return False, f"Target missing expected entity '{expected_name}' for {contract_dir} question"
            
        for eid in r["question"]["entities"]:
            if eid != expected_eid:
                if is_distinct_entity(mapping[eid], tgt):
                    return False, f"Target contains unexpected opposite entity '{mapping[eid]}' for {contract_dir} question"
                    
        if expected_adj not in tgt:
            return False, f"Target missing expected adj '{expected_adj}' for {contract_dir} question"
            
        if forbidden_adj in tgt:
            return False, f"Target contains forbidden opposite adj '{forbidden_adj}' for {contract_dir} question"
            
        return True, ""
        
    return False, f"Unknown question type {q_type}"

def check_numerical_preservation(r):
    inp = r["input_text"]
    mapping = r["entity_mapping"]
    
    if r["representation"] != "numerical" or "values" not in r:
        return True, ""
        
    from phase3.template_system.lexicon import get_lexicon
    lexicon = get_lexicon(r["language"])
    attr_noun = lexicon["attributes"][r["attribute"]]["noun"]
    
    # In our templates, numericals are always formatted like: `{name} की {attr} {val}` (Hindi) or `{name}को {attr} {val}` (Nepali)
    for eid, val in r["values"].items():
        if eid not in mapping: continue
        name = mapping[eid]
        
        # First ensure the value exists securely
        if not re.search(rf"\b{val}\b", inp):
            return False, f"Numerical value {val} entirely missing"
            
        if name not in inp:
            return False, f"Entity {name} missing"
            
        # The invariant: each numerical entity occurrence must be associated with exactly its expected numeric value.
        # We check the actual phrase generated by the template.
        pattern = rf"{re.escape(name)}(?:\s+की|को)\s+{re.escape(attr_noun)}\s+{val}\b"
        if not re.search(pattern, inp):
            return False, f"Entity '{name}' is not properly associated with value '{val}' in the generated text"
            
    return True, ""

def scan_unexpected_entities(r, pool):
    inp = r["input_text"]
    mapping = r["entity_mapping"]
    
    # Extract all possible names in the input
    # For every name in the ENTIRE language pool, if it is in the input, it MUST be in the mapping!
    flat_pool = set()
    for s in pool.values(): flat_pool.update(s)
    
    expected_names = set(mapping.values())
    
    for name in flat_pool:
        if name not in expected_names:
            if is_distinct_entity(name, inp):
                return False, f"Unexpected entity {name} leaked into input"
                
    return True, ""

def run_audit():
    validate_template_contract_consistency()
    
    overall_status = "PASS"
    
    for lang in ["hi", "ne"]:
        filepath = f"phase3/outputs/logical_{lang}.jsonl"
        records = []
        
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        records.append(json.loads(line))
        except Exception as e:
            print(f"Failed to read {filepath}: {e}")
            return
            
        print(f"\n{lang.upper()}:")
        
        rec_count = len(records)
        print(f"  records: {rec_count}")
        if rec_count != 7000: overall_status = "FAIL"
            
        family_counts = defaultdict(int)
        template_counts = defaultdict(int)
        split_counts = defaultdict(int)
        
        for r in records:
            tf = r["task_family"]
            tid = r["template_id"]
            family_counts[tf] += 1
            template_counts[f"{tf}::{tid}"] += 1
            split_counts[r["split"]] += 1
            
        families_pass = all(c == 1000 for c in family_counts.values()) and len(family_counts) == 7
        templates_pass = all(c == 250 for c in template_counts.values()) and len(template_counts) == 28
        splits_pass = (split_counts["train"] == 5600 and split_counts["val"] == 700 and split_counts["test"] == 700)
        
        print(f"  families: {'PASS' if families_pass else 'FAIL'}")
        if not families_pass: overall_status = "FAIL"
        print(f"  templates: {'PASS' if templates_pass else 'FAIL'}")
        if not templates_pass: overall_status = "FAIL"
        print(f"  splits: {'PASS' if splits_pass else 'FAIL'}")
        if not splits_pass: overall_status = "FAIL"
        
        pool = ENTITY_POOLS[lang]
        
        entity_split_leakage_errs = []
        unexpected_entity_leakage_errs = []
        linguistic_validation_errs = []
        semantic_answer_errs = []
        numerical_preservation_errs = []
        
        for r in records:
            # 1. Entity Split Leakage
            mapping = r["entity_mapping"]
            split = r["split"]
            allowed = set(pool[split])
            for abstract_id, name in mapping.items():
                if name not in allowed:
                    entity_split_leakage_errs.append(f"id={r['id']}: {name} not in {split} pool")
                    
            # 2. Unexpected Entity Leakage
            ok, msg = scan_unexpected_entities(r, pool)
            if not ok: unexpected_entity_leakage_errs.append(f"id={r['id']}: {msg}")
            
            # 3. Linguistic Validator
            try:
                validate_linguistic_record(r)
            except LinguisticValidationError as e:
                linguistic_validation_errs.append(f"id={r['id']}: {e}")
                
            # 4. Semantic Answer Failures
            ok, msg = independent_semantic_audit(r)
            if not ok:
                semantic_answer_errs.append(f"id={r['id']}: {msg}")
                
            # 5. Numerical Preservation Failures
            ok, msg = check_numerical_preservation(r)
            if not ok:
                numerical_preservation_errs.append(f"id={r['id']}: {msg}")
                
        def print_errs(name, errs):
            print(f"  {name}: {len(errs)}")
            for e in errs[:3]:
                print(f"    - {e}")
            if len(errs) > 0:
                return "FAIL"
            return ""
            
        if print_errs("entity split leakage", entity_split_leakage_errs): overall_status = "FAIL"
        if print_errs("unexpected entity leakage", unexpected_entity_leakage_errs): overall_status = "FAIL"
        if print_errs("linguistic validation failures", linguistic_validation_errs): overall_status = "FAIL"
        if print_errs("semantic answer failures", semantic_answer_errs): overall_status = "FAIL"
        if print_errs("numerical preservation failures", numerical_preservation_errs): overall_status = "FAIL"
        
    print(f"\nOverall: {overall_status}")

if __name__ == "__main__":
    run_audit()
