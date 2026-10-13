import json
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from phase3.template_system.validator import validate_linguistic_record, is_distinct_entity
from phase3.scripts.dataset.audit_dataset import independent_semantic_audit, check_numerical_preservation

def validate_entity_preservation(original, generated):
    # A. original.entity_mapping == generated.entity_mapping
    orig_mapping = original.get("entity_mapping", {})
    gen_mapping = generated.get("entity_mapping", {})
    if json.dumps(orig_mapping, sort_keys=True) != json.dumps(gen_mapping, sort_keys=True):
        return False, "Entity mapping changed."

    orig_input = original.get("input_text", "")
    orig_target = original.get("target_text", "")
    gen_input = generated.get("input_text", "")
    gen_target = generated.get("target_text", "")
    
    # B & C. Every entity name that occurs in the original input and is required must be preserved
    for eid, name in orig_mapping.items():
        if name in orig_input and name not in gen_input:
            return False, f"Entity '{name}' removed from generated input."
        if name in orig_target and name not in gen_target:
            return False, f"Entity '{name}' removed from generated target."

    # D. No entity name outside the original entity_mapping is introduced
    from phase3.template_system.entities import ENTITY_POOLS
    
    lang = original.get("language")
    pool = ENTITY_POOLS[lang]
    
    all_names = set()
    for cat in pool.values():
        all_names.update(cat)
        
    allowed_names = set(orig_mapping.values())
    
    for name in all_names:
        if name not in allowed_names:
            if is_distinct_entity(name, gen_input) or is_distinct_entity(name, gen_target):
                return False, f"Unexpected entity '{name}' introduced in generated text."

    return True, ""

def validate_metadata_integrity(original, generated):
    fields = [
        "id", "task_family", "difficulty", "attribute", "representation",
        "entities", "facts", "question", "ground_truth", "reasoning_path",
        "relation_pattern_id", "seed", "split", "values", "template_id", "language", "entity_mapping"
    ]
    for f in fields:
        orig_val = original.get(f)
        gen_val = generated.get(f)
        if json.dumps(orig_val, sort_keys=True) != json.dumps(gen_val, sort_keys=True):
            return False, f"Metadata field '{f}' changed."
            
    # Check that no OTHER logical fields were added or removed
    for k in original.keys():
        if k not in ["input_text", "target_text", "generation_metadata"] and k not in generated:
            return False, f"Metadata field '{k}' missing."
            
    return True, ""

def validate_pilot_ids(orig_records_list, gen_records_list):
    orig_ids = [r["id"] for r in orig_records_list]
    gen_ids = [r["id"] for r in gen_records_list]
    
    if len(set(gen_ids)) != len(gen_ids):
        return False, "Duplicate IDs in generated output."
        
    if set(orig_ids) != set(gen_ids):
        return False, "ID mismatch between original and generated pilot."
        
    if len(orig_ids) != len(gen_ids):
        return False, "Count mismatch."
        
    return True, ""

def full_acceptance_validation(original, generated):
    ok, msg = validate_entity_preservation(original, generated)
    if not ok: return False, msg
    
    ok, msg = validate_metadata_integrity(original, generated)
    if not ok: return False, msg
    
    try:
        validate_linguistic_record(generated)
    except Exception as e:
        return False, f"Linguistic Validation Failed: {e}"
        
    ok, msg = independent_semantic_audit(generated)
    if not ok: return False, f"Semantic Audit Failed: {msg}"
    
    ok, msg = check_numerical_preservation(generated)
    if not ok: return False, f"Numerical Audit Failed: {msg}"
    
    return True, ""

def run_pilot_validation(orig_file, gen_file):
    print(f"\n--- Validating {gen_file} against {orig_file} ---")
    
    with open(orig_file, 'r', encoding='utf-8') as f:
        orig_records = [json.loads(line) for line in f if line.strip()]
        
    with open(gen_file, 'r', encoding='utf-8') as f:
        gen_records = [json.loads(line) for line in f if line.strip()]
        
    ok, msg = validate_pilot_ids(orig_records, gen_records)
    if not ok:
        print(f"Pilot ID integrity: FAIL ({msg})")
        return False
        
    orig_dict = {r["id"]: r for r in orig_records}
    gen_dict = {r["id"]: r for r in gen_records}
    
    success = 0
    failures = []
    
    for rid, gen_rec in gen_dict.items():
        orig_rec = orig_dict[rid]
        ok, msg = full_acceptance_validation(orig_rec, gen_rec)
        if ok:
            success += 1
        else:
            failures.append(f"ID {rid}: {msg}")
            
    print(f"Validated: {success} passed, {len(failures)} failed.")
    for f in failures:
        print(f" - {f}")
        
    if len(failures) == 0:
        print("Pilot metadata integrity: PASS")
        print("Pilot ID integrity: PASS")
        return True
    else:
        print("Pilot metadata integrity: FAIL")
        return False

if __name__ == "__main__":
    if len(sys.argv) == 3:
        run_pilot_validation(sys.argv[1], sys.argv[2])
    else:
        print("Usage: python validation_utils.py <original_jsonl> <generated_jsonl>")
