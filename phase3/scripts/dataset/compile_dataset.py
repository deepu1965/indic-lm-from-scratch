import json
import os
import sys
import glob
from collections import Counter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from phase3.scripts.dataset.validation_utils import full_acceptance_validation

def compile_dataset(lang):
    baseline_file = f"phase3/outputs/logical_{lang}.jsonl"
    staging_dir = f"scratch/full_generation/{lang}/staging"
    output_file = f"phase3/outputs/reasoning_{lang}.jsonl"
    
    # 1. Load Original Phase 3.1 Baseline
    with open(baseline_file, 'r', encoding='utf-8') as f:
        baseline_records = [json.loads(line) for line in f if line.strip()]
        
    baseline_dict = {r["id"]: r for r in baseline_records}
    expected_ids = set(baseline_dict.keys())
    
    if len(expected_ids) != 7000:
        print(f"FAIL: Baseline contains {len(expected_ids)} records, expected 7000.")
        sys.exit(1)
        
    # 2. Load all Validated Staging Records
    staging_files = glob.glob(f"{staging_dir}/*.json")
    generated_records = []
    
    for sf in staging_files:
        with open(sf, 'r', encoding='utf-8') as f:
            generated_records.extend(json.load(f))
            
    # 3. Verify Exact ID Coverage
    generated_ids = [r["id"] for r in generated_records]
    generated_ids_set = set(generated_ids)
    
    if len(generated_ids) != len(generated_ids_set):
        print("FAIL: Duplicates found in generated staging records.")
        sys.exit(1)
        
    if generated_ids_set != expected_ids:
        missing = expected_ids - generated_ids_set
        extra = generated_ids_set - expected_ids
        print(f"FAIL: ID coverage mismatch. Missing {len(missing)}, Extra {len(extra)}")
        sys.exit(1)
        
    if len(generated_ids_set) != 7000:
        print(f"FAIL: Total generated records is {len(generated_ids_set)}, expected 7000.")
        sys.exit(1)
        
    # 4. Independent Final Validation
    print("Running independent final acceptance validation on all generated records...")
    for gen in generated_records:
        rid = gen["id"]
        orig = baseline_dict[rid]
        ok, msg = full_acceptance_validation(orig, gen)
        if not ok:
            print(f"FAIL: Record {rid} failed final compiler validation: {msg}")
            sys.exit(1)
            
    # Sort for deterministic output
    generated_records.sort(key=lambda x: x["id"])
    
    # Verify exact distributions
    families = Counter(r["task_family"] for r in generated_records)
    templates = Counter(r["template_id"] for r in generated_records)
    splits = Counter(r["split"] for r in generated_records)
    attributes = Counter(r["attribute"] for r in generated_records)
    representations = Counter(r["representation"] for r in generated_records)
    difficulties = Counter(r.get("difficulty", "unknown") for r in generated_records)
    
    expected_families = {
        "direct_comparison": 1000, "largest_smallest": 1000, "general_comparison": 1000,
        "two_hop_transitivity": 1000, "three_hop_transitivity": 1000, "reversed_relation": 1000, "mixed_equality": 1000
    }
    
    expected_splits = {"train": 5600, "val": 700, "test": 700}
    expected_attrs = {"height": 1750, "age": 1750, "price": 1750, "quantity": 1750}
    expected_reps = {"numerical": 1750, "relational": 5250}
    expected_diffs = {"easy": 2100, "medium": 2800, "hard": 2100}
    
    for k, v in expected_families.items():
        if families[k] != v: print(f"FAIL: Family {k} count {families[k]} != {v}"); sys.exit(1)
    
    for k, v in expected_splits.items():
        if splits[k] != v: print(f"FAIL: Split {k} count {splits[k]} != {v}"); sys.exit(1)
        
    for k, v in expected_attrs.items():
        if attributes[k] != v: print(f"FAIL: Attribute {k} count {attributes[k]} != {v}"); sys.exit(1)
        
    for k, v in expected_reps.items():
        if representations[k] != v: print(f"FAIL: Representation {k} count {representations[k]} != {v}"); sys.exit(1)
        
    for k, v in expected_diffs.items():
        if difficulties[k] != v: print(f"FAIL: Difficulty {k} count {difficulties[k]} != {v}"); sys.exit(1)
        
    # Derive expected per-template counts from the baseline (source of truth)
    baseline_template_counts = Counter(r["template_id"] for r in baseline_records)
    for t, expected_count in baseline_template_counts.items():
        actual_count = templates[t]
        if actual_count != expected_count:
            print(f"FAIL: Template {t} has {actual_count} records, expected {expected_count}.")
            sys.exit(1)
        
    if len(templates) != len(baseline_template_counts):
        extra = set(templates.keys()) - set(baseline_template_counts.keys())
        print(f"FAIL: Unexpected templates in generated data: {extra}"); sys.exit(1)
        
    # Write using clean 'w' mode
    with open(output_file, 'w', encoding='utf-8') as f:
        for rec in generated_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            
    print(f"SUCCESS: Compiled exactly 7000 {lang.upper()} records into {output_file}.")
    return True

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python compile_dataset.py <lang>")
        sys.exit(1)
    compile_dataset(sys.argv[1])
