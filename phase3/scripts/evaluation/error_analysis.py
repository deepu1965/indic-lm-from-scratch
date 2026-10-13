import json
import os
from collections import Counter
import random

def determine_error_category(record):
    if record["reasoning_correct"]:
        return None
        
    expected = record["independently_resolved_answer"]
    pred = record["predicted_answer"]
    
    # 1. Equality-specific errors
    if expected is None:
        gen_text = record.get("generated_text", "")
        # Remove spaces to do simple substring matching without BPE space issues
        gen_nospace = gen_text.replace(" ", "").replace("\u2581", "").strip("।。. \n\t")
        
        if not gen_nospace:
            return "no_recognizable_answer"
            
        eq_keywords = ["बराबर", "समान", "equal", "same"]
        has_eq = any(w in gen_nospace.lower() for w in eq_keywords)
        
        if not has_eq:
            return "equality_relation_missing"
        else:
            # We have the equality keyword, but reasoning_correct is False.
            # Since the evaluation record does not contain the entity_mapping,
            # we cannot reliably extract entities to determine if they are wrong or ambiguous.
            # Following instructions, we choose the conservative category instead of inventing evidence.
            return "no_recognizable_answer"
            
    # 2. Relational errors
    if pred == "AMBIGUOUS":
        return "ambiguous_multiple_entities"
        
    if pred is None:
        return "no_recognizable_answer"
        
    if pred != expected:
        return "wrong_entity"
        
    return "unknown"

def run_consistency_checks(data, filepath):
    if len(data) != 700:
        raise ValueError(f"Consistency Check Failed in {filepath}: Expected 700 records, got {len(data)}")
        
    seen_ids = set()
    req_fields = ["id", "task_family", "difficulty", "reasoning_correct", "exact_match", "independently_resolved_answer", "predicted_answer"]
    
    fam_counts = Counter()
    diff_counts = Counter()
    
    for row in data:
        rid = row.get("id")
        if rid in seen_ids:
            raise ValueError(f"Consistency Check Failed in {filepath}: Duplicate ID {rid}")
        seen_ids.add(rid)
        
        for rf in req_fields:
            if rf not in row:
                raise ValueError(f"Consistency Check Failed in {filepath}: Missing required field '{rf}' in record {rid}")
                
        if not isinstance(row["reasoning_correct"], bool):
            raise ValueError(f"Consistency Check Failed in {filepath}: reasoning_correct is not a boolean in record {rid}")
            
        if not isinstance(row["exact_match"], bool):
            raise ValueError(f"Consistency Check Failed in {filepath}: exact_match is not a boolean in record {rid}")
            
        fam_counts[row["task_family"]] += 1
        diff_counts[row["difficulty"]] += 1
        
    expected_fams = ["direct_comparison", "largest_smallest", "general_comparison", "two_hop_transitivity", "three_hop_transitivity", "reversed_relation", "mixed_equality"]
    for fam in expected_fams:
        if fam_counts[fam] != 100:
            raise ValueError(f"Consistency Check Failed in {filepath}: task family '{fam}' has {fam_counts[fam]} records, expected 100")
            
    if len(fam_counts) != 7:
        raise ValueError(f"Consistency Check Failed in {filepath}: found {len(fam_counts)} task families, expected 7")
        
    expected_diffs = {"easy": 210, "medium": 280, "hard": 210}
    for diff, exp_count in expected_diffs.items():
        if diff_counts[diff] != exp_count:
            raise ValueError(f"Consistency Check Failed in {filepath}: difficulty '{diff}' has {diff_counts[diff]} records, expected {exp_count}")
            
    if len(diff_counts) != 3:
        raise ValueError(f"Consistency Check Failed in {filepath}: found {len(diff_counts)} difficulties, expected 3")

def analyze_file(filepath: str, name: str):
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    run_consistency_checks(data, filepath)
        
    overall = {
        "total": len(data),
        "reasoning_correct": 0,
        "reasoning_incorrect": 0,
        "error_categories": Counter()
    }
    
    per_family = {}
    per_difficulty = {}
    
    exact_match_vs_reasoning = {
        "rc_true_em_false": [],
        "rc_false_em_false": [],
        "rc_true_em_true": []
    }
    
    examples = {
        "wrong_entity": [],
        "ambiguous_multiple_entities": [],
        "no_recognizable_answer": [],
        "equality_wrong_entities": [],
        "equality_relation_missing": []
    }
    success_examples = []
    
    for row in data:
        fam = row["task_family"]
        diff = row["difficulty"]
        
        if fam not in per_family:
            per_family[fam] = {"total": 0, "reasoning_correct": 0, "reasoning_incorrect": 0, "error_categories": Counter()}
        if diff not in per_difficulty:
            per_difficulty[diff] = {"total": 0, "reasoning_correct": 0, "reasoning_incorrect": 0, "error_categories": Counter()}
            
        per_family[fam]["total"] += 1
        per_difficulty[diff]["total"] += 1
        
        if row["reasoning_correct"] and row["exact_match"]:
            exact_match_vs_reasoning["rc_true_em_true"].append(row)
            success_examples.append(row)
        elif row["reasoning_correct"] and not row["exact_match"]:
            exact_match_vs_reasoning["rc_true_em_false"].append(row)
            success_examples.append(row)
        elif not row["reasoning_correct"] and not row["exact_match"]:
            exact_match_vs_reasoning["rc_false_em_false"].append(row)
            
        if row["reasoning_correct"]:
            overall["reasoning_correct"] += 1
            per_family[fam]["reasoning_correct"] += 1
            per_difficulty[diff]["reasoning_correct"] += 1
        else:
            overall["reasoning_incorrect"] += 1
            per_family[fam]["reasoning_incorrect"] += 1
            per_difficulty[diff]["reasoning_incorrect"] += 1
            
            cat = determine_error_category(row)
            overall["error_categories"][cat] += 1
            per_family[fam]["error_categories"][cat] += 1
            per_difficulty[diff]["error_categories"][cat] += 1
            
            if cat in examples:
                examples[cat].append(row)
                
    # Consistency check: error category counts sum to reasoning_incorrect
    sum_errors = sum(overall["error_categories"].values())
    if sum_errors != overall["reasoning_incorrect"]:
        raise ValueError(f"Consistency Check Failed in {filepath}: Error categories sum to {sum_errors}, expected {overall['reasoning_incorrect']}")
                
    return {
        "overall": overall,
        "per_family": per_family,
        "per_difficulty": per_difficulty,
        "exact_match_vs_reasoning": {
            "rc_true_em_false": len(exact_match_vs_reasoning["rc_true_em_false"]),
            "rc_false_em_false": len(exact_match_vs_reasoning["rc_false_em_false"]),
            "rc_true_em_true": len(exact_match_vs_reasoning["rc_true_em_true"])
        },
        "rc_true_em_false_examples": exact_match_vs_reasoning["rc_true_em_false"][:3],
        "examples": examples,
        "success_examples": success_examples
    }

def format_percentages(counts, total):
    return {k: f"{(v/total):.2%}" if total > 0 else "0.00%" for k, v in counts.items()}

def extract_representative_examples(examples_dict, k=2):
    random.seed(42)
    selected = {}
    for cat, ex_list in examples_dict.items():
        if ex_list:
            selected[cat] = random.sample(ex_list, min(k, len(ex_list)))
    return selected

def main():
    files = {
        "hi_pre": "phase3/outputs/evaluation/single_model_hi_pre.json",
        "hi_sft": "phase3/outputs/evaluation/single_model_hi_sft.json",
        "ne_pre": "phase3/outputs/evaluation/single_model_ne_pre.json",
        "ne_sft": "phase3/outputs/evaluation/single_model_ne_sft.json"
    }
    
    results = {}
    for name, path in files.items():
        results[name] = analyze_file(path, name)
        
    txt_lines = []
    def print_and_save(s):
        print(s)
        txt_lines.append(s)
        
    print_and_save("============================================================")
    print_and_save("OVERALL ERROR BREAKDOWN")
    print_and_save("============================================================")
    for name in ["hi_pre", "hi_sft", "ne_pre", "ne_sft"]:
        ov = results[name]["overall"]
        print_and_save(f"\nCondition: {name}")
        print_and_save(f"  Total: {ov['total']}")
        print_and_save(f"  Reasoning Correct: {ov['reasoning_correct']} ({ov['reasoning_correct']/ov['total']:.2%})")
        print_and_save(f"  Reasoning Incorrect: {ov['reasoning_incorrect']} ({ov['reasoning_incorrect']/ov['total']:.2%})")
        print_and_save("  Error Categories:")
        for cat, count in ov["error_categories"].items():
            pct = count / ov['reasoning_incorrect'] if ov['reasoning_incorrect'] > 0 else 0
            print_and_save(f"    - {cat}: {count} ({pct:.2%})")
            
    print_and_save("\n============================================================")
    print_and_save("EXACT MATCH VS REASONING CORRECTNESS")
    print_and_save("============================================================")
    for name in ["hi_sft", "ne_sft"]:
        em = results[name]["exact_match_vs_reasoning"]
        print_and_save(f"\nCondition: {name}")
        print_and_save(f"  RC=True,  EM=True : {em['rc_true_em_true']}")
        print_and_save(f"  RC=True,  EM=False: {em['rc_true_em_false']} (Correct reasoning, but non-exact wording)")
        print_and_save(f"  RC=False, EM=False: {em['rc_false_em_false']}")

    print_and_save("\n============================================================")
    print_and_save("PRETRAINED VS SFT COMPARISON")
    print_and_save("============================================================")
    for lang in ["hi", "ne"]:
        pre = results[f"{lang}_pre"]["overall"]
        sft = results[f"{lang}_sft"]["overall"]
        print_and_save(f"\n{lang.upper()} ERROR SHIFT (Pretrained -> SFT):")
        
        pre_cats = pre["error_categories"]
        sft_cats = sft["error_categories"]
        all_cats = set(pre_cats.keys()).union(set(sft_cats.keys()))
        
        for cat in sorted(all_cats):
            pre_count = pre_cats.get(cat, 0)
            pre_pct = pre_count / pre["total"]
            sft_count = sft_cats.get(cat, 0)
            sft_pct = sft_count / sft["total"]
            print_and_save(f"  {cat:<30}: {pre_pct:>6.2%} -> {sft_pct:>6.2%}")

    print_and_save("\n============================================================")
    print_and_save("ERROR BREAKDOWN BY TASK FAMILY (SFT ONLY)")
    print_and_save("============================================================")
    for name in ["hi_sft", "ne_sft"]:
        print_and_save(f"\nCondition: {name}")
        for fam, st in results[name]["per_family"].items():
            print_and_save(f"  {fam}:")
            print_and_save(f"    Reas Incorrect: {st['reasoning_incorrect']}/{st['total']}")
            for cat, count in st["error_categories"].items():
                print_and_save(f"      - {cat}: {count}")

    print_and_save("\n============================================================")
    print_and_save("ERROR BREAKDOWN BY DIFFICULTY (SFT ONLY)")
    print_and_save("============================================================")
    for name in ["hi_sft", "ne_sft"]:
        print_and_save(f"\nCondition: {name}")
        for diff, st in results[name]["per_difficulty"].items():
            print_and_save(f"  {diff}:")
            print_and_save(f"    Reas Incorrect: {st['reasoning_incorrect']}/{st['total']}")
            for cat, count in st["error_categories"].items():
                print_and_save(f"      - {cat}: {count}")

    # Build structured JSON
    json_output = {}
    for name in results:
        json_output[name] = {
            "overall": results[name]["overall"],
            "per_family": results[name]["per_family"],
            "per_difficulty": results[name]["per_difficulty"],
            "exact_match_vs_reasoning": results[name]["exact_match_vs_reasoning"],
            "representative_errors": extract_representative_examples(results[name]["examples"]),
            "success_examples": random.sample(results[name]["success_examples"], min(3, len(results[name]["success_examples"]))) if results[name]["success_examples"] else [],
            "rc_true_em_false_examples": results[name]["rc_true_em_false_examples"]
        }
        
    os.makedirs("phase3/outputs/evaluation", exist_ok=True)
    with open("phase3/outputs/evaluation/error_analysis.json", "w", encoding="utf-8") as f:
        json.dump(json_output, f, indent=2, ensure_ascii=False)
        
    with open("phase3/outputs/evaluation/error_analysis.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(txt_lines) + "\n")
        
    print("\nSaved to phase3/outputs/evaluation/error_analysis.json and .txt")
    print("\nALL CONSISTENCY CHECKS PASSED.")

if __name__ == "__main__":
    main()
