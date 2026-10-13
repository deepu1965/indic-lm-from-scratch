import json
import os
import sys

def aggregate_file(filepath: str, name: str) -> dict:
    if not os.path.exists(filepath):
        print(f"Error: {filepath} not found.")
        sys.exit(1)
        
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    # Consistency checks
    if len(data) != 700:
        raise ValueError(f"{name}: Expected 700 records, got {len(data)}")
        
    ids = set()
    for row in data:
        if row["id"] in ids:
            raise ValueError(f"{name}: Duplicate ID found: {row['id']}")
        ids.add(row["id"])
        
        required_fields = ["id", "task_family", "difficulty", "reasoning_correct", "exact_match"]
        for rf in required_fields:
            if rf not in row:
                raise ValueError(f"{name}: Missing required field '{rf}' in record {row['id']}")
                
        if not isinstance(row["reasoning_correct"], bool):
            raise ValueError(f"{name}: reasoning_correct is not a boolean in record {row['id']}")
        if not isinstance(row["exact_match"], bool):
            raise ValueError(f"{name}: exact_match is not a boolean in record {row['id']}")
            
    # Aggregations
    overall = {
        "total": len(data),
        "reasoning_correct": 0,
        "exact_match": 0
    }
    
    per_family = {}
    per_difficulty = {}
    
    for row in data:
        rc = 1 if row["reasoning_correct"] else 0
        em = 1 if row["exact_match"] else 0
        
        overall["reasoning_correct"] += rc
        overall["exact_match"] += em
        
        fam = row["task_family"]
        if fam not in per_family:
            per_family[fam] = {"total": 0, "reasoning_correct": 0, "exact_match": 0}
        per_family[fam]["total"] += 1
        per_family[fam]["reasoning_correct"] += rc
        per_family[fam]["exact_match"] += em
        
        diff = row["difficulty"]
        if diff not in per_difficulty:
            per_difficulty[diff] = {"total": 0, "reasoning_correct": 0, "exact_match": 0}
        per_difficulty[diff]["total"] += 1
        per_difficulty[diff]["reasoning_correct"] += rc
        per_difficulty[diff]["exact_match"] += em
        
    # Check sums
    if sum(v["total"] for v in per_family.values()) != 700:
        raise ValueError(f"{name}: Task family counts do not sum to 700")
    if sum(v["total"] for v in per_difficulty.values()) != 700:
        raise ValueError(f"{name}: Difficulty counts do not sum to 700")
        
    # Accuracies
    overall["reasoning_accuracy"] = overall["reasoning_correct"] / overall["total"]
    overall["exact_match_accuracy"] = overall["exact_match"] / overall["total"]
    
    for k, v in per_family.items():
        v["reasoning_accuracy"] = v["reasoning_correct"] / v["total"]
        v["exact_match_accuracy"] = v["exact_match"] / v["total"]
        
    for k, v in per_difficulty.items():
        v["reasoning_accuracy"] = v["reasoning_correct"] / v["total"]
        v["exact_match_accuracy"] = v["exact_match"] / v["total"]
        
    return {
        "overall": overall,
        "per_family": per_family,
        "per_difficulty": per_difficulty
    }

def main():
    files = {
        "hi_pre": "phase3/outputs/evaluation/single_model_hi_pre.json",
        "hi_sft": "phase3/outputs/evaluation/single_model_hi_sft.json",
        "ne_pre": "phase3/outputs/evaluation/single_model_ne_pre.json",
        "ne_sft": "phase3/outputs/evaluation/single_model_ne_sft.json"
    }
    
    results = {}
    for name, path in files.items():
        results[name] = aggregate_file(path, name)
        
    # Changes
    changes = {}
    for lang in ["hi", "ne"]:
        pre = results[f"{lang}_pre"]["overall"]
        sft = results[f"{lang}_sft"]["overall"]
        changes[lang] = {
            "reasoning_accuracy_change": sft["reasoning_accuracy"] - pre["reasoning_accuracy"],
            "exact_match_accuracy_change": sft["exact_match_accuracy"] - pre["exact_match_accuracy"]
        }
        
    final_output = {
        "metrics": results,
        "changes": changes
    }
    
    os.makedirs("phase3/outputs/evaluation", exist_ok=True)
    with open("phase3/outputs/evaluation/reasoning_summary.json", "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2)
        
    txt_lines = []
    
    def print_and_save(s):
        print(s)
        txt_lines.append(s)
        
    print_and_save("============================================================")
    print_and_save("OVERALL METRICS")
    print_and_save("============================================================")
    print_and_save(f"{'Condition':<15} {'Total':<8} {'Reas Correct':<15} {'Reas Acc':<15} {'Exact Match':<15} {'Exact Acc'}")
    for name in ["hi_pre", "hi_sft", "ne_pre", "ne_sft"]:
        ov = results[name]["overall"]
        print_and_save(f"{name:<15} {ov['total']:<8} {ov['reasoning_correct']:<15} {ov['reasoning_accuracy']:.2%} {ov['exact_match']:<15} {ov['exact_match_accuracy']:.2%}")
        
    print_and_save("\n============================================================")
    print_and_save("ABSOLUTE CHANGES (Pretrained -> SFT)")
    print_and_save("============================================================")
    for lang in ["hi", "ne"]:
        chg = changes[lang]
        print_and_save(f"{lang.upper()}:")
        print_and_save(f"  Reasoning Accuracy Change: {chg['reasoning_accuracy_change']:+.2%}")
        print_and_save(f"  Exact Match Acc Change:    {chg['exact_match_accuracy_change']:+.2%}")
        
    print_and_save("\n============================================================")
    print_and_save("PER TASK FAMILY")
    print_and_save("============================================================")
    for name in ["hi_pre", "hi_sft", "ne_pre", "ne_sft"]:
        print_and_save(f"\nCondition: {name}")
        for fam, st in results[name]["per_family"].items():
            print_and_save(f"  {fam:<25} Total:{st['total']:<4} Reas:{st['reasoning_accuracy']:>7.2%} Exact:{st['exact_match_accuracy']:>7.2%}")
            
    print_and_save("\n============================================================")
    print_and_save("PER DIFFICULTY")
    print_and_save("============================================================")
    for name in ["hi_pre", "hi_sft", "ne_pre", "ne_sft"]:
        print_and_save(f"\nCondition: {name}")
        for diff, st in results[name]["per_difficulty"].items():
            print_and_save(f"  {diff:<10} Total:{st['total']:<4} Reas:{st['reasoning_accuracy']:>7.2%} Exact:{st['exact_match_accuracy']:>7.2%}")

    with open("phase3/outputs/evaluation/reasoning_summary.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(txt_lines) + "\n")

if __name__ == "__main__":
    main()
