import json
import os
from phase3.scripts.evaluation.logical_resolver import resolve_expected_answer, get_template

import string

def get_target_answer_entity(record: dict) -> str:
    q_type = record["question"]["type"]
    gt = record["ground_truth"]
    
    if q_type == "compare" and gt.get("relation") == "equal":
        return None

    target_text = record["target_text"]
    mapping = record.get("entity_mapping", {})
    found_entities = []
    
    for eid, name in mapping.items():
        if name in target_text:
            found_entities.append(eid)
                
    if len(found_entities) == 1:
        return found_entities[0]
    elif len(found_entities) > 1:
        # Sort by length descending, and check if one is a substring of another? 
        # Actually our test showed 0 ambiguous cases for Nepali. Let's just return AMBIGUOUS.
        return "AMBIGUOUS"
    return None

def main():
    hi_path = "phase3/outputs/reasoning_hi.jsonl"
    ne_path = "phase3/outputs/reasoning_ne.jsonl"
    
    audit_results = {
        "total_records": 0,
        "independent_eq_gt": 0,
        "independent_neq_gt": 0,
        "independent_eq_target": 0,
        "independent_neq_target": 0,
        "gt_eq_target": 0,
        "gt_neq_target": 0,
        "equality_records": 0,
        "by_language": {"hi": {"total": 0, "indep_neq_gt": 0}, "ne": {"total": 0, "indep_neq_gt": 0}},
        "by_task_family": {},
        "by_difficulty": {},
        "by_representation": {},
        "by_question_direction": {}
    }
    
    discrepancy_examples = []
    
    for path in [hi_path, ne_path]:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                if record.get("split") != "test":
                    continue
                
                lang = record["language"]
                tf = record["task_family"]
                diff = record["difficulty"]
                rep = record["representation"]
                
                template = get_template(lang, tf, record["template_id"])
                q_dir = template.question_direction
                
                indep_res = resolve_expected_answer(record)
                indep_ans = indep_res["expected_answer_entity"]
                
                gt_ans = record["ground_truth"].get("answer_entity")
                target_ans = get_target_answer_entity(record)
                
                audit_results["total_records"] += 1
                audit_results["by_language"][lang]["total"] += 1
                
                if tf not in audit_results["by_task_family"]:
                    audit_results["by_task_family"][tf] = {"total": 0, "indep_neq_gt": 0}
                audit_results["by_task_family"][tf]["total"] += 1
                
                if diff not in audit_results["by_difficulty"]:
                    audit_results["by_difficulty"][diff] = {"total": 0, "indep_neq_gt": 0}
                audit_results["by_difficulty"][diff]["total"] += 1
                
                if rep not in audit_results["by_representation"]:
                    audit_results["by_representation"][rep] = {"total": 0, "indep_neq_gt": 0}
                audit_results["by_representation"][rep]["total"] += 1
                
                if q_dir not in audit_results["by_question_direction"]:
                    audit_results["by_question_direction"][q_dir] = {"total": 0, "indep_neq_gt": 0}
                audit_results["by_question_direction"][q_dir]["total"] += 1
                
                if record["ground_truth"].get("relation") == "equal":
                    audit_results["equality_records"] += 1
                    # equality may legitimately have no answer_entity, we skip the mismatch logic
                    continue
                
                if indep_ans == gt_ans:
                    audit_results["independent_eq_gt"] += 1
                else:
                    audit_results["independent_neq_gt"] += 1
                    audit_results["by_language"][lang]["indep_neq_gt"] += 1
                    audit_results["by_task_family"][tf]["indep_neq_gt"] += 1
                    audit_results["by_difficulty"][diff]["indep_neq_gt"] += 1
                    audit_results["by_representation"][rep]["indep_neq_gt"] += 1
                    audit_results["by_question_direction"][q_dir]["indep_neq_gt"] += 1
                    
                    if len(discrepancy_examples) < 10:
                        discrepancy_examples.append({
                            "id": record["id"],
                            "language": lang,
                            "task_family": tf,
                            "difficulty": diff,
                            "representation": rep,
                            "facts": record.get("facts", []),
                            "question": record["question"],
                            "ground_truth": record["ground_truth"],
                            "entity_mapping": record.get("entity_mapping", {}),
                            "target_text": record["target_text"],
                            "independently_derived_answer": indep_ans,
                            "template_id": record["template_id"],
                            "target_answer_derived": target_ans
                        })
                
                if indep_ans == target_ans:
                    audit_results["independent_eq_target"] += 1
                else:
                    audit_results["independent_neq_target"] += 1
                    
                if gt_ans == target_ans:
                    audit_results["gt_eq_target"] += 1
                else:
                    audit_results["gt_neq_target"] += 1
                    
    os.makedirs("phase3/outputs/evaluation", exist_ok=True)
    with open("phase3/outputs/evaluation/ground_truth_audit.json", "w", encoding="utf-8") as f:
        json.dump({"summary": audit_results, "representative_discrepancies": discrepancy_examples}, f, indent=4, ensure_ascii=False)
        
    with open("phase3/outputs/evaluation/ground_truth_audit.txt", "w", encoding="utf-8") as f:
        f.write("=== GROUND TRUTH AUDIT SUMMARY ===\n")
        f.write(f"Total Test Records Evaluated: {audit_results['total_records']}\n")
        f.write(f"Equality Records (excluded from mismatch counts): {audit_results['equality_records']}\n\n")
        
        f.write(f"Independent Answer == Ground Truth Answer Entity: {audit_results['independent_eq_gt']}\n")
        f.write(f"Independent Answer != Ground Truth Answer Entity: {audit_results['independent_neq_gt']}\n\n")
        
        f.write(f"Independent Answer == Target Text Entity: {audit_results['independent_eq_target']}\n")
        f.write(f"Independent Answer != Target Text Entity: {audit_results['independent_neq_target']}\n\n")
        
        f.write(f"Ground Truth Answer Entity == Target Text Entity: {audit_results['gt_eq_target']}\n")
        f.write(f"Ground Truth Answer Entity != Target Text Entity: {audit_results['gt_neq_target']}\n\n")
        
        f.write("=== BREAKDOWNS (Independent != GT) ===\n")
        for key in ["by_language", "by_task_family", "by_difficulty", "by_representation", "by_question_direction"]:
            f.write(f"\n{key}:\n")
            for k, v in audit_results[key].items():
                f.write(f"  {k}: {v['indep_neq_gt']} / {v['total']}\n")
                
if __name__ == "__main__":
    main()
