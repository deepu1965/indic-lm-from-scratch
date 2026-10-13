import re
from typing import Dict, Any
from .selector import select_template

class LinguisticValidationError(Exception):
    pass

def is_distinct_entity(name: str, text: str) -> bool:
    idx = 0
    while True:
        idx = text.find(name, idx)
        if idx == -1:
            return False
            
        # Check previous character
        if idx > 0:
            prev_char = text[idx - 1]
            if prev_char not in " \t\n.,!?।;:()-'\"":
                idx += 1
                continue
                
        end_idx = idx + len(name)
        if end_idx == len(text):
            return True
        next_char = text[end_idx]
        word_end = end_idx
        while word_end < len(text) and text[word_end] not in " \t\n.,!?।;:()-'\"":
            word_end += 1
        suffix = text[end_idx:word_end]
        if not suffix:
            return True
        allowed_suffixes = {"को", "लाई", "ले", "मा", "बाट", "भन्दा", "मध्ये"}
        if suffix in allowed_suffixes:
            return True
        idx += 1

def validate_linguistic_record(record: Dict[str, Any]):
    inp = record.get("input_text", "")
    tgt = record.get("target_text", "")
    
    if not inp.strip() or not tgt.strip():
        raise LinguisticValidationError("Empty input or target text.")
        
    gt = record["ground_truth"]
    mapping = record["entity_mapping"]
    q_type = record["question"]["type"]
    ans_eid = gt.get("answer_entity")
    rel = gt.get("relation")
    
    # 1. Check Missing Required Entity Names in Input
    for eid, name in mapping.items():
        if not is_distinct_entity(name, inp):
            raise LinguisticValidationError(f"Entity '{name}' missing from input text '{inp}'.")
            
    # 2. Check Numerical Value Preservation (Boundary safe)
    if record["representation"] == "numerical" and "values" in record and record["values"]:
        involved_eids = set()
        for f in record["facts"]:
            involved_eids.add(f["subject"])
            involved_eids.add(f["object"])
            
        for eid in involved_eids:
            val = record["values"][eid]
            # Use word boundary check
            if not re.search(rf"\b{val}\b", inp):
                raise LinguisticValidationError(f"Numerical value {val} for entity {mapping[eid]} missing from input text.")
                
    # Load lexicon to verify relation semantics in the target
    from .lexicon import get_lexicon
    lex = get_lexicon(record["language"])["attributes"][record["attribute"]]
    adj_greater = lex["adj_greater"]
    adj_less = lex["adj_less"]
    
    # 3. Check Equality Semantics
    if rel == "equal" or q_type == "equal":
        valid_eq = ["बराबर", "समान", "जत्तिकै"]
        if not any(w in tgt for w in valid_eq):
            raise LinguisticValidationError(f"Equality keyword missing in target '{tgt}'.")
        # Also ensure BOTH entities are named
        q_entities = record["question"]["entities"]
        for eid in q_entities:
            if mapping[eid] not in tgt:
                raise LinguisticValidationError(f"Equality target must contain both entities: missing {mapping[eid]}")
        # Ensure it doesn't accidentally claim one is greater
        if adj_greater in tgt:
            raise LinguisticValidationError(f"Equality target contains greater relation word '{adj_greater}'")
        if adj_less in tgt:
            raise LinguisticValidationError(f"Equality target contains less relation word '{adj_less}'")
    else:
        # Non-equality semantics
        t = select_template(record["task_family"], record["language"], record["id"])
        
        expected_ans_name = mapping[ans_eid] if ans_eid else ""
        
        # Determine intended semantic relation of the answer
        if q_type == "largest":
            expected_word = adj_greater
        elif q_type == "smallest":
            expected_word = adj_less
        elif q_type == "compare":
            if t.question_direction == "less":
                q_entities = record["question"]["entities"]
                lesser_eid = q_entities[0] if q_entities[1] == ans_eid else q_entities[1]
                expected_ans_name = mapping[lesser_eid]
                expected_word = adj_less
            elif t.question_direction == "greater":
                expected_word = adj_greater
            else:
                expected_word = "" # General
        else:
            expected_word = ""

        if expected_ans_name and expected_ans_name not in tgt:
            raise LinguisticValidationError(f"Target text does not contain expected answer entity '{expected_ans_name}'.")
            
        if expected_word and expected_word not in tgt:
            raise LinguisticValidationError(f"Target text does not contain expected relation word '{expected_word}'.")
            
        # Ensure the WRONG entity is NOT in the target (to prevent CoT or ambiguous targets)
        for eid, name in mapping.items():
            if expected_ans_name and name != expected_ans_name and is_distinct_entity(name, tgt):
                raise LinguisticValidationError(f"Target text unexpectedly contains entity '{name}' (reasoning leakage or wrong answer).")
                
    # 5. Check No Chain-of-Thought in Target
    if len(tgt.split()) > 15:
        raise LinguisticValidationError(f"Target text seems too long (possible CoT): {tgt}")
        
    # Check for meta-reasoning words in input/target
    banned_words = ["two-hop", "three-hop", "two steps", "three steps", "transitive reasoning"]
    for w in banned_words:
        if w in inp.lower() or w in tgt.lower():
            raise LinguisticValidationError(f"Meta-reasoning word '{w}' leaked into output.")
        
    return True
