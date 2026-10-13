from typing import Dict, Any, List
from .entities import assign_entities
from .lexicon import get_lexicon
from .selector import select_template

def render_example(record: Dict[str, Any], language: str) -> Dict[str, Any]:
    """
    Takes a Phase 3.1 LogicalRecord dict and renders it into input_text and target_text.
    """
    record_id = record["id"]
    seed = record["seed"]
    split = record["split"]
    task_family = record["task_family"]
    attr = record["attribute"]
    representation = record["representation"]
    
    abstract_ids = [e["id"] for e in record["entities"]]
    entity_mapping = assign_entities(language, split, abstract_ids, seed)
    
    lexicon = get_lexicon(language)
    attr_lex = lexicon["attributes"][attr]
    
    template = select_template(task_family, language, record_id)
    
    # 1. Render Facts
    fact_strings = []
    if representation == "numerical" and "values" in record and record["values"]:
        # Render a numerical statement for each unique entity that appears in the facts
        involved_entities = set()
        for f in record["facts"]:
            involved_entities.add(f["subject"])
            involved_entities.add(f["object"])
        
        # Sort to maintain determinism
        for eid in sorted(involved_entities):
            val = record["values"][eid]
            name = entity_mapping[eid]
            s = template.render_num_fact(name, val, attr_lex)
            fact_strings.append(s)
    else:
        # Relational rendering
        for f in record["facts"]:
            sub = entity_mapping[f["subject"]]
            obj = entity_mapping[f["object"]]
            rel = f["relation"]
            s = template.render_fact(sub, obj, rel, attr_lex)
            fact_strings.append(s)
            
    # 2. Render Question
    q_data = record["question"]
    q_type = q_data["type"]
    q_entities = [entity_mapping[eid] for eid in q_data["entities"]]
    q_str = template.render_question(q_type, q_entities, attr_lex)
    
    input_text = " ".join(fact_strings) + " " + q_str
    
    # 3. Render Answer
    gt = record["ground_truth"]
    ans_eid = gt.get("answer_entity")
    ans_name = entity_mapping[ans_eid] if ans_eid else ""
    gt_rel = gt.get("relation", "")
    
    if q_type == "compare":
        if gt_rel == "equal":
            ans_fmt = template.answer_formats["equal"]
            target_text = ans_fmt.format(e1=q_entities[0], e2=q_entities[1], attr=attr_lex["noun"])
        else:
            # We strictly identify the answer_entity and state that its attribute is GREATER or LESS,
            # depending entirely on the semantic direction of the question.
            # E.g., if question asks "who is greater?", target_text must be "ans_name is greater".
            # The linguistic layer ignores the logical relation ("less_than"), because answer_entity
            # correctly holds the mathematically correct entity that answers the question.
            if template.question_direction == "less":
                # For "who is less?" questions, we need the entity that is mathematically lesser.
                # Since phase 3.1 always sets answer_entity to the GREATER entity for compare questions,
                # the lesser entity is simply the other one in the question list!
                lesser_name = q_entities[0] if q_entities[1] == ans_name else q_entities[1]
                ans_fmt = template.answer_formats["less"]
                target_text = ans_fmt.format(ans=lesser_name, attr=attr_lex["noun"])
            else:
                # Default "greater" questions or "general"
                ans_fmt = template.answer_formats["greater"]
                target_text = ans_fmt.format(ans=ans_name, attr=attr_lex["noun"])
                
    elif q_type == "largest":
        ans_fmt = template.answer_formats["largest"]
        target_text = ans_fmt.format(ans=ans_name, attr=attr_lex["noun"])
    elif q_type == "smallest":
        ans_fmt = template.answer_formats["smallest"]
        target_text = ans_fmt.format(ans=ans_name, attr=attr_lex["noun"])
    else:
        target_text = ""
        
    # 4. Build output record
    out = dict(record)
    out["template_id"] = template.template_id
    out["language"] = language
    out["entity_mapping"] = entity_mapping
    out["input_text"] = input_text
    out["target_text"] = target_text
    
    return out
