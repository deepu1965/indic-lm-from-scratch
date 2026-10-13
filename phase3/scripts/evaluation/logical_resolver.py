import json
from phase3.logical_generator.structures import Fact
from phase3.logical_generator.solver import Solver
from phase3.logical_generator.relations import Relation
from phase3.template_system.templates import get_templates

_TEMPLATES_HI = {(t.task_family, t.template_id): t for t in get_templates("hi")}
_TEMPLATES_NE = {(t.task_family, t.template_id): t for t in get_templates("ne")}

def get_template(language: str, task_family: str, template_id: str):
    if language == "hi":
        return _TEMPLATES_HI[(task_family, template_id)]
    elif language == "ne":
        return _TEMPLATES_NE[(task_family, template_id)]
    raise ValueError(f"Unknown language {language}")

def resolve_expected_answer(record: dict) -> dict:
    """
    Independently derive the expected answer from a JSON record.
    Returns:
        {
            "expected_answer_entity": str | None,
            "expected_relation": str,
            "question_type": str,
            "question_direction": str
        }
    """
    facts_data = record.get("facts", [])
    facts = [Fact(subject=f["subject"], relation=f["relation"], object=f["object"]) for f in facts_data]
    
    solver = Solver(facts)
    
    q = record["question"]
    q_type = q["type"]
    entities = q.get("entities", [])
    
    template_id = record["template_id"]
    language = record["language"]
    task_family = record["task_family"]
    template = get_template(language, task_family, template_id)
    
    q_dir = template.question_direction # "greater", "less", "general"
    
    if q_type == "compare":
        e1, e2 = entities[0], entities[1]
        rel = solver.query(e1, e2)
        
        # Determine expected answer entity based on relation and question direction
        expected_entity = None
        if rel == Relation.GREATER_THAN:
            if q_dir in ("greater", "general"):
                expected_entity = e1
            elif q_dir == "less":
                expected_entity = e2
        elif rel == Relation.LESS_THAN:
            if q_dir in ("greater", "general"):
                expected_entity = e2
            elif q_dir == "less":
                expected_entity = e1
        elif rel == Relation.EQUAL:
            expected_entity = None
            
        return {
            "expected_answer_entity": expected_entity,
            "expected_relation": rel.value if rel else None,
            "question_type": q_type,
            "question_direction": q_dir
        }
        
    elif q_type in ("largest", "smallest"):
        # The solver.find_extreme expects "largest" or "smallest".
        extreme_type = "largest" if q_type == "largest" else "smallest"
        ans = solver.find_extreme(entities, extreme_type)
        
        return {
            "expected_answer_entity": ans,
            "expected_relation": extreme_type,
            "question_type": q_type,
            "question_direction": q_dir
        }
    else:
        raise ValueError(f"Unknown question type {q_type}")

