import hashlib
import random
from typing import List
from .templates import Template, get_templates

# Cache to avoid regenerating the balanced assignment array repeatedly
_ASSIGNMENT_CACHE = {}

def _get_balanced_assignment(task_family: str, language: str) -> List[int]:
    cache_key = f"{language}_{task_family}"
    if cache_key in _ASSIGNMENT_CACHE:
        return _ASSIGNMENT_CACHE[cache_key]
        
    # Generate exactly 250 of each template index [0, 1, 2, 3]
    assignment = []
    for i in range(4):
        assignment.extend([i] * 250)
        
    # Seed a deterministic RNG specifically for shuffling this family's templates
    seed_str = f"balance_{language}_{task_family}"
    seed_val = int(hashlib.md5(seed_str.encode()).hexdigest(), 16) % (2**32)
    rng = random.Random(seed_val)
    
    # Shuffle deterministically
    rng.shuffle(assignment)
    
    _ASSIGNMENT_CACHE[cache_key] = assignment
    return assignment

def select_template(task_family: str, language: str, record_id: str) -> Template:
    """
    Deterministically selects one of the 4 templates for the given task_family and language,
    guaranteeing exactly 250 instances per template for the 1000 records of a task family.
    
    Relies on record_id being a sequential string "000001" to "007000".
    """
    templates = get_templates(language)
    family_templates = [t for t in templates if t.task_family == task_family]
    
    if len(family_templates) != 4:
        raise ValueError(f"Expected 4 templates for {task_family} in {language}, found {len(family_templates)}")
        
    assignment = _get_balanced_assignment(task_family, language)
    
    # Calculate index within the task family (0 to 999)
    # id is 1-indexed globally across all families (1 to 7000)
    # Each family has exactly 1000 records.
    global_idx = int(record_id) - 1
    local_idx = global_idx % 1000
    
    chosen_template_idx = assignment[local_idx]
    
    return family_templates[chosen_template_idx]
