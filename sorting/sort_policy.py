# sorting/sort_policy.py

from typing import Dict, List
from llm_utils import LLM as query_llm, LLM_sort
from sorting.semantic_rules import categorize_object

def parse_sort_instruction(instruction: str):
    """
    Takes a raw natural language instruction like:
        "Put all red fruits in basket A and all tools in basket B"
    Returns a mapping: { category -> target_region }
    """
    # Simple rule parser (can be replaced with LLM or regex)
    # This splits by 'and' or commas.
    clauses = [c.strip() for c in instruction.split(" and ")]
    mapping = {}
    for clause in clauses:
        parts = clause.split(" in ")
        if len(parts) == 2:
            cat_phrase = parts[0].replace("Put all", "").strip()
            region = parts[1].strip()
            mapping[cat_phrase] = region
    return mapping

def generate_sort_plan(objects: List[str], instruction: str):
    """
    Build a list of pick-and-place plans for sorting objects
    based on semantic category rules + LLM grounding.
    """
    # Map semantic category -> target region from instruction
    cat_to_region = parse_sort_instruction(instruction)

    # assign objects to categories
    object_to_target = {}
    for obj in objects:
        cats = categorize_object(obj)
        # choose first matching category
        for cat in cats:
            if cat in cat_to_region:
                object_to_target[obj] = cat_to_region[cat]
                break

    # Create a list of robot actions
    plan = []
    for obj, region in object_to_target.items():
        plan.append(f'robot.pick_and_place("{obj}", "{region}")')
    print(plan)
    return plan
