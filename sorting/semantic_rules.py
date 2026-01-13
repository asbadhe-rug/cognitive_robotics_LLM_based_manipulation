# sorting/semantic_rules.py

# Define semantic categories for objects in the scene
# You can expand these later or load from JSON if needed

CATEGORIES = {
    "fruits": ["banana", "apple", "orange", "pear"],
    "tools": ["hammer", "screwdriver", "wrench", "pliers"],
    "fragile": ["glass_cup", "egg", "ceramic_plate"],
    "red": ["apple", "tomato", "red_ball"],
}

def categorize_object(obj_name: str):
    """
    Return a list of semantic categories an object name belongs to.
    """
    out = []
    for cat, items in CATEGORIES.items():
        if obj_name in items:
            out.append(cat)
    return out
