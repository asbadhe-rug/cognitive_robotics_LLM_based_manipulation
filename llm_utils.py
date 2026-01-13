from huggingface_hub import InferenceClient
import time
import os
import json

hugging_hub_token = os.getenv("HF_TOKEN")
if hugging_hub_token is None:
    raise ValueError("Get your Hugging Hub API token from here: https://huggingface.co/docs/hub/security-tokens.\nThen, set it in llm_utils.py.")

#llm_inference = InferenceClient("bigscience/bloom", token=hugging_hub_token)
llm_inference = InferenceClient("meta-llama/Llama-3.2-3B-Instruct", token=hugging_hub_token) #Llama-3.2

def LLM(query, prompt, **params):
    # This role forces the Instruct model to behave like a Base model
    messages = [
        {
            "role": "system", 
            "content": """You are a robotic manipulation assistant. Output ONLY robot.pick_and_place commands.

    CRITICAL: When arranging objects in patterns (circles, grids, etc.):
    - Count the total number of objects
    - Distribute them EVENLY across the shape
    - For circles: use angles of 360°/n for n objects
    - Never place multiple objects at the same coordinates

    Your output must start with 'robot.pick_and_place'."""
        },
        {"role": "user", "content": prompt + "\n\n" + query}
    ]
    
    response = llm_inference.chat_completion(
        messages=messages,
        max_tokens=512, # Shorten this to prevent long ramblings
        temperature=0,
        # 'stop' is our safety net: it cuts the LLM off if it starts talking
        stop=["#", "class", "```", "Note:", "\n\n"] 
    )
    
    content = response.choices[0].message.content.strip()
    
    # Final safety check: if it still put text before the command, strip it
    if "robot." in content:
        content = content[content.find("robot."):]
        
    return content


def LLMWrapper(query, context, verbose=False):
    query = query + ('.' if query[-1] != '.' else '') 
    resp = LLM(query, prompt=prompt_plan, stop_tokens=['#']).strip()
    resp_obj, resp_full = None, resp
    if 'parse_obj' in resp:
        steps = resp.split('\n')
        obj_query = [s for i, s in enumerate(steps) if 'parse_obj' in s][0].split('("')[1].split('")')[0]
        obj_query = context + '\n' + f'# {obj_query}.'
        resp_obj = LLM(obj_query, prompt=prompt_parse_obj, stop_tokens=['#', 'objects = [']).strip()
        resp_full = '\n'.join([resp, '\n' + obj_query, resp_obj])
    if verbose:
        print(query)
        print(resp_full)
    return resp, resp_obj

def LLM_geometric(instruction: str, objects_list: list, llm_fn, max_tokens=512):
    """
    Calls the LLM to output pick-and-place commands.
    """
    n_objects = len(objects_list)
    object_strs = [f'"{obj}"' for obj in objects_list]
    visual_context = "objects = [" + ", ".join(object_strs) + "]"
    
    # Add explicit count
    query = visual_context + f"\n# There are {n_objects} objects.\n# " + instruction
    
    content = llm_fn(query, prompt=prompt_geometric_arrangement, max_tokens=max_tokens)
    
    # Strip anything before robot command
    if "robot." in content:
        content = content[content.find("robot."):]
    
    # Split into lines and keep only valid commands
    step_cmds = [line.strip() for line in content.split('\n') 
                 if line.strip().startswith('robot.pick_and_place')]
    
    return "\n".join(step_cmds)

def validate_and_fix_parse(parsed: dict, instruction: str, n_objects: int) -> dict:
    """
    Validate object counts and basic structure only.
    NO special-case pattern matching.
    """
    
    # Only fix: object count mismatches
    if parsed.get("type") == "multiple":
        shapes = parsed.get("shapes", [])
        total_allocated = sum(s.get("objects", 0) for s in shapes)
        
        if total_allocated != n_objects and total_allocated > 0:
            print(f"⚠️  Object count mismatch: allocated {total_allocated}, have {n_objects}")
            
            # Redistribute proportionally
            for shape in shapes:
                ratio = shape.get("objects", 1) / total_allocated
                shape["objects"] = max(1, int(n_objects * ratio))
            
            # Fix rounding
            shapes[-1]["objects"] += n_objects - sum(s["objects"] for s in shapes)
            print(f"   Redistributed: {[s['objects'] for s in shapes]}")
    
    return parsed

def LLM_shape_parser(instruction: str, n_objects: int, llm_fn) -> dict:
    """
    LLM decomposes ANY instruction into geometric primitives.
    No hardcoded patterns - pure geometric reasoning.
    """
    
    system_prompt = """You are a geometric reasoning AI. Break down ANY instruction into simple primitives.

AVAILABLE PRIMITIVES: circle, line, diagonal, triangle, square, grid, arc

PARAMETERS:
- line: start=[x,y], end=[x,y], orientation='horizontal'/'vertical'
- circle: radius=<float>
- diagonal: start=[x,y], end=[x,y], direction='topleft-bottomright'/'topright-bottomleft'
- All shapes: offset=[x,y,z] to position them

WORKSPACE: X ∈ [-0.35, 0.35], Y ∈ [-0.85, -0.15], Z = 1.0

THINK GEOMETRICALLY: What primitives + positions create this pattern?

EXAMPLES:

Input: "make two parallel horizontal lines"
Total: 6
Reasoning: Two horizontal lines = 2 separate lines, positioned above/below center
{
  "type": "multiple",
  "shapes": [
    {"shape": "line", "params": {"orientation": "horizontal"}, "objects": 3, "offset": [0, -0.2, 0]},
    {"shape": "line", "params": {"orientation": "horizontal"}, "objects": 3, "offset": [0, 0.2, 0]}
  ]
}

Input: "create two concentric circles"
Total: 8
Reasoning: Two circles = small radius + large radius, both centered
{
  "type": "multiple",
  "shapes": [
    {"shape": "circle", "params": {"radius": 0.15}, "objects": 3, "offset": [0, 0, 0]},
    {"shape": "circle", "params": {"radius": 0.3}, "objects": 5, "offset": [0, 0, 0]}
  ]
}

Input: "arrange in an X shape"
Total: 6
Reasoning: X = two diagonals intersecting at center
{
  "type": "multiple",
  "shapes": [
    {"shape": "diagonal", "params": {"direction": "topleft-bottomright"}, "objects": 3, "offset": [0, 0, 0]},
    {"shape": "diagonal", "params": {"direction": "topright-bottomleft"}, "objects": 3, "offset": [0, 0, 0]}
  ]
}

Input: "spell the letter T"
Total: 7
Reasoning: T = horizontal line on top + vertical line below center
{
  "type": "multiple",
  "shapes": [
    {"shape": "line", "params": {"orientation": "horizontal"}, "objects": 4, "offset": [0, -0.3, 0]},
    {"shape": "line", "params": {"orientation": "vertical"}, "objects": 3, "offset": [0, 0.15, 0]}
  ]
}

Input: "create a plus sign"
Total: 8
Reasoning: Plus = vertical line + horizontal line, both centered
{
  "type": "multiple",
  "shapes": [
    {"shape": "line", "params": {"orientation": "vertical"}, "objects": 4, "offset": [0, 0, 0]},
    {"shape": "line", "params": {"orientation": "horizontal"}, "objects": 4, "offset": [0, 0, 0]}
  ]
}

Input: "arrange in a circle"
Total: 5
Reasoning: Single circle, use all objects
{"type": "single", "shape": "circle", "params": {"radius": 0.35}}

Input: "make a square with 4 and circle with rest"
Total: 7
Reasoning: User explicitly split: 4 for square, 3 for circle
{
  "type": "multiple",
  "shapes": [
    {"shape": "square", "params": {"radius": 0.2}, "objects": 4, "offset": [-0.15, 0, 0]},
    {"shape": "circle", "params": {"radius": 0.2}, "objects": 3, "offset": [0.15, 0, 0]}
  ]
}

KEY PRINCIPLES:
1. Use params (radius, orientation, direction) to control primitive behavior
2. Use offset to position shapes spatially
3. Distribute objects across shapes based on complexity
4. Think: "What simple primitives + positions = this pattern?"

Output ONLY valid JSON. No code, no explanations."""

    user_prompt = f"""Instruction: "{instruction}"
Total objects: {n_objects}

Think: What simple geometric shapes combine to create this pattern?
JSON output:"""
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    response = llm_inference.chat_completion(
        messages=messages,
        max_tokens=400,
        temperature=0.0,  # ✅ Deterministic
        stop=["\n\n", "```", "Instruction:", "To parse", "Here's", "def "]  # ✅ Stop at code/explanation
    )
    
    content = response.choices[0].message.content.strip()
    
    print("=" * 60)
    print("🔍 Shape Parser:")
    print(f"Input: {instruction}")
    print(f"Raw LLM Output:\n{content}")
    print("=" * 60)
    
    try:
        # Clean up response
        # Remove markdown code blocks
        if "```" in content:
            parts = content.split("```")
            for part in parts:
                if "{" in part and "}" in part:
                    content = part.strip()
                    if content.startswith("json"):
                        content = content[4:].strip()
                    break
        
        # Extract JSON only
        start = content.find("{")
        end = content.rfind("}") + 1
        if start == -1 or end <= start:
            raise ValueError("No JSON found in response")
        
        content = content[start:end]
        
        # Remove any trailing text after the JSON
        parsed = json.loads(content)

        parsed = validate_and_fix_parse(parsed, instruction, n_objects)
        
        # Validate structure
        if parsed.get("type") == "multiple":
            shapes = parsed.get("shapes", [])
            
            if not shapes:
                raise ValueError("Multiple type but no shapes")
            
            # Calculate "rest" allocations
            allocated = 0
            rest_indices = []
            
            for i, shape_def in enumerate(shapes):
                obj_count = shape_def.get("objects")
                if isinstance(obj_count, int):
                    allocated += obj_count
                elif isinstance(obj_count, str) and obj_count.lower() in ["rest", "remaining", "others"]:
                    rest_indices.append(i)
            
            # Distribute remaining objects to "rest" shapes
            remaining = n_objects - allocated
            if rest_indices and remaining > 0:
                per_rest = remaining // len(rest_indices)
                for i in rest_indices:
                    shapes[i]["objects"] = per_rest
                
                # Give any leftover to the last "rest" shape
                leftover = remaining % len(rest_indices)
                if leftover > 0:
                    shapes[rest_indices[-1]]["objects"] += leftover
            
            # Build summary
            shape_summary = [f"{s['shape']}({s['objects']})" for s in shapes]
            print(f"✅ Multiple shapes: {shape_summary}")
            
        elif parsed.get("type") == "single":
            obj_count = parsed.get('objects', n_objects)
            print(f"✅ Single shape: {parsed['shape']} with {obj_count} objects")
        else:
            raise ValueError("Invalid type, must be 'single' or 'multiple'")
        
        return parsed
        
    except Exception as e:
        print(f"⚠️  Failed to parse: {e}")
        print(f"Raw output: {content}")
        
        # Try to salvage with rule-based parsing as fallback
        instruction_lower = instruction.lower()
        
        # Check for "and" indicating multiple shapes
        if " and " in instruction_lower or " with the rest" in instruction_lower:
            print("🔧 Attempting rule-based parsing...")
            
            # Simple pattern matching
            import re
            
            # Try to find "N objects" patterns
            pattern = r"(\d+)\s+objects?"
            matches = re.findall(pattern, instruction_lower)
            
            if matches and len(matches) >= 1:
                # Found explicit count
                first_count = int(matches[0])
                remaining = n_objects - first_count
                
                # Detect shapes
                shape_keywords = {
                    'square': 'square', 'circle': 'circle', 'line': 'line',
                    'triangle': 'triangle', 'hexagon': 'hexagon', 'grid': 'grid'
                }
                
                found_shapes = []
                for keyword, shape in shape_keywords.items():
                    if keyword in instruction_lower:
                        found_shapes.append(shape)
                
                if len(found_shapes) >= 2:
                    print(f"✅ Parsed: {found_shapes[0]}({first_count}), {found_shapes[1]}({remaining})")
                    return {
                        "type": "multiple",
                        "shapes": [
                            {"shape": found_shapes[0], "params": {"radius": 0.25}, "objects": first_count},
                            {"shape": found_shapes[1], "params": {"radius": 0.3}, "objects": remaining}
                        ]
                    }
        
        # Final fallback
        return {"type": "single", "shape": "circle", "params": {"radius": 0.35}}

prompt_pick_and_place_detection = """
objects = ["scissors", "pear", "hammer", "mustard bottle", "tray"]
# put the bottle to the left side.
robot.pick_and_place("mustard bottle", "left side")
objects = ["banana", "foam brick", "strawberry", "tomato soup can", "pear", "tray"]
# move the fruit to the bottom right corner.
robot.pick_and_place("banana", "bottom right corner")
robot.pick_and_place("pear", "bottom right corner")
robot.pick_and_place("strawberry", "bottom right corner")
# now put the green one in the top side.
robot.pick_and_place("pear", "top side")
# undo the last step.
robot.pick_and_place("pear", "bottom right corner")
objects = ["potted meat can", "power drill", "chips can", "hammer", "tomato soup can", "tray"]
# put all cans in the tray.
robot.pick_and_place("potted meat can", "tray")
robot.pick_and_place("chips can", "tray")
robot.pick_and_place("tomato soup can", "tray")
objects = ["power drill", "strawberry", "medium clamp", "gelatin box", "tray"]
# move the clamp behind of the drill
robot.pick_and_place("medium clamp", "power drill", "behind")
# actually, I want it on the opposite side of the drill
robot.pick_and_place("medium clamp", "power drill", "front")
objects = ["chips can", "banana", "strawberry", "potted meat can", "pear", "tray"]
# put the red fruit left of the green one 
robot.pick_and_place("strawberry", "pear", "left")
""".strip()


prompt_pick_and_place_grounding = """
from robot_utils import pick_and_place
from camera_utils import find, scene_init

### start of trial
objects = scene_init()
# put the bottle to the left side.
bottle = find(objects, "bottle")[0]
pick_and_place(bottle, "left side")

### start of trial
objects = scene_init()
# move all the fruit to the bottom right corner.
fruits = find(objects, "fruit")
for fruit_instance in fruits:
    pick_and_place(fruit_instance, "bottom right corner")
# now put the small one in the right side.
small_fruit = find(fruits, "small fruit")[0]
pick_and_place(small_fruit, "right side")
# undo the last step.
pick_and_place(small_fruit, "bottom right corner")

### start of trial
objects = scene_init()
# put all cans in the tray.
cans = find(objects, "can")
for can_instance in cans:
    pick_and_place(can_instance, "tray")
""".strip()

prompt_geometric_arrangement = """
You are a robotic manipulation assistant. Given visible objects and an instruction,
output Python commands using robot.pick_and_place(object_name, [x, y, z]).

COORDINATE SYSTEM:
- X: -0.5 (LEFT BOUND) → 0.0 (CENTER) → 0.5 (RIGHT BOUND)
- Y: -0.85 (TOP/FAR from robot) → -0.5 (MIDDLE) → -0.15 (BOTTOM/NEAR robot)
- Z: Always 1.0 (table surface)

IMPORTANT:
- X: -0.35 to 0.35
- Y: -0.85 to -0.15 (MUST BE NEGATIVE!)
- Output ONLY robot.pick_and_place commands (no comments, no explanations)
- Prefer symmetry around X=0 unless instructed otherwise.

Rules:
- Always output robot.pick_and_place(object_name, [x, y, z])
- Coordinates must satisfy:
  x ∈ [-0.35,0.35], y ∈ [-0.85,-0.15], z = 1.0
- Do NOT use symbolic positions (left, right, middle, etc.)

# Example: Horizontal line (left to right)
objects = ["red_cube", "blue_cube", "green_cube"]
robot.pick_and_place("red_cube", [0.35, -0.5, 1.0])
robot.pick_and_place("blue_cube", [0.0, -0.5, 1.0])
robot.pick_and_place("green_cube", [-0.35, -0.5, 1.0])

# Example: Vertical line (far to near) with 3 objects
objects = ["apple", "banana", "orange"]
robot.pick_and_place("apple",  [0.0, -0.85, 1.0])
robot.pick_and_place("banana", [0.0, -0.5, 1.0])
robot.pick_and_place("orange", [0.0, -0.15, 1.0])

# Example: Circle with 4 objects (90° apart)
objects = ["ball_1", "ball_2", "ball_3", "ball_4"]
robot.pick_and_place("ball_1", [0.0, -0.85, 1.0])
robot.pick_and_place("ball_2", [0.35, -0.5, 1.0])
robot.pick_and_place("ball_3", [0.0, -0.15, 1.0])
robot.pick_and_place("ball_4", [-0.35, -0.5, 1.0])

# Example: Circle with 5 objects (72° apart)
objects = ["cube_1", "cube_2", "cube_3", "cube_4", "cube_5"]
robot.pick_and_place("cube_1", [0.0, -0.85, 1.0])
robot.pick_and_place("cube_2", [0.33, -0.63, 1.0])
robot.pick_and_place("cube_3", [0.33, -0.37, 1.0])
robot.pick_and_place("cube_4", [-0.33, -0.37, 1.0])
robot.pick_and_place("cube_5", [-0.33, -0.63, 1.0])

# Example: Circle with 6 objects (60° apart)
objects = ["item_1", "item_2", "item_3", "item_4", "item_5", "item_6"]
robot.pick_and_place("item_1", [0.0, -0.85, 1.0])
robot.pick_and_place("item_2", [0.30, -0.69, 1.0])
robot.pick_and_place("item_3", [0.30, -0.31, 1.0])
robot.pick_and_place("item_4", [0.0, -0.15, 1.0])
robot.pick_and_place("item_5", [-0.30, -0.31, 1.0])
robot.pick_and_place("item_6", [-0.30, -0.69, 1.0])

# Example: Triangle
objects = ["fork", "spoon", "knife"]
robot.pick_and_place("fork", [0.0, -0.2, 1.0])
robot.pick_and_place("spoon", [-0.35, -0.8, 1.0])
robot.pick_and_place("knife", [0.35, -0.8, 1.0])

# Example: Grid (2x2)
objects = ["item_1", "item_2", "item_3", "item_4"]
robot.pick_and_place("item_1", [-0.2, -0.7, 1.0])
robot.pick_and_place("item_2", [ 0.2, -0.7, 1.0])
robot.pick_and_place("item_3", [-0.2, -0.3, 1.0])
robot.pick_and_place("item_4", [ 0.2, -0.3, 1.0])

# Example: Diagonal line with 3 objects
objects = ["cube_1", "cube_2", "cube_3"]
robot.pick_and_place("cube_1", [-0.3, -0.8, 1.0])
robot.pick_and_place("cube_2", [ 0.0, -0.5, 1.0])
robot.pick_and_place("cube_3", [ 0.3, -0.2, 1.0])

# Example: Arc arrangement
objects = ["tool_1", "tool_2", "tool_3", "tool_4", "tool_5"]
robot.pick_and_place("tool_1", [-0.3, -0.5, 1.0])
robot.pick_and_place("tool_2", [-0.15, -0.7, 1.0])
robot.pick_and_place("tool_3", [ 0.0, -0.8, 1.0])
robot.pick_and_place("tool_4", [ 0.15, -0.7, 1.0])
robot.pick_and_place("tool_5", [ 0.3, -0.5, 1.0])
""".strip()

