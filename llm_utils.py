from huggingface_hub import InferenceClient
import time
import os

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
            "content": "You are a code completion engine for a robot. Output ONLY the next line of code. Do not use markdown. Do not explain. Your output must start with 'robot.pick_and_place'."
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
    Coordinates: X,Z: [0-1], Y: [-1, 0]
    """
    # Create visual context
    object_strs = [f'"{obj}"' for obj in objects_list]
    visual_context = "objects = [" + ", ".join(object_strs) + "]"
    
    # Build query
    query = visual_context + "\n# " + instruction
    
    # Call LLM
    content = llm_fn(query, prompt=prompt_geometric_arrangement, max_tokens=max_tokens)
    
    # Strip anything before robot command
    if "robot." in content:
        content = content[content.find("robot."):]
    
    # Split into lines and keep only valid commands
    step_cmds = [line.strip() for line in content.split('\n') 
                 if line.strip().startswith('robot.pick_and_place')]
    
    return "\n".join(step_cmds)


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

# Example: Vertical line (far to near)
objects = ["apple", "banana", "orange"]
robot.pick_and_place("apple",  [0.0, -0.85, 1.0])
robot.pick_and_place("banana", [0.0, -0.5, 1.0])
robot.pick_and_place("orange", [0.0, -0.15, 1.0])

# Example: Circle arrangement
objects = ["ball_1", "ball_2", "ball_3", "ball_4"]
robot.pick_and_place("ball_1", [0.0, -0.85, 1.0])
robot.pick_and_place("ball_2", [0.35, -0.5, 1.0])
robot.pick_and_place("ball_3", [0.0, -0.15, 1.0])
robot.pick_and_place("ball_4", [-0.35, -0.5, 1.0])

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

# Example: Diagonal line
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

