from huggingface_hub import InferenceClient
import time
import os

hugging_hub_token = os.environ.get("HF_TOKEN")
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
        max_tokens=64, # Shorten this to prevent long ramblings
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


def LLM_sort(objects, instruction, max_length=128):
    objects_str = ', '.join([f'"{o}"' for o in objects])
    query = prompt_sorting.format(objects_list=objects_str, instruction=instruction)

    resp = LLM(query, prompt="", max_length=max_length, stop_tokens=["#"]).strip()
    
    # Safety: only keep lines starting with robot.pick_and_place
    steps = [line for line in resp.split('\n') if line.startswith("robot.pick_and_place")]
    print(steps)
    return steps

prompt_sorting = """
objects = [{objects_list}]
# Sort objects according to the following instruction:
# {instruction}
# Only use robot.pick_and_place(...). 
# Use only these destinations: top left corner, top side, top right corner, left side, middle, right side, bottom left corner, bottom side, bottom right corner, tray.
# Output each step as a pick_and_place command.
# Do not explain or add text.
""".strip()

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