import time
from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import TerminalFormatter, HtmlFormatter
import tkinter as tk
from tkinter import simpledialog, messagebox, scrolledtext
import webbrowser
import tempfile
from PIL import Image
import cv2
import ast
import re
import math
#import streamlit as st

# pprint = lambda s: display(HTML(highlight(s, PythonLexer(), HtmlFormatter(full=True))))
#pprint = lambda s: print(highlight(s, PythonLexer(), TerminalFormatter()).strip())

from env.camera import Camera
from env.env import *
from env.objects import YcbObjects
from grconvnet import load_grasp_generator
from clip_utils import ClipInference
from sam_utils import SamInference
from llm_utils import llm_generate_plan
from env.objects import YCB_CATEGORIES as ADMISSIBLE_OBJECTS


# GUI stuff
# Function to create a text input dialog using Tkinter
def ask_for_user_input():
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    user_input = simpledialog.askstring("Input", "User Input: ")
    root.destroy()
    return user_input

# Function to display a message with the result
# def display_result(result):
#     root = tk.Tk()
#     root.withdraw()  # Hide the main window
#     formatted_code = highlight(result, PythonLexer(), HtmlFormatter(full=True, style='friendly'))
#     messagebox.showinfo("Plan", formatted_code)
#     root.destroy()
# Function to display a message with the result
# def display_result(result):
#     # Use Pygments to format the Python code
#     formatted_code = highlight(result, PythonLexer(), HtmlFormatter(full=True, style='friendly'))
#     # Create a temporary HTML file to display the result
#     with open("code.html", "w") as f:
#         f.write(formatted_code)
#     # Open the HTML file in the default web browser
#     webbrowser.open("code.html")



class RobotEnvUI:
    
    def __init__(self, 
                 n_objects: int, 
                 n_action_attempts: int = 3,
                 n_grasp_attempts: int = 4,
                 visualise_grasps: bool = False,
                 visualise_clip: bool = False,
                 ground_truth_segm: bool = True,
                 clip_prompt_eng: bool = False,
                 clip_this_is: bool = False,
                 selected_objects: list = None,
                 seed=None
):
        # init env
        center_x, center_y, center_z = CAM_X, CAM_Y, CAM_Z
        self.camera = Camera((center_x, center_y, center_z), (center_x, center_y, 0.785), 0.2, 2.0, (IMG_SIZE, IMG_SIZE), 40)
        self.env = Environment(self.camera, vis=True, asset_root='./env/assets', debug=False, finger_length=0.06)
        
        # constants
        self.TARGET_ZONE_POS = TARGET_ZONE_POS
        self.ADMISSIBLE_OBJECTS = ADMISSIBLE_OBJECTS
        self.ADMISSIBLE_LOCATIONS = list(self.env.TARGET_LOCATIONS.keys()) + ['tray']

        # load objects
        self.seed = None
        self.objects = YcbObjects('env/assets/ycb_objects',
                    mod_orn=['ChipsCan', 'MustardBottle', 'TomatoSoupCan'],
                    mod_stiffness=['Strawberry'],
                    seed=self.seed
        )
        self.objects.shuffle_objects()
        self.env.dummy_simulation_steps(10)
    
        # load GR-ConvNet grasp synthesis network
        self.grasp_generator = load_grasp_generator(self.camera)
        self.n_grasp_attempts = n_grasp_attempts

        self.clip_prompt_eng = clip_prompt_eng
        self.clip_this_is = clip_this_is
        self.visualise_clip = visualise_clip
        self.visualise_grasps = visualise_grasps
        
        # define LLM callable and params
        self.history = []
        self.n_action_attempts = n_action_attempts
        
        # load CLIP for vision-language grounding
        self.clip_model = ClipInference()

        # load object segmentation (groundtruth / SAM)
        self.ground_truth_segm = ground_truth_segm
        if not ground_truth_segm:
            # load SAM for segmentation - check sam_utils.py and https://github.com/facebookresearch/segment-anything
            raise NotImplementedError

        else:
            # load from simulator
            self.segment = lambda im: self.env.camera.get_cam_img()[-1]

        # spawn scene
        self.spawn(n_objects, selected_objects)
    
    def validate_coordinates(self, x, y, z):
        """
        Validate absolute world coordinates.

        Coordinate system:
        - x ∈ [0, 1]        (table width)
        - y ∈ [-1, 0]       (table depth, negative is farther)
        - z ≥ TABLE_HEIGHT  (absolute height)
        """

        TABLE_HEIGHT = 1.0   # must match env

        if not (-0.5 <= x <= 0.5):
            return False, f"X={x} out of range [0.0, 1.0]"

        if not (-1.0 <= y <= 0.0):
            return False, f"Y={y} out of range [-1.0, 0.0]"

        if z < TABLE_HEIGHT:
            return False, f"Z={z} below table height {TABLE_HEIGHT}"

        return True, "OK"
    
    def execute_llm_plan(self, llm_output):
        """
        Execute LLM plan with coordinate validation.
        """
        import ast
        import re
        
        step_cmds = [s.strip() for s in llm_output.split('\n') if s.strip()]

        for step_cmd in step_cmds:
            if not step_cmd.startswith("robot.pick_and_place"):
                print(f"Skipping line: {step_cmd}")
                continue

            try:
                # Parse command
                match = re.match(r'robot\.pick_and_place\((.*)\)', step_cmd)
                if not match:
                    print(f"Could not parse: {step_cmd}")
                    continue
                
                args_str = match.group(1)
                
                # Simple split for coordinates (assumes well-formed input)
                args = []
                current = ""
                depth = 0
                in_str = False
                
                for char in args_str:
                    if char in '"\'':
                        in_str = not in_str
                    elif char == '[' and not in_str:
                        depth += 1
                    elif char == ']' and not in_str:
                        depth -= 1
                    elif char == ',' and depth == 0 and not in_str:
                        args.append(current.strip())
                        current = ""
                        continue
                    current += char
                
                if current.strip():
                    args.append(current.strip())
                
                if len(args) < 2:
                    print(f"Invalid args: {step_cmd}")
                    continue
                
                obj_name = ast.literal_eval(args[0])
                target = ast.literal_eval(args[1])
                how = ast.literal_eval(args[2]) if len(args) > 2 else None
                
                # VALIDATE COORDINATES if target is a list
                if isinstance(target, (list, tuple)) and len(target) == 3:
                    valid, msg = self.validate_coordinates(target[0], target[1], target[2])
                    if not valid:
                        print(f"⚠️  Invalid coordinates in {step_cmd}")
                        print(f"   {msg}")
                        print(f"   Skipping this command.")
                        continue
                
                # Validate object exists
                if obj_name not in self.obj_name_to_id:
                    print(f"Object '{obj_name}' not found. Skipping.")
                    continue
                
                # Execute
                attempt = 0
                success = False
                while attempt < self.n_action_attempts and not success:
                    try:
                        success = self.pick_and_place(obj_name, target, how)
                    except Exception as e:
                        print(f"Error: {e}")
                        success = False

                    if not success:
                        print(f"Attempt {attempt+1}/{self.n_action_attempts} failed")
                        for _ in range(30):
                            self.env.step_simulation()
                        self._step()
                    else:
                        print(f"✓ {obj_name} placed successfully")
                    
                    attempt += 1

                if not success:
                    print(f"✗ Failed after {self.n_action_attempts} attempts: {step_cmd}")

            except Exception as e:
                print(f"Error processing {step_cmd}: {e}")

    def pick_and_place(self, obj_name, target, how=None):
        """
        obj_name: str
        target:
            - [x, y, z] absolute coordinates
            - str object name (ONLY if how is provided)
        """

        if obj_name not in self.obj_name_to_id:
            print(f"Object '{obj_name}' not found")
            return False

        obj_id = self.obj_name_to_id[obj_name]

        # Relative placement: object -> object
        if how is not None:
            if isinstance(target, str) and target in self.obj_name_to_id:
                target_id = self.obj_name_to_id[target]
                return self.step(obj_id, target_id, how)
            else:
                raise ValueError("Relative placement requires target object name")

        # Absolute placement: coordinates only
        if isinstance(target, (list, tuple)) and len(target) == 3:
            return self.step(obj_id, list(target))

        raise ValueError(f"Invalid target: {target}")


    def spawn(self, n_objects, selected_objects):
        self.n_objects = n_objects
        self.selected_objects = selected_objects
        print(selected_objects)
        print("Hello world")

        if selected_objects is not None:
            for obj_name in selected_objects:
                path, mod_orn, mod_stiffness = self.objects.get_obj_info(obj_name)
                self.env.load_isolated_obj(path, obj_name, mod_orn, mod_stiffness)
        else:
            for obj_name in self.objects.obj_names[:self.n_objects]:
                path, mod_orn, mod_stiffness = self.objects.get_obj_info(obj_name)
                self.env.load_isolated_obj(path, obj_name, mod_orn, mod_stiffness)
            
        self.env.dummy_simulation_steps(10)
        self._step()
        self.init_obj_state = self.obj_state
        self.obj_ids = self.env.obj_ids

    def reset_scene(self, new=False):
        if new:
            self.spawn(self.n_objects)
            return
        self.reset()

    def reset(self):
        assert self.init_obj_state is not None, "Have to spawn once to initialize state"
        self.env.set_obj_state(self.init_obj_state)
        self.env.dummy_simulation_steps(10)
        self._step()
        self.init_obj_state = self.obj_state
        self.obj_ids = self.env.obj_ids

    def _step(self):
        self.env.reset_robot()
        self.env.dummy_simulation_steps(10)
        self.env.update_obj_states()
        #self.env.dummy_simulation_steps(10)
        self.obj_state = self.env.get_obj_states()
        clip_out = self.run_clip()
        if clip_out is None:
            print("No more objects left, exiting")
            self.env.close()
            return  # exit safely
        else:
            masks, categories, objIds = clip_out
        self.clip_names = list(self.obj_name_to_id.keys())
        self.setup_grasps(objIds, masks)
        self.env.dummy_simulation_steps(10)

    # run inference with CLIP for zero-shot object recognition
    def run_clip(self, visualise_clip=None):
        visualise_clip = visualise_clip or self.visualise_clip

        img, _, seg = self.camera.get_cam_img()

        # exit if no more objects
        if np.unique(seg).shape[0] <= 2:
            return None  # ✅ Return None, not ValueError

        # segmentation
        if not self.ground_truth_segm:
            # fill in SAM code here
            pass

        # Call CLIP to zero-shot recognize objects
        prompt_categories =  ";".join(self.ADMISSIBLE_OBJECTS)
        clip_out = self.clip_model.get_most_similar(img, seg,
            prompt_categories, 
            mode="object", 
            prompt_engineering=self.clip_prompt_eng,
            this_is=self.clip_this_is,
            show=self.visualise_clip)

        self.obj_name_to_id = {}
        for i, x in enumerate(clip_out):
            name = f"{x['category']}_{i}"
            self.obj_name_to_id[name] = x['objID']

        masks = [x['mask'] for x in clip_out]
        objIds = [x['objID'] for x in clip_out]
        categories = [x['category'] for x in clip_out]

        return masks, categories, objIds

    # run inference with GR-ConvNet grasp generator 
    def setup_grasps(self, obj_ids, masks=None, visualise_grasps=None):
        visualise_grasps = visualise_grasps or self.visualise_grasps
        
        rgb, depth, seg = self.env.camera.get_cam_img()    
        img_size = self.grasp_generator.IMG_WIDTH
        if  img_size != self.env.camera.width: 
            rgb = cv2.resize(rgb, (img_size, img_size))
            depth = cv2.resize(depth, (img_size, img_size))  

        # @TODO: Alternatively pass masks from SAM for non groundtruth segm here
        if self.ground_truth_segm:
            masks = [seg == obj_id for obj_id in obj_ids]
        else:
            assert masks is not None

        #for obj_id in self.env.obj_ids:
        for obj_id, mask in zip(obj_ids, masks):
            # mask = seg == obj_id
            if img_size != self.env.camera.width:
                mask = np.array(Image.fromarray(mask).resize((img_size, img_size), Image.LANCZOS))
            if obj_id not in self.env.obj_ids:
                continue
            grasps = self.grasp_generator.predict_grasp_from_mask(rgb,
                                                           depth,
                                                           mask,
                                                           n_grasps=self.n_grasp_attempts, 
                                                           show_output=False
            )
            if obj_id not in self.env.obj_ids:
                continue
            self.env.set_obj_grasps(obj_id, grasps)
        
        if visualise_grasps:
            LID =[]
            for obj_id in obj_ids:
                grasps = self.env.get_obj_grasps(obj_id)
                color = np.random.rand(3).tolist()
                for g in grasps:
                    LID = self.env.draw_predicted_grasp(g,color=color,lineIDs=LID)
            
            time.sleep(5)
            self.env.remove_drawing(LID)
            self.env.dummy_simulation_steps(10)

    def step(self, what, where, how=None):
        """
        what: obj_id
        where: [x, y, z] absolute coordinates
        """

        assert what in self.obj_ids, f"Invalid objID {what}"
        assert isinstance(where, (list, tuple)) and len(where) == 3, \
            f"Invalid placement target: {where}"

        target_loc = list(where)

        success_grasp, success_target = self.env.put_obj_in_loc(what, target_loc)

        # Retry logic (unchanged)
        if success_grasp and success_target:
            success = True
        elif success_grasp:
            for _ in range(self.n_action_attempts):
                self.env.dummy_simulation_steps(10)
                if self.env.place_in_loc(target_loc):
                    success = True
                    break
            else:
                success = False
        else:
            success = False

        self._step()
        return success



    def get_visual_ctx(self):
        return f"""objects = [{', '.join([f'"{name}"' for name in self.clip_names])}, "tray"]"""

    def run(self):
        self.history = self.get_visual_ctx()

        while True:
            user_input = ask_for_user_input()
            if not user_input:
                continue

            # Special commands
            if user_input == ":clear":
                self.history = self.get_visual_ctx()
                self.env.dummy_simulation_steps(10)
                continue
            elif user_input == ":reset":
                self.reset_scene(new=False)
                self.history = self.get_visual_ctx()
                self.env.dummy_simulation_steps(10)
                continue
            elif user_input == ":new":
                self.reset_scene(new=True)
                self.history = self.get_visual_ctx()
                self.env.dummy_simulation_steps(10)
                continue
            elif user_input == ":exit":
                print("Exiting demo")
                self.env.close()
                break

            objects = self.clip_names
            
            try:
                # Use multi-strategy approach with full debug output
                llm_output = llm_generate_plan(
                    objects=objects,
                    instruction=user_input
                )
                
                print("\n" + "="*80)
                print("GENERATED COMMANDS:")
                print("="*80)
                print(highlight(llm_output, PythonLexer(), TerminalFormatter()).strip())
                print("="*80)
                
                
                # Execute
                self.execute_llm_plan(llm_output)
                self._step()

            except Exception as e:
                print(f"\n{'!'*80}")
                print(f"⚠️ EXECUTION FAILED")
                print(f"{'!'*80}")
                print(f"Error: {e}")
                import traceback
                traceback.print_exc()
        

