import os, random, json, numpy as np
from ui import RobotEnvUI
from llm_utils import llm_generate_coordinates

# =========================
# CONFIG
# =========================
LLMS = [
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.2",
    "gpt-3.5-turbo"
]

SHAPES = ["circle", "square", "triangle", "line"]
NUM_TRIALS = 5

X_BOUNDS = (-0.3, 0.3)
Y_BOUNDS = (-0.8, -0.2)
OBJECT_POOL = ["TennisBall", "Strawberry", "Hammer", "Banana", "Pear", "MustardBottle"]

RESULTS_FILE = "simulation_results.json"

# =========================
# HELPER FUNCTIONS
# =========================
def llm_output_consistency(coord_lists):
    """Calculate average std deviation across x and y for multiple trials"""
    if not coord_lists:
        return None
    n_objects = len(coord_lists[0])
    xs, ys = [[] for _ in range(n_objects)], [[] for _ in range(n_objects)]
    for trial in coord_lists:
        for i, c in enumerate(trial):
            xs[i].append(c["x"])
            ys[i].append(c["y"])
    stds = [np.mean([np.std(xs[i]), np.std(ys[i])]) for i in range(n_objects)]
    return np.mean(stds)

def save_results(results):
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=4)

# =========================
# INIT STORAGE
# =========================
if os.path.exists(RESULTS_FILE):
    with open(RESULTS_FILE, "r") as f:
        results = json.load(f)
else:
    results = {
        "metrics": {llm: {shape: [] for shape in SHAPES} for llm in LLMS},
        "per_object": {obj: {"placement": [], "grasp": []} for obj in OBJECT_POOL},
        "summary": {}
    }

# =========================
# TEST LOOP
# =========================
for llm_name in LLMS:
    print(f"\n=== Testing LLM: {llm_name} ===")
    
    for shape in SHAPES:
        print(f"\n--- Shape: {shape} ---")
        coords_across_trials = []
        
        for trial in range(NUM_TRIALS):
            print(f"Trial {trial+1}/{NUM_TRIALS}")
            
            # Select objects
            selected_objects = random.sample(OBJECT_POOL, 4)
            n_objects = len(selected_objects)
            
            # Initialize simulation
            env = RobotEnvUI(n_objects, llm_name, selected_objects=selected_objects, visualise_clip=False)
            
            # Ask LLM for coordinates
            instruction = f"Arrange these objects in a {shape} pattern"
            coords = llm_generate_coordinates(selected_objects, instruction, llm_name)
            coords_across_trials.append(coords)
            
            # LLM success = all coordinates within bounds
            llm_success = all(X_BOUNDS[0] <= c["x"] <= X_BOUNDS[1] and
                              Y_BOUNDS[0] <= c["y"] <= Y_BOUNDS[1]
                              for c in coords)
            
            # Execute plan
            placement_success_count = 0
            grasp_success_count = 0
            
            for c in coords:
                try:
                    success = env.pick_and_place(c["obj"], [c["x"], c["y"], 1.0])
                    placement_success_count += int(success)
                    grasp_success_count += int(success)
                    results["per_object"][c["obj"]]["placement"].append(int(success))
                    results["per_object"][c["obj"]]["grasp"].append(int(success))
                except Exception as e:
                    success = False
            
            placement_success_rate = placement_success_count / n_objects
            grasp_success_rate = grasp_success_count / n_objects
            
            # Calculate LLM consistency up to this trial
            consistency = llm_output_consistency(coords_across_trials)
            
            # Store trial metrics
            trial_metrics = {
                "trial": trial + 1,
                "llm_success": llm_success,
                "placement_success": placement_success_rate,
                "grasp_success": grasp_success_rate,
                "coords": coords,
                "llm_consistency": consistency
            }
            
            results["metrics"][llm_name][shape].append(trial_metrics)
            
            # Save after each trial
            save_results(results)
            
            env.reset_scene(new=True)
            env.env.close()

# =========================
# FINAL SUMMARY
# =========================
for llm_name in LLMS:
    results["summary"][llm_name] = {}
    for shape in SHAPES:
        trials = results["metrics"][llm_name][shape]
        if not trials: continue
        mean_placement = np.mean([t["placement_success"] for t in trials])
        mean_grasp = np.mean([t["grasp_success"] for t in trials])
        mean_llm = np.mean([t["llm_success"] for t in trials])
        consistency = trials[0]["llm_consistency"]
        
        results["summary"][llm_name][shape] = {
            "placement_success": mean_placement,
            "grasp_success": mean_grasp,
            "llm_success": mean_llm,
            "llm_consistency": consistency
        }

save_results(results)
print("\n✅ All metrics saved incrementally to", RESULTS_FILE)
