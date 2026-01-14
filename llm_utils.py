import os, json, re, math
import matplotlib.pyplot as plt
from huggingface_hub import InferenceClient

# =========================
# LLM SETUP
# =========================
hugging_hub_token = os.getenv("HF_TOKEN")
client = InferenceClient("meta-llama/Meta-Llama-3-8B-Instruct", token=hugging_hub_token)

# =========================
# THE GEOMETRY ENGINE (Python Side)
# =========================
def generate_points(schema, n, params):
    cx, cy = params.get("center_x", 0.0), params.get("center_y", -0.5)
    size = params.get("size", 0.15)
    points = []

    if schema == "circle":
        for i in range(n):
            angle = 2 * math.pi * i / n
            points.append({"x": cx + size * math.cos(angle), "y": cy + size * math.sin(angle)})
            
    elif schema == "square":
        pts_per_side = n / 4
        for i in range(n):
            side = int(i // pts_per_side)
            t = (i % pts_per_side) / pts_per_side
            if side == 0: x, y = cx - size + 2*size*t, cy + size # Top
            elif side == 1: x, y = cx + size, cy + size - 2*size*t # Right
            elif side == 2: x, y = cx + size - 2*size*t, cy - size # Bottom
            else: x, y = cx - size, cy - size + 2*size*t # Left
            points.append({"x": x, "y": y})

    elif schema == "triangle":
        # Distribute points across 3 sides
        pts_per_side = n / 3
        # Vertices of an equilateral triangle
        v = [(cx, cy + size), (cx + size, cy - size), (cx - size, cy - size)]
        for i in range(n):
            side = int(i // pts_per_side)
            t = (i % pts_per_side) / pts_per_side
            p1, p2 = v[side % 3], v[(side + 1) % 3]
            points.append({"x": p1[0] + (p2[0]-p1[0])*t, "y": p1[1] + (p2[1]-p1[1])*t})

    elif schema == "line":
        for i in range(n):
            t = i / (n - 1) if n > 1 else 0.5
            points.append({"x": cx - size + 2*size*t, "y": cy})

    return points

# =========================
# LLM COORDINATOR
# =========================
def llm_generate_coordinates(objects, instruction):
    # We provide the list of objects in the prompt so the LLM can pick from them
    prompt = f"""
    [SYSTEM]
    You are a robotic controller. Select the best schema, parameters, and subset of objects.
    WORKSPACE: X[-0.3, 0.3], Y[-0.8, -0.2]. Center is (0, -0.5).
    
    [AVAILABLE OBJECTS]
    {objects}

    [SCHEMAS]
    - "circle", "square", "triangle", "line"

    [TASK]
    Instruction: "{instruction}"
    
    [OUTPUT]
    Return ONLY a JSON object. 
    "selected_objects" must be a subset of the AVAILABLE OBJECTS list.
    
    Example: {{
      "selected_objects": ["apple", "banana"], 
      "schema": "line", 
      "params": {{"center_x": 0.0, "center_y": -0.5, "size": 0.1}}
    }}
    """

    try:
        response = client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=250
        )
        
        raw_text = response.choices[0].message.content.strip()
        print(f"--- [DEBUG] LLM RESPONSE ---\n{raw_text}")

        start = raw_text.find('{')
        end = raw_text.rfind('}') + 1
        decision = json.loads(raw_text[start:end])
        
        # 1. Use the subset the LLM chose
        selected = decision.get("selected_objects", objects)
        n = len(selected)
        
        # 2. Generate points for ONLY that many objects
        coords = generate_points(decision["schema"], n, decision["params"])
        
        # 3. Map the selected objects to the generated points
        return [{"obj": selected[i], "x": round(c["x"], 3), "y": round(c["y"], 3)} 
                for i, c in enumerate(coords)]

    except Exception as e:
        print(f"[DEBUG] Parser Error: {e}")
        return []

def llm_generate_plan(objects, instruction):
    positions = llm_generate_coordinates(objects, instruction)
    if not positions: return "Error: Could not generate plan."

    # Visualization
    xs, ys = [p['x'] for p in positions], [p['y'] for p in positions]
    plt.figure(figsize=(4,4)); plt.scatter(xs, ys); plt.xlim(-0.4, 0.4); plt.ylim(-0.9, -0.1); plt.show()

    return "\n".join([f'robot.pick_and_place("{p["obj"]}", [{p["x"]}, {p["y"]}, 1.0])' for p in positions])