import os, json, re, math
import matplotlib.pyplot as plt
from huggingface_hub import InferenceClient
from openai import OpenAI

# =========================
# LLM SETUP
# =========================
hugging_hub_token = os.getenv("HF_TOKEN")
OPENAI_KEY = os.getenv("OPENAI_KEY")

def get_client(model_name):
    if model_name in ["meta-llama/Meta-Llama-3-8B-Instruct", "mistralai/Mistral-7B-Instruct-v0.2"]:
        client = InferenceClient(model_name, token=hugging_hub_token)

    elif model_name == "gpt-3.5-turbo":
        client = OpenAI(api_key=os.getenv("OPENAI_KEY"))
    
    return client


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
            points.append({
                "x": cx + size * math.cos(angle), 
                "y": cy + size * math.sin(angle)
            })
            
    elif schema == "square":
        # Evenly distribute points around perimeter
        perimeter = 4 * (2 * size)  # Total perimeter length
        for i in range(n):
            # Distance along perimeter for this point
            dist = (i / n) * perimeter
            
            if dist < 2 * size:  # Top edge
                x = cx - size + dist
                y = cy + size
            elif dist < 4 * size:  # Right edge
                x = cx + size
                y = cy + size - (dist - 2 * size)
            elif dist < 6 * size:  # Bottom edge
                x = cx + size - (dist - 4 * size)
                y = cy - size
            else:  # Left edge
                x = cx - size
                y = cy - size + (dist - 6 * size)
            
            points.append({"x": x, "y": y})

    elif schema == "triangle":
        # Equilateral triangle vertices
        h = size * math.sqrt(3)  # Height from center to vertex
        vertices = [
            (cx, cy + h),                           # Top
            (cx + size * 1.5, cy - h/2),           # Bottom-right
            (cx - size * 1.5, cy - h/2)            # Bottom-left
        ]
        
        # Calculate side length
        side_length = 3 * size
        perimeter = 3 * side_length
        
        for i in range(n):
            # Distance along perimeter
            dist = (i / n) * perimeter
            
            # Determine which edge and position along it
            if dist < side_length:  # Edge 0 -> 1
                t = dist / side_length
                p1, p2 = vertices[0], vertices[1]
            elif dist < 2 * side_length:  # Edge 1 -> 2
                t = (dist - side_length) / side_length
                p1, p2 = vertices[1], vertices[2]
            else:  # Edge 2 -> 0
                t = (dist - 2 * side_length) / side_length
                p1, p2 = vertices[2], vertices[0]
            
            points.append({
                "x": p1[0] + (p2[0] - p1[0]) * t,
                "y": p1[1] + (p2[1] - p1[1]) * t
            })

    elif schema == "line":
        for i in range(n):
            t = i / (n - 1) if n > 1 else 0.5
            points.append({
                "x": cx - size + 2 * size * t, 
                "y": cy
            })
    
    elif schema == "grid":
        # Create a grid layout
        rows = int(math.sqrt(n))
        cols = math.ceil(n / rows)
        
        idx = 0
        for row in range(rows):
            for col in range(cols):
                if idx >= n:
                    break
                x = cx - size + (2 * size * col / (cols - 1) if cols > 1 else 0)
                y = cy - size + (2 * size * row / (rows - 1) if rows > 1 else 0)
                points.append({"x": x, "y": y})
                idx += 1

    return points

# =========================
# LLM COORDINATOR (WITH METRICS TRACKING)
# =========================
def llm_generate_coordinates(objects, instruction, model_name):
    """
    Generate coordinates using LLM.
    Returns: (coordinates, decision_metadata)
    """
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
    "selected_objects" must be a subset of the AVAILABLE OBJECTS list. You MUST select ALL AVAILABLE and ONLY objects that MATCH the subset requirement.
    Do NOT include any object whose attributes does not match.
    
    Example for prompt "Make a line with red items": {{
      "selected_objects": ["apple", "apple", "TomatoSoupCan"], 
      "schema": "line", 
      "params": {{"center_x": 0.0, "center_y": -0.5, "size": 0.1}}
    }}

    DO NOT RETURN EMPTY VALUES
    """

    client =  get_client(model_name)

    try:
        if model_name in  ["meta-llama/Meta-Llama-3-8B-Instruct", "mistralai/Mistral-7B-Instruct-v0.2"] :
            response = client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=400
            )
            raw_text = response.choices[0].message.content.strip()
        else:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a robotic controller. Return ONLY JSON as instructed."},
                    {"role": "user", "content": prompt}
                ],
            )
            raw_text = response.choices[0].message.content.strip()
        
        
        print(f"--- [DEBUG] LLM RESPONSE ---\n{raw_text}")

        start = raw_text.find('{')
        end = raw_text.rfind('}') + 1
        decision = safe_json_loads(raw_text[start:end])

        params = decision.get("params", {})

        # Normalize radius -> size
        if "radius" in params and "size" not in params:
            params["size"] = params["radius"]
            
        decision["params"] = params
        
        # 1. Use the subset the LLM chose
        selected = decision.get("selected_objects", objects)
        n = len(selected)
        
        # 2. Generate points for ONLY that many objects
        coords = generate_points(decision["schema"], n, decision["params"])

        plt.figure(figsize=(6,6))
        plt.scatter([p['x'] for p in coords], [p['y'] for p in coords], s=200, c='blue', alpha=0.6, edgecolors='black')
        for i, p in enumerate(coords): 
            plt.annotate(f"{i}: {selected[i]}", (p['x'], p['y']), 
                        textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)

        # Draw workspace bounds
        plt.axhline(y=-0.8, color='red', linestyle='--', alpha=0.3)
        plt.axhline(y=-0.2, color='red', linestyle='--', alpha=0.3)
        plt.axvline(x=-0.3, color='red', linestyle='--', alpha=0.3)
        plt.axvline(x=0.3, color='red', linestyle='--', alpha=0.3)
        plt.plot(0, -0.5, 'r+', markersize=15)  # Center marker

        plt.xlim(-0.4, 0.4)
        plt.ylim(-0.9, -0.1)
        plt.grid(alpha=0.3)
        plt.title(f"Schema: {decision['schema']}")
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.gca().set_aspect('equal')
        plt.tight_layout()
        plt.show()
        
        # 3. Map the selected objects to the generated points
        result = [{"obj": selected[i], "x": round(c["x"], 3), "y": round(c["y"], 3)} 
                for i, c in enumerate(coords)]
        
        # 4. Return both coordinates and decision metadata for metrics tracking
        return result, decision

    except Exception as e:
        print(f"[DEBUG] Parser Error: {e}")
        return [], None

def llm_generate_plan(objects, instruction, model_name):
    client = get_client(model_name)
    
    positions, decision = llm_generate_coordinates(objects, instruction, model_name)
    if not positions: 
        return "Error: Could not generate plan.", None

    plan = "\n".join([f'robot.pick_and_place("{p["obj"]}", [{p["x"]}, {p["y"]}, 1.0])' for p in positions])
    
    return plan, decision


def safe_json_loads(s: str):
    # remove // comments
    s = re.sub(r'//.*$', '', s, flags=re.MULTILINE)
    # remove trailing commas
    s = re.sub(r',\s*([}\]])', r'\1', s)
    return json.loads(s)