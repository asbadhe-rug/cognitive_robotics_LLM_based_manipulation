from ui import RobotEnvUI

models = ["meta-llama/Meta-Llama-3-8B-Instruct","gpt-3.5-turbo", "mistralai/Mistral-7B-Instruct-v0.2" ]
selected_objects = ["TennisBall", "Strawberry", "Strawberry", "FoamBrick", "Strawberry", "Strawberry", "Pear", "Pear"]
inputs = [
    # Basic geometric arrangements
    "Place all objects in a straight line along the X-axis.",
    "Arrange the objects in a circle around the center of the table.",
    "Form a triangle using the first three objects.",
    "Make a square with the first four objects.",
    
    # Semantic differentiation
    "Put all red fruits in a circle and the rest in a line.",
    "Place strawberries together in a small cluster and pears on the left.",
    "Put all cans in a line and the fruits in a triangle.",
    "Group objects by color: red objects in a circle, yellow objects in a square.",
    
    # Mixed but feasible instructions
    "Line up the first five objects from left to right along the table.",
    "Cluster all strawberries near the center in a small circle.",
    "Distribute the pears evenly in a square formation on the right side.",
    
    # Minimal subset selection
    "Select any three red objects and form a triangle.",
    "Choose any four objects and make a square in the workspace."
]

if selected_objects:
    n_objects = len(selected_objects)
else:
    n_objects = 8

for model in models:
    for inp in inputs:
        demo = RobotEnvUI(n_objects, model, user_input=inp, selected_objects=selected_objects, visualise_clip=False)
        demo.run()
