from ui import RobotEnvUI

models = ["meta-llama/Meta-Llama-3-8B-Instruct","gpt-3.5-turbo", "mistralai/Mistral-7B-Instruct-v0.2" ]
selected_objects = ["TennisBall", "Strawberry", "Strawberry", "Hammer", "Strawberry", "Strawberry", "Pear", "Pear"]
if selected_objects:
    n_objects = len(selected_objects)
else:
    n_objects = 8
demo = RobotEnvUI(n_objects, models[1], selected_objects=selected_objects, visualise_clip=False)

demo.run()