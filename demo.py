from ui import RobotEnvUI


selected_objects = ["TennisBall", "Strawberry", "Strawberry", "Hammer", "Strawberry", "Strawberry", "Banana", "Banana"]
if selected_objects:
    n_objects = len(selected_objects)
else:
    n_objects = 8
demo = RobotEnvUI(n_objects, selected_objects=selected_objects, visualise_clip=False)

demo.run()