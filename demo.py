from ui import RobotEnvUI
n_objects = 4
specific_objects = ['Banana', 'Strawberry', 'Banana', 'Strawberry']
demo = RobotEnvUI(n_objects, selected_objects=specific_objects, visualise_clip=False)

demo.run()