from ui import RobotEnvUI

n_objects = 8
clone_name = "tennis ball"
demo = RobotEnvUI(n_objects, clone_name="TennisBall", visualise_clip=False)

demo.run()