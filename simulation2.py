from ui import RobotEnvUI

models = ["meta-llama/Meta-Llama-3-8B-Instruct","gpt-3.5-turbo", "mistralai/Mistral-7B-Instruct-v0.2" ]
selected_objects = ["TomatoSoupCan", "ChipsCan", "CrackerBox", "FoamBrick", "MasterChefCan", "MustardBottle", "Pear", "Pear"]
inputs_level_1 = [
    "Make a circle", "Arrange all objects to create a square", "Form a triangle"
]

object_config = [["Strawberry", "Strawberry", "Pear", "Pear", "Strawberry"], ["Strawberry", "Strawberry", "Pear", "Pear", "Strawberry"], 
["Strawberry", "Strawberry", "Pear", "Pear"], ["Strawberry", "Strawberry", "Pear", "Pear", "Strawberry"], ["Strawberry", "Strawberry", "Pear"], ["Strawberry", "Strawberry", "Pear", "Pear"]
]

object_config2 = [["Pear, Strawberry, Pear, Strawberry, Pear, TennisBall, TennisBall"],
                 ["TennisBall, TennisBall, TennisBall, TennisBall, Pear, Strawberry"],
                 ["Banana, Banana, Banana, MustardBottle, Strawberry"],
                 ["Pear, Pear, Pear, Strawberry, Strawberry, Strawberry, TennisBall, TomatoSoupCan"],
                 ["Pear, Strawberry, Banana, TennisBall, TennisBall"],
                 ["Strawberry, Strawberry, Strawberry, Pear, Pear, Pear, Banana, Hammer"],
                ]


if selected_objects:
    n_objects = len(selected_objects)
else:
    n_objects = 8

for model in models:
    i = -1
    for inp in inputs_level_1:
        i+=1
        demo = RobotEnvUI(n_objects, model, user_input=inp, selected_objects=object_config[i], visualise_clip=False)
        demo.run()

        demo = RobotEnvUI(n_objects, model, user_input=inp, selected_objects=object_config[i+1], visualise_clip=False)
        demo.run()
        i+=1
