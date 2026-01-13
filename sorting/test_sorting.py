# sorting/test_sorting.py

from sort_policy import generate_sort_plan

# Example object list from the PyBullet environment
objects = [
    "banana", "apple", "hammer", "screwdriver", "glass_cup", "orange"
]

instruction = "Put all red fruits in basket_A and all tools in basket_B"

plan = generate_sort_plan(objects, instruction)
print("Sorting Plan:")
for step in plan:
    print(step)

# Expected:
# robot.pick_and_place("apple", "basket_A")
# robot.pick_and_place("banana", "basket_A")
# robot.pick_and_place("hammer", "basket_B")
# robot.pick_and_place("screwdriver", "basket_B")
