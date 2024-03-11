import os

crib_cache = {}

def get_crib_code(exercise_num):
    filename = f"src/cribs/Ex_{exercise_num}.txt"
    if exercise_num not in crib_cache:
        if not os.path.exists(filename):
            return "pass"
        with open(filename, 'r') as f:
            crib_cache[exercise_num] = f.read()
    return crib_cache[exercise_num]
