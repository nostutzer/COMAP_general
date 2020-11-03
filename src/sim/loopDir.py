import os


for filename in os.listdir("/home/sagittarius/Documents/COMAP_general/COMAP_general/src/sim"):
    if "2" in filename:
        print("Contains 2", filename)
    else:
        print("Does not contain 2", filename)
