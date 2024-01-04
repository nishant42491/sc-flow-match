'''import os
import  shutil

for i in os.listdir("logs/train/runs"):
    for j in os.listdir("logs/train/runs/"+i):
        for k in os.listdir("logs/train/runs/"+i+"/"+j):
            if "wandb" in k:
                shutil.rmtree("logs/train/runs/"+i+"/"+j+"/"+k)'''