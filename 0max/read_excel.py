import pandas as pd
import wandb
import os
import time

root = "C:\\Users\\<NAME>\\Desktop\\0max"

# 0.login to wandb
wandb_run = wandb.init(entity="", project="", name="")

i = 0
while True:
    i += 1
    # 1.download the excel form you email
    #######################
    # 2.read the excel file
    df = pd.read_excel(os.path.join(root, 'data.xlsx'))
    # 3.log the data to wandb
    wandb.log(df, step=i) # https://docs.wandb.ai/quickstart/
    # 4.define the period to check for new data
    time.sleep(86400)  # 24h:86400 3h:10800
