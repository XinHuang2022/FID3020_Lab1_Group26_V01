import os
import modal
import hopsworks
import pandas as pd

project = hopsworks.login()
fs = project.get_feature_store()

titanic_df = pd.read_csv("https://raw.githubusercontent.com/XinHuang2022/FID3020_Lab1_Group26_V01/main/titanic_new_V2.csv")
 
titanic_fg = fs.get_or_create_feature_group(
    name="titanic_modal",
    version=1,
    primary_key=["pclass","age","sibSp","parch","is_male"],
    description="Titanic Survival dataset")

titanic_fg.insert(titanic_df, write_options={"wait_for_job" : False})

print('Done')