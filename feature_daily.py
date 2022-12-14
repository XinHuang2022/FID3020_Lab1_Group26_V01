import os
import modal
import random

LOCAL=True

if LOCAL == False:
    stub = modal.Stub()
    image = modal.Image.debian_slim().pip_install(["hopsworks==3.0.4"]) 
    #image = modal.Image.debian_slim().apt_install(["libgomp1"]).pip_install(["hopsworks", "seaborn", "joblib", "scikit-learn"])
    @stub.function(image=image, schedule=modal.Period(days=1), secret=modal.Secret.from_name("scalableML"))
    def f():
        g()


# def empirical_family_members(feature):
    # """
    # These distributions are valid for both passengers that survived and for the group of passengers that died.
    # """ 
    
    # random_nr = random.uniform(0, 1)
    
    # if feature == 'sibSp':
    
        # # Empirically we know:
        # if random_nr >= 0 and random_nr <= 0.659664:
            # return 0
        # if random_nr >= 0.659664 and random_nr <= 0.915966:
            # return 1
        # if random_nr >= 0.915966 and random_nr <= 0.950980:
            # return 2
        # if random_nr >= 0.950980 and random_nr <= 0.967787:
            # return 3
        # if random_nr >= 0.967787 and random_nr <= 0.992997:
            # return 4
        # if random_nr >= 0.992997 and random_nr <= 1.0:
            # return 5
    
    # elif feature == 'parch':
        
        # # Empirically we know:
        # if random_nr >= 0 and random_nr <= 0.729692:
            # return 0
        # if random_nr >= 0.729692 and random_nr <= 0.883754:
            # return 1
        # if random_nr >= 0.883754 and random_nr <= 0.978992:
            # return 2
        # if random_nr >= 0.978992 and random_nr <= 0.985994:
            # return 3
        # if random_nr >= 0.985994 and random_nr <= 0.991597:
            # return 4
        # if random_nr >= 0.991597 and random_nr <= 0.998599:
            # return 5
        # if random_nr >= 0.998599 and random_nr <= 1.0:
            # return 6
# def empirical_family_members(feature):
    
    # if feature == 'sibSp':
    



def generate_passenger(status, titanic_df):
    """
    Returns a synthetic passenger
    """
    import pandas as pd
    import random
    import numpy as np

    # df = pd.DataFrame({"pclass": [random.randint(low_class,higher_class)],
                       # "age": [random.randint(age_min,age_max)*0.5],
                       # "sibSp": [empirical_family_members('sibSp')],
                       # "parch": [empirical_family_members('parch')],
                       # "is_male": [random.randint(0,1)]
                      # })
                      
    survived_df = titanic_df.loc[titanic_df['survived'] == 1]
    not_survived_df = titanic_df.loc[titanic_df['survived'] == 0]

    N_survive = survived_df.shape[0]
    N_not_survive = not_survived_df.shape[0]
    
    if status == 'survived':
        rand_index_1 = np.random.randint(0, N_survive)
        rand_passenger_pclass = survived_df['pclass'].iat[rand_index_1]
        rand_index_2 = np.random.randint(0, N_survive)
        rand_passenger_age = survived_df['age'].iat[rand_index_2]
        rand_index_3 = np.random.randint(0, N_survive)
        rand_passenger_sibsp = survived_df['sibsp'].iat[rand_index_3]
        rand_index_4 = np.random.randint(0, N_survive)
        rand_passenger_parch = survived_df['parch'].iat[rand_index_4]
        rand_index_5 = np.random.randint(0, N_survive)
        rand_passenger_is_male = survived_df['is_male'].iat[rand_index_5]
        df = pd.DataFrame({"pclass": [rand_passenger_pclass],
                           "age": [rand_passenger_age],
                           "sibsp": [rand_passenger_sibsp],
                           "parch": [rand_passenger_parch],
                           "is_male": [rand_passenger_is_male]
                          })
        df['survived'] = 1
    elif status == 'drowned':
        rand_index_1 = np.random.randint(0, N_not_survive)
        rand_passenger_pclass = not_survived_df['pclass'].iat[rand_index_1]
        rand_index_2 = np.random.randint(0, N_not_survive)
        rand_passenger_age = not_survived_df['age'].iat[rand_index_2]
        rand_index_3 = np.random.randint(0, N_not_survive)
        rand_passenger_sibsp = not_survived_df['sibsp'].iat[rand_index_3]
        rand_index_4 = np.random.randint(0, N_not_survive)
        rand_passenger_parch = not_survived_df['parch'].iat[rand_index_4]
        rand_index_5 = np.random.randint(0, N_not_survive)
        rand_passenger_is_male = not_survived_df['is_male'].iat[rand_index_5]
        df = pd.DataFrame({"pclass": [rand_passenger_pclass],
                           "age": [rand_passenger_age],
                           "sibsp": [rand_passenger_sibsp],
                           "parch": [rand_passenger_parch],
                           "is_male": [rand_passenger_is_male]
                          })
        df['survived'] = 0
    return df


def get_random_passenger():
    """
    Returns a DataFrame containing one random passenger
    """
    import pandas as pd
    import random

    titanic_df = pd.read_csv("https://raw.githubusercontent.com/XinHuang2022/FID3020_Lab1_Group26_V01/main/titanic_new_V2.csv")

    survived_df = generate_passenger("survived", titanic_df)
    drowned_df = generate_passenger("drowned", titanic_df)

    # randomly pick one of these 2 and write it to the featurestore
    pick_random = random.uniform(0,2)
    if pick_random >= 1:
        titanic_df = survived_df
        print("Survived passenger added")
    else:
        titanic_df = drowned_df
        print("Drowned passenger added")

    return titanic_df


def g():
    import hopsworks
    import pandas as pd

    project = hopsworks.login()
    fs = project.get_feature_store()

    titanic_df = get_random_passenger()

    titanic_fg = fs.get_feature_group(name="titanic_modal",version=1)
    titanic_fg.insert(titanic_df, write_options={"wait_for_job" : False})

if __name__ == "__main__":
    if LOCAL == True :
        g()
    else:
        with stub.run():
            f()
