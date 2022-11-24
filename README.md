# FID3020_Lab1_Group26_V01
Source codes for the Serverless ML with Titanic dataset

In the file "feature-pipeline.py", a feature pipeline based on the cleaned Titanic dataset is built and is available through a Hopsworks Feature Store.

In the file "training-pipeline.py", a binary classifier model is trained for predicting the survival status of a passenger on the Titanic based on the passenger's specific information. For the training of the model, we take the five primary features as: 'pclass' (the passenger's class in the Titanic boat), 'age' (the passenger's age), 'sibsp' (the number of the passenger's siblings and spouse on board), 'parch' (the number of the passenger's parents and children on board), and 'is_male' (a binary variable indicating the passenger's sex, 1 for male and 0 for female).
Specifically we trained a logistic regression model with the rescaled data, and on a testing set with 92 passengers, the model predicted correctly on 43 True Survival cases and 37 True Not-survived cases, with an accuracy around (43+37)/92 = 0.87. The confusion matrix for the testing set is available in the root directory of this repository with the name "confusion_matrix.png".

Under the folder "hugging-face-titanic", the file "app.py" downloads our model from Hopsworks, and provides a User Interface to allow users to enter or select feature values to predict if a passenger with the provided features would survive or not.

In the file "feature_daily.py", we define a synthetic passenger data generator, and our feature pipeline is updated to include also the new synthetic passengers.

In the file "batch-inference-pipeline.py", a batch inference pipeline is built to predict if the synthetic passengers survived or not. 

Under the folder "huggingface-spaces-titanic-monitor", the file named "app.py" builds a Gradio application to show the most recent synthetic passenger prediction
and outcome, together with a confusion matrix summarizing historical prediction performance.

The two public URLs with gradi UI on Huggingface space are as follows:

1. Interactive UI for entering feature values and predicting if a passenger would survive the titanic or not:

https://c8347bed360d308a.gradio.app


2. Dashboard UI showing a prediction of survival for the most recent passenger added to the Feature Store,
    and the outcome (label) if that passenger survived or not:

https://6870c2a1d159526b.gradio.app

