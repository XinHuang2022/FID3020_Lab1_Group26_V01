# FID3020_Lab1_Group26_V01

## Feature Engineering
In the notebook `FeatureEngineering.ipynb` we visualize the data, perform feature engineering (pick out features, remove feature rows with NaN for relevant features, test normalization of the features etc).

For the training of the model, we decided the use the five primary features as:
* 'pclass' (the passenger's class in the Titanic boat)
* 'age' (the passenger's age)
* 'sibsp' (the number of the passenger's siblings and spouse on board)
* 'parch' (the number of the passenger's parents and children on board)
* 'is_male' (a binary variable indicating the passenger's sex, 1 for male and 0 for female).

This choice was made because of the following reasons.
* 'passenger_id' has no predictive power and we hence got rid off it.
* 'survived' is our label, so we had to keep it.
* 'pclass' seemed very informative after looking at the pairplot (see notebook).
* 'name' might contain information, so we decided to extract the most relevant information (that is Mr, Mrs, Miss). However, this is almost the same information as is contained within 'sex' and 'sibsp' and we decided not to keep it.
* 'sex' also seemed very informative after looking at the pairplot (see notebook).
* 'age' also displayed a slight difference between survived and drowned passengers, so we decided to keep it. However, we first removed rows with NaN-values. This might not be optimal as most unknown ages correspond to poor people that did not survive, but we didn't want to assign these people an arbitrary age as this would bias the model.
* 'sibsp' and 'parch' both seemed like interesting features to keep, although they don't look super-informative according to the pairplot.
* 'ticket' did not seem relevant for the prediction we wanted to make. It might contain information related to 'pclass', however, and thus we thought that 'pclass' would be sufficient to keep.
* 'fare' is more informative than 'pclass', but contains similar information about the passengers. For this demonstration, we decided not to keep it as we wanted a simpler model, but we are aware that it could have improve the model and is something we would have liked to investigate more.
* 'cabin' also contains similar information as 'pclass' and 'fare'. We started by extracting only the floor ('A',...,'G','Unknown') but then decided not to keep it as we would have to use one-hot-encoding, creating mulitple new feature columns. This would probably not have been a problem, and keeping this feature is something we would have liked to investigate.
* 'embarked' did not seem to relevant, so we decided to not use it.   

In terms of normalizing the features, we decided not to do it. For some reason, the model performed worse after scaling the feature values, so we decided not to do it.

## Classification model choice
We trained a logistic regression model with the aforementioned features. We used a testing set with 92 passengers, out of which the model predicted correctly on 43 True Survival cases and 37 True Not-survived cases, with an accuracy around (43+37)/92 = 0.87. The confusion matrix for the testing set is available in the root directory of this repository with the name "confusion_matrix.png".

As shown in the notebook `FeatureEngineering.ipynb` (towards the end) we also did some tests using a RandomForestClassifier. It performed better than the LogisticRegression and could have been implemented into the training pipeline for a better model. 

## Source codes for the Serverless ML with Titanic dataset
In the file `feature-pipeline.py`, a feature pipeline based on the cleaned Titanic dataset is built and is available through a Hopsworks Feature Store.

In the file `training-pipeline.py`, a binary classifier model is trained for predicting the survival status of a passenger on the Titanic based on the passenger's specific information. See above for the feature engineering methods applied.

Under the folder **hugging-face-titanic**, the file `app.py` downloads our model from Hopsworks, and provides a User Interface to allow users to enter feature values to predict if a passenger with the provided features would survive or not.

In the file `feature_daily.py`, we define a synthetic passenger data generator, and our feature pipeline is updated to include also the new synthetic passengers.

In the file `batch-inference-pipeline.py`, a batch inference pipeline is built to predict if the synthetic passengers survived or not.

Under the folder **huggingface-spaces-titanic-monitor**, the file named `app.py` builds a Gradio application to show the most recent synthetic passenger prediction and outcome, together with a confusion matrix summarizing historical prediction performance.

## Links to UI

The two public URLs with gradi UI on Huggingface space are as follows:

1. Interactive UI for entering feature values and predicting if a passenger would survive the titanic or not:

https://c8347bed360d308a.gradio.app

2. Dashboard UI showing a prediction of survival for the most recent passenger added to the Feature Store,
    and the outcome (label) if that passenger survived or not:

https://6870c2a1d159526b.gradio.app
