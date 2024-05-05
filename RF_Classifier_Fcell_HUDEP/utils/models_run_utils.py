import numpy as np
import pandas as pd


def extract_features(df):
    # Assuming the features start from the 13th column (index 13)
    #features_cols = df.columns.to_list()[12:]
    #X = df[features_cols]
    features_cols = df.columns.to_list()[6:]
    return df[features_cols]

def make_predictions(models_dict, X):
    if X.size == 0:
        # Handle the case when the DataFrame is empty
        return {}, {}, {}

    model_predictions = {}
    for model_name, model in models_dict.items():
        # Use the model to make predictions on X
        prediction = model.predict(X)
        #probability = model.predict_proba(X)
        # Store the prediction in the dictionary
        model_predictions[model_name] = prediction
    

    # Dictionary of model probabilities
    model_probabilities = {}
    for model_name, model in models_dict.items():
        prediction = model.predict(X)
        probability = model.predict_proba(X)

        # Calculate selected probabilities
        selected_probability = []
        for i, pred_class in enumerate(prediction):
            if pred_class == 0:
                prob_value = 1 - probability[i][0]
            else:
                prob_value = probability[i][1]
            selected_probability.append(prob_value)

            # Store the probabilities in the dictionary
        model_probabilities[model_name] = selected_probability
    return model_predictions, model_probabilities
    
# Average Predictions for each compounds
def create_prediction_dataframe(df, model_probabilities):

    # Extract the desired columns from the input file DataFrame
    prediction_df = df[['DT#', 'PA#', 'LK', 'SM']].copy()

    # Calculate the average positional probability and standard deviation for all models
    # and store them in 'Predicted Active' and 'Predicted Active Std Dev' columns respectively
    averages = []

    for i in range(len(df)):
        probs = [model_probabilities[model][i] for model in model_probabilities]
        avg_prob = np.mean(probs)
        averages.append(avg_prob)


    prediction_df['Average Predicted Probability'] = averages
    prediction_df['Predicted Active'] = prediction_df['Average Predicted Probability'].apply(lambda x: 'NO' if x < 0.3 else 'YES')
    #prediction_df.drop(columns=['Average Predicted Probability Mean'], inplace=True)
    return prediction_df

