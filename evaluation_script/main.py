import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, explained_variance_score, r2_score

def evaluate(test_annotation_file, user_submission_file, phase_codename, **kwargs):
    # Load the ground truth labels and predictions into pandas dataframes
    ground_truth = pd.read_csv(test_annotation_file)
    prediction = pd.read_csv(user_submission_file)

    # Merge the two dataframes on the "id" and "word" columns
    merged_df = pd.merge(ground_truth, prediction, on=['Item_Identifier', 'Outlet_Identifier'])

    # define independent dan dependent variables
    # X = merged_df['independent_var1']
    y_true = merged_df['Item_Outlet_Sales22']
    y_pred = merged_df['Item_Outlet_Sales']

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)
    evs = explained_variance_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    output = {}
    if phase_codename == "test":
        print("Evaluating for Test Phase")
        output["result"] = [
            {
                "test_split": {
                    'Mean-absolute-error': mae,
                    'Mean-squared-error' : mse,
                    'Root-mean-squared-error' : rmse,
                    'R-squared' : r2,
                    'Explained-variance-score' : evs,
                    'Mean-absolute-percentage-error(%)' : mape,
                }
            }
        ]
        # To display the results in the result file
        output["submission_result"] = output["result"][0]["test_split"]
        print("Completed evaluation for Test Phase")
    return output
