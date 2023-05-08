import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, explained_variance_score, r2_score
import statsmodels.api as sm
from scipy.stats import linregress

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

    # y_mean=np.mean(y_true)
    # TSS = np.sum((y_true - y_mean)**2)
    # RSS = np.sum((y_true - y_pred)**2)
    # r2 = 1 - (RSS/TSS)

    r2 = sm.OLS(y_true, y_pred).fit().rsquared
    slope, intercept, r_value, p_value, std_err = linregress(y_true, y_pred)
    evs = r_value ** 2
    
    # evs = explained_variance_score(y_true, y_pred)
    # mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    wa = (mae * 0.2) + (mse * 0.3) + (rmse * 0.3) + (r2 * 0.1) + (evs * 0.1)

    output = {}
    if phase_codename == "dev":
        print("Evaluating for Test Phase")
        output["result"] = [
            {
                "test_split": {
                    'Mean-absolute-error': mae,
                    'Mean-squared-error' : mse,
                    'Root-mean-squared-error' : rmse,
                    'R-squared' : r2,
                    'Explained-variance-score' : evs,
                    'Average' : wa,
                }
            }
        ]
        # To display the results in the result file
        output["submission_result"] = output["result"][0]["test_split"]
        print("Completed evaluation for Test Phase")
    return output
