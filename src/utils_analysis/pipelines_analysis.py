from src.utils_analysis.config import *

def eval_prophet(df, data_type, data_folder, csv_filename='model_accuracy.csv', tgt_category='charge_disposition_cat', ytrue='case_count', yhat='predicted_case_count'):
    """
    evaluate prediction results by group with MAPE
    """
    logging.info(f"eval_prophet() Evaluating {data_type} predictions with MAPE")

    # group the dataframe by category such as Dismissals and Guilty Plea
    grouped = df.groupby(tgt_category)
    # columns having the desired target and prediction data
    cols = [ytrue, yhat]

    logging.info("Scaling data for consistent error comparisons")

    # initialize empty data struct to hold results
    results = []

    for name, group in grouped:
        # scale the target and prediction values
        scaled = pd.DataFrame(scaler.fit_transform(group[cols]), columns=cols).reset_index(drop=True)
        # extract values as lists, excluding the very first entry
        y_true = scaled[ytrue].values[1:]
        y_pred = scaled[yhat].values[1:]
        # compare lists in MAPE
        error = MSE(y_true, y_pred)
        logging.info('Model Accuracy for {} is {:.2f}'.format(name, error))
        # store current results in dictionary, then add to a list of results
        result = {'Category': name, 'MSE': error}
        results.append(result)

    # assemble results into a dataframe and export it
    results = pd.DataFrame(results)
    data_file = os.sep.join([data_folder, csv_filename])
    results.to_csv(data_file, index=False)
    logging.info(f'eval_prophet() Wrote to disk {data_file}')
