import pandas as pd
import numpy as np
import random as rd
import copy
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge
from sklearn import preprocessing



##################### Aymeric BERTHOUMIEU #####################

def DataPreparation(data, limitCategoricalNumerical:int=2): ### Aymeric BERTHOUMIEU
    """
    :param data: (pandas DataFrame) data to process
    :param limitCategoricalNumerical: (int) maximal number of values to consider the column as categorical.
    :return: pandas DataFrame with no missing value and categorical variables handled
    """
    workingData = data.copy()
    # Check for missing values
    columnsToFill = data.columns[data.isnull().any()].tolist()  # list of columns containing NaN
    for header in columnsToFill:
        columnType = data[header].dtypes  # type of the data contained in this column
        indicesMissing = data.index[data[header].isnull()].tolist()  # list of indices with value NaN at column header
        uniqueValues = data[header].value_counts() / len(
            data)  # differents values contained in data and their frequency
        if (len(uniqueValues) < limitCategoricalNumerical) or (columnType == str):
            # in that case, we consider we are facing categorical values
            # missing value will be replaced following the statistical distribution of values in this column
            replaceValues = rd.choices(uniqueValues.index, uniqueValues.values,
                                       k=len(indicesMissing))  # pick values following the distribution
            workingData[header][indicesMissing] = replaceValues
        else:
            # in that case, data is considered as numerical
            # missing values will be replaced by mean of the values
            columnMean = data[header].mean(axis=0)
            if columnType == int:
                columnMean(int(round(columnMean)))
            workingData[header][indicesMissing] = [columnMean for i in range(len(indicesMissing))]

    # Handle categorical values
    workingData = pd.get_dummies(workingData)

    return workingData

##################### Bich-Tien PHAN #######################

def feature_selection(df, target, test_split, α=5, thresh=50): ### Bich-Tien PHAN
    """
    :param df: (pandas DataFrame) data to process
    :param target: (string) label name to predict
    :param test_split: (float) percentage of test data
    :param α: (float) ridge regression hyperparameter, the maximal value to determine the optimal value of alpha for ridge regression
    :param thresh: (float) parameter to select the significant features
    :return: list of colums to keep after ridge regression
    """
    #Prepocessing for ridge regression
    X = df.drop(columns=target)
    X = preprocessing.scale(X)
    y = df[target].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_split)

    coefs = []
    error = []
    ridge_reg = Ridge()

    n_alphas = 200
    alphas = np.logspace(-5, α , n_alphas)
    # iterate lambdas
    for alpha in alphas:
        # training
        ridge_reg.set_params(alpha=alpha)
        ridge_reg.fit(X_train, y_train)
        coefs.append(ridge_reg.coef_)
        error.append([alpha, np.mean((ridge_reg.predict(X_test) - y_test) ** 2)])
    final_alpha = min(error, key = lambda t: t[1])[0]
    
    ridge_reg.set_params(alpha=final_alpha)
    ridge_reg.fit(X_train, y_train)
    
    # Here we select the features based on its coefficient determined from ridge regression in comparison to the highest absolute value of coefficient.
    # The threshold determines the features considered. 
    abs_coef = abs(ridge_reg.coef_)
    abs_coef_percent = abs_coef / max(abs_coef) * 100 # Percentage vis-a-vis the highest absolute coefficient 
    get_features = np.where(abs_coef_percent > thresh)[0]
    col_df = list(df.columns)
    list_features = [col_df[i] for i in get_features if col_df[i] != target]
    return list_features

def mean_absolute_percentage_error(y_true, y_pred): ### Bich-Tien PHAN
    """
    :param y_true: (array) true value of label
    :param y_pred: (array) predicted value of label
    :return: (float) mean absolute percentage error
    """
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

##################### Jéhoiakim KINGNE #####################

def separate_predictors_and_label(data_df,label_column_name): ### By Jéhoiakim KINGNE
    """
    :param data_df: (pandas dataframe) data available for prediction
    :param label_column_name: (string) label name
    :return: (2 pandas dataframe) features and label data
    """
    features_columns = [column_name for column_name in data_df.columns if column_name != label_column_name]
    features_data_df, label = data_df[features_columns], data_df[[label_column_name]]
    return features_data_df, label

def split_data_train_test(data_df,label_column_name, test_size=0.3, random_state=42): ### By Jéhoiakim KINGNE
    """
    :param data_df: (pandas dataframe) data available for prediction
    :param label_column_name: (string) label name
    :param test_size: (float) percentage of test data
    :param random_state: (int) random_state
    :return: (2 pandas dataframe) features and label data
    """
    features_data_df, label = separate_predictors_and_label(data_df,label_column_name)
    X_train, X_test, y_train, y_test = train_test_split(features_data_df, label,
                                                        test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

def apply_training(X_train, y_train, model): ### By Jéhoiakim KINGNE
    """
    :param X_train: (pandas dataframe) features data
    :param y_train: (pandas dataframe) label data
    :param model: (object) initial model
    :return: (object) trained model
    """
    model.fit(X_train,y_train)
    return model

def apply_testing(X_test, model): ### By Jéhoiakim KINGNE
    """
    :param X_test: (pandas dataframe) features data
    :param model: (object) trained model
    :return: (array) predicted label
    """
    return model.predict(X_test)

def train_test_stage(data_df, label_column_name, model): ### By Jéhoiakim KINGNE
    """
    :param data_df: (pandas dataframe) features data
    :param label_column_name: (string) label name
    :param model: (object) initial model
    :return: (4 arrays) predicted and true labels for training and testing sets
    """
    X_train, X_test, y_train, y_test = split_data_train_test(data_df,label_column_name)
    model = apply_training(X_train, y_train, model)
    y_test_pred = apply_testing(X_test, model)
    y_train_pred = apply_testing(X_train, model)
    return y_test, y_test_pred, y_train, y_train_pred

def assess_prediction(y_test,y_pred): ### By Jéhoiakim KINGNE
    """
    :param y_test: (array) true labels
    :param y_pred: (array) predicted labels
    :return: (float) root means squared error
    """
    return mean_squared_error(y_test,y_pred,squared=False) #RMSE

# def apply_kfold_cross_validation(data_df, label_column_name, model, kernel_list,
#                                  n_folds=10, scale_data = True, random_state=42, shuffle=True): ### By Jéhoiakim KINGNE
#     X,y = separate_predictors_and_label(data_df,label_column_name)
#     mean_scores = dict()
#     if scale_data:
#         scaler = MinMaxScaler(feature_range=(0, 1))
#         X = pd.DataFrame(scaler.fit_transform(X), columns = X.columns)
#     cross_val = KFold(n_splits=n_folds, random_state=random_state, shuffle=shuffle)
#     for kernel in kernel_list:
#         current_model = model(kernel=kernel)
#         for train_index, test_index in cross_val.split(X):
#             #print("Train Index: ", train_index, "\n")
#             #print("Test Index: ", test_index)
#             X_train, X_test, y_train, y_test = X.iloc[train_index], X.iloc[test_index], y.iloc[train_index], y.iloc[test_index]
#             current_model.fit(X_train, y_train)
#             if kernel in mean_scores:
#                 mean_scores[kernel] += current_model.score(X_test, y_test)/n_folds
#             else:
#                 mean_scores[kernel] = current_model.score(X_test, y_test)/n_folds
#     return mean_scores

def apply_kfold_cross_validation(data_df, label_column_name, model, alpha, test_split, threshold_list,
                                 n_folds=10, scale_data = True, random_state=42, shuffle=True): ### By Jéhoiakim KINGNE
    """
    :param data_df: (pandas dataframe) data available for prediction
    :param label_column_name: (string) label name
    :param model: (object) initial model
    :param alpha: (float) ridge regression hyperparameter
    :param test_split: (float) percentage of test data
    :param threshold_list: (float) parameter to select the significant features
    :param n_folds: (int) number of folds
    :param scale_data: (boolean) boolean value to decide if we normalize the data or not
    :param random_state: (int) random state
    :param shuffle: (boolean) boolean value to decide if we shuffle the data or not
    :param param random_state: (int) random_state
    :return: (2 pandas dataframe) features and label data
    """
    X,y = separate_predictors_and_label(data_df,label_column_name)
    mean_scores = dict()
    if scale_data:
        scaler = MinMaxScaler(feature_range=(0, 1))
        X = pd.DataFrame(scaler.fit_transform(X), columns = X.columns)
    cross_val = KFold(n_splits=n_folds, random_state=random_state, shuffle=shuffle)
    for threshold in threshold_list:
        selected_features = feature_selection(data_df, label_column_name, test_split, α=alpha, thresh=threshold)
        current_model = model
        new_X = X[selected_features]
        for train_index, test_index in cross_val.split(new_X):
            X_train, X_test, y_train, y_test = new_X.iloc[train_index], new_X.iloc[test_index], y.iloc[train_index], y.iloc[test_index]
            current_model.fit(X_train, y_train)
            if threshold in mean_scores:
                mean_scores[threshold] += current_model.score(X_test, y_test)/n_folds
            else:
                mean_scores[threshold] = current_model.score(X_test, y_test)/n_folds
    return mean_scores



# def grid_search_parameters(alpha_list, threshold_list, kernel_list, estimator_list, model_list):
#    params = dict()
#    i=0
#    for alpha in alpha_list:
#        for threshold in threshold_list:
#            for model in model_list:
#                if 'kernel' in model._get_param_names():
#                    for kernel in kernel_list:
#                        params['run_'+str(i)] =




if __name__ == '__main__':
    path_to_data = 'prostate.data'  # path to data
    label_name_to_predict = 'lpsa'  # name of the label to predict
    maximum_categorical_values = 7  # for data preparation, maximal number of possible different values for a features
                                    # to be categorical
    test_split = 0.3  # split between train and test datasets
    linear_regression = True
    alpha = 5  # parameter for Ridge Regression selection, represent 10**alpha. Used only when linear_regression == True

    # import data managing if it is a .csv or .data file
    try:
        my_data = pd.read_csv(path_to_data, sep=r'\t')
    except:
        my_data = pd.read_table(path_to_data, sep=r'\t')

    # preparation of the data (filling Nan, dealing with categorical features ...)
    preprocessed_data = DataPreparation(my_data, limitCategoricalNumerical=maximum_categorical_values)
    X_train, X_test, y_train, y_test = split_data_train_test(preprocessed_data, label_name_to_predict)
    
    #### We will apply the cross validation on this training data ####
    preprocessed_train_data = pd.concat([X_train, y_train], axis=1, sort=False)
    if linear_regression:
        model = LinearRegression()  # defining the model
        threshold_list = [i*5 for i in range(1, 20)]  # list of thresholds to try
        # cross validation in order to find the best threshold
        kfold_mean_scores = apply_kfold_cross_validation(preprocessed_train_data, label_name_to_predict, model, alpha, test_split, threshold_list)
        best_threshold = min(kfold_mean_scores, key=kfold_mean_scores.get)  # selection of the best threshold
        # feature selection
        feature_columns = feature_selection(preprocessed_train_data, label_name_to_predict, test_split, α=alpha, thresh=best_threshold)
        preprocessed_data = preprocessed_data[feature_columns + [label_name_to_predict]]
    else:
        model = RandomForestRegressor() # if the user doesn't want a linear regression, a random forest is applied

    # training and prediction
    y_test, y_test_pred, y_train, y_train_pred = train_test_stage(preprocessed_data, label_name_to_predict, model)

    # computation of metrics
    mse_test_set = assess_prediction(y_test, y_test_pred)
    mse_train_set = assess_prediction(y_train, y_train_pred)

    mape_train_set = mean_absolute_percentage_error(y_train.values.reshape(1,len(y_train))[0], y_train_pred)
    mape_test_set = mean_absolute_percentage_error(y_test.values.reshape(1,len(y_test))[0], y_test_pred)

    print('Root Mean Squared Error on the training set', mse_train_set)
    print('Root Mean Squared Error on the testing set', mse_test_set)

    print('Mean Absolute Percentage Error on the training set', mape_train_set)
    print('Mean Absolute Percentage Error on the testing set', mape_test_set)
