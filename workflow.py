import pandas as pd
import numpy as np
import random as rd
import copy
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
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

def feature_selection(df, target, test_split, α=50, thresh=50):
    X = df.drop(columns=target)
    X = preprocessing.scale(X)
    y = df[target].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_split)

    ridge_reg = Ridge(alpha=α)
    ridge_reg.fit(X_train, y_train)
    abs_coef = abs(ridge_reg.coef_)
    abs_coef_percent = abs_coef / max(abs_coef) * 100
    get_features = np.where(abs_coef_percent > thresh)[0]
    col_df = list(df.columns)
    list_features = [col_df[i] for i in get_features]
    return list_features

##################### Jéhoiakim KINGNE #####################

def separate_predictors_and_label(data_df,label_column_name): ### By Jéhoiakim KINGNE
    features_columns = [column_name for column_name in data_df.columns if column_name != label_column_name]
    features_data_df, label = data_df[features_columns], data_df[[label_column_name]]
    return features_data_df, label

def split_data_train_test(data_df,label_column_name, test_size=0.3, random_state=42): ### By Jéhoiakim KINGNE
    features_data_df, label = separate_predictors_and_label(data_df,label_column_name)
    X_train, X_test, y_train, y_test = train_test_split(features_data_df, label,
                                                        test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

def apply_training(X_train, y_train, model): ### By Jéhoiakim KINGNE
    model.fit(X_train,y_train)
    return model

def apply_testing(X_test, model): ### By Jéhoiakim KINGNE
    return model.predict(X_test)

def train_test_stage(data_df, label_column_name, model): ### By Jéhoiakim KINGNE
    X_train, X_test, y_train, y_test = split_data_train_test(data_df,label_column_name)
    model = apply_training(X_train, y_train, model)
    y_pred = apply_testing(X_test, model)
    return y_test, y_pred

def assess_prediction(y_test,y_pred): ### By Jéhoiakim KINGNE
    return mean_squared_error(y_test,y_pred)

def apply_kfold_cross_validation(data_df, label_column_name, model, kernel_list,
                                 n_folds=10, scale_data = True, random_state=42, shuffle=True): ### By Jéhoiakim KINGNE
    X,y = separate_predictors_and_label(data_df,label_column_name)
    mean_scores = dict()
    if scale_data:
        scaler = MinMaxScaler(feature_range=(0, 1))
        X = pd.DataFrame(scaler.fit_transform(X), columns = X.columns)
    cross_val = KFold(n_splits=n_folds, random_state=random_state, shuffle=shuffle)
    for kernel in kernel_list:
        #print('kernel used', kernel)
        current_model = model(kernel=kernel)
        for train_index, test_index in cross_val.split(X):
            #print("Train Index: ", train_index, "\n")
            #print("Test Index: ", test_index)
            X_train, X_test, y_train, y_test = X.iloc[train_index], X.iloc[test_index], y.iloc[train_index], y.iloc[test_index]
            current_model.fit(X_train, y_train)
            if kernel in mean_scores:
                mean_scores[kernel] += current_model.score(X_test, y_test)/n_folds
            else:
                mean_scores[kernel] = current_model.score(X_test, y_test)/n_folds
    return mean_scores


if __name__ == '__main__':
    path_to_data = ''
    my_data = pd.read_csv(path_to_data)


    label_name_to_predict = 'MEDV'
    maximum_categorical_values = 2

    kernel_list = ['linear', 'poly', 'rbf', 'sigmoid']
    model = SVR

    preprocessed_data = DataPreparation(my_data, limitCategoricalNumerical=maximum_categorical_values)
    best_kernel = 'linear'
    test_split = 0.3
    feature_columns = feature_selection(preprocessed_data, label_name_to_predict, test_split)
    preprocessed_data = preprocessed_data[feature_columns]

    y_test, y_pred = train_test_stage(preprocessed_data, label_name_to_predict, model(kernel=best_kernel))
    assess_prediction(y_test, y_pred)

    print('O')
