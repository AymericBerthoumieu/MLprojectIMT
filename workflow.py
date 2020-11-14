import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error












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
