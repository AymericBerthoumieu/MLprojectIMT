import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge


def ridge_test(df, target, test_split, α):
    X = df.drop(columns=target)
    X = preprocessing.scale(X)
    y = df[target].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_split)

    ridge_df = pd.DataFrame({'variable': list(df.columns)[:-1]})
    ridge_train_pred = []
    ridge_test_pred = []

    # iterate lambdas
    for alpha in range(α):
        # training
        ridge_reg = Ridge(alpha=alpha)
        ridge_reg.fit(X_train, y_train)
        var_name = str(alpha)
        ridge_df[var_name] = ridge_reg.coef_
        # prediction
        ridge_train_pred.append(ridge_reg.predict(X_train))
        ridge_test_pred.append(ridge_reg.predict(X_test))
    ridge_df = ridge_df.set_index('variable').T
    ridge_df.plot(figsize=(10, 5))
    plt.axhline(y=0, color='black', linestyle='--')
    plt.xlabel('α')
    plt.ylabel('Ridge coef')
    plt.show()
    # return ridge_df

