import pandas as pd
import random as rd
import copy


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