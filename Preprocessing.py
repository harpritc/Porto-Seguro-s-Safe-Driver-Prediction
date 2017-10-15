import pandas as pd
import numpy as np
import sys


def remove_missing(df):
    df_clean = df.dropna()
    return df_clean


def convert_string_data_to_int(df):
    columns = df.columns.values

    # for each column
    for column in columns:
        text_to_int_dict = {}

        def convert_to_int(val):
            return text_to_int_dict[val]

        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            column_contents = df[column].values.tolist()
            unique_elements = set(column_contents)
            x = 0

            for unique in unique_elements:
                if unique.find('?') != -1:
                    # Replace ? with None, apply imputer later
                    text_to_int_dict[unique] = None
                elif unique not in text_to_int_dict:
                    text_to_int_dict[unique] = x
                    x += 1

            df[column] = list(map(convert_to_int, df[column]))

    return df



def normalize(df_classRemoved,df_original):
    df_classRemoved = (df_classRemoved-df_classRemoved.mean())/df_classRemoved.std()
    df_classRemoved['Class'] = df_original['Class'].values
    return df_classRemoved


def main(argv):
    if len(argv) < 2:
        print('Usage: preprocessing.py <InputDataSet> <OutputDataset>')
        print('Example: preprocessing.py ')
        sys.exit()

    inputDataSet = argv[0]
    outputDataSet = argv[1]

    # df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data")
    df = pd.read_csv(inputDataSet)

    nof_columns = df.shape[1]
    columns = []
    for index in range(0, nof_columns - 1):
        columns.append('Feature{}'.format(index))
    columns.append('Class')

    df = pd.read_csv(inputDataSet, names=columns)

    df_remove_missing = remove_missing(df);
    # convert string data to int
    df_numeric = convert_string_data_to_int(df_remove_missing)
    #df_numeric_clean = remove_missing(df_numeric)

    df_ClassRemoved = df_numeric.drop(df_numeric.columns[-1], axis=1)

    #df_normalize = normalize(df_ClassRemoved, df_numeric)
    df_normalize = normalize(df_ClassRemoved,df_numeric)

    df_normalize_clean =  df_normalize.dropna()
    df_normalize_clean.to_csv(outputDataSet, index=False)
    print(df_normalize)


if __name__ == "__main__":
    main(sys.argv[1:])
