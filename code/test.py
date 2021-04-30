import argparse

import logging
import sys

import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error, r2_score

from module import Model


def test(test_path, path_name):

    logger.debug(" load model ")

    model = Model()
    model = model.load(path_name)

    logger.debug(" load test data ")
    data_test = pd.read_csv(test_path)

    data_test = data_test.to_numpy()

    X_test = data_test[:,:-1]
    Y_test = data_test[:,-1]

    logger.debug(" make prediction ")
    y_pred = model.predict(X_test)

    logger.debug(" calculating r2 score ")
    y_test = Y_test

    result_metrics = {"r2_score": r2_score(y_test,y_pred),
                      "mse" : mean_squared_error(y_test,y_pred)}
    logger.debug(result_metrics)




if __name__ == "__main__":

    logger = logging.getLogger(" my test logger ")
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)

    parser = argparse.ArgumentParser()
    parser.add_argument("--test_path",default="../data/test_path.csv",type=str)
    parser.add_argument("--path_name", default ="../data/model_path", type=str)
    args = parser.parse_args()
    test(args.test_path, args.path_name)

