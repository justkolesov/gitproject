import argparse

import logging
import sys

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from dataloader import load_data

from module import Model



def train(test_size, file_name,path_name,test_path):
    '''
     train data for training from csv file housing.csv

     Args:
         test_size: size of test samples
         file_name : namefile of data
         path_name : filename where model will be downloaded
     '''


    logger.warning("Get_data")
    try:
        data = load_data(file_name)
    except FileNotFoundError:
        logger.exception("file not found")

    logger.warning("Preprocess data")
    # preprocessing (utils.py)

    logger.warning("Split data to train and test")
    data_train,data_test = train_test_split(data, test_size = test_size, random_state = 5)

    logger.warning("test data download")
    data_test = pd.DataFrame(data_test)
    data_test.to_csv(test_path, index=False)

    logger.warning("download test data")


    logger.warning("Initialize")
    model = Model()

    logger.warning("Fit model")
    model.trainee(data_train[:,:-1], data_train[:,-1])
    model.save(path_name)

    logger.warning(f"Model is saved to {path_name}")


if __name__ == "__main__":

    logger = logging.getLogger("my logger")
    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    logger.addHandler(stdout_handler)
    logger.setLevel(logging.WARNING)

    parser = argparse.ArgumentParser()
    parser.add_argument("--test_size",type=int, default = 0.2)
    parser.add_argument("--path_name", type = str, default = "./data/model_path")
    parser.add_argument("--file_name", type = str, default = "./data/regression.csv")
    parser.add_argument("--test_path", type=str, default = "./data/test_path.csv")
    args = parser.parse_args()
    train(args.test_size, args.file_name, args.path_name, args.test_path)
