import pytest

from sklearn.model_selection import train_test_split
import numpy as np
import sklearn

from module import Model
import dataloader
import train
import test

@pytest.fixture(scope="function")
def fixture_of_model():
    yield Model()

@pytest.fixture(scope="function")
def fixture_of_fail(request):
    filename = 'pythregression.csv'
    return filename

def test_load_data(fixture_of_fail):
    data = dataloader.load_data(fixture_of_fail)
    assert data[:,:-1].shape == (len(data[:,:-1]), 10) and data[:,-1].shape == (len(data[:,:-1]), )


def test_score_predictions(fixture_of_model, fixture_of_fail):
    data = dataloader.load_data(fixture_of_fail)
    data_train,data_test = train_test_split(data,test_size = 0.2, random_state = 5)
    fixture_of_model.trainee(data_train[:,:-1],data_train[:,-1])
    pred = fixture_of_model.predict(data_test[:,:-1])
    r2_score = sklearn.metrics.r2_score(data_test[:,-1],pred)
    assert r2_score >= 0.7


#@pytest.mark.parametrize()
def test_predict():
    raise NotImplementedError