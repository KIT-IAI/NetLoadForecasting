import json
import os.path
from glob import glob

from forecasting_framework.model.bo_cnn import BoCNN
from forecasting_framework.model.bo_mlp import BoMLP
from forecasting_framework.model.bo_prob_cnn import BoProbCNN
from forecasting_framework.model.bo_prob_mlp import BoProbMLP

folder_models = '../models'


#
# Wrapper for execution of the models
#


def launch_all_execute_train_and_test(train_all):
    """
    Launches every model with train true

    train_all: parameter to force to train every modelfile
    """
    models = []

    dirname = os.path.dirname(__file__)
    folder = os.path.join(dirname, folder_models)
    pattern = os.path.join(folder, '*/*.json')

    for file_name in glob(pattern, recursive=True):
        with open(file_name) as f:
            json_object = json.load(f)
            if json_object['train'] or train_all:
                print(file_name)
                models.append(_toModel(json_object))

    for model in models:
        print(model.name + " train started!")
        model.execute_train()
        print(model.name + " train done!")
        print(model.name + " forecast started!")
        model.execute_forecast()
        print("Model is cleared  forecast done!")


def launch_specific_execute_train_and_test(path):
    """
    Launches specific model
    """

    models = []

    dirname = os.path.dirname(__file__)
    folder = os.path.join(dirname, folder_models)
    path_str = os.path.join(folder, str(path))

    for file_name in glob(path_str):
        with open(file_name) as f:
            json_object = json.load(f)

            print(file_name)
            models.append(_toModel(json_object))

    for model in models:
        print(model.name + " train started!")
        model.execute_train()
        print(model.name + " train done!")
        print(model.name + " forecast started!")
        model.execute_forecast()
        print("Model is cleared  forecast done!")


def _toModel(json_object):
    if json_object['modelname'] == "cnn":
        return BoCNN(**json_object)

    if json_object['modelname'] == "mlp":
        return BoMLP(**json_object)

    if json_object['modelname'] == "prob_cnn":
        return BoProbCNN(**json_object)

    if json_object['modelname'] == "prob_mlp":
        return BoProbMLP(**json_object)
