import json
import os.path


#
# Superclass of all models e.G. of bayesian optimization
#

class Model:
    _folder_data_const = '../datamodifiers/data'
    _folder_result_relative = '../models/model_results'

    _dirname = os.path.dirname(__file__)
    _folder = os.path.join(_dirname, _folder_data_const)

    _result_folder = os.path.join(_dirname, _folder_result_relative)
    _model = None

    def __init__(self, data_path,
                 modelparams, features, forecast_variables, modelname, name, train, folder, plot_diagnostic=False,
                 save_diagnostics=False):
        self.data_path = data_path
        self.modelparams_dict = modelparams
        self.plot_diagnostic = plot_diagnostic
        self.save_diagnostics = save_diagnostics
        self.features = features
        self.forecast_variables = forecast_variables
        self.modelname = modelname
        self.name = name
        self.folder = folder
        self.train = train

    #
    # basic train function
    #
    def execute_train(self):
        pass

    #
    # basic forecastfunction function
    #
    def execute_forecast(self):
        pass

    #
    # plots diagnostic/ saves them into folder
    #
    def plot_diagnostic(self):
        pass

    def save_diagnostic(sel, name, folder=None, folder_results=_result_folder):
        if folder == None:
            if not os.path.exists(folder_results):
                os.mkdir(folder_results)

        else:
            folder_test = os.path.join(folder_results, folder)
            if not os.path.exists(folder_test):
                os.makedirs(folder_test)

    def to_json(self, additional_path=None):
        if additional_path is not None:
            if not os.path.exists(additional_path):
                os.makedirs(additional_path)
        with open(os.path.join(additional_path, self.name + ".json"), "w") as outfile:
            json.dump(self, outfile, default=lambda o: o.__dict__,
                      sort_keys=True, indent=4)

    #
    # save function pkl and csv
    #
    def save_pkl(self, df, name, folder=None, folder_results=_result_folder):
        if folder == None:
            if not os.path.exists(folder_results):
                os.mkdir(folder_results)
            df.to_pickle(folder_results + "/" + name + ".pkl")
        else:
            folder_test = os.path.join(folder_results, folder)
            if not os.path.exists(folder_test):
                os.makedirs(folder_test)
            df.to_pickle(folder_results + "/" + folder + "/" + name + ".pkl")

    def save_csv(self, df, name, folder=None, folder_results=_result_folder):

        if folder == None:
            if not os.path.exists(folder_results):
                os.mkdir(folder_results)
            df.to_csv(folder_results + "/" + name + ".csv")
        else:
            folder_test = os.path.join(folder_results, folder)
            if not os.path.exists(folder_test):
                os.makedirs(folder_test)
            df.to_csv(folder_results + "/" + folder + "/" + name + ".csv")

    def test_path(self, name, folder, folder_results=_result_folder):

        folder_test = os.path.join(folder_results, folder)

        return os.path.exists(os.path.join(folder_test, name + ".pkl"))

    def save_dict(self, dict, name, folder=None, folder_results=_result_folder):

        print(name)
        if folder == None:
            if not os.path.exists(folder_results):
                os.mkdir(folder_results)

            with open(folder_results + "/" + name + ".json", 'w') as fp:
                json.dump(dict, fp)
        else:
            folder_test = os.path.join(folder_results, folder)
            if not os.path.exists(folder_test):
                os.makedirs(folder_test)
            with open(folder_results + "/" + folder + "/" + name + ".json", 'w') as fp:
                json.dump(dict, fp)
