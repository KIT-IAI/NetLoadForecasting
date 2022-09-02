import pathlib
import sys

sys.path.append(".")
import argparse

import forecasting_framework.model.model_wrapper as mw
import forecasting_framework.datamodifier.datamodifier_wrapper as dw

#
# If you want to create Data outcomment the dw.launch line
#


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Parser to start specific model out of CommandLine")
    parser.add_argument('--datapath', type=pathlib.Path, help="Enter your path to model.json")

    args = parser.parse_args()

    if args.datapath == None:
        #
        # Run the following lines only if data generation is needed.
        # Therefore, the line dw.launch() can be out-commented if no data generation needed.
        #
        print("Data modification started !")
        dw.launch()
        print("Data modification done !")

        #
        # Run the following lines only if training is needed.
        # Therefore, the line mw.launch_all_execute_train_and_test(train_all=False)
        # can be out-commented if training needed.
        # The train all parameter could be used to train every model not depending on  its train parameter.
        #
        print("Training started !")
        mw.launch_all_execute_train_and_test(train_all=False)
        print("Training done !")
    else:

        #
        # Run the following lines only if data generation is needed.
        # Therefore, the line dw.launch() can be out-commented if no data generation needed.
        #
        print("Data modification started !")
        dw.launch()
        print("Data modification done !")

        #
        # Run the following lines only if training is needed.
        # Therefore, the line mw.launch_all_execute_train_and_test(train_all=False)
        # can be out-commented if training needed.
        #
        print("Training started !")
        mw.launch_specific_execute_train_and_test(args.datapath)
        print("Training done !")
