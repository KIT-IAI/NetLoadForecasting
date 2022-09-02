from time import sleep

from tensorboard import program

#
# Get the best performing runs and take the results, by take the path to the tensorboard folder.
#
if __name__ == '__main__':
    tb = program.TensorBoard()
    #
    # insert path
    #
    tb.configure(argv=[None, '--logdir', "Replace it with fitting path"])
    launch = tb.launch()
    sleep(1000)
