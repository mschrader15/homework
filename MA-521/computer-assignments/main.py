import json
from argparse import ArgumentParser
import numpy as np

# creating the command line arguments 
parser = ArgumentParser()
parser.add_argument('--input', type=str, help="file defining inputs to the algorithm")
parser.add_argument('--algorithm', type=str, default='CG')


def conjugate_gradient(input_dict):
    



def check_inputs(input_dict):
    # loop through potential inputs
    for _input in ['n', 'Q', 'q', 'c', 'A', 'a', 'B', 'b', 'ax', 'bx']:
        
        # trying to read in the data and convert it into a numpy array
        try:
            # converting the string from the input file into a integer 
            # and then creating a numpy array object from it
            input_dict[_input] = np.array(map(int, input_dict[_input]), dtype=int)
        
        # catching a missing input
        except KeyError:
            # raise an Exception if a necessary input is missing 
            if _input in ['Q', 'q', 'n', 'c', 'ax', 'bx']:
                raise Exception(f"{_input} is missing from the input file. That is not allowed")
            # else continue on
            pass


def read_inputs(file_path):
    # read in the input file, assuming that it is in json format
    with open(file_path, 'rb') as f:
        return json.load(f)



# the main entry point of the script
if __name__ == "__main__":

    # reading the command line arguments 
    args = parser.parse_args()

    # reading in the input file
    inputs = read_inputs(args.input)

    # checking the inputs and creating array objects
    check_inputs(inputs)

    # run algorithm of choice
    if args.algorithm == 'CG':

        conjugate_gradient(inputs)