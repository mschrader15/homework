from argparse import ArgumentParser
from functools import wraps
from time import process_time
from datetime import datetime
from numpy import array, matmul, add, subtract, mean, array_equal, zeros_like
import sys

# the setrecursionlimit function is
# used to modify the default recursion
# limit set by python.
sys.setrecursionlimit(10 ** 6)

# creating the command line arguments
parser = ArgumentParser()
parser.add_argument('--input', type=str, help="file defining inputs to the algorithm")
parser.add_argument('--algorithm', type=str, default='CG')

POTENTIAL_INPUTS = ['n', 'Q', 'q', 'c', 'A', 'a', 'B', 'b', 'ax', 'bx']


# wrapper function to time optimization algorithms
def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = process_time()
        result = f(*args, **kw)
        te = process_time()
        return te - ts, result
    return wrap

# class DescentMethod:


class ConjugateGradient:
    MAX_ITERATIONS = 10 ** 5
    ERROR_TOLERANCE = 0.01

    def __init__(self, input_dict):
        # unpacking the input dict:
        self.n, self.Q, self.q, self.c, self.A, self.a, self.B, self.b, self.ax, self.bx = \
            (input_dict[_input] for _input in POTENTIAL_INPUTS)

        # creating objects for all necessary variables in the G-G Method
        self.x_k = self.initial_x()
        self.g_k = self.g_k_1_func()
        self.d_k = -1 * self.g_k.copy()
        self.d_k_Q_d_k = self.fast_d_k_Q_d_k()
        self.zeros = zeros_like(self.g_k)
        self.alpha_k = None
        self.beta_k = None
        self.error_tolerance = ConjugateGradient.ERROR_TOLERANCE ** 2

    def g_k_1_func(self):
        # g_0 = -gradient f(x_0) else gradient f(x_0)
        # return subtract(matmul(self.Q, self.x_k.T), self.q) if k > 0 else subtract(self.q, matmul(self.Q, self.x_k.T))
        return subtract(matmul(self.Q, self.x_k.T), self.q)

    def fast_d_k_Q_d_k(self, ):
        # this denominator term appears in both alpha_k and beta_k. Only compute once
        return matmul(matmul(self.d_k.T, self.Q), self.d_k)

    def alpha_k_func(self, ):
        return matmul(-1 * self.g_k, self.d_k) / self.d_k_Q_d_k

    def x_k_func(self, ):
        return add(self.x_k, self.alpha_k * self.d_k)

    def d_k_func(self, ):
        return add(-1 * self.g_k, self.beta_k * self.d_k)

    def beta_k_func(self, ):
        return matmul(matmul(self.g_k.T, self.Q), self.d_k) / self.d_k_Q_d_k

    def initial_x(self, ):
        # return mean(array([self.ax, self.bx]), axis=0).astype(int)
        return zeros_like(self.ax)

    def check_stopping_conditions(self, k):
        if k > ConjugateGradient.MAX_ITERATIONS:
            return "maximum iterations achieved", True
        elif array_equal(self.g_k, self.zeros):
            return "g_k = 0", True
        else:
            return None, False

    @timing
    def cg(self, ):
        # run the conjugate gradient function
        stop = False
        k = 0
        reason = None
        while not stop:
            # calculate d_k * Q * d_k.T first and only once, as it appears multiple times in subsequent functions
            self.d_k_Q_d_k = self.fast_d_k_Q_d_k()
            # calculate alpha (with last g_k)
            self.alpha_k = self.alpha_k_func()
            # calculate x_k_1 with g_k
            self.x_k = self.x_k_func()
            # calculate g_k_1
            self.g_k = self.g_k_1_func()
            # calculate beta_k_1
            self.beta_k = self.beta_k_func()
            # calculate d_k_1
            self.d_k = self.d_k_func()
            # check if stopping condition has been met
            reason, stop = self.check_stopping_conditions(k)
            k += 1
        return reason

    def optimize(self, output_file_path):
        # run the cg function and record the excution time
        cpu_time, reason = self.cg()
        # write the output of the function to the specified file path
        self.write_output(cpu_time, output_file_path)

    def fn(self, x):
        # evaluate the function at the specified point
        return 0.5 * matmul(matmul(x.T, self.Q, ), x) + matmul(self.q.T, x) + self.c

    def check_bounds(self, x_star):
        # checking to see whether the solution is inside the bounds
        for x, x_min, x_max in zip(x_star, self.ax, self.bx):
            if not x_min <= x <= x_max:
                # exit the loop if x_star is not within the bounds and report false
                return False
        return True

    def check_1st_order(self,):
        # necessary 1st order condition
        # gradient(f(x*)) * d_k >= 0
        # g_k holds value from when cg() terminated, so it is = gradient(f(x*))
        return matmul(self.g_k, self.d_k) >= 0

    def write_output(self, cpu_time, file_path):
        # create a list for the rows to write
        write_strings = []
        # these objects need to be restructured to print
        for item in [self.x_k, self.fn(self.x_k)]:
            # making the array more "pretty" to print
            if isinstance(item.tolist(), list):
                # create a spaced list of lists
                _string = [" ".join(inner_item) if isinstance(inner_item, list) else str(inner_item)
                           for inner_item in item.tolist()]
                # add the brackets
                _string[0] = "".join(['[', _string[0]])
                _string[-1] = "".join([_string[-1], ']'])
                # add the newly composed string to the list of row
                write_strings.append(" ".join(_string))
            else:
                write_strings.append(str(item))

        # add in the CPU time
        write_strings.append(str(cpu_time))

        # check the x_star bounds and add that
        write_strings.append("inside bounds" if self.check_bounds(self.x_k) else "outside")

        # check 1st order necessary condition
        write_strings.append("1st Order Condition satisfied" if self.check_1st_order() else
                             "1st Order Condition not satisfied")

        # write to file
        write_string = "\n".join([" = ".join([variable, value]) for variable, value
                                  in zip(['x_star', 'f(x_star)', 'CPU time', 'inside bounds?', 'first order check'],
                                         write_strings)])

        # add in the time
        write_string = datetime.now().strftime("%b %d %Y %H:%M:%S") + "\n" + write_string

        with open(file_path, 'w') as f:
            f.write(write_string)


def check_inputs(input_dict):
    # loop through potential inputs
    for _input in POTENTIAL_INPUTS:

        # trying to read in the data and convert it into a numpy array
        try:
            # creating a numpy array object from inputs that should be matrices
            if isinstance(input_dict[_input], list):
                input_dict[_input] = array(input_dict[_input], dtype=int)
            else:
                # this is statement is used to catch the key errors below
                input_dict[_input] = input_dict[_input]

        # catching a missing input
        except KeyError:
            # raise an Uncaught Exception if a necessary input is missing
            if _input in ['Q', 'q', 'n', 'c', 'ax', 'bx']:
                raise Exception(f"{_input} is missing from the input file. That is not allowed")
            # else continue on
            input_dict[_input] = None
            pass


def read_inputs(file_path):
    # read text file
    with open(file_path, 'r') as f:
        file_str = f.read()

    # creating a python dictionary for the inputs
    input_dict = {}
    # split at line breaks and again at "=" if it is in the string
    split_file = [line.split('=') if '=' in line else line for line in file_str.split('\n')[:-1]]

    # add split matrix lines back together
    save_index = 0
    for i, item in enumerate(split_file):
        if isinstance(item, list):
            save_index = i
            continue
        split_file[save_index].append(item)

    # removed the lines without an = sign
    split_file = [item for item in split_file if isinstance(item, list)]

    # loop through the cleaned input list
    for item in split_file:
        # the variable name is letter left of "="
        variable = item[0].strip()
        # remove the brackets in the input file and split at spaces. then convert from string to integer
        values = [list(map(int, val.strip().strip(']').strip('[').split())) for val in item[1:]]
        # reformat the list of lists if necessary
        values = values if len(values) > 1 else values[0]
        input_dict[variable] = values if len(values) > 1 else values[0]

    # return the inputs
    return input_dict


# the main entry point of the script
if __name__ == "__main__":

    # reading the command line arguments
    cmd_args = parser.parse_args()

    # reading in the input file
    inputs = read_inputs(cmd_args.input)

    # checking the inputs and creating array objects
    check_inputs(inputs)

    # run algorithm of choice
    if cmd_args.algorithm == 'CG':
        # create the conjugate gradient object
        cg = ConjugateGradient(input_dict=inputs)

        # run the conjugate gradient algorithm
        cg.optimize("output.txt")
