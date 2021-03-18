from math import sqrt
from argparse import ArgumentParser
from functools import wraps
from time import process_time
from datetime import datetime
from numpy import array, matmul, add, subtract, mean, array_equal, zeros_like
from numpy.linalg import inv, norm
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


class Optimizer:
    MAX_ITERATIONS = 10 ** 5
    ERROR_TOLERANCE = 1e-5

    def __init__(self, input_dict):
        # unpacking the input dict:
        self.n, self.Q, self.q, self.c, self.A, self.a, self.B, self.b, self.ax, self.bx = \
            (input_dict[_input] for _input in POTENTIAL_INPUTS)

        # creating objects for all necessary variables in the G-G Method
        self.x_k = self.initial_x()
        self.g_k = None
        self.d_k = None
        self.d_k_Q_d_k = None
        self.zeros = zeros_like(self.g_k)
        self.alpha_k = None
        self.beta_k = None
        self.error_tolerance = Optimizer.ERROR_TOLERANCE ** 2

    # def g_k_1_func(self):
    #     # g_0 = -gradient f(x_0) else gradient f(x_0)
    #     # return subtract(matmul(self.Q, self.x_k.T), self.q) if k > 0 else subtract(self.q, matmul(self.Q, self.x_k.T))
    #     return add(matmul(self.Q, self.x_k.T), self.q)
    #
    # def fast_d_k_Q_d_k(self, ):
    #     # this denominator term appears in both alpha_k and beta_k. Only compute once
    #     return matmul(matmul(self.d_k.T, self.Q), self.d_k)
    #
    # def alpha_k_func(self, ):
    #     return matmul(-1 * self.g_k, self.d_k) / self.d_k_Q_d_k
    #
    # def x_k_func(self, ):
    #     return add(self.x_k, self.alpha_k * self.d_k)
    #
    # def d_k_func(self, ):
    #     return add(-1 * self.g_k, self.beta_k * self.d_k)
    #
    # def beta_k_func(self, ):
    #     return matmul(matmul(self.g_k.T, self.Q), self.d_k) / self.d_k_Q_d_k

    def gradient(self, x):
        return matmul(self.Q, x) + self.q

    def initial_x(self, ):
        # return mean(array([self.ax, self.bx]), axis=0).astype(int)
        return zeros_like(self.ax)

    def check_stopping_conditions(self, k):
        if k > Optimizer.MAX_ITERATIONS:
            return "maximum iterations achieved", True
        elif array_equal(self.g_k, self.zeros):
            return "g_k = 0", True
        else:
            return None, False

    def golden_section_search(self, x, g):
        r = (sqrt(5) + 1) / 2
        fn = lambda alpha: self.fn(x + alpha * g)
        a = 0
        b = self.fn(x)
        c = b - (b - a) / r
        d = a + (b - a) / r
        while abs(b - a) > 0.001:
            if fn(c) < fn(d):
                b = d
            else:
                a = c
            c = b - (b - a) / r
            d = a + (b - a) / r
        return (b + a) / 2

    @timing
    def steepest_descent(self):
        stop = False
        reason = None
        self.x_k = self.initial_x()
        k = 0
        while not stop:
            g_k = self.gradient(self.x_k)
            a_k = self.golden_section_search(self.x_k, g_k)
            self.x_k = self.x_k - a_k * g_k
            if norm(g_k) < self.error_tolerance:
                stop = True
                reason = "solution found"
            elif k > self.MAX_ITERATIONS:
                stop = True
                reason = "Max Iterations Reached"
            k += 1
        return reason

    # @timing
    # def accelerated_sd(self, ):
    # # This is not working
    #     stop = False
    #     k = 0
    #     reason = None
    #     lambda_k = 0
    #     x_tilda_k = zeros_like(self.ax)
    #     self.x_k = zeros_like(self.ax)
    #     while not stop:
    #         g_k_1 = self.gradient(self.x_k)
    #         lambda_k_1 = (1 + (1 + 4 * lambda_k)**(1/2)) / 2
    #         alpha_k = (1 - lambda_k) / lambda_k_1
    #         x_tilda_k_1 = self.x_k - alpha_k * g_k_1.T
    #         self.x_k = (1 - alpha_k) * x_tilda_k_1 + alpha_k * x_tilda_k
    #         if norm(g_k_1) < self.error_tolerance:
    #             stop = True
    #             reason = "solution found"
    #         elif k > self.MAX_ITERATIONS:
    #             stop = True
    #             reason = "Max Iterations"
    #         else:
    #             lambda_k = lambda_k_1
    #             x_tilda_k = x_tilda_k_1
    #             k += 1
    #     return reason

    @timing
    def cg(self, ):
        # run the conjugate gradient function
        stop = False
        k = 0
        reason = None
        self.x_k = zeros_like(self.ax)
        self.d_k = self.q - matmul(self.Q, self.x_k)
        self.g_k = -1 * self.d_k
        while not stop:
            # calculate d_k.T * Q * d_k first and only once, as it appears multiple times in subsequent functions
            self.d_k_Q_d_k = matmul(matmul(self.d_k.T, self.Q), self.d_k)
            # calculate alpha
            self.alpha_k = matmul(-1 * self.g_k.T, self.d_k) / self.d_k_Q_d_k
            # calculate x_k_1 with g_k
            x_k_1 = self.x_k + self.alpha_k * self.d_k
            # calculate g_k_1
            g_k_1 = matmul(self.Q, x_k_1) + self.q
            # calculate beta_k_1
            self.beta_k = matmul(matmul(g_k_1.T, self.Q), self.d_k) / self.d_k_Q_d_k
            # calculate d_k_1
            self.d_k = add(-1 * self.g_k, self.beta_k * self.d_k)
            # pass x_k_1 and g_k_1 onto the next
            self.x_k = x_k_1
            self.g_k = g_k_1
            # check if stopping condition has been met
            reason, stop = self.check_stopping_conditions(k)
            k += 1
        return reason

    @timing
    def newton(self):
        k = 0
        stop = False
        reason = None
        self.x_k = zeros_like(self.ax)
        Q_inv = inv(self.Q)
        while not stop:
            g_k_1 = matmul(self.Q, self.x_k) + self.q
            x_k_1 = self.x_k - matmul(Q_inv, g_k_1.T)
            if norm(x_k_1 - self.x_k) < self.error_tolerance:
                stop = True
                reason = "solution found"
            elif k > self.MAX_ITERATIONS:
                stop = True
                reason = "Max Iterations"
            else:
                self.x_k = x_k_1
            k += 1
        return reason

    def optimize(self, output_file_path, method):
        # select the function based on user input
        methods = {'cg': self.cg, 'newton': self.newton, 'steepest descent': self.steepest_descent}
        # run the selected function and record the excution time
        cpu_time, reason = methods[method]()
        # write the output of the function to the specified file path
        self.write_output(cpu_time, output_file_path, method)

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

    def check_1st_order(self, ):
        # necessary 1st order condition
        # gradient(f(x*)) * d_k >= 0
        # g_k holds value from when cg() terminated, so it is = gradient(f(x*))
        return matmul(self.g_k, self.d_k) >= 0

    def write_output(self, cpu_time, file_path, method):
        # create a list for the rows to write
        write_strings = [method]
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
                                  in zip(['method', 'x_star', 'f(x_star)', 'CPU time', 'inside bounds?',
                                          'first order check'],
                                         write_strings)])

        # add in the time
        write_string = "\n------------------------------------------------------------------\n" + \
                       datetime.now().strftime("%b %d %Y %H:%M:%S") + "\n" + write_string

        with open(file_path, 'a+') as f:
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
    # if cmd_args.algorithm == 'CG':
    # create the conjugate gradient object
    cg = Optimizer(input_dict=inputs)

    # run the conjugate gradient algorithm
    cg.optimize("output.txt", cmd_args.algorithm)
