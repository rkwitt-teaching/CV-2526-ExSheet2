"""Submission for exercise sheet 2

SUBMIT this file as submission_<STUDENTID>.py where
you replace <STUDENTID> with your student ID, e.g.,
submission_1234567.py
"""
from torch.autograd import Function
import torch


# Exercise 2.1
def assignment_ex1():
    # YOUR CODE GOES HERE
    pass


# Exercise 2.2
def assignment_ex2():
    # YOUR CODE GOES HERE
    pass


# Exercice 2.3
def assignment_ex3():
    # YOUR CODE GOES HERE
    class Cube(Function):
        @staticmethod
        def forward(ctx, input):
            # YOUR CODE GOES HERE
            pass

        @staticmethod
        def backward(ctx, grad_output):
            # YOUR CODE GOES HERE
            pass

    return Cube
