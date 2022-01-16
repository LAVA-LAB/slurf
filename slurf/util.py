import math


def leq(a, b):
    return math.isclose(a, b) or a < b


def is_inbetween(a, b, c):
    return leq(a, b) and leq(b, c)
