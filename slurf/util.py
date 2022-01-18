import math


def leq(a, b):
    return math.isclose(a, b) or a < b


def is_inbetween(a, b, c):
    return leq(a, b) and leq(b, c)


def is_precise_enough(lb, ub, precision, ind_precisions, prop):
    rel_error = 2 * (ub - lb) / (ub + lb)
    if prop in ind_precisions:
        return rel_error <= ind_precisions[prop]
    else:
        return rel_error <= precision
