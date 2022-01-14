import os
import math

testfile_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources")


def _path(folder, file):
    """
    Internal method for simpler listing of examples.
    :param folder: Folder.
    :param file: Example file.
    :return: Complete path to example file.
    """
    return os.path.join(testfile_dir, folder, file)


dummy_file = _path("", "dft.drn")
tiny_pctmc = _path("", "tiny.sm")
mini_pctmc = _path("", "mini.sm")
tandem_pctmc_jani = _path("", "tandem.jani")
dft_and = _path("", "and_param.dft")
nonmonotonic_dft = _path("", "nonmonotonic_param.dft")


def inbetween(a, b, c):
    return (math.isclose(a, b) or a < b) and (math.isclose(c, b) or b < c)
