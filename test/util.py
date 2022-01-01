import os

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
sir10_pctmc = _path("", "sir10.sm")
dft_and = _path("", "and_param.dft")
nonmonotonic_dft = _path("", "nonmonotonic_param.dft")
