from datetime import datetime
import os


def path(root_dir, folder, file):
    """
    Internal method for simpler listing of examples.
    :param folder: Folder.
    :param file: Example file.
    :return: Complete path to example file.
    """
    
    return os.path.join(root_dir, folder, file)


def getTime():

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")

    return current_time


def getDateTime():
    
    now = datetime.now()

    return now.strftime("%Y-%M-%d_%H-%M-%S")