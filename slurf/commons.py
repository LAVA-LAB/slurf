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


def create_output_folder(root_dir, modelfile):
    output_root_dir = path(root_dir, "output", "")
    output_subfolder = modelfile.replace(".", "_") + '_date=' + getDateTime()
    output_path = path(output_root_dir, output_subfolder, "")
    os.mkdir(output_path)
    
    return output_path


def getTime():

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")

    return current_time


def getDateTime():
    
    now = datetime.now()

    return now.strftime("%Y-%M-%d_%H-%M-%S")