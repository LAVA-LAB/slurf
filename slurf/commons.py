from datetime import datetime
import os
import pandas as pd


def path(root_dir, folder, file):
    """
    Internal method for simpler listing of examples.
    
    Parameters
    ----------
    :folder: Folder.
    :file: Example file.
    ----------
    
    Returns
    ----------
    Complete path to example file.
    ----------
    """
    
    return os.path.join(root_dir, folder, file)


def create_output_folder(root_dir, modelfile):
    """
    Create the specified folder.
    """
    
    output_root_dir = path(root_dir, "output", "")
    output_subfolder = modelfile.replace(".", "_") + '_date=' + getDateTime()
    output_path = path(output_root_dir, output_subfolder, "")
    os.mkdir(output_path)
    
    return output_path


def getTime():
    """
    Returns the current time, given a datetime object.
    """

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")

    return current_time


def getDateTime():
    """
    Returns a formatted string of the datetime.
    """
    
    now = datetime.now()

    return now.strftime("%Y-%m-%d_%H-%M-%S")


def print_stats(stats):
    """
    Print interesting statistics returned by Storm.
    """
    
    print('-----------------------------------------')
    print('MODEL STATISTICS:')
    series = pd.Series(stats, name='time')
    print(series)
    print('-----------------------------------------')
    
    return series


def set_solution_df(exact, solutions):
    """
    Define the Pandas DF for storing the solution vectors and return this.
    """
    
    if not exact:
        solutions = [[tuple(solutions[i,j,:]) 
                      for j in range(solutions.shape[1])] 
                      for i in range(solutions.shape[0])]
    
    df = pd.DataFrame(solutions)
    df.index.names = ['Sample']
    
    return df


def set_output_path(root_dir, args):
    """
    Define and return the full path to the output folder.
    """
        
    output_folder = args.modelfile_nosuffix + "_N=" + str(args.Nsamples)
    output_path = create_output_folder(root_dir, output_folder)
        
    return output_path


def append_new_line(file_name, text_to_append):
    """
    Append given text as a new line at the end of file
    """
    
    # Open the file in append & read mode ('a+')
    with open(file_name, "a+") as file_object:
        # Move read cursor to the start of file.
        file_object.seek(0)
        # If file is not empty then append '\n'
        data = file_object.read(100)
        if len(data) > 0:
            file_object.write("\n")
        # Append text at the end of file
        file_object.write(text_to_append)
    
        
def intersect(i1, i2):
    """
    Check if two intervals i1, i2 intersect.
    """
    
    return max(i1[0], i2[0]) < min(i1[1], i2[1])