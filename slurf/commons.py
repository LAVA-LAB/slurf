from datetime import datetime


def getTime():

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")

    return current_time

def getDateTime():
    
    now = datetime.now()

    return now.strftime("%Y-%M-%d_%H-%M-%S")