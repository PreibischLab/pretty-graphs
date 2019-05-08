import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

# function for loading one data-set
def load_data(file_path, skiprows_=0):
    data = np.loadtxt(file_path, delimiter = '\t', skiprows=skiprows_)
    # print(data.shape) 
    # handles some weird cases, e.g. when there is no data in the file
    if (len(data.shape) < 2):
        data = data[None, :]
    if (data.shape[1] == 0):
        I = np.array([0])
    else:
        I = data[:, -1]
    return I

# create title for the image
def create_title(path, name_id=-1):
    # get the name of the initial image
    image_name = path.split("/")[name_id]
    # create the full title 
    title = image_name[:-4] # drop extension
    return title

# how good is fitter-meter?
def fitter_meter(y, y_hat):
    return [mean_absolute_error(y, y_hat), 
            np.sqrt(mean_squared_error(y, y_hat))]