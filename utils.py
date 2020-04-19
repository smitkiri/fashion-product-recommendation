import cv2
import sys
import numpy as np
from pickle import dump, load
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
from keras.applications.vgg16 import VGG16
from datetime import datetime # library to compute run of various algorithm

def drawProgressBar(percent, barLen = 20):
    '''
    Draws a progress bar, something like [====>    ] 20%
    
    Parameters
    ------------
    precent: float 
             percentage completed, between 0 and 1
    
    barLen: int
            Length of progress bar
    '''
    # Carriage return, returns to the begining of line to owerwrite
    sys.stdout.write("\r")
    sys.stdout.write("Progress: [{:<{}}] {:.0f}%".format("=" * int(barLen * percent) + ">", 
                                                         barLen, percent * 100))
    sys.stdout.flush()


def read_images(path, filenames, extension = None, shape = None, labels = None, verbose = 0):
    '''
    Returns list of images by reading them as arrays, filenames that we read 
    successfully and associated label indexes if labels are provided.
    
    Parameters
    -----------
    path: string
          Path of the directory where the images are stored
    
    filenames: list of strings
               Names of images to be read
               
    exension: string
              If extension not present in filenames list, 
              provide the extension of images to be read.
    
    shape: tuple
           If specified, only those images with matching shape will be returned. 
    
    labels: list
            List of labels associated with images. 
            Used if shape is specified.
    
    verbose: int
             If 1, print the progress of reading images.
    '''
    path = path.replace('\\', '/')
    if path[-1] != '/':
        path = path + '/'
    
    # Creating an empty list of labels to return if labels are specified
    if labels is not None:
        if len(filenames) != len(labels):
            raise ValueError("Number of labels not equal to number of files to be read")
        y = []
            
    if not extension:
        extension = ''
    else:
        extension = extension.lower()
        if extension[0] != '.':
            extension = '.' + extension

    images = []
    extracted_filenames = []
    total_files = len(filenames)
    
    for idx in range(len(filenames)):
        try:
            img = cv2.imread(path + str(filenames[idx]) + extension)
            
            if verbose == 1: 
                drawProgressBar(idx / total_files)
            
            if shape is not None:
                if img.shape == shape:
                    images.append(img)
                    extracted_filenames.append(filenames[idx])
                    if labels is not None:
                        y.append(idx)
            
            else:
                images.append(img)
                extracted_filenames.append(filenames[idx])
                if labels is not None:
                    y.append(idx)
        except:
            sys.stdout.write('\nSkipping ' + str(filenames[idx]) + extension + 
                                ', no such file in directory ' + path + '\n\r')
    
    if verbose == 1:
        print("\n")
    
    # Converting list of arrays into multi-dimensional array 
    # if all images have the same shape
    if shape is not None:
        images = np.stack(images)
    
    if labels is not None:
        return images, extracted_filenames, y
    else:
        return images, extracted_filenames


def save_pickle(file, variable):
    '''
    Saves variable as a pickle file
    
    Parameters
    -----------
    file: str
          File name/path in which the variable is to be stored
    
    variable: object
              The variable to be stored in a file
    '''
    if file.split('.')[-1] != "pickle":
        file += ".pickle"
        
    with open(file, 'wb') as f:
        dump(variable, f)
        print("Variable successfully saved in " + file)


def open_pickle(file):
    '''
    Returns the variable after reading it from a pickle file
    
    Parameters
    -----------
    file: str
          File name/path from which variable is to be loaded
    '''
    if file.split('.')[-1] != "pickle":
        file += ".pickle"
    
    with open(file, 'rb') as f:
        return load(f)


def resize_image(image, dimension = (80, 60, 3)):
    '''
    Resizes the image to a specified lower dimension
    
    Parameters
    -----------
    image: numpy array
           The image which is to be resized
    
    dimension: tuple / list
               The target size
    '''
    if image.shape != dimension:
            image = cv2.resize(image, (dimension[1], dimension[0]))
    
    return image  


def get_embeddings(image, dimension = (80, 60, 3), model = None):
    '''
    Returns the embeddings of given input image
    
    Parameters
    -----------
    image: numpy array
           The image for which embeddings are needed.
           Recommended shape (80, 60, 3)
           
    dimension: tuple / list
               The target size if the image is to be resized
               
    model: keras model with predict function
           The model to be used to get image embeddings, by default, uses VGG16
    '''
    if image.ndim != 3:
        raise AttributeError("The image should be 3 - dimensional")
    
    image = resize_image(image, dimension)
    image = image.reshape(1, dimension[0], dimension[1], dimension[2])
        
    if model is None:
        model = VGG16(weights = 'imagenet', input_shape = dimension, include_top = False)
        model.trainable = False
    
    return model.predict(image)


def plot_images(images, nrows = None, ncols = None, figsize = None, ax = None, 
                axis_style = 'on', bgr2rgb = True):
    '''
    Plots a given list of images and returns axes.Axes object
    
    Parameters
    -----------
    images: list
            A list of images to plot
            
    nrows: int
           Number of rows to arrange images into
    
    ncols: int
           Number of columns to arrange images into
    
    figsize: tuple
             Plot size (width, height) in inches
           
    ax: axes.Axes object
        The axis to plot the images on, new axis will be created if None
        
    axis_style: str
                'off' if axis are not to be displayed
    '''
    N = len(images)
    if not isinstance(images, (list, np.ndarray)):
        raise AttributeError("The images parameter should be a list of images, "
                             "if you want to plot a single image, pass it as a "
                             "list of single image")

    # Setting nrows and ncols as per parameter input
    if nrows is None:
        if ncols is None:
            nrows = N
            ncols = 1
        else:
            nrows = int(np.ceil(N / ncols))
    else:
        if ncols is None:
            ncols = int(np.ceil(N / nrows))
    
    if ax is None:
        _, ax = plt.subplots(nrows, ncols, figsize = figsize)
    
    if len(images) == 1:
        if bgr2rgb == True:
            images[0] = cv2.cvtColor(images[0], cv2.COLOR_BGR2RGB)
    
        ax.imshow(images[0])
        ax.axis(axis_style)
        
        return ax
    
    else:
        for i in range(nrows):
            for j in range(ncols):
                if (i * ncols + j) < N:
                    img = images[i * ncols + j]
                    
                    if bgr2rgb == True:
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    # For this condition, ax is a 2d array else a 1d array
                    if nrows >1 and ncols > 1: 
                        ax[i][j].imshow(img)
                    
                    else:
                        ax[i + j].imshow(img)
                
                if nrows > 1 and ncols > 1:
                    ax[i][j].axis(axis_style)
                else:
                    ax[i + j].axis(axis_style)
        
        return ax


def recommend_images(image, n, database_images, transform = 'pca', pca_fit = None, 
                     cluster_fit = None, distance_metric = "euclidean"):          
    '''
    image: numpy array
           The input image to compare other images to
    
    n: int
       Number of similar images to return
                       
    database_images: multi-dimensional numpy array
                     database of images to compare input image to
    
    transform: str
               The transformation to be used, must be pca or embeddings
                     
    pca_fit: sklearn.decomposition.PCA object
             The pca object to transform the image into lower dimensions
             
    cluster_fit: clustering object with predict method
                 If used, it first predicts the cluster of images in the databse 
                 that the input image belongs to, narrowing the search space
    
    distance_metric: str
                     The distance metric to use to compare input image with database images
    '''
    if transform.lower() == 'pca':
        if pca_fit is None:
            raise AttributeError("If transform is PCA, must pass pca_fit object")
        
        image = resize_image(image)
        image = image.reshape(image.shape[0] * image.shape[1] * image.shape[2])
        image = image / 255.0
        image = pca_fit.transform(image.reshape(1,-1))
        
    elif transform.lower() == 'embeddings':
        image = get_embeddings(image)
        image = image.reshape(1, image.shape[1] * image.shape[2] * image.shape[3])
    
    else:
        raise AttributeError("transform should either be 'pca' or 'embeddings'")
        
    if cluster_fit is not None:
        image_label = cluster_fit.predict(image)[0]
        database_labels = cluster_fit.predict(database_images)
        image_indices = np.argwhere(database_labels == image_label)
        database_images = database_images[database_labels == image_label]
    
    else:
        image_indices = None
        
    distances = pairwise_distances(database_images, image, metric = distance_metric).flatten()
    top_n_idx = [image_indices[idx][0] if image_indices is not None else idx for idx in distances.argsort()[:n]]
    
    return top_n_idx
   

def get_time():
    return datetime.now()
