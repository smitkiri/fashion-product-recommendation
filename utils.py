# -*- coding: utf-8 -*-

import cv2
import numpy as np
from pickle import dump, load
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt

PICKLE_PATH = ".\\pickles\\"

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
            
            if verbose == 1 and (idx + 1) % 1000 == 0: 
                print("Extracted", idx+1, "images out of", total_files)
            
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
            print('Skipping ' + str(filenames[idx]) + extension + 
                                ', no such file in directory ' + path)
    
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


def get_embeddings(image, resize = False):
    '''
    Returns the VGG16 embeddings of given input image
    
    Parameters
    -----------
    image: numpy array
           The image for which embeddings are needed.
           Must be of shape (80, 60, 3)
           
    resize: boolean
            Wether to resize higher dimensional image to (80,60,3)
    '''
    if image.shape != (80, 60, 3) and resize == True:
        if image.ndim == 3:
            image = cv2.resize(image, (60, 80))
        else:
            raise AttributeError("The image should be 3 - dimensional")
    
    if image.shape == (80, 60, 3):
        image = image.reshape(1, 80, 60, 3)
    
    if image.shape != (1, 80, 60, 3):
        raise AttributeError("The image shape should be (80, 60, 3)")
        
    model = open_pickle(PICKLE_PATH + 'embedded_images_sub')['model']
    embedded_image = model.predict(image)
    return embedded_image[0]


def get_pca_transform(image, n_components = 145, resize = False):
    '''
    Returns the top 2 components of the image
    
    Parameters
    -----------
    image: numpy array
           The image for which embeddings are needed.
    
    components: The number of components needed for the image.
                Must be either 145 or 1000.
    '''
    if n_components == 145:
        pca = open_pickle(PICKLE_PATH + 'pca_transformation_results_145')['object']
    
    elif n_components == 1000:
        pca = open_pickle(PICKLE_PATH + 'pca_transformation_results_1000')['object']
    
    else:
        raise ValueError("Number of components must be 145 or 1000")
    
    if resize == True:
        image = cv2.resize(image, (60, 80))
    
    if image.ndim == 3:
        image = image.reshape(1, image.shape[0] * image.shape[1] * image.shape[2])
        
    if image.ndim > 3 or image.ndim < 2:
        raise AttributeError("The input image must be 2 or 3 dimensional")
        
    if image.shape[1] != pca.n_features_:
        raise AttributeError("Shape mismatch, expected array of shape (n, " +
                             str(pca.n_features_) + ") but got " + str(image.shape))

    return pca.transform(image)


def recommend_images(image, n, transformation, resize = False, 
                     use_kmeans = True, metric = 'euclidean'):
    '''
    Returns top n images similar to given image
    
    Parameters
    -----------
    image: numpy array
           The input image to compare other images to
    
    n: int
       Number of similar images to return
       
    transformation: str
                    The transformation to apply to input image.
                    Currently supported values: 'pca_145', 'pca_1000', 
                                                'embeddings_sub', 'embeddings_all'
                    
    resize: boolean
            Used if to determine if the image size is greater than (80, 60, 3)
    
    use_kmeans: boolean
                If True, uses K-Means to first predict the cluster of images
                it belongs to, narrowing the search space
                
    metric: str
            The distance metric to use
    '''
    valid_transformations = ['pca_145', 'pca_1000', 'embeddings_sub', 'embeddings_all']
    
    # Check for argument validity
    if transformation.lower() not in valid_transformations:
        raise ValueError("Currently supported values for transformation: 'pca_145'," 
                         " 'pca_1000', 'embeddings_sub', 'embeddings_all'")
    
    transformation = transformation.lower().split('_')
    
    if transformation[0] == 'pca':
        image = get_pca_transform(image, int(transformation[1]))
        pca = open_pickle(PICKLE_PATH + "pca_transformation_results_" + transformation[1])
        X = pca['transformed_images']
        y = np.array(pca['image_names'])
        y = y.reshape(y.shape[0], 1)
    
    else:
        # Get input image embeddings and convert them to shape (1, 1024)
        image = get_embeddings(image, resize)
        image = image.reshape(1, image.shape[0] * image.shape[1] * image.shape[2])
        
        # Get pre-computed embeddings from data and convert them to shape (n, 1024)
        embeddings = open_pickle(PICKLE_PATH + "embedded_images_" + transformation[1])
        X = embeddings['data']
        X = X.reshape(X.shape[0], X.shape[1] * X.shape[2] * X.shape[3])
        
        # Get image names corresponding to pre-computed embeddings
        y = np.array(embeddings['images'])
        y = y.reshape(y.shape[0], 1)
        
    if use_kmeans == True:
        if transformation[0] == 'embeddings':
            kmeans = open_pickle(PICKLE_PATH + "KMeans_embeddings_clusters_" +
                                 transformation[1] + "_12")
        else:
            kmeans = open_pickle(PICKLE_PATH + "KMeans_pca_" + transformation[1] + 
                                 "_clusters_12")
        
        # Get K-Means prediction and filter data accordingly
        image_label = kmeans.predict(image)
        X = X[kmeans.labels_ == image_label[0]]
        y = y[kmeans.labels_ == image_label[0]]
    
    # Calculate distances of test image from each one in dataset 
    # and make 2nd column as image names
    distances = pairwise_distances(X, image, metric = metric)
    distances = np.hstack((distances, y))
    
    # Sort distances matrix by 1st column (distances) and return top n image names
    top_n = distances[distances[:, 0].argsort()][:n, 1].astype(int)
    
    return top_n


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
        ax.imshow(images[0])
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
          