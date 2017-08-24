#------------------------------------------------------------------------------------------------
#Developed by Information Technologies Institute of the Center for Research and Technology Hellas
#This code was created by the team to participate in the NVIDIA AI City Challenge / Aug 2017
#Contact: giannakeris@iti.gr
#------------------------------------------------------------------------------------------------

#This script performs the DeepHOG pipeline on a given set of deepnet predictions 
#and their corresponding HOG boxes that have been computed in a previous step.
#Requires the PCA model, GMM model and the Neural Network model 
#that were used during training (those are provided).
#The output is a txt file per image containing corresponding boxes, classes and scores 
#which have now been extracted using the DeepHOG and Ensemble frameworks. 
#Please check out the paper for more details.

#Usage: Please edit the required paths 
#       in the section immediatly after the imports bellow.

import pandas
import numpy as np
import keras
import os
from tqdm import tqdm_notebook as tqdm
from scipy.stats import multivariate_normal
from sklearn.decomposition import PCA
from sklearn.externals import joblib

################################################EDIT HERE##########################################################

#enter the paths to calculated hog boxes and the deepnet's predictions
SET = '480'
hog_path = '/enter/hog/boxes/path/here/'
test_path = '/enter/deepnet/predictions/folder/here/'
OUTPUT_FOLDER = '/enter/here/output/folder/' #this folder needs to have two subfolders named 'deephog' & 'ensemble'

###################################################################################################################

df_labels = ['class', 'xmin', 'ymin', 'xmax', 'ymax', 'score']
num_of_words = 16

def load_gmm(folder):
    files = ["means.gmm.npy", "covs.gmm.npy", "weights.gmm.npy"]
    m = np.load(folder+files[0])
    c = np.load(folder+files[1])
    w = np.load(folder+files[2])
    return m, c, w
def likelihood_moment(x, ytk, moment):
    x_moment = np.power(np.float32(x), moment) if moment > 0 else np.float32([1])
    return x_moment * ytk

def likelihood_statistics(samples, means, covs, weights):
    gaussians, s0, s1, s2 = {}, {}, {}, {}
    #samples = zip(range(0, len(samples)), samples)
    g = [multivariate_normal(mean=means[k], cov=covs[k]) for k in range(0, len(weights))]
    for idx, sample in enumerate(samples):
        gaussians[idx] = np.array([g_k.pdf(sample) for g_k in g])
    for k in range(0, len(weights)):
        s0[k], s1[k], s2[k] = 0, 0, 0
        for idj, j in enumerate(samples):
            probabilities = np.multiply(gaussians[idj], weights)
            probabilities = probabilities / np.sum(probabilities)
            s0[k] = s0[k] + likelihood_moment(j, probabilities[k], 0)
            s1[k] = s1[k] + likelihood_moment(j, probabilities[k], 1)
            s2[k] = s2[k] + likelihood_moment(j, probabilities[k], 2)
    
    return s0, s1, s2

def fisher_vector_weights(s0, s1, s2, means, covs, w, T):
    return np.float32([((s0[k] - T * w[k]) / np.sqrt(w[k]) ) for k in range(0, len(w))])

def fisher_vector_means(s0, s1, s2, means, sigma, w, T):
    return np.float32([(s1[k] - means[k] * s0[k]) / (np.sqrt(w[k] * sigma[k])) for k in range(0, len(w))])

def fisher_vector_sigma(s0, s1, s2, means, sigma, w, T):
    return np.float32([(s2[k] - 2 * means[k]*s1[k]  + (means[k]*means[k] - sigma[k]) * s0[k]) / (np.sqrt(2*w[k])*sigma[k])  for k in range(0, len(w))])

def normalize(fisher_vector):
    v = np.sqrt(abs(fisher_vector)) * np.sign(fisher_vector)
    return v / np.sqrt(np.dot(v, v))

def fisher_vector(samples, means, covs, w):
    s0, s1, s2 =  likelihood_statistics(samples, means, covs, w)
    T = samples.shape[0]
    covs = np.float32([np.diagonal(covs[k]) for k in range(0, covs.shape[0])])
    a = fisher_vector_weights(s0, s1, s2, means, covs, w, T)
    b = fisher_vector_means(s0, s1, s2, means, covs, w, T)
    c = fisher_vector_sigma(s0, s1, s2, means, covs, w, T)
    fv = np.concatenate([np.concatenate(b), np.concatenate(c)])
    fv = normalize(fv)
    return fv

def transform_PCA(descs, pca):
    return pca.transform(descs)

#Load all necessary models
gmm = load_gmm(SET+'/')
pca = joblib.load(SET+'/pca.pkl')
weights = [each for each in os.listdir(SET) if each.endswith('.hdf5')]
model = keras.models.load_model(SET+'/'+weights[0])
classes = pandas.read_csv(SET+'/classes.txt', header=None).values.flatten().tolist()

#load images
images = os.listdir(test_path)

#start calculations and write output
for image in tqdm(images):
    deep_label = pandas.read_csv(test_path+image, sep=' ', header=None, names=df_labels)
    if (os.path.getsize(hog_path+os.path.splitext(image)[0]+'.hog')==0):
        deep_label.to_csv(OUTPUT_FOLDER+'/deephog/'+os.path.splitext(image)[0]+'.txt', header=False, index=False, sep=' ')
        deep_label.to_csv(OUTPUT_FOLDER+'/ensemble/'+os.path.splitext(image)[0]+'.txt', header=False, index=False, sep=' ')
    else:
        hog_boxes = pandas.read_csv(hog_path+os.path.splitext(image)[0]+'.hog', header=None)
        descs = transform_PCA(hog_boxes.as_matrix(), pca)
        fv = np.float32([fisher_vector(np.atleast_2d(desc), *gmm) for desc in descs])
        predictions = model.predict(fv)
        scores = pandas.DataFrame(np.amax(predictions, axis=1))
        labels = pandas.DataFrame([classes[i] for i in np.argmax(predictions, axis=1)])
        df = deep_label.drop(['class', 'score'], axis=1)
        data = pandas.concat([labels, df, scores], axis=1)
        data.columns = df_labels
        ensemble = pandas.concat([data[data['score'] >= deep_label['score']], deep_label[data['score'] < deep_label['score']]])
        ensemble.sort_index(inplace=True)
        data.to_csv(OUTPUT_FOLDER+'/deephog/'+os.path.splitext(image)[0]+'.txt', header=False, index=False, sep=' ')
        ensemble.to_csv(OUTPUT_FOLDER+'/ensemble/'+os.path.splitext(image)[0]+'.txt', header=False, index=False, sep=' ')

