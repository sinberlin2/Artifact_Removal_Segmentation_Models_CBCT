from msd_pytorch import (MSDRegressionModel)
from metrics5 import Metrics
from radial2axial5 import Radial2Axial

from timeit import default_timer as timer
from torch.utils.data import DataLoader
from os import environ
from unet_regr_model import UNetRegressionModel
from sacred import Experiment, Ingredient
from sacred.observers import MongoObserver
from image_dataset import ImageDataset
from natsort import natsorted
#from torchsummary import summary
from pytorch_model_summary import summary
import glob
import os.path
import numpy as np
import torch as t
import random
import imageio as io
import tifffile

# Copyright 2019 Centrum Wiskunde & Informatica, Amsterdam

#-----------------------------------------------------------------
# Author: Jordi Minnema
# Contact: jordi@cwi.nl
# Github: https://github.com/Jomigi/Cone_angle_artifact_reduction
# License: MIT

# This script is intended to train a convolutional neural network 
# (either MS-D Net or U-Net for high cone-angle artifact 
# reduction in walnut CBCT scans.
#----------------------------------------------------------------

##########################################################################################
#                                 Network Parameters                                     #
##########################################################################################

# Number of input channels
in_channels = 1  
# Number of output channels
out_channels = 1                 
# Use of reflection padding
reflect = True                      
# The number of epochs to train
epochs = 10                   
# The mini-batch size
batch_size = 1       
# CNN architecture used('msd' or 'unet')
architecture= 'unet'

# Dilations of the convolutional kernels in the MS-D Net 
dilation_f = 4
if dilation_f == 4:
    dilations = [1,2,4,8,16]              
elif dilation_f == 0:
    dilations = [1, 1, 1, 1,1]
# Depth of the MS-D Net
depth = 40            
# Width of the MS-D Net
width = 1                          
loss_f='L2'
# Whether a pre-trained model should be loaded
load_model = True             
print(load_model, 'load model')
if load_model== True:
# Whether the model should be trained
    train =False
else:
    train=True                     

# # metrics calculation
# phantoms = [9]  #,13,16,19,33,36,37  #[35,27,15,3,39,7,33]  inherit test_scans
# #architecture  defined
# #pos defined
# #it=[1] #7,8,10 inherit
# mode='horizontal'  #default to horizontal
# #maybe dsc and things

##########################################################################################                      
#                           Dataset Parameters                                           #
##########################################################################################

# Position of the X-ray source when acquiring the cone-beam CT data (1,2 or 3)
pos = 1
# Iteration number for documentation purposes
it  = 1             

# Path to input and target CBCT scans
dataset_dir = "/bigstore/felix/WalnutsRadialSlices/"   
# Path to store results
run_folder= '{}_pos{}_width{}_depth{}_dil{}_ep{}_it{}/'.format(architecture, pos, width, depth, dilation_f,epochs, it)
results_dir = "/bigstore/felix/shannon/ConeBeamResults/"  + run_folder
if not os.path.exists(results_dir):
     os.makedirs(results_dir)  
# Path to store network parameters
model_path= '/bigstore/felix/shannon/'
network_path = os.path.join(
        model_path, "saved_nets/")

best_models_path= os.path.join(
        network_path, "best_models/")
if not os.path.exists(best_models_path):
	os.makedirs(best_models_path)

# The CBCT scans used
input_scans = list(range(1,43))   

##########################################################################################
#                     Separate set into training, validation and test set                #
##########################################################################################

# Number of CBCT scans used for training
training_nb = 28
# Number of CBCT scans used for validation
val_nb = 7
# Number of CBCT scans used for testing
test_nb = 7

# Selection seed for reproducibility
selection_seed = 123456             
np.random.seed(selection_seed)

# Determine CBCT scans for training
training_scans = np.random.choice(
        input_scans, training_nb, replace=False)
for i in training_scans:
    input_scans.remove(i)

# Determine CBCT scans for validation
val_scans = np.random.choice(
        input_scans, val_nb, replace=False)
for i in val_scans:
    input_scans.remove(i)

# Determine CBCT scans for testing
if test_nb > 0:
    test_scans = np.random.choice(input_scans, test_nb, replace=False)
else:
    test_scans = val_scans


# Apply random seed
np.random.seed(selection_seed)

#########################################################################################
#                                  Loading Data                                         #
#########################################################################################

# Create training set
inp_imgs = []
tgt_imgs = []

for i in sorted(training_scans):
    inp_imgs.extend(natsorted(glob.glob(os.path.join(dataset_dir, 
        'Walnut{}/fdk_pos{}_*.tif*'.format(i, pos)))))
    tgt_imgs.extend(natsorted(glob.glob(os.path.join(dataset_dir, 
        'Walnut{}/iterative_iter50_*.tif*'.format(i)))))

train_ds = ImageDataset(inp_imgs, tgt_imgs)
print('Training set size', str(len(train_ds)))

# Create validation set
inp_imgs = []
tgt_imgs = []

for i in sorted(val_scans):
    inp_imgs.extend(natsorted(glob.glob(os.path.join(dataset_dir, 
        'Walnut{}/fdk_pos{}_*.tif*'.format(i, pos)))))
    tgt_imgs.extend(natsorted(glob.glob(os.path.join(dataset_dir, 
        'Walnut{}/iterative_iter50_*.tif*'.format(i)))))
      
val_ds = ImageDataset(inp_imgs, tgt_imgs)
print('Validation set size', str(len(val_ds)))

# Create test set       
inp_imgs = []
tgt_imgs = []
test_ds = []
test_size = 0

for i in sorted(test_scans): 
    inp_imgs = natsorted(glob.glob(os.path.join(dataset_dir, 
        'Walnut{}/fdk_pos{}_*.tif*'.format(i, pos))))
    tgt_imgs = natsorted(glob.glob(os.path.join(dataset_dir,
        'Walnut{}/iterative_iter50_*.tif*'.format(i))))
    
    # Create an additional list in order to process 2D slices for evaluation. 
    # This list is necessary to remember which slices correspond to each walnut. 
    test_ds.append(ImageDataset(inp_imgs, tgt_imgs))
    test_size += len(ImageDataset(inp_imgs, tgt_imgs))
                 
print('Test set size', str(len(ImageDataset(inp_imgs, tgt_imgs))))


#########################################################################################
#                                      Create Model                                     #
#########################################################################################

# Create MS-D Net 
if architecture== 'msd':
    model = MSDRegressionModel(in_channels, out_channels, depth, width,
                               dilations = dilations, loss = loss_f, parallel=True)
    print(model)
# Create U-Net
elif architecture== 'unet':
    model = UNetRegressionModel(network_path, in_channels, out_channels, depth, width, 
        loss_function=loss_f, dilation=dilations, reflect=True, conv3d=False)
print(model)
#print(summary(model) )#input_size=(channels, H, W))
#########################################################################################
#                                      Train Model                                      #
#########################################################################################

if train==True:
    print('Training model..')
    # Define dataloaders
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
 
    # Normalize input data such that it has zero mean and a standard deviation of 1.
    model.set_normalization(train_dl)

    # Print how a random network performs on the validation dataset:
    print("Initial validation loss: {}".format(model.validate(val_dl)))

# Try loading a precomputed network if wanted:
if load_model == True:
    try:
        print('trying to load')
        best_epoch_model=7   #print(best_model_path + 'radial_msd_depth80_it5_epoch59.pytorch')
        file_name ='{}_pos{}_width{}_depth{}_dil{}_ep{}_it{}_epoch{}.pytorch'.format(architecture, pos, width, depth, dilation_f,epochs, it, best_epoch_model)
        #print(best_models_path)        
        model.load(best_models_path + file_name)

        
        print("Network loaded")
    except:
        print("Loading failed")
        pass

# Train network
if train==True:
    print("Training...")
    best_validation_error = 10e6
    start = timer()
    best_epoch = 0
    
    for epoch in range(epochs):
        print("epoch", epoch)
        startd = timer()
        
        # Train        
        model.train(train_dl, 1)
        
        # Compute training error
        train_error = model.validate(train_dl)
        print("    *Training error:", train_error)

        # Compute validation error
        validation_error = model.validate(val_dl)
        print("    *Validation error: {}".format(validation_error))
                   
        endd = timer()
        print('Training time epoch {}: {}'.format(epoch, endd-startd))
     
        # Save network if worthwile
        if validation_error < best_validation_error:
            best_validation_error = validation_error
            run_network_path= network_path + run_folder 
            if not os.path.exists(run_network_path):
                os.makedirs(run_network_path)
            model.save(run_network_path + '{}_depth{}_it{}_epoch{}.pytorch'.format(architecture, depth, it, epoch), epoch)

            best_epoch = epoch
               
    end = timer()
    
    # Print final validation loss and total training time
    val_loss= model.validate(val_dl)
    train_time= end-start
    print("Final validation loss: {}".format(val_loss))
    print("Total training time:   {}".format(train_time))
    
    #print for recording by guildai 
    
    print("- val_loss: %f" % val_loss)
    print("- train_time: %f" % train_time)

    # Save network:
    model.save(best_models_path + '{}_pos{}_width{}_depth{}_dil{}_ep{}_it{}_epoch{}.pytorch'.format(architecture, pos, width, depth, dilation_f,epochs, it, best_epoch), best_epoch)

#########################################################################################
#                               Apply trained network to the test set                   #
#########################################################################################

# Allocate processed image
img = np.zeros((709, 501, 501), dtype=np.float32) 

# Iterate over walnuts scans in test set
for walnut_idx, ds in enumerate(test_ds):    

    test_dl = DataLoader(ds, batch_size=1, shuffle=False)
    start = timer() 

    # Iterate over each slice
    for idx, inp in enumerate(test_dl):        
        model.forward(inp[0],inp[1])
        output = model.output.data.cpu().numpy()
        img[idx, :, :] = output[0,:,:,:]
        
        # Save the result
        path_results = os.path.join(results_dir, 
                '{}_pos{}_it{}_depth{}_phantom{}/'.format(
                    architecture, pos, it, depth, sorted(test_scans)[walnut_idx])) #we could only keep the phantom as name
        if not os.path.exists(path_results):
            os.makedirs(path_results)
        io.imsave(path_results + '/slice{:05}.tif'.format(idx), img[idx,:,:].astype("float32"))
    # Print time required for processing
    end = timer()
    print('Processing time of walnut{}:'.format(test_scans[walnut_idx]), end-start)
#########################################################################################
#                               Generate axial from radial images                  #
#########################################################################################
print(test_scans)
phantoms = test_scans #[9]  #,13,16,19,33,36,37  #[35,27,15,3,39,7,33]  inherit test_scans
#iterates through all paramters except phantoms

for phantom in phantoms:
    radial_phantom_generator=Radial2Axial(phantom, architecture, [depth],  [pos], [it], run_folder)
    radial_phantom_generator.generate_rad_2_ax()
#########################################################################################
#                               Calculate metrics                  #
#########################################################################################


metrics_loader= Metrics(phantoms, architecture, pos, [it], depth, width, dilation_f, run_folder)  #import runfolder
SSIM,SSIM_ROI,MSE, MSE_ROI, DSC_low, DSC_ROI_low, DSC_high, DSC_ROI_high, PSNR, PSNR_ROI= metrics_loader.calculate_metrics()
print("- SSIM: %f" % SSIM)
print("- SSIM_ROI: %f" % SSIM_ROI)
print("- MSE: %f" % MSE)
print("- MSE_ROI: %f" % MSE_ROI)
print("- DSC_low: %f" % DSC_low)
print("- DSC_ROI_low: %f" % DSC_ROI_low)
print("- DSC_high: %f" % DSC_high)
print("- DSC_ROI_high: %f" % DSC_ROI_high)
print("- PSNR: %f" % PSNR)
print("- PSNR_ROI: %f" % PSNR_ROI)



# # metrics calculation
# print(test_ds)
# phantoms = test_ds #[9]  #,13,16,19,33,36,37  #[35,27,15,3,39,7,33]  inherit test_scans
# #architecture  defined
# #pos defined
# #it=[1] #7,8,10 inherit
# #mode='horizontal'  #default to horizontal
#
# metrics_loader= Metrics(phantoms, architecture, pos, it)
# results= metrics_loader.calculate_metrics()
