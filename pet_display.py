import os
os.system("clear all")

from matplotlib import pyplot
import numpy as np
import nibabel as nib

def normalize(arr): #   3D array, dims are like (?, ?, ?)
        print("Shape of array to normalize = ",arr.shape)
        print("Max value in array = ",np.max(arr))
        print("Min value in array = ",np.min(arr))
        arr = (arr-np.min(arr)) / (np.max(arr) - np.min(arr))
        print("New max value in array = ",np.max(arr))
        print("New min value in array = ",np.min(arr))
        print("Normalization complete.")        #       The values should be between 0 and 1
        return arr

# load and prepare training images
pet_stack = []
pet_stack = np.asarray(nib.load('/Users/kajewys/Workspace/Roncali/y90_img/y90pet.nii').get_fdata())

normalize(pet_stack)

# plot images from the training dataset, want 275 to 350
for i in range(9):
	pyplot.subplot(3, 3, 1 + i)
	pyplot.axis('off')
	pyplot.imshow(pet_stack[:,:,250+i], cmap='afmhot')
pyplot.show()