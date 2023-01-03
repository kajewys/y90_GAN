from matplotlib import pyplot
import nibabel as nib
import cv2
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import LeakyReLU

def normalize(arr):
        print("\nShape of array to normalize = ",arr.shape)
        print("Max value in array = ",np.max(arr))
        print("Min value in array = ",np.min(arr))
        arr = 2*(arr-np.min(arr)) / (np.max(arr) - np.min(arr)) -1
        print("New max value in array = ",np.max(arr))
        print("New min value in array = ",np.min(arr))
        print("Normalization complete.\n")        #       The values should be between -1 and 1
        return arr

# Load and normalize training PET images as a NumPy array
# (Have two directories for local vs server)
# Pick image slices (based on 3D Slicer view)
pet_stack = np.asarray(nib.load('/Users/kajewys/Workspace/Roncali/y90_img/y90pet.nii').get_fdata())
#pet_stack = np.asarray(nib.load('/home/kajewys/gan/y90pet.nii').get_fdata())
pet_stack = normalize(pet_stack)
pet_stack = pet_stack[:,:,292:308]

# Plot 9 images from the training dataset as sanity check
pyplot.figure()
pyplot.suptitle("9 images from training set")
for i in range(9):
	pyplot.subplot(3, 3, 1 + i)
	pyplot.axis('off')
	pyplot.imshow(pet_stack[:,:,4+i], cmap='afmhot')
pyplot.savefig('PET_input_demo_3x3.png', dpi=100)
pyplot.show()
pyplot.close()

# Downsize to 28x28 and interpolate
pet_stack_mini = np.empty([28,28,1])
for n in range(pet_stack.shape[2]):
	img = cv2.resize(pet_stack[:,:,n], dsize=(28, 28), interpolation=cv2.INTER_NEAREST)
	img = np.expand_dims(img, axis=-1)
	pet_stack_mini = np.append(pet_stack_mini, img, axis=-1)
	img = cv2.resize(pet_stack[:,:,n], dsize=(28, 28), interpolation=cv2.INTER_AREA)
	img = np.expand_dims(img, axis=-1)
	pet_stack_mini = np.append(pet_stack_mini, img, axis=-1)
	img = cv2.resize(pet_stack[:,:,n], dsize=(28, 28), interpolation=cv2.INTER_CUBIC)
	img = np.expand_dims(img, axis=-1)
	pet_stack_mini = np.append(pet_stack_mini, img, axis=-1)
	img = cv2.resize(pet_stack[:,:,n], dsize=(28, 28), interpolation=cv2.INTER_BITS)
	img = np.expand_dims(img, axis=-1)
	pet_stack_mini = np.append(pet_stack_mini, img, axis=-1)
	img = cv2.resize(pet_stack[:,:,n], dsize=(28, 28), interpolation=cv2.INTER_LINEAR)
	img = np.expand_dims(img, axis=-1)
	pet_stack_mini = np.append(pet_stack_mini, img, axis=-1)

del pet_stack
# 80 slices of 28x28 PET images result from numerous interpolation modes
pet_stack_mini = pet_stack_mini[:,:,1:]
print("Shape of downsampled stack = ",pet_stack_mini.shape)

# Plot 9 images from the downsampled stack
pyplot.figure()
pyplot.suptitle("9 downsampled training images")
for i in range(9):
	pyplot.subplot(3, 3, 1 + i)
	pyplot.axis('off')
	pyplot.imshow(pet_stack_mini[:,:,int(np.random.choice(80,1))], cmap='afmhot')
pyplot.show()

# Define discriminator model
def define_discriminator():
	model = Sequential()
	model._name = "Discriminator"
	# First convolutional layer, 3x3 kernel with padding (keeps dimensions), channels last input shape
	model.add(Conv2D(16, (3,3), padding='same', input_shape=(28,28,1)))
	# LeakyReLU useful for GANs due to sparse gradients (alpha is small negative slope)
	model.add(LeakyReLU(alpha=0.1))
	# Dropout layer reduces overfitting (some nodes set 0, rest boosted to maintain sum)
	model.add(Dropout(0.25))
	# Second convolutional layer, same params as first one
	model.add(Conv2D(16, (3,3), padding='same'))
	model.add(LeakyReLU(alpha=0.1))
	model.add(Dropout(0.25))
	# Flatten (turn output into column vector)
	model.add(Flatten())
	model.add(Dense(1, activation='sigmoid'))
	# Compile model, Adam is a good starting choice
	opt = Adam(learning_rate=0.0002, beta_1=0.5)
	# Try BCE as loss function
	model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
	model.summary()
	return model

# Define generator model (want 28x28 img as output)
def define_generator(latent_dim):
	# Linear stack of layers
	model = Sequential()
	model._name = "Generator"
	# Size of output space for Dense layer (fully-connected)
	# Start with 7x7 img from latent space, 16 outputs
	n_nodes = 7*7*16
	model.add(Dense(n_nodes, input_dim=latent_dim))
	# Add activation of LeakyReLU, 0.1 slope
	model.add(LeakyReLU(alpha=0.1))
	model.add(Reshape((7, 7, 16)))
	# Upsample to 14x14 through Deconvolution
	# new size is stride + kernel - 2
	model.add(Conv2DTranspose(16, (3,3), strides=2, padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	# Upsample to 28x28, another Deconvolution
	model.add(Conv2DTranspose(16, (3,3), strides=2, padding='same'))
	# Activation
	model.add(LeakyReLU(alpha=0.1))
	# Last convolutional layer
	model.add(Conv2D(1, (7,7), activation='sigmoid', padding='same'))
	# load previously trained weights
	model.load_weights('generator_model_350.h5')
	model.summary()
	return model
 
# Define combined generator and discriminator model for updating generator
def define_gan(g_model, d_model):
	# Make weights in the discriminator not trainable
	d_model.trainable = False
	# Connect models
	model = Sequential()
	model.add(g_model)
	model.add(d_model)
	# Compile model
	opt = Adam(learning_rate=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt)
	return model

def load_real_samples():
	# Load PET dataset: original (150x150x16), now (28x28x80)
	trainX = pet_stack_mini.reshape(pet_stack_mini.shape[2],pet_stack_mini.shape[0],pet_stack_mini.shape[1])
	print("Real input shape: ",trainX.shape)
	# Add channels dimension and change into float32
	X = np.expand_dims(trainX, axis=-1)
	X = X.astype('float32')
	print("Expanded real input shape: ",X.shape)
	return X
'''
# Carlotta's way of loading npy data
def load_real_samples():
	# load dataset
	#(trainX, trainy), (_, _) = load_data()
    #trainX = np.load("/home/kajewys/gan_pet/Dataset_center.npy")
    trainX = np.load("/Users/kajewys/Workspace/Roncali/TrainingImages/Dataset.npy")
    #trainy = np.load("./TrainingDataset/Dataset_center.npy" )
    # expand to 3d, e.g. add channels
    X = np.expand_dims(trainX, axis=-1)
	# convert from ints to floats
    X = X.astype('float32')
    # scale from [0,255] to [-1,1]
    X = (X - 127.5) / 127.5
    #return [X, trainy], X, trainy
    return X
'''
 # Select real samples
def generate_real_samples(dataset, n_samples):
	# Choose random instances
	ix = np.random.randint(0, dataset.shape[0], n_samples)
	# Retrieve selected images
	X = dataset[ix]
	# Generate 'real' class labels (1)
	y = np.ones((n_samples, 1))
	return X, y

# Generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples):
	# Generate points
	x_input = np.random.randn(latent_dim * n_samples)
	# Reshape into a batch of inputs for the network
	x_input = x_input.reshape(n_samples, latent_dim)
	return x_input
 
# Use generator to generate n fake examples, with class labels
def generate_fake_samples(g_model, latent_dim, n_samples):
	x_input = generate_latent_points(latent_dim, n_samples)
	# Predict outputs
	X = g_model.predict(x_input)
	# Create 'fake' class labels (0)
	y = np.zeros((n_samples, 1))
	return X, y
 
# Create and save a plot of generated images
def save_plot(examples, epoch, n=3):
	pyplot.figure()
	for i in range(n * n):
		pyplot.subplot(n, n, 1 + i)
		pyplot.axis('off')
		pyplot.imshow(examples[i, :, :, 0], cmap='afmhot')
	filename = 'PET_plot_%03d.png' % (epoch+1)
	pyplot.savefig(filename)
	pyplot.close()

# Evaluate discriminator, plot generated images, save generator model
def summarize_performance(epoch, g_model, d_model, dataset, latent_dim, n_samples=80):
	# Prepare real samples
	X_real, y_real = generate_real_samples(dataset, n_samples)
	# Evaluate discriminator on real examples
	_, acc_real = d_model.evaluate(X_real, y_real, verbose=0)
	# Prepare fake examples
	x_fake, y_fake = generate_fake_samples(g_model, latent_dim, n_samples)
	# Evaluate discriminator on fake examples
	_, acc_fake = d_model.evaluate(x_fake, y_fake, verbose=0)
	# Summarize discriminator performance
	print('>Accuracy real: %.0f%%, fake: %.0f%%' % (acc_real*100, acc_fake*100))
	# save plot
	save_plot(x_fake, epoch)
	# Save the generator model tile file
	filename = 'generator_model_%03d' % (epoch + 1)
	#g_model.save(filename)
	tf.keras.models.save_model(g_model,filename,save_format='h5')
	del g_model
 
# Train generator and discriminator
def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=1000, n_batch=80):
	bat_per_epo = int(dataset.shape[0] / n_batch)
	half_batch = int(n_batch / 2)
	n = 0
	# Manually enumerate epochs
	for i in range(n_epochs):
		# Enumerate batches over the training set
		for j in range(bat_per_epo):
			# Get randomly selected 'real' samples
			X_real, y_real = generate_real_samples(dataset, half_batch)
			# Generate 'fake' examples
			X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
			# Create training set for discriminator
			if (n==0):
				print("X_real: ",X_real.shape)
				print("X_fake: ",X_fake.shape)
				print("y_real: ",y_real.shape)
				print("y_fake: ",y_fake.shape)
				n = 1
			X, y = np.vstack((X_real, X_fake)), np.vstack((y_real, y_fake))
			# Update discriminator model weights
			d_loss, _ = d_model.train_on_batch(X, y)
			# Prepare points in latent space as generator input
			X_gan = generate_latent_points(latent_dim, n_batch)
			# Create inverted labels for fake samples
			y_gan = np.ones((n_batch, 1))
			# Update generator via discriminator's error
			g_loss = gan_model.train_on_batch(X_gan, y_gan)
			# Summarize loss on this batch
			print('>%d, %d/%d, d=%.3f, g=%.3f' % (i+1, j+1, bat_per_epo, d_loss, g_loss))
			#del X_real, X_fake, y_fake, y_real, X, y
		# Evaluate model performance every N epochs
		if (i+1) % 100 == 0 or i == 0:
			summarize_performance(i, g_model, d_model, dataset, latent_dim)
 
# Choose size of latent space, want smaller latent space than num samples
latent_dim = 10
# Create discriminator
d_model = define_discriminator()
# Create generator
g_model = define_generator(latent_dim)
# Create GAN
gan_model = define_gan(g_model, d_model)
# Load image data
dataset = load_real_samples()
print("Shape of dataset: ", dataset.shape)

# Train model
train(g_model, d_model, gan_model, dataset, latent_dim)