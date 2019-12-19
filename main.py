import argparse
import cv2 as cv
import os
import pylab as pl
import numpy as np

parser = argparse.ArgumentParser(description='Compute a fusion of several images to obtain an HDR image')
parser.add_argument("folder", type=str, help='folder where the images are stored')
args = parser.parse_args()
folder = args.folder

#### PARAMETERS ####
wc, ws, we = 1, 1, 1
sigmaE = 0.2
####################

files = os.listdir(folder)
I = np.array([cv.imread(folder + "/" + f) for f in files])
gray = np.array([cv.cvtColor(im, cv.COLOR_RGB2GRAY) for im in I])
I = I / 255.0
N, w, h, c = I.shape

def plot_images(images, cmap='viridis'):
	pl.figure(figsize=(15,12))
	n=len(images)
	for i in range(n):
		pl.subplot((n+1)//2,2,i+1)
		cmap = 'gray' if len(images[i].shape)==2 else cmap
		pl.imshow(images[i],cmap=cmap)
		pl.xticks([])
		pl.yticks([])
	pl.tight_layout(pad=0,h_pad=0,w_pad=0)
	pl.show()

# Compute C
C = abs(np.array([cv.Laplacian(im, cv.CV_16S, ksize=3) for im in gray]))+1e-5

# Compute S
S = np.sqrt(((I - np.expand_dims(I.sum(3), 3)) ** 2).sum(3))

#S = np.zeros(C.shape)
#for i in range(len(I)):
#	R=I[i][:,:,0]
#	G=I[i][:,:,1]
#	B=I[i][:,:,2]
#	mu=(R+G+B)/3
#	S[i] = np.sqrt(((R - mu)**2 + (G - mu)**2 + (B - mu)**2)/3)

# Compute E
E = np.exp(-((I - 0.5)**2 / (2*sigmaE**2)).sum(3))

# Compute W
W = C**wc * S**ws * E**we
W /= W.sum(0)
W = W.reshape(N, w, h, 1)

R = (W*I).sum(0)

plot_images(np.concatenate((I, [R])))
# plot_images(C)
