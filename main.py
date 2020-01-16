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
name=files[0].split('_')[0]
I = np.array([cv.imread(folder + "/" + f) for f in files])
I = I[:,:,:,[2, 1, 0]]
gray = np.array([cv.cvtColor(im, cv.COLOR_RGB2GRAY) for im in I])
I = I / 255.0
N, w, h, c = I.shape

def plot_images(images, cmap='viridis'):
	pl.figure(figsize=(15,12))
	n=len(images)
	for i in range(n):
		pl.subplot((n+1)//2,2,i+1)
		im = images[i].squeeze()
		cmap = 'gray' if len(im.shape)==2 else cmap
		pl.imshow(im)
		pl.xticks([])
		pl.yticks([])
	pl.tight_layout(pad=0,h_pad=0,w_pad=0)
	pl.show()

def plot_pyramids(pyr):
	ims = []
	K = len(pyr)
	N = pyr[0].shape[0]
	sh = []
	if len(pyr[0][0].shape) == 2:
		for k in range(K):
			sh.append(pyr[k].shape)
			pyr[k] = np.expand_dims(pyr[k], 3)
	for i in range(N):
		w, h, c = pyr[0][i].shape
		w2 = w + pyr[1][i].shape[0]
		im = np.zeros((w2, h, c))
		im[:w,:h,:] = pyr[0][i]
		x = 0
		for k in range(1, K):
			a, b, _ = pyr[k][i].shape
			im[-a:, x:x+b,:] = pyr[k][i]
			x += b
		ims.append(im)
	if len(sh) > 0:
		for k in range(K):
			pyr[k] = pyr[k].reshape(sh[k])
	plot_images(ims)

# Compute C
C = abs(np.array([cv.Laplacian(im, cv.CV_16S, ksize=3) for im in gray]))+1e-5

# Compute S
S = np.sqrt(((I - np.expand_dims(I.sum(3), 3)) ** 2).sum(3))+1e-5

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

# Bad R
R = (np.expand_dims(W, 3)*I).sum(0)
plot_images(np.concatenate((I, [R])))

# Bad R with gaussian
W = np.array([cv.GaussianBlur(wi, (9,9), 5) for wi in W])
# plot_images(W)
R = (np.expand_dims(W, 3)*I).sum(0)
plot_images(np.concatenate((I, [R])))

# Bad R with cross-bilateral
# VERY SLOW BECAUSE PYTHON .......
sx, sc, sk = 5, 0.3, 4
dx = np.array([[(-sk+i)**2 + (-sk+j)**2 for j in range(2*sk+1)] for i in range(2*sk+1)])
dx = np.exp(- 0.5 * dx / sx**2)
W2 = W.copy()
for i in range(sk, w-sk):
	for j in range(sk, h-sk):
		for k in range(N):
			patch = W[k, i-sk:i+sk+1, j-sk:j+sk+1] 
			dc = np.exp(- 0.5 * np.linalg.norm(I[k, i-sk:i+sk+1, j-sk:j+sk+1] - I[k,i,j], axis=2)**2 / sc**2)
			coeff = dx*dc
			coeff /= coeff.sum()
			W2[k,i,j] = (patch * coeff).sum() + 1e-7
	print(i, "/", w)
W2 /= W2.sum(0)
plot_images(W2)
R = (np.expand_dims(W2, 3)*I).sum(0)
plot_images(np.concatenate((I, [R])))

GI, GW = [I], [W]
while min(GW[-1].shape[1], GW[-1].shape[2]) > 4:
	GW.append(np.array([cv.pyrDown(w) for w in GW[-1]]))
	GI.append(np.array([cv.pyrDown(im) for im in GI[-1]]))

# plot_pyramids(GW)
# plot_pyramids(GI)

LI = []
for l in range(len(GI)-1):
	gi_up = np.array([cv.pyrUp(GI[l+1][n], dstsize=(GI[l][n].shape[1], GI[l][n].shape[0])) for n in range(N)])
	LI.append(gi_up - GI[l])
LI.append(GI[-1])

# plot_pyramids(LI)

# Ireconstruct = np.array(LI[-1])
# for l in range(2, len(LI)+1):
# 	li_up = np.array([cv.pyrUp(Ireconstruct[n], dstsize=(LI[-l][n].shape[1], LI[-l][n].shape[0])) for n in range(N)])
# 	Ireconstruct = li_up - LI[-l]
# 	plot_images(Ireconstruct)

LR = []
for l in range(len(GW)):
	LR.append((np.expand_dims(GW[l], 3) * LI[l]).sum(0))

# plot_pyramids([np.expand_dims(lr, 0) for lr in LR])

R = LR[-1]
for l in range(2, len(LR)+1):
	r_up = cv.pyrUp(R, dstsize=(LR[-l].shape[1], LR[-l].shape[0]))
	R = r_up - LR[-l]
R = np.minimum(1, np.maximum(0, R))

cv.imwrite("results/" + name + "_res.jpg", 256*R[:,:,[2,1,0]])
plot_images(np.concatenate((I, [R]), 0))