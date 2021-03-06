import argparse
import cv2 as cv
import os
import pylab as pl
import numpy as np

parser = argparse.ArgumentParser(description='Compute a fusion of several images to obtain an HDR image')
parser.add_argument("folder", type=str, help='folder where the images are stored')
parser.add_argument("-s", "--save", help="save result images and weight maps", action="store_true")
parser.add_argument("-p", "--plot_more", help="plot intermediate images as W or pyramids", action="store_true")
parser.add_argument("-f", "--fusion", help="Choose the algorithm of fusion:\
	0 -> Naive fusion\
	1 -> Gaussian Blur on W\
	2 -> Cross-Bilateral filter on W and I\
	3 (default) -> Using Laplacian pyramids", type=int, default=3)
args = parser.parse_args()
folder = args.folder
save = args.save
plot_more = args.plot_more
fusion = args.fusion
if fusion > 3 or fusion < 0: fusion = 3

#### PARAMETERS ####
ind = wc, ws, we = 1, 1, 1
sigmaE = 0.2
####################

files = os.listdir(folder)
if '.DS_Store' in files:
	files.remove('.DS_Store')
name=files[0].split('_')[0]
print(files)
I = np.array([cv.imread(folder + "/" + f) for f in files])
I = I[:,:,:,[2, 1, 0]]
gray = np.array([cv.cvtColor(im, cv.COLOR_RGB2GRAY) for im in I])
I = I / 255.0
N, w, h, c = I.shape

def to01(im):
	mi = im.min((0,1))
	ma = im.max((0,1))
	return (im - mi) / (ma - mi)

def plot_images(images, cmap='viridis', clip=False):
	pl.figure(figsize=(15,12))
	n=len(images)
	for i in range(n):
		pl.subplot((n+1)//2,2,i+1)
		im = images[i].squeeze()
		cmap = 'gray' if len(im.shape)==2 else cmap
		pl.imshow(to01(im) if clip else im)
		pl.xticks([])
		pl.yticks([])
	pl.tight_layout(pad=0,h_pad=0,w_pad=0)
	pl.show()

def plot_pyramids(pyr, clip=False):
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
		im[:w,:h,:] = to01(pyr[0][i]) if clip else pyr[0][i]
		x = 0
		for k in range(1, K):
			a, b, _ = pyr[k][i].shape
			im[-a:, x:x+b,:] = to01(pyr[k][i]) if clip else pyr[k][i]
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
# Compute E
E = np.exp(-((I.mean(3) - 0.5)**2 / (2*sigmaE**2)))
# Compute W
W = C**wc * S**ws * E**we
W /= W.sum(0)

# Eventually save it
if save:
	for i in range(N):
		cv.imwrite(f"results/{name}_w{i}.jpg", 256*W[i])

def naive(I, W, plotW=False, save=False):
	# Bad R
	if plotW: plot_images(W)
	R = (np.expand_dims(W, 3)*I).sum(0)
	if save: cv.imwrite("results/" + name + "_base.jpg", 256*R[:,:,[2,1,0]])
	plot_images(np.concatenate((I, [R])))

def gaussian(I, W, plotW=False, save=False):
	# Bad R with gaussian
	W = np.array([cv.GaussianBlur(wi, (9,9), 5) for wi in W])
	if plotW: plot_images(W)
	R = (np.expand_dims(W, 3)*I).sum(0)
	if save: cv.imwrite("results/" + name + "_gauss.jpg", 256*R[:,:,[2,1,0]])
	plot_images(np.concatenate((I, [R])))

def cross_bil(I, W, plotW=False, save=False):
	# Bad R with cross-bilateral
	# VERY SLOW BECAUSE PYTHON .......
	sx, sc, sk = 5, 0.18, 4
	dx = np.array([[(-sk+i)**2 + (-sk+j)**2 for j in range(2*sk+1)] for i in range(2*sk+1)])
	dx = np.exp(- 0.5 * dx / sx**2)
	W2 = W.copy()
	K = len(I)
	for i in range(sk, w-sk):
		for j in range(sk, h-sk):
			patch = W[:, i-sk:i+sk+1, j-sk:j+sk+1] 
			dc = np.exp(- 0.5 * np.linalg.norm(I[:, i-sk:i+sk+1, j-sk:j+sk+1] - I[:,i,j].reshape((K,1,1,3)), axis=3)**2 / sc**2)
			coeff = dx*dc
			coeff /= coeff.sum((1,2), keepdims=True)
			W2[:,i,j] = (patch * coeff).sum((1,2)) + 1e-7
		print(i, "/", w)
		# print(patch)
	W2 /= W2.sum(0)
	plot_images(W2)
	R = (np.expand_dims(W2, 3)*I).sum(0)
	if save: cv.imwrite("results/" + name + "_cross.jpg", 256*R[:,:,[2,1,0]])
	plot_images(np.concatenate((I, [R])))

def pyramid(I, W, plot=False, save=False, ind=(1,1,1)):
	GI, GW = [I], [W]
	while min(GW[-1].shape[1], GW[-1].shape[2]) > 4:
		GW.append(np.array([cv.pyrDown(w) for w in GW[-1]]))
		GI.append(np.array([cv.pyrDown(im) for im in GI[-1]]))

	if plot:
		plot_pyramids(GW)
		plot_pyramids(GI)

	LI = []
	for l in range(len(GI)-1):
		gi_up = np.array([cv.pyrUp(GI[l+1][n], dstsize=(GI[l][n].shape[1], GI[l][n].shape[0])) for n in range(N)])
		LI.append(GI[l] - gi_up)
	LI.append(GI[-1])

	if plot: plot_pyramids(LI, clip=True)

	LR = []
	for l in range(len(GW)):
		LR.append((np.expand_dims(GW[l], 3) * LI[l]).sum(0))

	if plot: plot_pyramids([np.expand_dims(lr, 0) for lr in LR], clip=True)

	R = LR[-1]
	for l in range(2, len(LR)+1):
		r_up = cv.pyrUp(R, dstsize=(LR[-l].shape[1], LR[-l].shape[0]))
		R = r_up + LR[-l]
	R = np.clip(R, 0, 1)

	if save: cv.imwrite("results/" + name + "_res"+str(ind)+".jpg", 256*R[:,:,[2,1,0]])
	plot_images(np.concatenate((I, [R]), 0))

if fusion == 0: naive(I, W, plotW=plot_more, save=save)
elif fusion == 1: gaussian(I, W, plotW=plot_more, save=save)
elif fusion == 2: cross_bil(I, W, plotW=plot_more, save=save)
elif fusion == 3: pyramid(I, W, plot=plot_more, save=save, ind=(wc,ws,we))