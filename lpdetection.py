import sys, os
import keras
import cv2
import traceback

from src.keras_utils 			import load_model
from glob 						import glob
from os.path 					import splitext, basename
from src.utils 					import im2single
from src.keras_utils 			import load_model, detect_lp
from src.label 					import Shape, writeShapes
import numpy as np

def adjust_pts(pts,lroi):
	return pts*lroi.wh().reshape((2,1)) + lroi.tl().reshape((2,1))


def detect_license(Ivehicle,wpod_net):

	# try:

	lp_threshold = .9

	#print ('Searching for license plates using WPOD-NET')

	ratio = float(max(Ivehicle.shape[:2]))/min(Ivehicle.shape[:2])
	side  = int(ratio*288.)
	bound_dim = min(side + (side%(2**4)),608)
	#print ("\t\tBound dim: %d, ratio: %f" % (bound_dim,ratio))

	Llp,LlpImgs,_ = detect_lp(wpod_net,im2single(Ivehicle),bound_dim,2**4,(160,160),lp_threshold)

	if len(LlpImgs):
		Ilp = LlpImgs[0]
		Ilp = cv2.cvtColor(Ilp, cv2.COLOR_BGR2GRAY)
		Ilp = cv2.cvtColor(Ilp, cv2.COLOR_GRAY2BGR)

		s = Shape(Llp[0].pts)

		#print("shape:::::",Llp[0].pts)
		width = (Llp[0].pts[0][2]-Llp[0].pts[0][3])**2+(Llp[0].pts[1][2]-Llp[0].pts[1][3])**2
		height =(Llp[0].pts[0][3]-Llp[0].pts[0][0])**2+(Llp[0].pts[1][3]-Llp[0].pts[1][0])**2
		# print("+++++++++++++width:" + str(width))
		# print("+++++++++++++height:" + str(height))
		if (width < 0.005):
			return None, None, None

		scale =np.sqrt(width/height)
		# cv2.imwrite('lp.png',Ilp*255.)
		return Ilp*255.,Llp,scale
	else:
		return None, None, None
	# except:
	# 	traceback.print_exc()



