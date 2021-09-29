import numpy as np
import cv2
import os
import argparse

from numpy.core.records import recarray
from conv import Conv2d
from Pyramid import *
def img_blending(pyramid1:Pyramid,pyramid2:Pyramid,mask:Pyramid):
    h_,w_,c_=pyramid1.GaussianPyramid[-1].shape
    start=pyramid1.GaussianPyramid[-1]*mask.GaussianPyramid[-1]+pyramid2.GaussianPyramid[-1]*(np.ones((h_,w_,c_),float)-mask.GaussianPyramid[-1])
    # cv2.imshow("start",start)
    Blend_lp=[]
    h=pyramid2.height_Laplacian
    for i in range(h):
        h_,w_,c_=pyramid1.GaussianPyramid[i].shape
        Blend_im=pyramid1.LaplacianPyramid[h-i-1]*mask.GaussianPyramid[i]+pyramid2.LaplacianPyramid[h-i-1]*(np.ones((h_,w_,c_),float)-mask.GaussianPyramid[i])
        # cv2.imshow("blend_im"+str(i),Blend_im)
        Blend_lp.append(Blend_im)
    img=start
    for i in range(pyramid1.height_Gaussian-1):
        rec_img=pyramid1.conv2d.pyrUp(img)
        cv2.imshow("before"+str(i),rec_img)
        rec_img=rec_img+Blend_lp[-i-1]
        cv2.imshow("after"+str(i),rec_img)
        img=rec_img
    return img
if __name__ == '__main__':
    orange_dir="/Users/puyuandong613/Downloads/ImageBlending-master/orange.png"
    apple_dir="/Users/puyuandong613/Downloads/ImageBlending-master/apple.png"
    save_dir="/Users/puyuandong613/Downloads/ImageBlending-master/blended_img.png"
    orange_pyramid=Pyramid(orange_dir)
    orange_pyramid.build_GaussianPyramid()
    orange_pyramid.build_LaplacianPyramid()
    apple_pyramid=Pyramid(apple_dir)
    apple_pyramid.build_GaussianPyramid()
    apple_pyramid.build_LaplacianPyramid()
    # apple_pyramid.output_GaussianPyramid()
    # apple_pyramid.output_LaplacianPyramid()
    # orange_pyramid.output_GaussianPyramid()
    # orange_pyramid.output_LaplacianPyramid()
    masks=[]
    h_=IMAGE_H
    w_=IMAGE_W
    for i in range(apple_pyramid.height_Gaussian):
        h_=IMAGE_H/2**i
        w_=IMAGE_W/2**i
        mask=np.zeros(shape=(int(h_),int(w_),3),dtype=float)
        len=mask.shape[1]//2
        mask[:,0:len]=[1.0,1.0,1.0]
        masks.append(mask)
    mask_pyramid=Pyramid()
    mask_pyramid.GaussianPyramid=masks
    blended_img=img_blending(orange_pyramid,apple_pyramid,mask_pyramid)
    # blended_img=img_blending(apple_pyramid,orange_pyramid,mask_pyramid)
    print("blended_img.shape",blended_img.shape)
    cv2.imwrite(save_dir,blended_img*255)
    cv2.imshow("blended_img",blended_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()