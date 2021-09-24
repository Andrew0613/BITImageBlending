import numpy as np
import cv2
import os
import argparse
from conv import Conv2d
from Pyramid import *
def img_blending(pyramid1:Pyramid,pyramid2:Pyramid,mask:Pyramid):
    h_,w_,c_=pyramid1.GaussianPyramid[-1].shape
    start=pyramid1.GaussianPyramid[-1]*mask.GaussianPyramid[-1]+pyramid2.GaussianPyramid[-1]*(np.ones((h_,w_,c_),float)-mask.GaussianPyramid[-1])
    Blend_lp=[]
    h=pyramid2.height_Laplacian
    for i in range(h):
        h_,w_,c_=pyramid1.GaussianPyramid[i].shape
        Blend_im=pyramid1.LaplacianPyramid[h-i-1]*mask.GaussianPyramid[i]+pyramid2.LaplacianPyramid[h-i-1]*(np.ones((h_,w_,c_),float)-mask.GaussianPyramid[i])
        Blend_lp.append(Blend_im)
    img=start

    for i in range(h):
        rec_img=mask.conv2d.pyrUp(img)
        rec_img=rec_img+Blend_lp[h-i-1]
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
    # orange_pyramid.output_LaplacianPyramid()
    # apple_pyramid.output_LaplacianPyramid()
    mask=np.zeros(shape=(IMAGE_H,IMAGE_W,3),dtype=float)
    len=mask.shape[1]//2
    mask[:,0:len,:]=[1.0,1.0,1.0]
    mask_pyramid=Pyramid()
    mask_pyramid.img=mask
    mask_pyramid.build_GaussianPyramid()
    blended_img=img_blending(orange_pyramid,apple_pyramid,mask_pyramid)
    print("blended_img.shape",blended_img)
    cv2.imwrite(save_dir,blended_img*255)
    cv2.imshow("blended_img",blended_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()