import numpy as np
import cv2
import os
from conv import Conv2d
IMAGE_H=256
IMAGE_W=256
class Pyramid():
    def __init__(self,img_dir=""):
        self.kernel=np.array([
            [1, 4, 6, 4, 1],
            [4,16,24,16, 4],
            [6,24,36,24, 6],
            [4,16,24,16, 4],
            [1, 4, 6, 4, 1],
        ])/256
        self.GaussianPyramid=[]
        self.LaplacianPyramid=[]
        self.img_dir=img_dir
        self.img=None
        self.conv2d=Conv2d(self.kernel)
        if img_dir!="" and os.path.exists(img_dir):
            self.img=self.img_read()
    def img_read(self):
        img = cv2.imread(self.img_dir)/255
        img_ = cv2.resize(img, (IMAGE_H, IMAGE_W))
        return img_
    def build_GaussianPyramid(self,height=4):
        self.height_Gaussian=height+1
        img=self.img
        self.GaussianPyramid.append(img)
        for i in range(height):
            new_img=self.conv2d.pyrDown(img)
            self.GaussianPyramid.append(new_img)
            img=new_img
    def output_GaussianPyramid(self):
        for i in range(self.height_Gaussian):
            cv2.imshow("img"+str(i),self.GaussianPyramid[i])
            # print(self.GaussianPyramid[i].shape)
    def build_LaplacianPyramid(self):
        self.height_Laplacian=self.height_Gaussian-1
        img=self.GaussianPyramid[-1]
        # print(len(self.GaussianPyramid))
        l=self.height_Laplacian
        print("length of GaussianPyramid:",l)
        for i in range(l):
            rec_img=self.conv2d.pyrUp(img)
            print("rec_img.shape:",rec_img.shape)
            res_img=self.GaussianPyramid[l-i-1]-rec_img
            img=rec_img
            self.LaplacianPyramid.append(res_img)
    def output_LaplacianPyramid(self):
        for i in range(self.height_Laplacian):
            cv2.imshow("res_img"+str(i),self.LaplacianPyramid[i])