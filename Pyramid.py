import numpy as np
import cv2
import os
from conv import Conv2d
IMAGE_H=256
IMAGE_W=256
class Pyramid():
    def __init__(self,img_dir=""):
        self.kernel1=np.array([
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
        self.conv2d=Conv2d(self.kernel1)
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
            # new_img=cv2.pyrDown(img)
            self.GaussianPyramid.append(new_img)
            img=new_img
    def output_GaussianPyramid(self):
        for i in range(self.height_Gaussian):
            cv2.imshow("img"+str(i),self.GaussianPyramid[i])
            # print(self.GaussianPyramid[i].shape)
    def build_LaplacianPyramid(self):
        self.height_Laplacian=self.height_Gaussian-1
        
        # print(len(self.GaussianPyramid))
        l=self.height_Laplacian
        print("length of GaussianPyramid:",l)
        for i in range(l):
            img=self.GaussianPyramid[-1-i]
            # rec_img=cv2.pyrUp(img)
            rec_img=self.conv2d.pyrUp(img)
            res_img=(self.GaussianPyramid[l-i-1]-rec_img)
            img=rec_img
            self.LaplacianPyramid.append(res_img)
    def output_LaplacianPyramid(self):
        for i in range(self.height_Laplacian):
            cv2.imshow("res_img"+str(i),self.LaplacianPyramid[i])
if __name__ == '__main__':
    dir="/Users/puyuandong613/Downloads/ImageBlending-master/lena.jpg"
    save_dir="/Users/puyuandong613/Downloads/ImageBlending-master/new_lena.jpg"
    pyramid=Pyramid(img_dir=dir)
    # #使用cv2
    # img=cv2.resize(cv2.imread(dir), (IMAGE_H, IMAGE_W))/255
    # new_img=cv2.pyrDown(img)
    # rec_img=cv2.pyrUp(new_img)
    # res_img=img-rec_img
    # cv2.imshow("1",new_img)
    # cv2.imshow("11",rec_img)
    # cv2.imshow("111",res_img)
    # # cv2.imwrite(save_dir,res_img*255)
    # #使用自定义
    # new_img=pyramid.conv2d.pyrDown(img)
    # rec_img=pyramid.conv2d.pyrUp(new_img)
    # res_img=img-rec_img
    # cv2.imshow("2",new_img)
    # cv2.imshow("22",rec_img)
    # cv2.imshow("222",res_img)

    pyramid.build_GaussianPyramid()
    pyramid.build_LaplacianPyramid()
    cv2.imshow("1",pyramid.GaussianPyramid[1])
    cv2.imshow("2",pyramid.LaplacianPyramid[-1])
    # pyramid.output_LaplacianPyramid()
    # pyramid.output_GaussianPyramid()
    cv2.waitKey(0)
    cv2.destroyAllWindows()