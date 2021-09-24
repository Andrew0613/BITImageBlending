import numpy as np
import cv2
class Conv2d:
    def __init__(self,kernel, in_plains=3, out_plains=3,):
        kernel_param_num = kernel.shape[0] * kernel.shape[1]
        # self.kernel = np.ones(kernel_size) / kernel_param_num
        self.kernel = kernel
        self.kernel_size = kernel.shape
        self.in_plains = in_plains
        self.out_plains = out_plains
        self.H = 0
        self.W = 0
        self.C = out_plains
    def convolution(self,img,stride=1, padding=0):
        h, w, c = img.shape
        self.H = (h - self.kernel_size[0] + 2*padding)//stride + 1
        self.W = (w - self.kernel_size[1] + 2*padding)//stride + 1
        tensor = np.zeros([self.H, self.W,self.C])
        
        h_ = h+2*padding
        w_ = w+2*padding
        img_padded = np.zeros([h_, w_, c])
        img_padded[padding:h_-padding,
                    padding:w_-padding,:] = img
        img_new = np.zeros([self.H, self.W,self.C])
        for channel in range(c):
            for i in range(self.H):
                for j in range(self.W):
                    i_ = i*stride
                    j_ = j*stride
                    a = img_padded[i_:i_ +
                                    self.kernel_size[0], j_:j_+self.kernel_size[1],channel]
                    img_new[i, j,channel] = np.sum(a*self.kernel)
        tensor = img_new
        return tensor
    def pyrDown(self,img,stride=2,padding=2):
        new_img=self.convolution(img,stride,padding)
        return new_img
    def pyrUp(self,img,stride=1,padding=2):#unpooling
        unpooled_img=self.unpooling(img)
        # cv2.imshow("unpooled_img",unpooled_img)
        new_img=self.convolution(unpooled_img,stride,padding)
        return new_img*4
    def unpooling(self,img):
        new_img=[]
        for i in range(len(img)):
            array=[]
            for j in range(len(img[i])):
                array.append(img[i][j])
                array.append([0,0,0])
                # array.append(img[i][j])
            new_img.append(np.array(array))
            # new_img.append(np.array(array))
            arr=[[0,0,0]]*2*len(img[i])
            arr=np.array(arr)
            new_img.append(arr)
        new_img=np.array(new_img)
        # print("new_img shape:",new_img.shape)
        return new_img
