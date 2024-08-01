import os
import cv2
import time
import numpy as np
import time
import matplotlib.pyplot as plt

from multiprocessing import Pool

class Camera_Undistortion:

    def __init__(self,
                 image_target_size=(1280,1056),
                 camera='1',
                 root_dir='data',
                 data_dir='output',
                 image_dir='board_ir',
                 params_dir='board_ir_parameters') -> None:
        self.image_target_size = image_target_size
        self.params_path = os.path.join(data_dir,'camera'+camera,params_dir)
        print(self.params_path)
        
        self.image_path = os.path.join(root_dir,'camera'+camera,image_dir)
        assert os.path.exists(self.params_path)
        assert os.path.exists(self.image_path)
        print("Reading camera parameters from file: ",self.params_path)
        print("Reading distorted images from file: ", self.image_path)
        self.saved_path = os.path.join(data_dir,'camera'+camera,image_dir+'_und')
        if not os.path.exists(self.saved_path):
            os.makedirs(self.saved_path)
        print("Output undistorted images saved directory:\n",self.saved_path)
        
    def get_params(self,):
        self.Kint = np.loadtxt(os.path.join(self.params_path,'Kint.txt'))
        self.Kdist = np.loadtxt(os.path.join(self.params_path,'Kdist.txt'))

    def get_images(self,):
        files = os.listdir(self.image_path)
        num = len(files)
        print("Total number of files: ",num)
        self.fnames = np.array([os.path.join(self.image_path, files[i]) for i in range(num)])
        test_fname = files[0]
        if test_fname.endswith('png'):
            image = cv2.imread(self.fnames[0])
            self.image_size = image.shape[:2]
        else:
            image = np.loadtxt(self.fnames[0])
            self.image_size = image.shape
    
    def prepare(self,):
        self.get_params()
        self.get_images()
        if self.image_size != self.image_target_size:
            self.Kint[0] = self.Kint[0]/self.image_target_size[1]*self.image_size[1]
            self.Kint[1] = self.Kint[1]/self.image_target_size[0]*self.image_size[0]
        else:
            pass
        print("Kint:\n",self.Kint)
        self.Knew, self.roi = cv2.getOptimalNewCameraMatrix(self.Kint,
                                                            self.Kdist,
                                                            self.image_size[::-1],
                                                            0,
                                                            self.image_size[::-1])
        
        print("Knew:\n",self.Knew)
        print("roi:\n",self.roi)

    def undistort_image(self,fname):
        try:
            if fname.endswith('png'):
                image = cv2.imread(fname)
            else:
                image = np.loadtxt(fname)
            image_name = fname.split('/')[-1]
            t0 = time.time()
            image_und = cv2.undistort(image, self.Kint, self.Kdist, None, self.Knew)
            x, y, w, h = self.roi
            image_und = image_und[y:y+h, x:x+w]
            image_und = cv2.resize(image_und,[*self.image_size[::-1]])
            if fname.endswith('png'):
                cv2.imwrite(os.path.join(self.saved_path, image_name), image_und)
                print(time.time()-t0)
                # np.savetxt(os.path.join(self.params_path,'Knew.txt'),self.Knew)
            else:
                plt.imshow(image_und)
                plt.axis('off')
                plt.savefig(os.path.join(self.saved_path,image_name),bbox_inches='tight',pad_inches=0)
                np.savetxt(os.path.join(self.saved_path,image_name),image_und)
                time.sleep(20)
        except FileNotFoundError:
            pass

    def undistort_images(self,num_processes=16):
        self.prepare()
        pool = Pool(num_processes)
        pool.map(self.undistort_image,self.fnames)
        pool.close()
        pool.join()
            

if __name__=="__main__":
    # tool = Camera_Undistortion(image_target_size=(1080,1920),
    #                            camera='1',
    #                            root_dir='0905_calib_copy/0/data',
    #                            data_dir='0905_calib_copy/0/output')
    # tool.undistort_images()
    tool = Camera_Undistortion(image_target_size=(1080,1920),
                               camera='2',
                               root_dir='0325_calib/1/data',
                               data_dir='0325_calib/1/output',
                            #    image_dir='board_color',
                            #    params_dir='board_color_parameters')
                               image_dir='board_ir',
                               params_dir='board_ir_parameters')
    tool.undistort_images()


