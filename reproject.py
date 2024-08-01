import os
import cv2
import glob
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

from multiprocessing import Pool

class Image_Reprojection:

    def __init__(self,
                 data_dir='save3mm_0328_test/data',
                 output_dir='save3mm_0328_test/output',
                 image_dir='board_ir',
                 mode='depth',
                 image_target_size=(1280,1080),
                 image_size=(1280,1080),
                 image_mode='wavue',
                 orbb_image_form='ir', 
                 orbb_type='femto',
                 camera_position = 'center') -> None:
        
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.image_dir = image_dir
        self.mode = mode
        self.image_target_size = image_target_size
        self.image_size = image_size
        self.stereo_params_dir = os.path.join(output_dir,'stereo_parameters')
        self.image_mode = image_mode
        self.orbb_image_form = orbb_image_form
        self.orbb_type = orbb_type
        self.camera_position = camera_position

    def get_paths(self,):
        if self.orbb_image_form == 'color':
            self.camera1_color_dir = os.path.join(self.data_dir,'camera1',self.image_dir)
            f = lambda x: x.replace('color','depth')
        else:
            self.camera1_ir_dir = os.path.join(self.data_dir,'camera1',self.image_dir)
            f = lambda x: x.replace('ir','depth')
        self.camera2_ir_dir = os.path.join(self.data_dir,'camera2',self.image_dir)
        image_cues = self.image_dir.split('_')
        
        depth_cues = np.array(list(map(f,image_cues))).astype('<U6')
        depth_cues = np.insert(depth_cues,-1,'filled')
        depth_cues = '_'.join(depth_cues)
        self.camera1_depth_dir = os.path.join(self.data_dir,'camera1','board_depth_filled')
        # self.camera1_depth_dir = os.path.join(self.data_dir,'camera1','board_depth')
        print('Camera 1 depth diretory: ',self.camera1_depth_dir)
        self.saved_dir = os.path.join(self.output_dir,'camera2','matched_'+self.image_dir+'_'+self.mode)
        if not os.path.exists(self.saved_dir):
            os.makedirs(self.saved_dir)
        print("Output matched images saved directory:\n",self.saved_dir)
            
        
    def get_params(self,):
        self.Kint1 = np.loadtxt(os.path.join(self.output_dir,'camera1',self.image_dir+'_parameters','Kint.txt'))
        self.Kint2 = np.loadtxt(os.path.join(self.output_dir,'camera2',self.image_dir+'_parameters','Kint.txt'))
        print("Kint1:\n",self.Kint1)
        print("Kint2:\n",self.Kint2)
        if self.image_size != self.image_target_size:
            self.Kint2[0] = self.Kint2[0]/self.image_target_size[1]*self.image_size[1]
            self.Kint2[1] = self.Kint2[1]/self.image_target_size[0]*self.image_size[0]
        else:
            pass
        self.R = np.loadtxt(os.path.join(self.stereo_params_dir,'R.txt'))
        self.T = np.loadtxt(os.path.join(self.stereo_params_dir,'T.txt'))
        self.T[0] = self.T[0] * 1000
        self.T[1] = self.T[1] * 1000
        self.T[2] = self.T[2] * 1000
        self.Kint2[1][2] = self.Kint2[1][2]# +22
        print("Kint1:\n",self.Kint1)
        print("Kint2:\n",self.Kint2)
        print("Rotation matrix camera1 to 2:\n",self.R)
        print("Translation matrix camera 1 to 2:\n",self.T)
    
    @staticmethod
    def to_camera(p,H):
        return np.matmul(np.linalg.inv(H),p.T)
    
    def get_matched_image(self,image1,image2,depth):
        v,u = np.mgrid[0:image1.shape[0],0:image1.shape[1]]
        
        p1 = np.c_[u.flatten(),v.flatten(),np.ones(len(u.flatten()))]*depth.flatten()[:,np.newaxis]
        Pc1 = Image_Reprojection.to_camera(p1, self.Kint1)
        Pc2 = np.matmul(self.R,Pc1)+self.T[:,np.newaxis]
        p2 = np.matmul(self.Kint2,Pc2)
        u2 = p2[0]/p2[2]
        v2 = p2[1]/p2[2]
        yy,xx = np.mgrid[:image2.shape[0],:image2.shape[1]]
        matched_image = griddata((u2,v2),image1.flatten(),(xx,yy))
        matched_image = np.nan_to_num(matched_image)

        return matched_image

    def prepare(self,):
        self.get_paths()
        self.get_params()
        camera1_depth_files = np.array(glob.glob(os.path.join(self.camera1_depth_dir,'*.raw')))
        ind = np.array([int(fname.split('/')[-1].split('.')[0].split('_')[2]) for fname in camera1_depth_files])
        # camera1_depth_files = np.array(glob.glob(os.path.join(self.camera1_depth_dir,'*.txt')))
        # ind = np.array([int(fname.split('/')[-1].split('.')[0]) for fname in camera1_depth_files])
        # # Single Gemini camera
        # camera1_depth_files = np.array(glob.glob(os.path.join(self.camera1_depth_dir,'*.txt')))
        # ind = np.array([int(fname.split('/')[-1].split('.')[0]) for fname in camera1_depth_files])
        ind = np.argsort(ind)
        camera1_depth_files = camera1_depth_files[ind]
        num = len(camera1_depth_files)
        self.fnames = camera1_depth_files
        
        print("Number of files: ",num)

    def reproject_image(self,fname):
        # camera1_depth_image = np.loadtxt(fname)
        # camera1_depth_image = np.fromfile(fname,'float32').reshape(480,640)
        if self.orbb_type == 'femto':
            camera1_depth_image = np.fromfile(fname,'float32').reshape(480,640)
        elif self.orbb_type == 'zed':
            print(fname)
            camera1_depth_image = np.fromfile(fname,'float32').reshape(720,1280)
        else:
            print(fname)
            camera1_depth_image = np.fromfile(fname,'float32').reshape(360,640)
        if self.image_mode == 'wavue':
            if self.image_target_size == (1280,1080):
                if self.camera_position == 'right':
                    camera1_depth_image = cv2.rotate(camera1_depth_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    camera1_depth_image = cv2.resize(camera1_depth_image, [1080, 1440])
                    camera1_depth_image = camera1_depth_image[80:-80, :]
                    camera1_depth_image = cv2.flip(camera1_depth_image, 1)
                else:
                    if self.orbb_type == 'gemini':
                        camera1_depth_image = cv2.resize(camera1_depth_image,[2276, 1280])
                        camera1_depth_image = camera1_depth_image[:,598:-598]
                        # camera1_depth_image = cv2.flip(camera1_depth_image, 1)
                    elif self.orbb_type == 'femto':
                        camera1_depth_image = cv2.resize(camera1_depth_image,[1706, 1280])
                        camera1_depth_image = camera1_depth_image[:,313:-313]
                        camera1_depth_image = cv2.flip(camera1_depth_image, 1)
                    else:
                        camera1_depth_image = cv2.resize(camera1_depth_image,[2276, 1280])
                        camera1_depth_image = camera1_depth_image[:,598:-598]
            elif self.image_target_size == (1920,1080):
                camera1_depth_image = cv2.resize(camera1_depth_image,[1440, 1920])
                camera1_depth_image = camera1_depth_image[:, 180:-180]
            else:
                if self.orbb_type == 'femto':
                        camera1_depth_image = cv2.resize(camera1_depth_image,[1920, 1440])
                        camera1_depth_image = camera1_depth_image[180:-180,:]
                        camera1_depth_image = cv2.flip(camera1_depth_image, 1)
                else:
                    camera1_depth_image = cv2.resize(camera1_depth_image,[1920, 1080])
                    # camera1_depth_image = camera1_depth_image[:,598:-598]
                if self.orbb_type == 'gemini':
                    camera1_depth_image = camera1_depth_image / 5

        # # Single Gemini camera
        # timestamp = int(fname.split('/')[-1].split('.')[0])
        timestamp = int(fname.split('/')[-1].split('.')[0].split('_')[2])
        if self.orbb_image_form == 'color':
            camera1_image_name = os.path.join(self.camera1_color_dir,'{}.png'.format(timestamp))
        else:
            camera1_image_name = os.path.join(self.camera1_ir_dir,'{}.png'.format(timestamp))
              
        camera2_ir_image_name = os.path.join(self.camera2_ir_dir,'{}.png'.format(timestamp))
        if os.path.exists(camera1_image_name) and os.path.exists(camera2_ir_image_name):
            if self.mode == 'depth':
                camera1_image = camera1_depth_image
            else:
                camera1_image = cv2.imread(camera1_image_name)[:,:,0]
                if self.image_target_size == (1280,1080):
                    if self.orbb_type == 'gemini':
                        camera1_image = cv2.resize(camera1_image,[2276, 1280])
                        camera1_image = camera1_image[:,598:-598]
                        # camera1_depth_image = cv2.flip(camera1_depth_image, 1)
                    elif self.orbb_type == 'femto':
                        camera1_image = cv2.resize(camera1_image,[1706, 1280])
                        camera1_image = camera1_image[:,313:-313]
                        camera1_image = cv2.flip(camera1_image, 1)
                    else:
                        camera1_image = cv2.resize(camera1_image,[2276, 1280])
                        camera1_image = camera1_image[:,598:-598]
                elif self.image_target_size == (1920,1080):
                    camera1_image = cv2.resize(camera1_image,[1440, 1920])
                    camera1_image = camera1_image[:,180:-180]
                else:
                    if self.orbb_type == 'femto':
                        camera1_image = cv2.resize(camera1_image,[1920, 1440])
                        camera1_image = camera1_image[180:-180,:]
                        camera1_image = cv2.flip(camera1_image, 1)
                    else:
                        camera1_image = cv2.resize(camera1_image,[1920, 1080])
                        # camera1_image = camera1_image[:,598:-598]
                    # if self.orbb_type == 'gemini':
                    # camera1_depth_image = camera1_depth_image / 5
            camera2_ir_image = cv2.imread(camera2_ir_image_name)
            camera2_ir_image = cv2.cvtColor(camera2_ir_image, cv2.COLOR_RGB2GRAY)
            self.image_size = camera2_ir_image.shape
            matched_image = self.get_matched_image(camera1_image,camera2_ir_image,camera1_depth_image)

            if self.mode == 'depth':
                matched_image = matched_image.astype('f4')
                matched_image.tofile(os.path.join(self.saved_dir,'{}.txt'.format(timestamp)))
            else:
                matched_image = np.tile(np.expand_dims(matched_image,-1),(1,1,3))
                matched_image = matched_image.astype('int')
                cv2.imwrite(os.path.join(self.saved_dir,'{}.png'.format(timestamp)),matched_image)
            # time.sleep(20)

    def reproject(self,num_processes=5):
        self.prepare()
        pool = Pool(num_processes)
        pool.map(self.reproject_image,self.fnames)
        pool.close()
        pool.join()

if __name__=="__main__":

    # tool = Image_Reprojection(data_dir='0814_calib/femto_left/data',
    #                             output_dir='0814_calib/femto_left/output', 
    #                             orbb_image_form='color',
    #                             orbb_type='femto')
    # tool.reproject()
    
    # camera_index = '1'
    # base_dir = '1012_calib/'
    # camera_type = 'dabai'
    # tool = Image_Reprojection(data_dir=base_dir + camera_index + '/data',
    #                             output_dir=base_dir + camera_index +'/output',
    #                             image_dir='board_color',
    #                             image_target_size=(1080,1920),
    #                             image_size=(1080,1920),
    #                             mode='color', 
    #                             orbb_image_form='color',
    #                             orbb_type=camera_type)
    # tool.reproject()
    tool = Image_Reprojection(data_dir='0206_calib/0/data',
                                output_dir='0206_calib/0/output',
                                # image_dir='board_color',
                                image_dir='board_ir',
                                image_target_size=(1080,1920),
                                image_size=(1080,1920),
                                # image_target_size=(1920,1080),
                                # image_size=(1920,1080),
                                mode='color', 
                                orbb_image_form='color',
                                orbb_type='dabai')
    tool.reproject()