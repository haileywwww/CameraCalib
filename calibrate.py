import os
import cv2
import glob
import numpy as np
from PIL import Image, ImageEnhance

class Camera_Calibration:

    def __init__(
            self,
            num_h=5,
            num_w=8,
            size=39.5,
            eps=200,
            tol=.001,
            kernel_size=11,
            image_target_size=(1280,1080),
            camera='1',
            orbb_type='femto',
            image_form='ir',
            data_dir='save3mm_0328_test/data',
            saved_dir='save3mm_0328_test/output',
            image_dir='board_ir',
            mode='resize',
            points_file_dir='./save3mm_0328_test/',
            points_file_name='_points_0328.txt'):

        self.num_h = num_h
        self.num_w = num_w
        self.size = size
        self.eps = eps
        self.tol = tol
        self.kernel_size = kernel_size
        self.image_target_size = image_target_size
        self.camera = camera
        self.orbb_type = orbb_type
        self.image_form = image_form
        self.image_path = os.path.join(data_dir,'camera'+camera,image_dir)
        self.saved_image_path = os.path.join(saved_dir,'camera'+camera,image_dir+'_corners')
        self.saved_params_path = os.path.join(saved_dir,'camera'+camera,image_dir+'_parameters')
        self.saved_points_path = os.path.join(saved_dir,'camera'+camera,image_dir+'_points')
        self.mode = mode
        self.points_file_dir = points_file_dir
        self.points_file_name = points_file_name
        assert os.path.exists(self.image_path)
        print("Reading images from:\n",self.image_path)
        if not os.path.exists(self.saved_image_path):
            os.makedirs(self.saved_image_path)
        print("Output corner images saved directory:\n",self.saved_image_path)
        if not os.path.exists(self.saved_params_path):
            os.makedirs(self.saved_params_path)
        print("Output camera parameters saved directory:\n",self.saved_params_path)
        if not os.path.exists(self.saved_points_path):
            os.makedirs(self.saved_points_path)
        print("Output corner points saved directory:\n",self.saved_points_path)

    def get_criteria(self,):
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, self.eps, self.tol)

    def get_objp(self,):
        w,h = self.num_w-1, self.num_h-1
        objp = np.zeros((w*h,3), np.float32)
        objp[:,:2] = np.mgrid[0:h,0:w].T.reshape(-1,2)
        objp = self.size/1e3 * objp
        self.objp = objp

    def get_images(self,):
        if self.mode == 'load':
            # name_dict = {'1':'orbb','2':'wavue'}
            name_dict = {'1':'color','2':'wavue'}
            image_name, corners = self.get_coordinates(self.points_file_dir + name_dict[self.camera] + self.points_file_name)
            
            # num = len(os.listdir(self.image_path))
            # fnames = np.array(glob.glob(os.path.join(self.image_path,'*.png')))
            # ind = np.array([int(fname.split('/')[-1].split('.')[0]) for fname in fnames])
            # ind = np.argsort(ind)
            self.fnames = image_name
            print(self.fnames)
            #np.array([os.path.join(self.image_path,'{}.png').format(i+1) for i in range(num)])
        else:
            image_names = []
            num = 100
            for i in range(num):
                if self.image_form == 'color':
                    image_name = self.image_path + '/' + 'Color_' + str(i) + '_' + str(int(self.camera)-1) + '.png'
                else:
                    image_name = self.image_path + '/' + 'Ir_' + str(i) + '_' + str(int(self.camera)-1) + '.png'
                if os.path.exists(image_name):
                    image_names.append(image_name)
            self.fnames = image_names

    def get_coordinates(self,fname):
        image_name = []
        coords = []
        for t in open(fname).read().split():
            if not 'png' in t:
                a, b = t.strip('()').split(',')
                coords.append((int(a), int(b)))
            else:
                image_name.append(t)
        coords = np.array(coords).reshape(-1,(self.num_h-1)*(self.num_w-1),1,2)
        
        return image_name, coords
    
    def find_corners(self,):
        self.get_criteria()
        self.get_images()
        self.get_objp()
        w,h = self.num_w-1, self.num_h-1
        objpoints = []
        imgpoints = []
        count = 0
        print("Number of files:",len(self.fnames))

        if self.mode == 'load':
            for fname in self.fnames:
                timestamp = 0
                if self.camera == '1':
                    timestamp = fname.split('/')[-1].split('.')[0].split('_')[1]
                    fname = os.path.join(self.image_path, timestamp + '.png')
                elif self.camera == '2':
                    timestamp = fname.split('/')[-1].split('.')[0]
                    # fname = os.path.join(self.image_path, timestamp + '.png')
                    # timestamp = fname.split('/')[-1].split('.')[0].split('_')[1]
                    fname = os.path.join(self.image_path, timestamp + '.png')
                print(fname)
                image = cv2.imread(fname)
                self.image_size = image.shape[:2]
                if self.image_size != self.image_target_size:
                    # image = cv2.resize(image,[*self.image_target_size[::-1]])
                    if self.image_target_size == (1280,1080):
                        if self.orbb_type == 'femto':
                            image = cv2.resize(image, [1706, 1280])
                            image = image[:, 313:-313]
                            image = cv2.flip(image, 1)
                        elif self.orbb_type == 'gemini':
                            image = cv2.resize(image, [2276, 1280])
                            image = image[:, 598:-598]
                        elif self.orbb_type == 'dabai':
                            image = cv2.resize(image, [2276, 1280])
                            image = image[:, 598:-598]
                    elif self.image_target_size == (1920,1080):
                        image = cv2.resize(image, [1440, 1920])
                        image = image[:, 180:-180]
                    else:
                        if self.orbb_type == 'femto':
                            image = cv2.resize(image, [1920, 1440])
                            image = image[180:-180, :]
                            image = cv2.flip(image, 1)
                        else:
                            image = cv2.resize(image, [1920, 1080])
                else:
                    pass
                gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

                count += 1
                # name_dict = {'1':'orbb','2':'wavue'}
                name_dict = {'1':'color','2':'wavue'}
                image_name, corners_all = self.get_coordinates(self.points_file_dir + name_dict[self.camera] + self.points_file_name)
                corners = corners_all[count-1]
                corners = corners.astype('f4')
                objpoints.append(self.objp)
                imgpoints.append(corners)
                cv2.drawChessboardCorners(image, (h,w), corners, True)
                cv2.namedWindow('findCorners', cv2.WINDOW_NORMAL)
                cv2.imshow('findCorners',image)
                cv2.imwrite(os.path.join(self.saved_image_path, 'corners_' + timestamp + '.png'),image)
                cv2.waitKey(200)

        else:
            for fname in self.fnames:
                img_name = fname.split('/')[-1]
                image = cv2.imread(fname)
                self.image_size = image.shape[:2]
                if self.image_size != self.image_target_size:
                    image = cv2.resize(image,[*self.image_target_size[::-1]])
                else:
                    pass
                gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        
                if self.mode == 'manual':
                    print("gray:",gray.shape)
                    corners = []
                    def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
                        if event == cv2.EVENT_LBUTTONDOWN:
                            xy = "%d,%d" % (x, y)
                            print(x,y)
                            corners.append([[x,y]])
                            cv2.circle(image, (x, y), 2, (0, 0, 255))
                            cv2.putText(image, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,1.0, (0,0,255))
                            cv2.imshow("image", image)
                    cv2.namedWindow("image")
                    cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN)
                    while(1):
                        cv2.imshow("image", image)
                        key = cv2.waitKey(5) & 0xFF
                        if key == ord('q'):
                            break
                    corners = np.asarray(corners).reshape(-1,1,2).astype('f4')
                    objpoints.append(self.objp)
                    imgpoints.append(corners)
                    cv2.drawChessboardCorners(image, (h,w), corners, True)
                    cv2.namedWindow('findCorners', cv2.WINDOW_NORMAL)
                    cv2.imshow('findCorners',image)
                    cv2.imwrite(self.saved_image_path + '/' + 'corners_' + img_name, image)
                    cv2.waitKey(200)

                elif self.mode == 'resize':
                    gray = cv2.resize(gray,[640,480])
                    ret, corners = cv2.findChessboardCorners(gray, (h,w),None)
                    if ret:
                        count += 1
                        cv2.cornerSubPix(gray,corners,(self.kernel_size,self.kernel_size),(-1,-1),self.criteria)
                        corners[:,0,0] = corners[:,0,0]/640*self.image_target_size[1]
                        corners[:,0,1] = corners[:,0,1]/480*self.image_target_size[0]
                        objpoints.append(self.objp)
                        imgpoints.append(corners)
                        cv2.drawChessboardCorners(image, (h,w), corners, ret)
                        cv2.namedWindow('findCorners', cv2.WINDOW_NORMAL)
                        cv2.imshow('findCorners',image)
                        cv2.imwrite(self.saved_image_path + '/' + 'corners_' + img_name, image)
                        cv2.waitKey(200)

                        
                else:
                    ret, corners = cv2.findChessboardCorners(gray, (h,w),None)
                    if ret:
                        count += 1
                        cv2.cornerSubPix(gray,corners,(self.kernel_size,self.kernel_size),(-1,-1),self.criteria)
                        objpoints.append(self.objp)
                        imgpoints.append(corners)
                        cv2.drawChessboardCorners(image, (h,w), corners, ret)
                        cv2.namedWindow('findCorners', cv2.WINDOW_NORMAL)
                        cv2.imshow('findCorners',image)
                        cv2.imwrite(self.saved_image_path + '/' + 'corners_' + img_name, image)
                        cv2.waitKey(200)

    
        print("Number of files with corners found: ",count)
        cv2.destroyAllWindows()
        objpoints = np.array(objpoints).astype('f4')
        imgpoints = np.array(imgpoints).astype('f4')
        objpoints.tofile(os.path.join(self.saved_points_path,'objpoints.bin'))
        imgpoints.tofile(os.path.join(self.saved_points_path,'imgpoints.bin'))

        return objpoints, imgpoints
    
    def get_camera_matrices(self,):
        objpoints, imgpoints = self.find_corners()
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, 
                                                           imgpoints, 
                                                           self.image_target_size[::-1], 
                                                           None, None)
        R = cv2.Rodrigues(rvecs[-1])[0]
        T = tvecs[-1]
        print("error:",ret)
        print("Kint:\n",mtx)      
        print("Kdist:\n",dist)  
        print("Kext R:\n",R)
        print("Kext T:\n",T)
        np.savetxt(os.path.join(self.saved_params_path,'Kint.txt'),mtx)
        np.savetxt(os.path.join(self.saved_params_path,'Kdist.txt'),dist)  

if __name__=="__main__":
    # camera_index = '1'
    # base_dir = '1012_calib/'
    # date = '1012'
    # camera_type = 'dabai'
    # tool = Camera_Calibration(size=20.5,
    #                           image_target_size=(1080,1920), 
    #                           camera='1', 
    #                           orbb_type=camera_type,
    #                           image_form='color',
    #                           data_dir=base_dir + camera_index + '/data', 
    #                           saved_dir=base_dir + camera_index + '/output',
    #                           mode='load',
    #                           image_dir='board_color',
    #                           points_file_dir=base_dir + camera_index + '/',
    #                           points_file_name='_points_' + date + '.txt')
    # tool.get_camera_matrices()
    # tool = Camera_Calibration(size=20.5,
    #                           image_target_size=(1080,1920), 
    #                           camera='2', 
    #                           image_form='color',
    #                           data_dir=base_dir + camera_index + '/data', 
    #                           saved_dir=base_dir + camera_index + '/output',
    #                           mode='load',
    #                           image_dir='board_color',
    #                           points_file_dir=base_dir + '/',
    #                           points_file_name='_points_' + date + '.txt')


    base_dir = '0326_calib/'
    camera_index = '0/'
    tool = Camera_Calibration(
                              size=40,
                            #   size=20.5,
                            #   image_target_size=(1920,1080),
                              image_target_size=(1080,1920), 
                              camera='1', 
                              orbb_type='dabai',
                              image_form='color',
                              data_dir=base_dir + camera_index + 'data', 
                              saved_dir=base_dir + camera_index + 'output',
                              mode='load',
                            #   image_dir='board_color',
                              image_dir='board_ir',
                              points_file_dir=base_dir + camera_index,
                              points_file_name='_points_0325.txt')
    tool.get_camera_matrices()
    tool = Camera_Calibration(size=40,
                            #   size=20.5,
                            #   image_target_size=(1920,1080),
                              image_target_size=(1080,1920), 
                              camera='2', 
                              image_form='color',
                              data_dir=base_dir + camera_index + 'data', 
                              saved_dir=base_dir + camera_index + 'output',
                              mode='load',
                            #   image_dir='board_color',
                              image_dir='board_ir',
                              points_file_dir=base_dir + camera_index,
                              points_file_name='_points_0325.txt')
    tool.get_camera_matrices()
