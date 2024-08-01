import os 
import cv2
import numpy as np

class Stereo_Calibration:
    def __init__(self,
                 num_h=5,
                 num_w=8,
                 image_target_size=(1280,1080),
                 eps=200,
                 tol=.001,
                 data_dir='save3mm_0328_test/output',
                 image_dir='board_ir',
                 ) -> None:
        
        self.h = num_h-1
        self.w = num_w-1
        self.image_target_size = image_target_size
        self.eps = eps
        self.tol = tol
        self.data_dir = data_dir
        self.points_dir = image_dir+'_points'
        self.params_dir = image_dir+'_parameters'
        self.saved_dir = os.path.join(data_dir,'stereo_parameters')
        if not os.path.exists(self.saved_dir):
            os.makedirs(self.saved_dir)
        print("Output parameters saved directory:\n",self.saved_dir)

    def get_points(self,camera):
        objp_path = os.path.join(self.data_dir,'camera'+camera,self.points_dir,'objpoints.bin')
        imgp_path = os.path.join(self.data_dir,'camera'+camera,self.points_dir,'imgpoints.bin')
        objpoints = np.fromfile(objp_path,'f4').reshape(-1,self.w*self.h,3)
        imgpoints = np.fromfile(imgp_path,'f4').reshape(-1,self.w*self.h,1,2)
        
        return objpoints, imgpoints
    
    def get_params(self,camera):
        params_path = os.path.join(self.data_dir,'camera'+camera,self.params_dir)
        Kint = np.loadtxt(os.path.join(params_path,'Kint.txt'))
        Kdist = np.loadtxt(os.path.join(params_path,'Kdist.txt'))

        return Kint, Kdist
    
    def calibrate(self,):
        camera1_objpoints, camera1_imgpoints = self.get_points('1')
        camera2_objpoints, camera2_imgpoints = self.get_points('2')
        assert np.all(camera1_objpoints) == np.all(camera2_objpoints)
        Kint1, Kdist1 = self.get_params('1')
        Kint2, Kdist2 = self.get_params('2')
        stereocalibration_flags = cv2.CALIB_FIX_INTRINSIC
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, self.eps, self.tol) 
        ret, CM1, dist1, CM2, dist2, R, T, E, F = cv2.stereoCalibrate(camera1_objpoints, camera1_imgpoints, camera2_imgpoints, Kint1, Kdist1,Kint2, Kdist2, self.image_target_size[::-1], criteria = criteria, flags = stereocalibration_flags)
        
        print("Rotation matrix camera 1 to 2:\n",R)
        print("Translation matrix camera 1 to 2:\n",T)
        np.savetxt(os.path.join(self.saved_dir,'R.txt'),R)
        np.savetxt(os.path.join(self.saved_dir,'T.txt'),T)

if __name__=="__main__":
    # tool = Stereo_Calibration(num_h=9, 
    #                           num_w=12, 
    #                           image_target_size=(480,640),
    #                           data_dir='save0524_calib/output')
    # tool.calibrate()


    tool = Stereo_Calibration(
                              image_target_size=(1080,1920),
                            #   image_target_size=(1920,1080),
                              data_dir='0326_calib/0/output',
                              image_dir='board_ir')
                            #   image_dir='board_color')
    tool.calibrate()
