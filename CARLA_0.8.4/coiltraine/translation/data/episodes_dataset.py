import os.path
from data.base_dataset import BaseDataset, get_params, get_transform, normalize
from data.image_folder import make_dataset
from PIL import Image
from input import data_parser
from input import splitter
from configs import g_conf
import glob
import json
import copy
from coilutils.general import sort_nicely
import torch


class AlignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.imitation=opt.imitation
        print('new dataset!')

        self.A_paths,self.B_paths,self.Control_paths = load_lists(self.root,self.imitation)    

        

        ### instance maps
        if not opt.no_instance:
            self.dir_inst = os.path.join(opt.dataroot, opt.phase + '_inst')
            self.inst_paths = sorted(make_dataset(self.dir_inst))

        ### load precomputed instance-wise encoded features
        if opt.load_features:                              
            self.dir_feat = os.path.join(opt.dataroot, opt.phase + '_feat')
            print('----------- loading features from %s ----------' % self.dir_feat)
            self.feat_paths = sorted(make_dataset(self.dir_feat))

        self.dataset_size = len(self.A_paths) 
      
    def __getitem__(self, index):        
        ### input A (label maps)
        A_path = self.A_paths[index]
        #print(A_path)
        A = Image.open(A_path)        
        params = get_params(self.opt, A.size)
        if self.opt.label_nc == 0:
            transform_A = get_transform(self.opt, params)
            A_tensor = transform_A(A.convert('RGB'))
            #print(A_tensor)
        else:
            transform_A = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
            A_tensor = transform_A(A) * 255.0

        B_tensor = inst_tensor = feat_tensor = 0
        ### input B (real images)
        if self.opt.isTrain or self.opt.use_encoded_image:
            B_path = self.B_paths[index]   
            B = Image.open(B_path).convert('RGB')
            transform_B = get_transform(self.opt, params)      
            B_tensor = transform_B(B)

        ### if using instance maps        
        if not self.opt.no_instance:
            inst_path = self.inst_paths[index]
            inst = Image.open(inst_path)
            inst_tensor = transform_A(inst)

            if self.opt.load_features:
                feat_path = self.feat_paths[index]            
                feat = Image.open(feat_path).convert('RGB')
                norm = normalize()
                feat_tensor = norm(transform_A(feat))
       
        controls = 0

        if self.imitation:
            controls = self.Control_paths[index].copy()
            for k,v in controls.items():
                v = torch.tensor([v,])
                controls[k]= v.float()


        input_dict = {'label': A_tensor, 'inst': inst_tensor, 'image': B_tensor, 
                      'feat': feat_tensor, 'path': A_path,'controls':controls}

        return input_dict

    def __len__(self):
        return len(self.A_paths) // self.opt.batchSize * self.opt.batchSize

    def name(self):
        return 'AlignedDataset'

def load_lists(path,imitation=False):

    episodes_list = glob.glob(os.path.join(path, 'episode_*'))
    sort_nicely(episodes_list)
    # Do a check if the episodes list is empty
    if len(episodes_list) == 0:
        raise ValueError("There are no episodes on the training dataset folder %s" % path)

    left_lists = []
    front_lists = []
    controls_lists = []

    for episode in episodes_list:
        left_lists.extend(sorted(glob.glob(os.path.join(episode+'/LeftCameraRGB','*.png'))))
        front_lists.extend(sorted(glob.glob(os.path.join(episode+'/CameraRGB','*.png'))))
        assert(len(left_lists) == len(front_lists))
        if imitation:
            available_measurements_dict = data_parser.check_available_measurements(episode)
            measurements_list = sorted(glob.glob(os.path.join(episode,'Controls', '*.json')))
            #print(measurements_list)
            for measurement in measurements_list:

                #data_point_number = measurement.split('_')[-1].split('.')[0]

                with open(measurement) as f:
                    measurement_data = json.load(f)

                # depending on the configuration file, we eliminated the kind of measurements
                # that are not going to be used for this experiment
                # We extract the interesting subset from the measurement dict

                speed = data_parser.get_speed(measurement_data)

                directions = measurement_data['directions']
                final_measurement = get_final_measurement(speed, measurement_data, 0,
                                                                directions,
                                                                available_measurements_dict)
                controls_lists.append(final_measurement)
        if imitation:
            print('left:{}'.format(len(left_lists)))
            print('control:{}'.format(len(controls_lists)))
            assert(len(left_lists) == len(controls_lists))
    return left_lists,front_lists,controls_lists


def get_final_measurement(speed, measurement_data, angle,
                           directions, avaliable_measurements_dict):
    """
    Function to load the measurement with a certain angle and augmented direction.
    Also, it will choose if the brake is gona be present or if acceleration -1,1 is the default.
    Returns
        The final measurement dict
    """
    
    measurement_augmented = copy.copy(measurement_data)

    if 'gameTimestamp' in measurement_augmented:
        time_stamp = measurement_augmented['gameTimestamp']
    else:
        time_stamp = measurement_augmented['elapsed_seconds']

    final_measurement = {}
    # We go for every available measurement, previously tested
    # and update for the measurements vec that is used on the training.
    for measurement, name_in_dataset in avaliable_measurements_dict.items():
        # This is mapping the name of measurement in the target dataset
        final_measurement.update({measurement: measurement_augmented[name_in_dataset]})

    # Add now the measurements that actually need some kind of processing
    final_measurement.update({'speed_module': speed / g_conf.SPEED_FACTOR})
    final_measurement.update({'directions': directions})
    final_measurement.update({'game_time': time_stamp})

    return final_measurement            

                
