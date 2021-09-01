import os
import json
import numpy as np
import pickle
import pandas as pd

def match_format(output_array):
    output_array = np.array(output_array)
    xy = []; score = []

    for i in range(15):
        xy.append(output_array[0][i][0])
        xy.append(output_array[0][i][1])
        score.append(output_array[0][i][2])

    output_dict = {}
    output_dict['data'] = []
    output_dict['label'] = "99"    # demo class : 99
    output_dict['label_index'] = 99
    for frame in range(output_array.shape[0]):
        output_dict['data'].append(  {
                            'frame_index':frame,
                            'skeleton':[ {
                                        'pose': xy,
                                        'score': score
                                        }  ] } )

    folder_list = ['sample']
    
    return output_dict, folder_list



### HRNet output (dict) -> STGCN input (npy,pkl)
class MakeNumpy:
    def __init__(self, output_dict, folder_list):
        self.N = 1     # length of file names in each folder w/o null
        self.C = 3                    # number of channels : 3  (x,y,score)
        self.T = 280                  # maximum frame sequence
        self.V = 15                   # number of joints
        self.M = 1                    # number of object

        self.label_list = []
        
        self.total_data_numpy = np.zeros((self.N, self.C, self.T, self.V, self.M))
        
        self.folder_list = folder_list
        self.video_info = output_dict
        self.tuple_pkl = (0,0)

    def fill_data_numpy(self):
        for frame_index,frame_info in enumerate(self.video_info['data']):
            # frame 각각의 순서 index에서 data_numpy 채움
            skeleton_info = frame_info['skeleton'][0]

            pose = skeleton_info['pose']
            score = skeleton_info['score']

            self.total_data_numpy[0,0, frame_index, :, 0] = pose[0::2]
            self.total_data_numpy[0,1, frame_index, :, 0] = pose[1::2]
            self.total_data_numpy[0,2, frame_index, :, 0] = score
            
        return self.total_data_numpy

    def save_tuple_to_pkl(self):
        # pkl 만들기 위해 list에 label_index를 append 함
        self.label_list.append(self.video_info['label_index'])

        label_dic = {11:0, 2:1, 3:2, 4:3, 5:4, 13:5, 7:6, 8:7, 9:8, 10:9}
        self.label_list = [label_dic.get(n,n) for n in self.label_list]

        self.tuple_pkl = (self.folder_list, self.label_list)
        with open('./data/pickle_demo.pkl','wb') as f: 
            pickle.dump(self.tuple_pkl,f)
        
        return self.tuple_pkl

    def save_total_npy(self):
        np.save('./data/array_demo',self.total_data_numpy)
