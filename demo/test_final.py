from rcnn import KeypointDataset, get_frame, get_video, collate_fn, pred_keypoints
from stgcn import match_format, MakeNumpy
import torch

#!/usr/bin/env python
import argparse
import sys
sys.path.append('./torchlight')

# torchlight
import torchlight
from torchlight import import_class

import warnings
warnings.filterwarnings('ignore')

if __name__ == "__main__":
    ### Keypoint Detection using RCNN
    if torch.cuda.is_available():
        DEVICE = torch.device('cuda')
    else:
        DEVICE = torch.device('cpu')
        print(DEVICE)

    video_path = '../test_video_wide.mp4'
    frame_path= '../test'

    if get_video(video_path, frame_path)==True:
        print('video successfully fetched!')
    
    #frame_path = './images/images_2/20201113_dog-footup-000733.mp4' # 비디오 프레임들이 저장된 경로
    model_path = '../models/RCNN_ep5_1.4756307702064515.pt'

    pred_key = pred_keypoints(frame_path, model_path, DEVICE)

    ### Make input data for STGCN
    output_dict, folder_list = match_format(pred_key)

    sample = MakeNumpy(output_dict, folder_list)
    sample_total_npy = sample.fill_data_numpy()
    sample_tuple_pkl = sample.save_tuple_to_pkl()
    sample.save_total_npy()

    ### Action Recognition using STGCN
    parser = argparse.ArgumentParser(description='Processor collection')

    # region register processor yapf: disable
    processors = dict()
    processors['recognition'] = import_class('processor.recognition.REC_Processor')
  
    #endregion yapf: enable

    # add sub-parser
    subparsers = parser.add_subparsers(dest='processor')
    for k, p in processors.items():
        subparsers.add_parser(k, parents=[p.get_parser()])

    # read arguments
    arg = parser.parse_args()

    # start
    Processor = processors[arg.processor]
    p = Processor(sys.argv[2:])
    
    p.start()


