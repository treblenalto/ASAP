import streamlit as st
PAGE_CONFIG = {"page_title":"ASAP","page_icon":":dog:","layout":"centered"}
st.set_page_config(**PAGE_CONFIG)

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

import os
import warnings
warnings.filterwarnings('ignore')

def main():
  ##### APPLICATION #####
  st.title('반려동물 행동 분석 서비스🐾')
  st.subheader("반려견을 담은 동영상을 올려 주세요!")

  video_file = st.file_uploader("동영상 선택", type=["mp4", "avi", "mpeg"])
  
  if video_file is None:
    st.write('📽️선택된 영상이 없습니다🐕‍🦺')

  else: 
    file_details = {"FileName":video_file.name,"FileType":video_file.type}
    # st.write(file_details)

    video_path = os.path.join(".",video_file.name)
    with open(video_path,"wb") as f: 
      f.write(video_file.getbuffer())       
    st.success("반려견 관찰중...")

    ##### Keypoint Detection with RCNN #####
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    frame_path= '../test'
    model_path = '../models/RCNN_ep46_1.0672876937389373_30000:40000_.pt'

    if get_video(video_path, frame_path)==True:
      print('video successfully fetched!')
    # predict keypoints
    pred_key = pred_keypoints(frame_path, model_path, DEVICE)

    ### DATA CONVERSION #####
    output_dict, folder_list = match_format(pred_key)

    sample = MakeNumpy(output_dict, folder_list)
    sample_total_npy = sample.fill_data_numpy()
    sample_tuple_pkl = sample.save_tuple_to_pkl()
    sample.save_total_npy()

    ### Action Recognition using STGCN ###
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

    # st.write('반려견의 현재 행동🦴')
    result = '<p style="font-size: 18px;">반려견의 현재 행동🦴: </p>'
    st.markdown(result, unsafe_allow_html=True)

    video_file = open(video_path, 'rb')
    video_bytes = video_file.read()
    st.video(video_bytes)

if __name__ == '__main__':
	main()
