import os
import numpy as np
from typing import List
import matplotlib.pyplot as plt

class SingleModelConfig:
  def __init__(self,
               
               input_size: List[int] = [384, 288],
               kpd: float = 4.0,
               epochs: int = 15,
               sigma: float = 3.0,
               num_joints: int = 15,
               batch_size: int = 16,
               random_seed: int = 2021,
               test_ratio: float = 0.1,
               learning_rate: float = 1e-3,
               save_folder: str = '',
              #  main_dir: str = main_dir,
               loss_type: str = "MSE",
               target_type: str = "gaussian",
               post_processing: str = "dark",
               debug: bool = False,
               shift: bool = False,
               init_training: bool = False,
              
              
    ):


    self.save_folder = save_folder
    if not os.path.exists(self.save_folder) and self.save_folder != '':
      os.makedirs(self.save_folder, exist_ok=True)

    self.epochs = epochs
    self.seed = random_seed
    self.lr = learning_rate
    self.loss_type = loss_type
    self.num_joints = num_joints
    self.batch_size = batch_size
    self.test_ratio = test_ratio
    self.init_training = init_training

    
    self.kpd = kpd

    self.sigma = sigma
    self.shift = shift
    self.debug = debug
    
    self.target_type = target_type
    self.image_size = np.array(input_size)
    self.output_size = self.image_size//4
    self.post_processing = post_processing



    self.joints_name = {
          0: 'nose', 1: 'middle_forehead', 2: 'lip_tail', 3: 'middle_lower_lip', 4: 'neck',
          5: 'right_foreleg_start', 6: 'left_foreleg_start', 7: 'right_foreleg_ankle', 8: 'left_foreleg_ankle',
          9: 'right_femur', 10: 'left_femer', 11: 'right_hindleg_ankle', 12: 'left_hindleg_ankle',
          13: 'tail_start', 14: 'tail_end', 
    }

    self.joint_pair = [
          (0, 1),  (0,3), (2,3), (3, 4), (4, 5),
          (4, 6), (5, 7), (6, 8), (4, 13), (13, 9), 
          (13, 10), (13,14), (9, 11), (10,12)
    ]

    self.flip_pair = [
          (9, 10), (7,8), (5, 6), (11, 12)
     
    ]

    self.joints_weight = np.array(
            [
                1.3,           # 코
                1.3,           #이마 중앙 
                1.3,           # 입꼬리(입끝)
                1.3,           # 아래 입술 중앙
                1.3,           # 목
                1.3, 1.3,      # 앞다리 시작
                1.3, 1.3,      # 앞다리 발목
                1.3, 1.3,      # 대퇴골
                1.3, 1.3,      # 뒷다릭 발목
                1.3,           # 꼬리 시작
                1.3,           #꼬리 끝
            ],
            dtype=np.float32
        ).reshape((self.num_joints, 1))

    
    cmap = plt.get_cmap("rainbow")
    colors = [cmap(i) for i in np.linspace(0, 1,  self.num_joints + 2)]
    colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]
    self.joint_colors = {k: colors[k] for k in range(self.num_joints)}