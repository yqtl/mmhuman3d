import numpy as np
from mmhuman3d.core.conventions.keypoints_mapping import convert_kps
from mmhuman3d.core.conventions.keypoints_mapping import KEYPOINTS_FACTORY, convert_kps
from mmhuman3d.data.data_structures.human_data import HumanData
keypoints_agora = np.zeros((100, 127, 3))
keypoints_human_data, mask = convert_kps(keypoints_agora, src='agora', dst='human_data')
human_data = HumanData()
human_data['keypoints2d_mask'] = mask
human_data['keypoints2d'] = keypoints_human_data
#assert mask.all()==1
human_data.dump('./dumped_human_data.npz')
another_human_data = HumanData()
another_human_data.load('./dumped_human_data.npz')
print(another_human_data)