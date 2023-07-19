#python3 convert_kpts.py PATH_TO_OPENPOSE_keypoints.json PATH_TO_OUTPUT_humandata.npz
import argparse
import json
import numpy as np
from mmhuman3d.core.conventions.keypoints_mapping import convert_kps
from mmhuman3d.data.data_structures.human_data import HumanData

def main(json_file, output_file):
    # Load keypoints from the JSON file.
    with open(json_file, 'r') as f:
        keypoints_human_data = json.load(f)

    # Extract pose keypoints
    pose_keypoints_2d = keypoints_human_data["people"][0]["pose_keypoints_2d"]

    # Reshape to (1, 25, 3) - assuming OpenPose outputs 25 keypoints
    pose_keypoints_2d = np.array(pose_keypoints_2d).reshape(1, 25, 3)

    # Convert the keypoints to the 'coco' format.
    human_data_converted, mask = convert_kps(pose_keypoints_2d, src='openpose_25', dst='human_data')
    #assert mask.all()==1
    human_data = HumanData()
    human_data['keypoints2d_mask'] = mask
    human_data['keypoints2d'] = human_data_converted
    #print(human_data)
    human_data.dump(output_file)
    another_human_data = HumanData()
    another_human_data.load(output_file)
    print(another_human_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('json_file', help='Path to the JSON file containing keypoints')
    parser.add_argument('output_file', help='Path to the output file where converted keypoints will be saved')
    args = parser.parse_args()

    main(args.json_file, args.output_file)