import json

OP_TO_COCO_MAPPING = {
    'nose_openpose': 'nose',
    'neck_openpose': None,
    'right_shoulder_openpose': 'right_shoulder',
    'right_elbow_openpose': 'right_elbow',
    'right_wrist_openpose': 'right_wrist',
    'left_shoulder_openpose': 'left_shoulder',
    'left_elbow_openpose': 'left_elbow',
    'left_wrist_openpose': 'left_wrist',
    'pelvis_openpose': None,
    'right_hip_openpose': 'right_hip_extra',
    'right_knee_openpose': 'right_knee',
    'right_ankle_openpose': 'right_ankle',
    'left_hip_openpose': 'left_hip_extra',
    'left_knee_openpose': 'left_knee',
    'left_ankle_openpose': 'left_ankle',
    'right_eye_openpose': 'right_eye',
    'left_eye_openpose': 'left_eye',
    'right_ear_openpose': 'right_ear',
    'left_ear_openpose': 'left_ear',
    'left_bigtoe_openpose': None,
    'left_smalltoe_openpose': None,
    'left_heel_openpose': None,
    'right_bigtoe_openpose': None,
    'right_smalltoe_openpose': None,
    'right_heel_openpose': None
}

COCO_KEYPOINTS = [
    'nose',
    'left_eye',
    'right_eye',
    'left_ear',
    'right_ear',
    'left_shoulder',
    'right_shoulder',
    'left_elbow',
    'right_elbow',
    'left_wrist',
    'right_wrist',
    'left_hip_extra',
    'right_hip_extra',
    'left_knee',
    'right_knee',
    'left_ankle',
    'right_ankle',
]

# Read the OpenPose JSON file
with open('keypoints.json', 'r') as f:
    data = json.load(f)

# Get the pose keypoints
openpose_keypoints = data['people'][0]['pose_keypoints_2d']

# Initialize the COCO keypoints
coco_keypoints = [0] * 17 * 3  # In COCO, we have 17 keypoints with x, y, and visibility.

for i, key in enumerate(OP_TO_COCO_MAPPING.keys()):
    coco_key = OP_TO_COCO_MAPPING[key]
    if coco_key is not None:
        # Get the index of the COCO keypoint
        coco_index = COCO_KEYPOINTS.index(coco_key)
        # Get the OpenPose keypoint data (x, y, confidence)
        op_data = openpose_keypoints[i*3:i*3+3]
        # Set the x, y in COCO format and set visibility to 2 if confidence > 0.5 else 0.
        coco_keypoints[coco_index*3:coco_index*3+3] = [op_data[0], op_data[1], 2 if op_data[2] > 0.5 else 0]

# Now we replace the OpenPose keypoints with COCO keypoints in the data
data['people'][0]['pose_keypoints_2d'] = coco_keypoints

# Save the converted data to a new JSON file
with open('keypoints_COCO.json', 'w') as f:
    json.dump(data, f)

