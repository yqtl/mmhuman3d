` python3 tools/smplify.py --input humandata.npz --input_type keypoints2d --config configs/smplify/smplify.py --body_model_dir data/body_models/smpl/SMPL_NEUTRAL.pkl --output <OUTPUT_PATH> --show_path <OUTPUT_PATH>`
Failed.
The default method in tools/smplify.py does not read the converted humandata.npz.
First convert openpose to humandata.npz, then if use default args, src='human_data', dst='smpl_45', the loaded humandata are all zeros.
Therefore directly do kpts convert inside smplify.py, with src='openpose_25', dst='smpl_45'.

Just make sure the correct data is loaded.

Another issue is, passing keypoints2D results in errors here.
`smplify_output = smplify(**human_data, return_joints=True)`
Created an issue ticket in the original repo.

Below are the git diff.
Apptainer> git diff tools/smplify.py
diff --git a/tools/smplify.py b/tools/smplify.py
index 8302fb7..c0fc5dc 100644
--- a/tools/smplify.py
+++ b/tools/smplify.py
@@ -1,3 +1,4 @@
+import json
 import argparse
 import os
 import time
@@ -87,14 +88,33 @@ def main():
         keypoints_src = keypoints_src[..., :3]
     else:
         raise KeyError('Only support keypoints2d and keypoints3d')
+    keypoints, mask = convert_kps(
+        keypoints_src,
+        mask=keypoints_src_mask,
+        src=args.keypoint_type,
+        dst=smplify_config.body_model['keypoint_dst'])
+
+    json_file='keypoints.json'
+    with open(json_file, 'r') as f:
+        keypoints_human_data = json.load(f)
+
+    # Extract pose keypoints
+    pose_keypoints_2d = keypoints_human_data["people"][0]["pose_keypoints_2d"]

+    # Reshape to (1, 25, 3) - assuming OpenPose outputs 25 keypoints
+    pose_keypoints_2d = np.array(pose_keypoints_2d).reshape(1, 25, 3)
+
+    keypoints, mask = convert_kps(pose_keypoints_2d, src='openpose_25', dst='smpl_45')
+    #dummy
     keypoints, mask = convert_kps(
         keypoints_src,
         mask=keypoints_src_mask,
         src=args.keypoint_type,
         dst=smplify_config.body_model['keypoint_dst'])
+    print(keypoints.shape)
+    print(mask.shape)
     keypoints_conf = np.repeat(mask[None], keypoints.shape[0], axis=0)
-
+    #print(keypoints)
     batch_size = args.batch_size if args.batch_size else keypoints.shape[0]

     keypoints = torch.tensor(keypoints, dtype=torch.float32, device=device)
@@ -135,10 +155,12 @@ def main():
             use_one_betas_per_video=args.use_one_betas_per_video,
             num_epochs=args.num_epochs))

+    #print(dict(smplify_config))
     smplify = build_registrant(dict(smplify_config))

     # run SMPLify(X)
     t0 = time.time()
+    #print(human_data)
     smplify_output = smplify(**human_data, return_joints=True)
     t1 = time.time()
     print(f'Time:  {t1 - t0:.2f} s')