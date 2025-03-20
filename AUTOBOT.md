some ugly autobot commands for convenience until I've implemented some shell script functionality 
for autobot.

nstructions for running code on AUTOBOT:

Copying the data to the head node:
```
rsync -anv --exclude='*archived*' --exclude='*.gif'  proccloth/ eycai@autobot.vision.cs.cmu.edu:/project_data/held/eycai/data/proccloth
```

Copying data from head node to GPU node:
```
rsync -anv data/ autobot-0-25:/scratch/eycai/data
```

Copying the code to the head node:
```
rsync -anv --exclude='*scripts/logs/*' --exclude='.git/*' --exclude='*scripts/wandb*' --exclude='*.ckpt' --exclude='*notebooks/*' non-rigid/ eycai@autobot.vision.cs.cmu.edu:code/non-rigid
```


RUNNING EVAL FROM GPU NODE:
```
singularity exec --nv -B /home/eycai/code/non-rigid:/opt/eycai/code/non-rigid -B /scratch/eycai/data:/opt/eycai/data -B /scratch/eycai/logs:/opt/eycai/logs /scratch/eycai/singularity/tax3d.sif bash -c "cd /opt/eycai/code/non-rigid/scripts && ./eval.sh 1 gzc40qe1 dataset.data_dir='/opt/eycai/data/proccloth/' coverage=True"
```

TRAINING TAX3DV2 FROM GPU NODE:
```
singularity exec --nv -B /home/eycai/code/non-rigid:/opt/eycai/code/non-rigid -B /scratch/eycai/data:/opt/eycai/data -B /scratch/eycai/logs:/opt/eycai/logs /scratch/eycai/singularity/tax3d.sif bash -c "cd /opt/eycai/code/non-rigid/scripts && CUDA_VISIBLE_DEVICES=0 ./train2.sh 0 cross_point_relative disabled dataset.data_dir=/opt/eycai/data/ dataset.train_size=400 dataset.cloth_geometry=multi dataset.cloth_pose=random dataset.hole=single dataset.num_anchors=2 dataset.scene_transform_type=random_flat_upright model.scene_anchor=False model.center_type=scene_center model.action_context_center_type=scene_center model.diffuse_ref_frame=True model.tax3dv2=True model.extra_features=True model.noisy_goal_origin=True model.scale_inputs=none resources.num_workers=16"
```



TRAINING TAX3DV1 FROM GPU NODE:
```
singularity exec --nv -B /home/eycai/code/non-rigid:/opt/eycai/code/non-rigid -B /scratch/eycai/data:/opt/eycai/data -B /scratch/eycai/logs:/opt/eycai/logs /scratch/eycai/singularity/tax3d.sif bash -c "cd /opt/eycai/code/non-rigid/scripts && CUDA_VISIBLE_DEVICES=0 ./train2.sh 0 cross_point_relative online dataset.data_dir=/opt/eycai/data/ dataset.train_size=400 dataset.cloth_geometry=multi dataset.cloth_pose=random dataset.hole=single dataset.num_anchors=2 dataset.scene_transform_type=random_flat_upright model.scene_anchor=False model.center_type=anchor_center model.rel_pose=True model.oracle=True model.x_encoder=pn model.x0_encoder=pn model.y_encoder=pn resources.num_workers=16"
```



TRAIN DP3:

```

singularity exec --nv -B /home/eycai/code/non-rigid:/opt/eycai/code/non-rigid -B /scratch/eycai/data:/opt/eycai/data -B /scratch/eycai/logs:/opt/eycai/logs /scratch/eycai/singularity/tax3d.sif bash -c "cd /opt/eycai/code/non-rigid/third_party/3D_Diffusion_Policy && CUDA_VISIBLE_DEVICES=1 bash scripts/train_policy.sh dp3 dedo_proccloth autobot_test 1 0"
```