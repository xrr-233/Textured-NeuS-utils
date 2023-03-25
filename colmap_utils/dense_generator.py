import os
import shutil
import subprocess

"""
$ colmap image_undistorter \
    --image_path $DATASET_PATH/images \
    --input_path $DATASET_PATH/sparse/0 \
    --output_path $DATASET_PATH/dense \
    --output_type COLMAP \
    --max_image_size 2000

$ colmap patch_match_stereo \
    --workspace_path $DATASET_PATH/dense \
    --workspace_format COLMAP \
    --PatchMatchStereo.geom_consistency true

$ colmap stereo_fusion \
    --workspace_path $DATASET_PATH/dense \
    --workspace_format COLMAP \
    --input_type geometric \
    --output_path $DATASET_PATH/dense/fused.ply

$ colmap poisson_mesher \
    --input_path $DATASET_PATH/dense/fused.ply \
    --output_path $DATASET_PATH/dense/meshed-poisson.ply
"""
def run_colmap(basedir):
    logfile_name = os.path.join(basedir, 'colmap_output_dense.txt')
    logfile = open(logfile_name, 'w')

    if os.path.exists(os.path.join(basedir, 'dense')):
        shutil.rmtree(os.path.join(basedir, 'dense'))
    os.makedirs(os.path.join(basedir, 'dense'), exist_ok=True)

    image_undistorter_args = [
        'colmap', 'image_undistorter',
        '--image_path', os.path.join(basedir, 'images'),
        '--input_path', os.path.join(basedir, 'sparse', '0'),
        '--output_path', os.path.join(basedir, 'dense'),
    ]
    feat_output = (subprocess.check_output(image_undistorter_args, universal_newlines=True))
    logfile.write(feat_output)
    print('Undistorted')

    stereo_matcher_args = [
        'colmap', 'patch_match_stereo',
        '--workspace_path', os.path.join(basedir, 'dense'),
    ]

    match_output = (subprocess.check_output(stereo_matcher_args, universal_newlines=True))
    logfile.write(match_output)
    print('Stereo matched')

    stereo_fusion_args = [
        'colmap', 'stereo_fusion',
        '--workspace_path', os.path.join(basedir, 'dense'),
        '--output_path', os.path.join(basedir, 'dense', 'fused.ply'),
    ]

    map_output = (subprocess.check_output(stereo_fusion_args, universal_newlines=True))
    logfile.write(map_output)
    print('Stereo fused')

    poisson_mesher_args = [
        'colmap', 'poisson_mesher',
        '--input_path', os.path.join(basedir, 'dense', 'fused.ply'),
        '--output_path', os.path.join(basedir, 'dense', 'meshed-poisson.ply'),
    ]

    match_output = (subprocess.check_output(poisson_mesher_args, universal_newlines=True))
    logfile.write(match_output)
    logfile.close()
    print('Stereo matched')

    print('Finished running COLMAP, see {} for logs'.format(logfile_name))