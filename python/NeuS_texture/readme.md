MVSNet input from SfM output

We provide a script to convert COLMAP SfM result to R/MVSNet input. After recovering SfM result and undistorting all images, COLMAP should generate a dense folder COLMAP/dense/ containing an undistorted image folder COLMAP/dense/images/ and an undistorted camera folder COLMAP/dense/sparse/. Then, you can apply the following script to generate the R/MVSNet input:

python colmap2mvsnet.py --dense_folder COLMAP/dense

The depth sample number will be automatically computed using the inverse depth setting. If you want to generate the MVSNet input with a fixed depth sample number (e.g., 256), you could specified the depth number via --max_d 256.

只能在linux上跑，改txt后缀和image文件夹，跑出来后有cams、blended_images和pair.txt