import os
import shutil
import sys
import matplotlib_inline as plt_inline



def main()

    upload_folder = 'upload'
    result_folder = 'results'

    if os.path.isdir(upload_folder):
        shutil.rmtree(upload_folder)
    if os.path.isdir(result_folder):
        shutil.rmtree(result_folder)
    os.mkdir(upload_folder)
    os.mkdir(result_folder)

    # upload images
    uploaded = files.upload()
    for filename in uploaded.keys():
    dst_path = os.path.join(upload_folder, filename)
    print(f'move {filename} to {dst_path}')
    shutil.move(filename, dst_path)
    # use sys to move up a directory
    sys.path.append('..')
