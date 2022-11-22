<div align='center'>
<h1>FamilyPhotoResurrection</h1>
Resurrect your Family Photos with ESRGAN neural networks.

**Version 1.0.0**

**Created by Graham Waters**
<div align='center'>
<!-- add badges for the issues, release, latest updates, and stars/forks -->

[![GitHub issues](https://img.shields.io/github/issues/grahamwaters/FamilyPhotoResurrection)](https://img.shields.io/github/issues/grahamwaters/FamilyPhotoResurrection)
[![GitHub release (latest by date)](https://img.shields.io/github/v/release/grahamwaters/FamilyPhotoResurrection)](https://img.shields.io/github/v/release/grahamwaters/FamilyPhotoResurrection)
[![GitHub last commit](https://img.shields.io/github/last-commit/grahamwaters/FamilyPhotoResurrection)](https://img.shields.io/github/last-commit/grahamwaters/FamilyPhotoResurrection)
[![GitHub stars](https://img.shields.io/github/stars/grahamwaters/FamilyPhotoResurrection)](https://img.shields.io/github/stars/grahamwaters/FamilyPhotoResurrection)
[![GitHub forks](https://img.shields.io/github/forks/grahamwaters/FamilyPhotoResurrection)](https://img.shields.io/github/forks/grahamwaters/FamilyPhotoResurrection)
<!-- add view count to the repo -->
![ViewCount](https://views.whatilearened.today/views/github/grahamwaters/FamilyPhotoResurrection.svg)

</div>
</div>

## What is this?

This project aims to take the photos from the past and project them into the eyes of the present using neural networks, and machine learning techniques to super enhance them.

## Installation

Create a new conda environment and install the requirements.

```bash
conda create -n tf tensorflow
conda activate tf
pip install -r requirements.txt
```


## How does it work?

The project uses the ESRGAN neural network to upscale the images.

## How to use it?

1. Clone the repository

```bash
git clone https://github.com/xinntao/Real-ESRGAN.git
```
2. Install the requirements

```python
# Set up the environment
!pip install basicsr
!pip install facexlib
!pip install gfpgan
!pip install -r requirements.txt
```

3. Download the pretrained model

```bash
!wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -P experiments/pretrained_models
```

Now you have a copy of the full pretrained model ready to spin up on your local computer. You can use the pretrained model to upscale your images.

## What makes this Different from the base project by Google Colab?

This repo implements recursive image enhancement, which means that the image is enhanced multiple times to get the best possible result. This is done by using the ESRGAN model multiple times on the same image.

```python
%cd Real-ESRGAN
import time
recursive_mode = True

print("checking mode...")
time.sleep(1)
epoch = 1
if recursive_mode:
  running = True

  while running:
    print(f"running epoch:{epoch}")
    try:
      #!python inference_realesrgan.py -n RealESRGAN_x4plus -i upload --outscale 1  # if faces include --face_enhance
      if epoch == 1:
          !python inference_realesrgan.py -n RealESRGAN_x4plus -i upload --outscale 1  # if faces include --face_enhance
      if epoch == 2:
          !python inference_realesrgan.py -n RealESRGAN_x4plus -i upload --outscale 2  # if faces include --face_enhance
      if epoch == 3:
          !python inference_realesrgan.py -n RealESRGAN_x4plus -i upload --outscale 3  # if faces include --face_enhance
      if epoch == 4:
          !python inference_realesrgan.py -n RealESRGAN_x4plus -i upload --outscale 4  # if faces include --face_enhance
      if epoch == 5:
          !python inference_realesrgan.py -n RealESRGAN_x4plus -i upload --outscale 5  # if faces include --face_enhance
      if epoch == 6:
          !python inference_realesrgan.py -n RealESRGAN_x4plus -i upload --outscale 6  # if faces include --face_enhance
      if epoch == 7:
          !python inference_realesrgan.py -n RealESRGAN_x4plus -i upload --outscale 7  # if faces include --face_enhance
      if epoch == 8:
          !python inference_realesrgan.py -n RealESRGAN_x4plus -i upload --outscale 8  # if faces include --face_enhance
      if epoch == 9:
          !python inference_realesrgan.py -n RealESRGAN_x4plus -i upload --outscale 9  # if faces include --face_enhance
      if epoch == 10:
          !python inference_realesrgan.py -n RealESRGAN_x4plus -i upload --outscale 10  # if faces include --face_enhance
      if epoch == 11:
          !python inference_realesrgan.py -n RealESRGAN_x4plus -i upload --outscale 11  # if faces include --face_enhance
      if epoch == 12:
          !python inference_realesrgan.py -n RealESRGAN_x4plus -i upload --outscale 12  # if faces include --face_enhance
      if epoch == 13:
          !python inference_realesrgan.py -n RealESRGAN_x4plus -i upload --outscale 13  # if faces include --face_enhance
      if epoch == 14:
          !python inference_realesrgan.py -n RealESRGAN_x4plus -i upload --outscale 14  # if faces include --face_enhance
      if epoch == 15:
          !python inference_realesrgan.py -n RealESRGAN_x4plus -i upload --outscale 15  # if faces include --face_enhance
      if epoch == 16:
          !python inference_realesrgan.py -n RealESRGAN_x4plus -i upload --outscale 16  # if faces include --face_enhance
      #print(f"successfully outscaled this file to --{epoch}")
      #finished = input("stop? y/n")
      #if finished == 'y':
      #  break
    except Exception as e:
      print(e)
      try:
        !python inference_realesrgan.py -n RealESRGAN_x4plus -i upload --outscale 5 --half # if faces include --face_enhance
      except Exception as e:
        print(f"reached the enhancement limit for this photo at epoch:{epoch}")
        running = False
        print(e)
        break
    if epoch>20:
      break
    epoch+=1
else:
  print(f"running single epoch:{epoch}")
  try:
    !python inference_realesrgan.py -n RealESRGAN_x4plus -i upload --outscale 5  # if faces include --face_enhance
    print("successfully outscaled all files to --5")
  except Exception as e:
    print(e)
    try:
      !python inference_realesrgan.py -n RealESRGAN_x4plus -i upload --outscale 5 --half # if faces include --face_enhance
    except Exception as e:
      print(f"finished the enhancement for this photo at epoch:{epoch}")
      running = False
      print(e)
    epoch+=1



#!python inference_realesrgan.py -n RealESRGAN_x4plus -i upload --outscale 4 --half --face_enhance
#!python inference_realesrgan.py -n RealESRGAN_x4plus -i upload --outscale 5 --half
#!python inference_realesrgan.py -n RealESRGAN_x4plus -i upload --outscale 1 --half
# !python inference_realesrgan.py --model_path experiments/pretrained_models/RealESRGAN_x4plus.pth --input upload --netscale 4 --outscale 3.5 --half --face_enhance
%cd ..
```

If your installation was successful, you should see the following output:

```bash
Successfully built basicsr filterpy future grpcio
Installing collected packages: yapf, tensorboard-plugin-wit, pyasn1, lmdb, addict, werkzeug, torch, tifffile, tensorboard-data-server, rsa, pyasn1-modules, protobuf, oauthlib, markdown, llvmlite, imageio, grpcio, future, cachetools, absl-py, torchvision, scikit-image, requests-oauthlib, numba, google-auth, google-auth-oauthlib, filterpy, tb-nightly, facexlib, basicsr, gfpgan
Successfully installed absl-py-1.3.0 addict-2.4.0 basicsr-1.4.2 cachetools-5.2.0 facexlib-0.2.5 filterpy-1.4.5 future-0.18.2 gfpgan-1.3.8 google-auth-2.14.1 google-auth-oauthlib-0.4.6 grpcio-1.50.0 imageio-2.22.4 llvmlite-0.39.1 lmdb-1.3.0 markdown-3.4.1 numba-0.56.4 oauthlib-3.2.2 protobuf-3.20.3 pyasn1-0.4.8 pyasn1-modules-0.2.8 requests-oauthlib-1.3.1 rsa-4.9 scikit-image-0.19.3 tb-nightly-2.12.0a20221122 tensorboard-data-server-0.6.1 tensorboard-plugin-wit-1.8.1 tifffile-2022.10.10 torch-1.13.0 torchvision-0.14.0 werkzeug-2.2.2 yapf-0.32.0
```

# Acknowledgements

This project is based on the following projects:

  * [Real-ESRGAN](https://github.com/styler00dollar/Colab-ESRGAN)
  * [ESRGAN](https://colab.research.google.com/github/ml4a/ml4a/blob/master/examples/models/ESRGAN.ipynb)
