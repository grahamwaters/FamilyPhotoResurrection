# FamilyPhotoResurrection
Resurrect your Family Photos with ESRGAN neural networks.

## What is this?

This project aims to take the photos from the past and project them into the eyes of the present using neural networks, and machine learning techniques to super enhance them.

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
