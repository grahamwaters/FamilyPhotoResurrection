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