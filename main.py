import os
import shutil
import sys
# import matplotlib_inline as plt_inline
import os
import sys
import subprocess
import zipfile
import requests
import glob
import shutil
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import PIL
from PIL import Image
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
import clip
import torch
from torchvision import transforms
# import keras
from keras.preprocessing import image
from tqdm import tqdm
from alive_progress import alive_bar


def main():

    # from ml4a import image
    # from ml4a.models import esrgan

    # instead of using ml4a, I will use the ESRGAN model directly via a downloaded copy, and CLIP to do the image comparison and selection. I will also use the PIL library to do the image manipulation. I will also use the os library to do the file manipulation.

    # load the model into memory
    print("Loading ESRGAN model")

    # load the CLIP model into memory
    # print("Loading CLIP model")
    # model, preprocess = clip.load("ViT-B/32", device)

    # if the model is not already downloaded, download it
    # check the models folder for the model
    # if it is not there, download it
    # if it is there, load it

    if not os.path.exists("models"):
        os.mkdir("models")
    elif not os.path.exists("models/ESRGAN"):
        os.mkdir("models/ESRGAN")
    else:
        print("ESRGAN model likely already downloaded")

    if not os.path.exists("models/ESRGAN/RRDB_ESRGAN_x4.pth"):
        # download the ESRGAN model
        print("Downloading ESRGAN model")
        r = requests.get("https://data.vision.ee.ethz.ch/cvl/DIV2K/models/RRDB_ESRGAN_x4.pth", allow_redirects=True)
        open("RRDB_ESRGAN_x4.pth", "wb").write(r.content)

        # download the ESRGAN code
        print("Downloading ESRGAN code")
        r = requests.get("https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip", allow_redirects=True)

        # unzip the ESRGAN code
        print("Unzipping ESRGAN code")
        with zipfile.ZipFile("DIV2K_train_HR.zip", "r") as zip_ref:
            zip_ref.extractall()

        # move the ESRGAN code to the correct folder
        print("Moving ESRGAN code")
        # make a folder called 'models'
        os.mkdir("models")
        # move the ESRGAN model to the models folder
        shutil.move("RRDB_ESRGAN_x4.pth", "models/RRDB_ESRGAN_x4.pth")


    # load the model into memory (it is in esrgan/RRDB_ESRGAN_x4.pth)
    print("Loading ESRGAN model")
    #!model = torch.load("./models/ESRGAN/RRDB_ESRGAN_x4.pth") # I am getting a invalid load key, '<'. error here
    # copilot: I think this is because the model is not in the correct format. I think it is a PyTorch model, and I am trying to load it into TensorFlow. I think I need to convert it to a TensorFlow model. I think I can do that with the following code:
    # model = tf.keras.models.load_model("./models/ESRGAN/RRDB_ESRGAN_x4.pth")
    try:
        model = tf.keras.models.load_model("./models/ESRGAN/RRDB_ESRGAN_x4.pth", compile=False)
    except Exception as e:
        print(e)
        # download the ESRGAN model
        print("Downloading ESRGAN model")
        r = requests.get("https://data.vision.ee.ethz.ch/cvl/DIV2K/models/RRDB_ESRGAN_x4.pth", allow_redirects=True)
        open("./models/ESRGAN/RRDB_ESRGAN_x4.pth", "wb").write(r.content)

        # try to load the model again
        SAVED_MODEL_PATH = "https://tfhub.dev/captain-pool/esrgan-tf2/1"
        model = hub.load(SAVED_MODEL_PATH)

    model.eval() # set to evaluation mode which means that the model will not be trained and will not be updated with new data. This is important because we want to use the model as is, and not have it change as we use it.


    input_images_directory = './input_images'
    output_images_directory = './output_images'

    # create the input and output directories if they don't exist
    if not os.path.exists(input_images_directory):
        os.makedirs(input_images_directory)

    if not os.path.exists(output_images_directory):
        os.makedirs(output_images_directory)

    # take the input images and convert them to the correct format for the ESRGAN model
    input_images = glob.glob(input_images_directory + '/*')



    number_of_images = len(input_images)

    with alive_bar(number_of_images) as bar:
        for input_image in input_images:
            # load the image
            image = PIL.Image.open(input_image)

            # resize the image to 256x256
            image = image.resize((256, 256))

            # save the image
            image.save(input_image)

            bar()


    # Now that the images are in the correct format, we can run them through the ESRGAN model
    input_images = glob.glob(input_images_directory + '/*')

    # create a list of the output images
    output_images = []

    # create a list of the output image names
    output_image_names = []

    # begin the loop
    with alive_bar(number_of_images) as bar:
        # take each image and magnify it using the ESRGAN model 4x and save the output
        for input_image in input_images:
            # load the image
            image = PIL.Image.open(input_image)

            # convert the image to a numpy array
            image = np.array(image)

            # convert the image to a tensor
            image = torch.from_numpy(image).to(device).float()

            # convert the image to the correct format for the ESRGAN model
            image = image.permute(2, 0, 1).unsqueeze(0)

            # run the image through the ESRGAN model
            with torch.no_grad():
                output = model(image)

            # convert the output to a numpy array
            output = output.squeeze(0).permute(1, 2, 0).cpu().numpy()

            # convert the output to an image
            output = PIL.Image.fromarray(np.uint8(output.clip(0, 1) * 255))

            # save the output image
            output_image_name = os.path.basename(input_image)
            output_image_name = os.path.splitext(output_image_name)[0]
            output_image_name = output_image_name + '_esrgan.png'
            output_image_path = os.path.join(output_images_directory, output_image_name)
            output.save(output_image_path)

            # add the output image to the list of output images
            output_images.append(output_image_path)

            # add the output image name to the list of output image names
            output_image_names.append(output_image_name)

            bar()




if __name__ == "__main__":
    main()