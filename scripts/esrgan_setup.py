# using the Waters_Photo_Enhancer.ipynb notebook as a guide, I created this script to automate the setup of the ESRGAN model
import os
import sys
import subprocess
import zipfile
import requests
import glob
import shutil

# download the ESRGAN model
print("Downloading ESRGAN model")
r = requests.get("https://data.vision.ee.ethz.ch/cvl/DIV2K/models/RRDB_ESRGAN_x4.pth", allow_redirects=True)
open("RRDB_ESRGAN_x4.pth", "wb").write(r.content)

# download the ESRGAN code
print("Downloading ESRGAN code")
r = requests.get("https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip", allow_redirects=True)
open("DIV2K_train_HR.zip", "wb").write(r.content)

# unzip the ESRGAN code
print("Unzipping ESRGAN code")
with zipfile.ZipFile("DIV2K_train_HR.zip", "r") as zip_ref:
    zip_ref.extractall()

# move the ESRGAN code to the correct folder
print("Moving ESRGAN code")
shutil.move("DIV2K_train_HR", "esrgan")
