# Water Surface Identification Using UNet/ UNet-RAU in Drone Imagery #

**This repo is originally folked from zhixuhao/unet**

*Please go through zhixuhao/unet/README.md for implementation details of UNet*

## Setup ##

###### Google Colab Pro ######

Steps:

1. Clone the folder.
2. Create a drive folder (Lets assume the its "unet-new").
3. Upload all files to Google Drive except UNetPro.ipynb
4. Upload UNetPro.ipynb to Google Colab Pro
5. In Colab go to top menu and select `Runtime > Change runtime type > Hardware accelerator` to `GPU` and save it
4. Go to top menu and select `Run all`

Adding Training & Testing Data

Here we test 04 datasets:
- ONR
- OFR
- BOTH
- DRONE

Each dataset has a training and a testing sub-datasets.
In each sub-dataset it has image and label folders.
- images - original images
- lable - ground truth images

Images were obtained from Puddle-1000 dataset available at Cow911/SingleImageWaterHazardDetectionWithRAU

updating main.py

base_path - Path of the data folder.
base_path = "/content/gdrive/My Drive/unet-new"

batch_size - Bach size

trainCount - Training image size. It is set to total number of training images by default.

EPOCHS
ecpoch_array - set of epoch to continue the training
ecpoch_array = [1000,20000]

- Train for 1000 epochs and evaluate.
- Then trian for 20000 epochs from begining and evaluate.

RESIZING
target_size - input image size to unet
target_size_channels - input image size with channels to unet

MODEL
model,model_name - select the model
currently it has the following models

- unet | unet without any modifications
- unet-1-rau
- unet-2-rau
- unet-3-rau
- unet-4-rau
- unet-8-rau

CallBack Class
It is used to evaluate the results while training.
Ex: If you specify 100, 200, 500 in the if statement and your original epochs count is 1000, it will evaluate results at 100, 200, 500 and finally at 1000. If nothing is specified it will evaluate only at 1000.

Select datasets to evaluate
Specify the datasets to evaluate. Ex. If you specify (0,1) only 0th dataset will be evaluated. If you specify (0,4) 0th,1st,2nd, and 3rd datasets will be evaluated. If you need to evaluate the 2nd dataset only specify (2,3)

Define datasets to evaluate
Specify the datasets indicated by each index. That is the original folder names in the 'data' folder.

- data.py - No need to change | contains data pre processing
- model.py - No need to change | contains unet models
- image_metrics.py - No need to change | contains evaluation matrics
