# Water Surface Identification Using UNet/ UNet-RAU in Drone Imagery #

**This repo is originally forked from [zhixuhao/unet](https://github.com/zhixuhao/unet)**

Please go through their ReadMe.md from [here](https://github.com/zhixuhao/unet/blob/master/README.md) for implementation details of UNet

FCN-RAU Water Surface Identification is available [here](https://github.com/Cow911/SingleImageWaterHazardDetectionWithRAU)

Puddle-1000 dataset is available [here](https://cloudstor.aarnet.edu.au/plus/s/oSeR8zogqzaXN6X)

## Setup - Google Colab Pro ##

**Steps**

1. Clone this repository.
2. Create a Google Drive folder (Lets assume the its `unet-new`).
3. Upload all files to Google Drive folder `unet-new` except UNetPro.ipynb
4. Upload UNetPro.ipynb to Google Colab Pro
5. In [Colab](https://colab.research.google.com/) go to `Menu bar` and select `Runtime > Change runtime type > Hardware accelerator` to `GPU` and save it.
6. Go to `Menu bar` and select `Run all`.
7. Provide accesss permission to your dive from Colab.

**Adding Training & Testing Data**

Here we test 04 `datasets`:
 - `ONR`
 - `OFR`
 - `BOTH`
 - `DRONE`

Each `dataset` has two `sub-datasets`:
 - `train` - used to train the model
 - `test` - used to test the model
 
In each `sub-dataset` it has `image` and `label` folders.
 - `images` - original images
 - `lable` - ground truth images

Images were obtained from `Puddle-1000` dataset available [here](https://cloudstor.aarnet.edu.au/plus/s/oSeR8zogqzaXN6X).

**Updating main.py**

`base_path` - Path of the data folder *Ex. base_path = "/content/gdrive/My Drive/unet-new"*

`batch_size` - Bach size

`trainCount` - Training image count. By default it is set to total number of training images.

EPOCHS

`ecpoch_array` - set of epoch to continue the training

*Ex. `ecpoch_array = [1000,20000]`*
- Train for 1000 epochs and evaluate.
- Then, trian for 20000 epochs from begining and evaluate.

RESIZING

`target_size` - input image size to unet
`target_size_channels` - input image size with channels to unet

MODEL

`model,model_name` - select the model and model name

Currently following models have been implemented:
- `unet` - unet without any modifications
- `unet-1-rau`
- `unet-2-rau`
- `unet-3-rau`
- `unet-4-rau`
- `unet-8-rau`

CallBack Class

It is used to evaluate the results while training.
Ex: If you specify `if((epoch == 100) or (epoch == 200) or (epoch == 500)):` in the if statement and your original epochs count is 1000, it will evaluate results at 100, 200, 500 and finally at 1000. If nothing is specified it will evaluate only at 1000.

Select datasets to evaluate

Specify the datasets to evaluate. Ex. If you specify `if(0,1)` only 0th dataset will be evaluated. If you specify `if(0,4)` 0th,1st,2nd, and 3rd datasets will be evaluated. If you need to evaluate the 2nd dataset only specify `if(2,3)`.

Define datasets to evaluate

Specify the datasets indicated by each index. That is the original folder names in the `data` folder which specify datasets.

- `data.py` - No need to change | contains data pre processing
- `model.py` - No need to change | contains unet models
- `image_metrics.py` - No need to change | contains evaluation matrics
