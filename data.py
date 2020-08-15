from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
import numpy as np 
import os
import glob
import skimage.io as io
import skimage.transform as trans
import cv2

# https://machinelearningmastery.com/reproducible-results-neural-networks-keras/
from numpy.random import seed
seed(123)
from tensorflow import set_random_seed
set_random_seed(456)

Sky = [128,128,128]
Building = [128,0,0]
Pole = [192,192,128]
Road = [128,64,128]
Pavement = [60,40,222]
Tree = [128,128,0]
SignSymbol = [192,128,128]
Fence = [64,64,128]
Car = [64,0,128]
Pedestrian = [64,64,0]
Bicyclist = [0,128,192]
Unlabelled = [0,0,0]

COLOR_DICT = np.array([Sky, Building, Pole, Road, Pavement,
                          Tree, SignSymbol, Fence, Car, Pedestrian, Bicyclist, Unlabelled])
test_list = None

def adjustData(img,mask,flag_multi_class,num_class):
    if(flag_multi_class):
        img = img / 255
        mask = mask[:,:,:,0] if(len(mask.shape) == 4) else mask[:,:,0]
        new_mask = np.zeros(mask.shape + (num_class,))
        for i in range(num_class):
            #for one pixel in the image, find the class in mask and convert it into one-hot vector
            #index = np.where(mask == i)
            #index_mask = (index[0],index[1],index[2],np.zeros(len(index[0]),dtype = np.int64) + i) if (len(mask.shape) == 4) else (index[0],index[1],np.zeros(len(index[0]),dtype = np.int64) + i)
            #new_mask[index_mask] = 1
            new_mask[mask == i,i] = 1
        new_mask = np.reshape(new_mask,(new_mask.shape[0],new_mask.shape[1]*new_mask.shape[2],new_mask.shape[3])) if flag_multi_class else np.reshape(new_mask,(new_mask.shape[0]*new_mask.shape[1],new_mask.shape[2]))
        mask = new_mask
    elif(np.max(img) > 1):
        img = img / 255
        mask = mask /255
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
    return (img,mask)



def trainGenerator(batch_size,train_path,image_folder,mask_folder,aug_dict,image_color_mode = "grayscale",
                    mask_color_mode = "grayscale",image_save_prefix  = "image",mask_save_prefix  = "mask",
                    flag_multi_class = False,num_class = 2,save_to_dir = None,target_size = None,seed = 1):
    '''
    can generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
    if you want to visualize the results of generator, set save_to_dir = "your path"
    '''
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes = [image_folder],
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix,
        seed = seed)
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes = [mask_folder],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        seed = seed)
    train_generator = zip(image_generator, mask_generator)
    for (img,mask) in train_generator:
        img,mask = adjustData(img,mask,flag_multi_class,num_class)
        yield (img,mask)

# def testGenerator(test_path,num_image = 30,target_size = (256,256),flag_multi_class = False,as_gray = True):
#     for i in range(num_image):
#         img = io.imread(os.path.join(test_path,"%d.png"%i),as_gray = as_gray)
#         img = img / 255
#         img = trans.resize(img,target_size)
#         img = np.reshape(img,img.shape+(1,)) if (not flag_multi_class) else img
#         img = np.reshape(img,(1,)+img.shape)
#         yield img

def validGenerator(test_path,num_image = None,target_size = None,flag_multi_class = False,as_gray = True):
    global valid_list_color
    valid_list_color = next(os.walk(test_path + '/image'))[2]

    if '.DS_Store' in valid_list_color:
        valid_list_color.remove('.DS_Store')

    img_color_array = []
    img_gt_array = []

    for i in valid_list_color:
        img_color = cv2.imread(os.path.join((test_path + '/image'),i),0)
        img_gt = cv2.imread(os.path.join((test_path + '/label'),i),0)

        img_color = img_color / 255
        img_gt = img_gt / 255

        img_color = trans.resize(img_color,target_size)
        img_gt = trans.resize(img_gt,target_size)

        # img_color = np.reshape(img_color,img_color.shape+(1,)) if (not flag_multi_class) else img_color
        # img_gt = np.reshape(img_gt,img_gt.shape+(1,)) if (not flag_multi_class) else img_gt

        img_color = np.reshape(img_color,img_color.shape + (1,))
        img_gt = np.reshape(img_gt,img_gt.shape + (1,))

        img_color_array.append(img_color)
        img_gt_array.append(img_gt)

    img_color_array = np.array(img_color_array)
    img_gt_array = np.array(img_gt_array)

    # print(img_color_array.shape,img_gt_array.shape)
        
    return (img_color_array,img_gt_array)

def testGenerator(test_path,num_image = None,target_size = None,flag_multi_class = False,as_gray = True):
    global test_list
    test_list = next(os.walk(test_path))[2]

    if '.DS_Store' in test_list:
        test_list.remove('.DS_Store')

    for i in test_list:
        img = cv2.imread(os.path.join(test_path,i),0)
        img = img / 255
        img = trans.resize(img,target_size)
        img = np.reshape(img,img.shape+(1,)) if (not flag_multi_class) else img
        img = np.reshape(img,(1,)+img.shape)
        yield img

def getCount(path=None):
    print(path)
    count_list = next(os.walk(path))[2]

    if '.DS_Store' in count_list:
        count_list.remove('.DS_Store')

    count = len(count_list)
    return count

def geneTrainNpy(image_path,mask_path,flag_multi_class = False,num_class = 2,image_prefix = "image",mask_prefix = "mask",image_as_gray = True,mask_as_gray = True):
    image_name_arr = glob.glob(os.path.join(image_path,"%s*.png"%image_prefix))
    image_arr = []
    mask_arr = []
    for index,item in enumerate(image_name_arr):
        img = io.imread(item,as_gray = image_as_gray)
        img = np.reshape(img,img.shape + (1,)) if image_as_gray else img
        mask = io.imread(item.replace(image_path,mask_path).replace(image_prefix,mask_prefix),as_gray = mask_as_gray)
        mask = np.reshape(mask,mask.shape + (1,)) if mask_as_gray else mask
        img,mask = adjustData(img,mask,flag_multi_class,num_class)
        image_arr.append(img)
        mask_arr.append(mask)
    image_arr = np.array(image_arr)
    mask_arr = np.array(mask_arr)
    return image_arr,mask_arr


def labelVisualize(num_class,color_dict,img):
    img = img[:,:,0] if len(img.shape) == 3 else img
    img_out = np.zeros(img.shape + (3,))
    for i in range(num_class):
        img_out[img == i,:] = color_dict[i]
    return img_out / 255


# def saveResult(save_path,npyfile,flag_multi_class = False,num_class = 2):
#     for i,item in enumerate(npyfile):
#         i = i+301
#         img = labelVisualize(num_class,COLOR_DICT,item) if flag_multi_class else item[:,:,0]
#         io.imsave(os.path.join(save_path,"%d.png"%i),img)


def saveResult(save_path,npyfile,flag_multi_class = False,num_class = 2):
    global test_list

    for i,item in enumerate(npyfile):
        img_name = test_list[i]
        img = labelVisualize(num_class,COLOR_DICT,item) if flag_multi_class else item[:,:,0]
        io.imsave(os.path.join(save_path,"%s"%img_name),img)


# if __name__ == '__main__':
#     test_path = "data/membrane/test_301_330_ep100"
#     a=testGenerator(test_path=test_path)
#     for i in a:
#         print(i.shape)