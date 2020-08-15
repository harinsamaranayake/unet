import os
import cv2
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

#### CHANGE
flag_resize_pred=None
# path_pred = '/Users/harinsamaranayake/Documents/Research/UNET/unet-master-puddle-data/data/membrane/pred_000_029_ep100'
# path_true = '/Users/harinsamaranayake/Documents/Research/Datasets/FCN8s/crop/label'
# path_pred = '/Users/harinsamaranayake/Documents/Research/UNET/unet-master-puddle-data/data/membrane/pred_301_330_ep100'
# path_true = '/Users/harinsamaranayake/Documents/Research/Datasets/FCN8s/crop/label'
#### END

true_img_list = []
pred_img_list = []

true_list_new=[]
pred_list_new=[]

true_img_length = 0
true_img_width = 0
pred_img_length = 0
pred_img_width = 0

#### Read from file
def read_from_file():
    true_img_list = []
    pred_img_list = []

    true_list_new=[]
    pred_list_new=[]

    text_file_path='/Users/harinsamaranayake/Documents/Research/Datasets/FCN8s/split/on_road_test_mask'
    p=np.genfromtxt(text_file_path,dtype='str')

    # obtaining the image sizes
    for i in range(p.shape[0]):
        img = p[i,0]

        pred_img = cv2.imread(path_pred+"/%s" % img)
        true_img = cv2.imread(path_true+"/%s" % img)

        # print('pred_img',pred_img.shape)
        # print('true_img',true_img.shape)

        pred_img_length = pred_img.shape[0]
        pred_img_width = pred_img.shape[1]
        true_img_length = true_img.shape[0]
        true_img_width = true_img.shape[1]
        # print(pred_img_length, pred_img_width, true_img_length, true_img_width)
        
        break

    # resizing to same shape
    for i in range(p.shape[0]):
        # gt imgage name
        img = p[i,0]

        pred_img = cv2.imread(path_pred+"/%s" % img)
        true_img = cv2.imread(path_true+"/%s" % img)

        if (flag_resize_pred):
            pred_img = cv2.resize(pred_img, (true_img_width, true_img_length))
        else:
            true_img = cv2.resize(true_img, (pred_img_width, pred_img_length))

        # Threshold images
        # ret_1,pred_img = cv2.threshold(pred_img,128,255,cv2.THRESH_BINARY)
        # ret_2,true_img = cv2.threshold(true_img,128,255,cv2.THRESH_BINARY)

        true_img_list.append(true_img)
        pred_img_list.append(pred_img)

    #converting to numpy arrays
    true_list_new = np.array(true_img_list)
    pred_list_new = np.array(pred_img_list)

    #flatterning the arrays
    true_list_new = true_list_new.flatten()
    pred_list_new = pred_list_new.flatten()

    return true_list_new,pred_list_new


#### Read from folder
def read_from_folder(path_true = None , path_pred = None):
    print('\n..........read_from_folder..........\n')

    flag_resize_pred=False

    true_img_list = []
    pred_img_list = []

    true_list_new=[]
    pred_list_new=[]

    # true_list = next(os.walk(path_true))[2]
    pred_list = next(os.walk(path_pred))[2]

    if '.DS_Store' in pred_list:
        pred_list.remove('.DS_Store')

    # to print sizes of gt and predicted images
    for img in pred_list:
        # Note : to obtain the name of the true mask img
        # Note : commment if the names of the both files are the same
        # pred_name = img
        # mask_name_part = pred_name.split("_")
        # mask_name = mask_name_part[0]+".png"

        # obtain the image sizes
        true_img = cv2.imread(path_true+"/%s" % img)
        pred_img = cv2.imread(path_pred+"/%s" % img)

        print('true_img',true_img.shape)
        print('pred_img',pred_img.shape)
        
        true_img_length = true_img.shape[0]
        true_img_width = true_img.shape[1]
        pred_img_length = pred_img.shape[0]
        pred_img_width = pred_img.shape[1]
        
        # print(true_img_length, true_img_width, pred_img_length, pred_img_width)
        
        break

    # to obtain all gt and predicted images
    for img in pred_list:
        true_img = cv2.imread(path_true+"/%s" % img)
        pred_img = cv2.imread(path_pred+"/%s" % img)

        # Making the both images same size
        if (flag_resize_pred):
            # resizing the predicted image to match the size of gt image 
            pred_img = cv2.resize(pred_img, (true_img_width, true_img_length))
        else:
            true_img = cv2.resize(true_img, (pred_img_width, pred_img_length))

        # Threshold images
        ret_2,true_img = cv2.threshold(true_img,127,255,cv2.THRESH_BINARY)
        ret_1,pred_img = cv2.threshold(pred_img,127,255,cv2.THRESH_BINARY)

        # View images
        # cv2.imshow('true_img',true_img)
        # cv2.imshow('pred_img',pred_img)
        # cv2.waitKey(0)

        # true_img=np.array(true_img)
        # pred_img=np.array(pred_img)
        # true_img=true_img.flatten()
        # pred_img=pred_img.flatten()
        
        true_img_list.append(true_img)
        pred_img_list.append(pred_img)

    # print('true_new_img',true_img.shape)
    # print('pred_new_img',pred_img.shape)

    #converting to numpy arrays
    true_list_new = np.array(true_img_list)
    pred_list_new = np.array(pred_img_list)

    #flatterning the arrays
    true_list_new = true_list_new.flatten()
    pred_list_new = pred_list_new.flatten()

    return true_list_new,pred_list_new

def confusion_metrics_method_01(true_list_new = None,pred_list_new = None):
    print('\n..........confusion_metrics_method_01..........\n')

    tp = 0
    fp = 0
    fn = 0
    tn = 0

    for i in range(0, len(true_list_new)):
        true_value = true_list_new[i]
        pred_value = pred_list_new[i]

        if((pred_value == 255) and (pred_value == true_value)):
            tp += 1
        elif((pred_value == 255) and (pred_value != true_value)):
            fp += 1
        elif((pred_value == 0) and (pred_value != true_value)):
            fn += 1
        elif ((pred_value == 0) and (pred_value == true_value)):
            tn += 1

        if ((i % 1000000) == 0):
            print(i)

    print('tp - water detected as water - 255\n')
    print('tp\t', tp, '\nfp\t', fp, '\nfn\t', fn, '\ntn\t', tn, '\n')

    try:
        # water : white
        precision = tp/(tp+fp)
        recall = tp/(tp+fn)

        intersection_water=tp
        union_water=tp + fp + fn
        iou_water=intersection_water/union_water

        print('precision_water\t\t', precision, '\trecall_water\t', recall,'\tiou_water\t',iou_water,'\n')

        # background : black
        precision = tn/(tn+fn)
        recall = tn/(tn+fp)

        intersection_ground=tn
        union_ground=tn + fp + fn
        iou_ground=intersection_ground/union_ground

        print('precision_ground\t', precision, '\trecall_ground\t', recall,'\tiou_ground\t',iou_ground,'\n')
   
    except:
        print('calculation error')

def list_value_finder():
    # List value finder. Get different values that exists within a list.
    print('START > List value finder')

    true_value_list = []
    pred_value_list = []

    for i in true_list_new:
        if i in true_value_list:
            pass
        else:
            true_value_list.append(i)

    for j in pred_list_new:
        if j in pred_value_list:
            pass
        else:
            pred_value_list.append(j)

    print(true_value_list)
    print(pred_value_list)

    print(len(true_list_new))
    print(len(pred_list_new))

def scikit_metrix(true_list_new = None,pred_list_new = None):
    print('\n..........scikit_metrix..........\n')

    y_true = true_list_new
    y_pred = pred_list_new

    cf = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix: \n\n", cf)

    p1 = cf[0, 0]/(cf[0, 0]+cf[0, 1])
    r1 = cf[0, 0]/(cf[0, 0]+cf[1, 0])
    p2 = cf[1, 1]/(cf[1, 1]+cf[1, 0])
    r2 = cf[1, 1]/(cf[1, 1]+cf[0, 1])

    print('\nprecision-01\t\t',p1, '\nrecall-01\t\t', r1, '\nprecision-02\t\t', p2, '\nrecall-02\t\t', r2, '\n')

    print("Accuracy\t: ", accuracy_score(y_true=y_true, y_pred=y_pred))

    # print("Precision : ", precision_score(y_true=y_true, y_pred=y_pred, pos_label=0))
    print("Precision\t: ", precision_score(y_true=y_true, y_pred=y_pred, pos_label=255))
    # print("Precision : ", precision_score(y_true=y_true, y_pred=y_pred,average='weighted'))

    print("Recall\t\t: ", recall_score(y_true=y_true, y_pred=y_pred, pos_label=255))
    print("F1_Score\t: ", f1_score(y_true=y_true, y_pred=y_pred, pos_label=255),'\n')

    print('Classification Report: \n', classification_report(y_true=y_true, y_pred=y_pred))

if __name__ == '__main__':
    path_true = '/Users/harinsamaranayake/Documents/Research/Datasets/FCN8s/split/both_road_test_mask'
    path_pred = '/Users/harinsamaranayake/Desktop/uner-master-new-results/data_1000/BOTH/pred'

    true_list_new,pred_list_new = read_from_folder(path_true = path_true , path_pred = path_pred)
    confusion_metrics_method_01(true_list_new = true_list_new,pred_list_new = pred_list_new)
    scikit_metrix(true_list_new = true_list_new,pred_list_new = pred_list_new)
    # list_value_finder()
    pass
