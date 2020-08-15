from model import *
from data import *
from image_metrics import *
import keras
import matplotlib.pyplot as plt

import tensorflow as tf
print('\nTF VERSION\n',tf.version.VERSION)

#..........IMPORTANT..........

base_path = "/content/gdrive/My Drive/unet-master-new-x/UNet_Pro_256_256/"

batch_size = 1
trainCount = None
epoch_array = [1100]

target_size = (256,256)
target_size_channels = (256,256,1)

model,model_name = unet(input_size = target_size_channels)

path_true = None
path_pred = None

#..........CallBack Class..........

class ClacMatrix(tf.keras.callbacks.Callback):
    base_path = None
    data_set = None
    testGene = None
    testCount = None
    target_siz = None

    def __init__(self,bp,ds,tg,tc,tz):
        self.base_path=bp
        self.data_set=ds
        self.testGene=tg
        self.testCount=tc
        self.target_size=tz
        print("\nbase_path:",base_path,"\ndata_set:",data_set,"\ntest_count:",testCount,"\ntarget_size:",target_size)

    def on_epoch_end(self, epoch, logs=None):
        # if((epoch == 100) or (epoch == 500) or (epoch == 1000) or (epoch == 2000) or (epoch == 3000) or (epoch == 4000) or (epoch == 5000)):
        if((epoch == 100) or (epoch == 500) or (epoch == 1000)):
            print("Epoch::",epoch)
            test_path = base_path + "data/" + data_set + "/test"
            gt_path = base_path + "data/" + data_set + "/test/label"
            save_path = base_path + "RESULTS/EP" + str(epoch) + "/" + data_set

            path_true = gt_path
            path_pred = save_path

            testGene = testGenerator(test_path = (test_path + '/image'),target_size = target_size)
            
            print("\npath_true:",path_true,"\npath_pred:",path_pred,"\nsave_path:",save_path)

            # PREDICT
            results = model.predict_generator(testGene,steps = testCount,verbose=1)
            saveResult(save_path,results)
            
            # EVALUATE
            true_list_new, pred_list_new = read_from_folder(path_true = path_true , path_pred = path_pred)
            
            try:
                # confusion_metrics_method_01(true_list_new = true_list_new,pred_list_new = pred_list_new)
                scikit_metrix(true_list_new = true_list_new,pred_list_new = pred_list_new)
            except:
                print("An exception occurred")


#.............................................................

for epoch in epoch_array:
    print('\nTrain Epochs Count\t:\t',epoch,'\n')

    # Select datasets to evaluate
    for i in range(0,1):
        train_path = None
        test_path = None
        save_path = None
        model_load_path = None
        data_set = None

        # Define datasets to evaluate
        if(i==0):
            data_set = "ONR"
        elif(i==1):
            data_set = "OFR"
        elif(i==2):
            data_set = "BOTH"
        elif(i==3):
            data_set = "DRONE"

        print('\n Dataset\t:\t',data_set,'\n')
            
        train_path =  base_path + "data/" + data_set + "/train"
        test_path = base_path + "data/" + data_set + "/test"
        gt_path = base_path + "data/" + data_set + "/test/label"
        # save_path = base_path + "RESULTS/EP" + str(epoch) + "/" + data_set
       
         #..........SetUp..........

        data_gen_args = dict(rotation_range=0.2,
                            width_shift_range=0.05,
                            height_shift_range=0.05,
                            shear_range=0.05,
                            zoom_range=0.05,
                            horizontal_flip=True,
                            fill_mode='nearest')
                            
        myGene = trainGenerator(batch_size = batch_size, train_path = train_path, image_folder = 'image', mask_folder = 'label', aug_dict = data_gen_args, save_to_dir=None,target_size = target_size)
        trainCount = getCount(path = (train_path+'/image'))
        print('\nntrainCount\t:\t',trainCount,'\n')

        testGene = testGenerator(test_path = (test_path + '/image'),target_size = target_size)
        testCount = getCount(path = (test_path + '/image'))
        print('\ntestCount\t:\t',testCount,'\n')

        validGene = validGenerator(test_path ,target_size = target_size)
        validCount = getCount(path = (test_path + '/image'))
        print('\nvalidCount\t:\t',validCount,'\n')

        #..........Training..........

        #..........IMPORTANT..........
        steps_per_epoch = trainCount
        #.............................
        
        print('\nbatchSize\t:\t',batch_size,'\n')
        print('\nstepsPerEpoch\t:\t',steps_per_epoch,'\n')

        model_load_path = base_path + "MODELS/" + model_name + "_" + str(target_size[0]) + "_" + str(target_size[1]) + "_bs_" + str(batch_size) + "_spe_" + str(steps_per_epoch) + "_ep_"  + str(epoch) + "_" + data_set + "_shuffle_true"
        print('\nmodelPath\t:\t',model_load_path,'\n')

        # check for loss
        model_checkpoint_1 = ModelCheckpoint((model_load_path+'.h5'), monitor='loss', verbose=2, save_best_only=True)

        # save model after x iterations
        model_load_path_epochs = model_load_path + "_@epoch_{epoch:06d}_" + '.h5'
        model_checkpoint_2 = ModelCheckpoint(model_load_path_epochs, monitor='loss', verbose=2, save_best_only=False, save_weights_only=False, mode='auto', period=5000)
        
        # path_true = gt_path
        # path_pred = save_path
        
        ClacMatrix_obj = ClacMatrix(base_path,data_set,testGene,testCount,target_size)
        
        history = model.fit_generator(generator = myGene, steps_per_epoch=steps_per_epoch, epochs=epoch, verbose = 1, callbacks=[model_checkpoint_1,model_checkpoint_2,ClacMatrix_obj],shuffle=True)

        #..........View..........

        # plt.plot(history.history['val_acc'])
        # plt.plot(history.history['val_acc'])
        # plt.title('Model Accuracy')
        # plt.ylabel('Accuracy')
        # plt.xlabel('Epoch')
        # plt.legend(['Train' 'Test'], loc='upper left')
        # plt.legend(['Train'], loc='upper left')
        # plt.show()

        #..........Prediction Drirect..........
        
        # results = model.predict_generator(testGene,steps = testCount,verbose=1)
        # saveResult(save_path,results)

        #..........Prediction After Loading MODEL..........

        # steps_per_epoch = 2
        # model_load_path = base_path + "MODELS/" + model_name + "_" + str(target_size[0]) + "_" + str(target_size[1]) + "_bs_" + str(batch_size) + "_spe_" + str(steps_per_epoch) + "_ep_"  + str(epoch) + "_" + data_set + "_shuffle_true.h5"
        # print("MODEL LOAD PATH",model_load_path)

        # model = load_model(model_load_path)

        # results = model.predict_generator(testGene,steps = testCount,verbose=1)
        # saveResult(save_path,results)

        #..........Evaluation..........

        # path_true = gt_path
        # path_pred = save_path

        # print(path_true,"\n",path_pred)
        
        # true_list_new, pred_list_new = read_from_folder(path_true = path_true , path_pred = path_pred)
        
        # try:
        #     # confusion_metrics_method_01(true_list_new = true_list_new,pred_list_new = pred_list_new)
        #     scikit_metrix(true_list_new = true_list_new,pred_list_new = pred_list_new)
        # except:
        #     print("An exception occurred")
        
print("ALL DONE")


