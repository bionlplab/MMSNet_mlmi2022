import sys
import os
from modelnew import Res, ensemble_res, Den, ensemble_model, ensemble_resden,ensemble_resden1, multiscale_Net, Multiscale_multimodel,triplescale_Net, Res1,Den1,mobv2, vgg_16, nas,naslarge, xception,ensemble_resden_double,ensemble_resden2,Res_double,Den_double,ensemble_resden_siamese,ensemble_resden_siamese1,ensemble_resden_siamese2
from modelnew import ensemble_resden_siamese3,ensemble_resden_siamese4,ensemble_resden_siamese_contrastive,ensemble_resden_siamese_contrastive_so,ensemble_resden_siamese_contra3
from data_load_cv_series_siames import load_data_series_siamese
import numpy as np
from keras.layers import Input
from keras.callbacks import ModelCheckpoint
import keras.backend as K
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix
import copy
from skimage.transform import resize
from ImageGenerator_cv_series_siamese import DataGenerator_seires_siamese
path = '/prj0129/mil4012/glaucoma' 


def weighted_binary_crossentropy(y_true, y_pred) :
    weight = 1 - K.sum(y_true) /(K.sum(y_true) + K.sum(1 - y_true))
    y_true = K.clip(y_true, K.epsilon(), 1-K.epsilon())
    y_pred = K.clip(y_pred, K.epsilon(), 1-K.epsilon())
    logloss = -(y_true * K.log(y_pred) * weight +  (1 - y_true) * K.log(1 - y_pred) * (1-weight))
    return K.mean(logloss, axis=-1)

def loss(margin=1):
    """Provides 'constrastive_loss' an enclosing scope with variable 'margin'.
  Arguments:
      margin: Integer, defines the baseline for distance for which pairs
              should be classified as dissimilar. - (default is 1).
  Returns:
      'constrastive_loss' function with data ('margin') attached.
  """

    # Contrastive loss = mean( (1-true_value) * square(prediction) +
    #                         true_value * square( max(margin-prediction, 0) ))
    def contrastive_loss(y_true, y_pred):
        """Calculates the constrastive loss.
      Arguments:
          y_true: List of labels, each label is of type float32.
          y_pred: List of predictions of same length as of y_true,
                  each label is of type float32.
      Returns:
          A tensor containing constrastive loss as floating point value.
      """

        square_pred = tf.math.square(y_pred)
        margin_square = tf.math.square(tf.math.maximum(margin - (y_pred), 0))
        return tf.math.reduce_mean(
            (1 - y_true) * square_pred + (y_true) * margin_square
        )

    return contrastive_loss

# for one output
def total_loss(margin=1):
    def to_loss(y_true, y_pred):
        """Calculates the constrastive loss.
      Arguments:
          y_true: List of labels, each label is of type float32.
          y_pred: List of predictions of same length as of y_true,
                  each label is of type float32.
      Returns:
          A tensor containing constrastive loss as floating point value.
      """

        square_pred = tf.math.square(y_pred)
        margin_square = tf.math.square(tf.math.maximum(margin - (y_pred), 0))
        return 0.5* tf.math.reduce_mean(
            (1 - y_true) * square_pred + (y_true) * margin_square
        ) + 0.5 * weighted_binary_crossentropy(y_true, y_pred)

    return to_loss

# for two outputs
def total_con_loss(margin=1):
    def to_loss(y_true, y_pred):
        """Calculates the constrastive loss.
      Arguments:
          y_true: List of labels, each label is of type float32.
          y_pred: List of predictions of same length as of y_true,
                  each label is of type float32.
      Returns:
          A tensor containing constrastive loss as floating point value.
      """

        square_pred = tf.math.square(y_pred[1])
        margin_square = tf.math.square(tf.math.maximum(margin - (y_pred[1]), 0))
        return 0.5* tf.math.reduce_mean(
            (1 - y_true) * square_pred + (y_true) * margin_square
        ) + 0.5 * weighted_binary_crossentropy(y_true, y_pred[0])

    return to_loss



def get_train_test_p_id(glaucoma_list,normal_list, fold, total_num_fold):
    num_glaucoma = len(glaucoma_list) // 2
    test_num_glaucoma = num_glaucoma // total_num_fold * 2
    
    num_normal = len(normal_list) // 2
    test_num_normal = num_normal // total_num_fold * 2

    if fold == total_num_fold:
        test_glaucoma = glaucoma_list[((fold-1) * test_num_glaucoma):,:]
        test_normal = normal_list[((fold-1) * test_num_normal):,:]
        train_glaucoma = glaucoma_list[0:((fold-1) * test_num_glaucoma),:]
        train_normal = normal_list[0:((fold-1) * test_num_normal),:]
    else:
        test_glaucoma = glaucoma_list[((fold-1) * test_num_glaucoma):fold * test_num_glaucoma,:]
        test_normal = normal_list[((fold-1) * test_num_normal):fold * test_num_normal,:]
        train_glaucoma = np.concatenate((glaucoma_list[0:((fold-1) * test_num_glaucoma),:], glaucoma_list[(fold * test_num_glaucoma):,:]), axis=0)
        train_normal = np.concatenate((normal_list[0:((fold-1) * test_num_normal),:], normal_list[(fold * test_num_normal):,:]), axis=0)
    
    valiation_glaucoma = train_glaucoma[int(0.8*len(train_glaucoma) // 2) * 2:,:]
    validation_normal = train_normal[(len(train_normal) - len(valiation_glaucoma)):,:] 

    train_glaucoma = train_glaucoma[0:(len(train_glaucoma)-len(valiation_glaucoma)) :]

    train_normal = train_normal[0:(len(train_normal) - len(validation_normal)),:]
    le_train_glaucoma = len(train_glaucoma)
    le_train_normal = len(train_normal)
    le_validation_glaucoma = len(valiation_glaucoma)
    le_validation_normal = len(validation_normal)
    
    le_test_glaucoma = len(test_glaucoma)
    le_test_normal = len(test_normal)
    

    
    train_name = np.concatenate((train_normal, train_glaucoma), axis=0)
    validation_name = np.concatenate((validation_normal, valiation_glaucoma), axis=0)
    test_name = np.concatenate((test_normal, test_glaucoma), axis=0)
    return train_normal,train_glaucoma,le_train_glaucoma, le_train_normal, validation_name, le_validation_glaucoma, le_validation_normal, test_name, le_test_glaucoma, le_test_normal



def train_simense(x_train, y_train, x_val, y_val, model, epochs, weights_path):
    print('the program start now')
    model_checkpoint = ModelCheckpoint(weights_path, monitor='val_loss', save_best_only=True)
    model.fit(x_train,y_train, validation_data=(x_val, y_val), batch_size= 24, epochs=epochs
              ,shuffle=True, callbacks=[model_checkpoint])
    print('fitting done')

def pair_generator(train_generator, train_labels, datagens, batch_size=32):
    while True:
        # shuffle once per batch
        indices = np.random.permutation(np.arange(len(train_labels)))
        num_batches = len(train_labels) // batch_size
        for bid in range(num_batches):
            batch_indices = indices[bid * batch_size : (bid + 1) * batch_size]
           # batch = [train_generator[i] for i in batch_indices]
            X1 = np.zeros((batch_size, 224, 224, 3))
            X2 = np.zeros((batch_size, 224, 224, 3))
            Y = np.zeros((batch_size, ))
            for i in range(batch_size):
                if datagens is None or len(datagens) == 0:
                    X1[i] = train_generator[0][batch_indices[i]]
                    X2[i] = train_generator[1][batch_indices[i]]
                else:
                  #  X1[i] = datagens[0].random_transform(train_generator[0][batch_indices[i]])
                    X1[i] = train_generator[0][batch_indices[i]]
                    X2[i] = datagens[1].random_transform(train_generator[1][batch_indices[i]])
                Y[i] = train_labels[[batch_indices[i]]]
            yield [X1, X2], Y

def train_simense_au(x_train, y_train, x_val, y_val, model, epochs, weights_path):
    print('the program start now')
    
    datagen_args = dict(rotation_range=10,
                    width_shift_range=0.1,
                    height_shift_range=0.1,
                    horizontal_flip=True)
    datagens = [ImageDataGenerator(**datagen_args),
            ImageDataGenerator(**datagen_args)]
    BATCH_SIZE = 32
    train_pair_gen = pair_generator(x_train, y_train, datagens, BATCH_SIZE)

    model_checkpoint = ModelCheckpoint(weights_path, monitor='val_loss', save_best_only=True)
    model.fit_generator(train_pair_gen, validation_data=(x_val, y_val), steps_per_epoch=len(y_train) // BATCH_SIZE, epochs=epochs
              ,shuffle=True, callbacks=[model_checkpoint])
    print('fitting done')    

def train(x_train, y_train, x_val, y_val, model, epochs, weights_path):
    print('the program start now')
    datagen = ImageDataGenerator(rotation_range=10, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)

    datagen.fit(x_train)
#    print('data tpye of x_train is', type(x_train), type(y_train))
    model_checkpoint = ModelCheckpoint(weights_path, monitor='val_loss', save_best_only=True)
    print('the program start to fit')
    model.fit_generator(datagen.flow(x_train, y_train, batch_size= 64), validation_data=(x_val, y_val), steps_per_epoch=len(x_train) // 64, epochs=epochs
                        , shuffle=True, callbacks=[model_checkpoint])
    print('fitting done')




def screening_sens_at_spec(y_true, y_pred, at_spec, eps=sys.float_info.epsilon): 
#     y_true, y_pred = get_screening_cases(predictions_all, reference_all)

    fpr, tpr, threshes = roc_curve(y_true, y_pred, drop_intermediate=False)
    spec = 1 - fpr

    operating_points_with_good_spec = spec >= (at_spec - eps)
    max_tpr = tpr[operating_points_with_good_spec][-1]

    operating_point = np.argwhere(operating_points_with_good_spec).squeeze()[-1]
    operating_tpr = tpr[operating_point]

    assert max_tpr == operating_tpr or (np.isnan(max_tpr) and np.isnan(operating_tpr)), f'{max_tpr} != {operating_tpr}'
    assert max_tpr == max(tpr[operating_points_with_good_spec]) or (np.isnan(max_tpr) and max(tpr[operating_points_with_good_spec])), \
        f'{max_tpr} == {max(tpr[operating_points_with_good_spec])}'

    return max_tpr

def test(x_test, y_test, model, weights):
#def test(x_test, y_test, model, weights):
    model.load_weights(weights)
    p_test = model.predict(x_test)
    
#     p_test = 1 - p_test
#     y_test = 1 - y_test
#    np.savetxt(weights[i][:-3]+'.txt', np.reshape(p_test,(len(p_test),)))
#     p_test = get_test(x_test, y_test, model, weights)
    p_classes = copy.deepcopy(p_test)
    p_classes[p_classes>=0.5]=1
    p_classes[p_classes<0.5]=0
    if len(p_test.shape) == 2:
        p_test = p_test[:, 0]
    if len(p_classes.shape) == 2:
        p_classes = p_classes[:, 0]

    print('the shape of test is', p_test.shape)
    accuracy = accuracy_score(y_test, p_classes)
    print('classification accuracy: ', accuracy)
    precision = precision_score(y_test, p_classes)
    print('precision: ', precision)
    recall = recall_score(y_test, p_classes)
    print('recall: ', recall)
    f1 = f1_score(y_test, p_classes)
    print('F1 score: ', f1)
    auc = roc_auc_score(y_test, p_test)
    print('AUC: ', auc)
    auc_90 = roc_auc_score(y_test, p_test, max_fpr=(1 - 0.9))
    print('AUC_90: ', auc_90)
    screening_sens_at_95_spec =  screening_sens_at_spec(y_test, p_test, 0.95)
    print('screening_sens_at_95_spec: ', screening_sens_at_95_spec)
    matrix = confusion_matrix(y_test, p_classes)
    print(matrix)
    result_den = np.concatenate((y_test, p_classes,p_test), axis=-1)

    
    return





if __name__ == '__main__':
    w_path2 = '/prj0129/mil4012/glaucoma/weights/glaucoma_DenseNet201.h5'
    w_path1 = '/prj0129/mil4012/glaucoma/weights/glaucoma_ResNet152.h5'
    w_path22 = '/prj0129/mil4012/glaucoma/weights/glaucoma_DenseNet201double3_ohtsnew.h5'
    w_path11 = '/prj0129/mil4012/glaucoma/weights/glaucoma_ResNet152double3_ohts.h5'
    model_path = '/prj0129/mil4012/glaucoma/weights/glaucoma_MultiNet1sp_5.h5'
#     w_path2 = 'glaucoma_DenseNet201LAG_5.h5'
#     w_path1 = 'glaucoma_ResNet152LAG_5.h5'
    #model = vgg16(img_size=(224, 224, 3), scale=1,dropout=False)
    #model.load_weights('vgg16_glaucoma.h5')
    #model.summary()
   # model = vgg_16(vgg_en='vgg_16',img_size=(224, 224, 3), dropout=False)
   # model = nas(nas_en ='nasmobile',img_size=(224, 224, 3), dropout=False)
  #  model = naslarge(naslarge_en = 'naslarge',img_size=(331, 331, 3), dropout=False)
   # model = xception(xcep_en = 'xception',img_size=(299, 299, 3), dropout=False)
#     model = mobv2(mob_en='mobv2',img_size=(224, 224, 3), dropout=False)
   # model = ensemble_vgg(img_size=(224, 224, 3), model_input=Input((224, 224, 3)),dropout=False)
   # model = ensemble_res(res_en=['res50','res101','res152'],img_size=(224, 224, 3), model_input=Input((224, 224, 3)),dropout=False)
#     model = Den(den_en='den201',img_size=(224, 224, 3), dropout=False)
   # model = Den_double(w_path2,den_en='den201',img_size=(224, 224, 6), dropout=False,flag = 1)
   # model = Res1(res_en='res152',img_size=(224, 224, 6), dropout=False)
   # model = Den1(den_en='den201',img_size=(224, 224, 6), dropout=False)
   # model = Res(res_en='res152',img_size=(224, 224, 3), dropout=False)
#     model = Res_double(w_path1,res_en='res152',img_size=(224, 224, 6), dropout=False,flag = 1)
  #  model = ensemble_model(model_en=['res152','den201'],img_size=(224, 224, 3), model_input=Input((224, 224, 3)),dropout=False)
  #  model = ensemble_resden(img_size=(224, 224, 3), model_input=Input((224, 224, 3)),dropout=False,flag=1)
    #model = ensemble_resden2(model_path,w_path1,w_path2,w_path11,w_path22,img_size=(224, 224, 6), model_input=Input((224, 224, 3)),dropout=False,flag = 0)
#     model = ensemble_resden_siamese(model_path,w_path1,w_path2,img_size=(224, 224, 3), model_input=Input((224, 224, 3)), dropout=False, flag = 0) #flag=1: imagenet, flag=0: ohts
#     model = ensemble_resden_siamese1(model_path,w_path1,w_path2,img_size=(224, 224, 3), model_input=Input((224, 224, 3)), dropout=False, flag = 1)
#     model = ensemble_resden_siamese2(model_path,w_path1,w_path2,img_size=(224, 224, 3), model_input=Input((224, 224, 3)), dropout=False, flag = 1)
#proposed
#     model = ensemble_resden_siamese3(model_path,w_path1,w_path2,img_size=(224, 224, 3), model_input=Input((224, 224, 3)), dropout=False, flag = 2)
   # model = ensemble_resden_siamese4(model_path,w_path1,w_path2,img_size=(224, 224, 3), model_input=Input((224, 224, 3)), dropout=False, flag = 1)
 # flag = 1: euclidean distance, flag=0: cosin
#     model = ensemble_resden_siamese_contrastive(model_path,w_path1,w_path2,img_size=(224, 224, 3), model_input=Input((224, 224, 3)), dropout=False, flag = 0)
# flag = 0: siamese + conv + side output + cos; flag =2: siamese +cov +cos
    model = ensemble_resden_siamese_contra3(model_path,w_path1,w_path2,img_size=(224, 224, 3), model_input=Input((224, 224, 3)), dropout=False, flag = 0)
# flag = 1: euclidean distance, flag=0: cosin
#     model =ensemble_resden_siamese_contrastive_so(model_path,w_path1,w_path2,img_size=(224, 224, 3), model_input=Input((224, 224, 3)), dropout=False, flag = 0)   
   # model = ensemble_resden1(w_path1,w_path2,img_size=(224, 224, 3), model_input=Input((224, 224, 3)),dropout=False,flag=1)
   # model = ensemble_resden_double(w_path1,w_path2,img_size=(224, 224, 6), model_input=Input((224, 224, 6)),dropout=False,flag=1)
  #  model = multiscale_Net(net='res152',img_size=(224, 224, 3), dropout=False, flag=1)
  #  model = Multiscale_multimodel(img_size=(224, 224, 3), dropout=False, flag=1)
  #  model = triplescale_Net(net='den201',img_size=(224, 224, 3), dropout=False, flag=0)
#     model.load_weights('glaucoma_ResNet152AREDS.h5')
    learning_rate = 5*1e-5
    epochs = 15

    weights_path = '/prj0129/mil4012/glaucoma/weights/DenseNet201conv_ohts_sosiamese_pair_au_cos_ce_5year_w55demon.h5'

    model.compile(optimizer=Adam(lr=learning_rate), loss=weighted_binary_crossentropy)   
    
    # #     #for contrasitive loss
#     margin = 1
#     model.compile(loss=loss(margin=margin), optimizer=Adam(lr=learning_rate), metrics=["binary_accuracy"])
#     model.compile(loss=loss(margin=margin), optimizer="RMSprop", metrics=["binary_accuracy"])

#     #for contrasitive loss + weighted_binary_corssentryop
#     margin = 1
#     model.compile(optimizer=Adam(lr=learning_rate),loss= total_con_loss(margin=margin),metrics=["binary_accuracy"])
#     model.compile(optimizer=Adam(lr=learning_rate),loss= total_loss(margin=margin),metrics=["binary_accuracy"])
#     model.compile(loss=loss(margin=margin), optimizer="RMSprop", metrics=["binary_accuracy"])
    
    label_path1 = os.path.join(path,'glaucoma_list_patient.csv')
    tmp = np.loadtxt(label_path1, dtype=np.str, delimiter=",")

    label_path2 = os.path.join(path,'normal_list_patient.csv')
    tmp_1 = np.loadtxt(label_path2, dtype=np.str, delimiter=",")

    tmp = tmp[1:,:] 
    tmp_1 = tmp_1[1:,:]
    fold = 1
    total_num_fold = 5
    x_size = 224
    y_size = 224
    train_normal,train_glaucoma,le_train_glaucoma, le_train_normal, validation_name, le_validation_glaucoma, le_validation_normal, test_name, le_test_glaucoma, le_test_normal = get_train_test_p_id(tmp, tmp_1, fold, total_num_fold)
    
    print('the number of training', (le_train_glaucoma + le_train_normal))
    print('the number of validation', len(validation_name))
    print('the number of testing', len(test_name))
    #print(test_name)


    val_images,val_labels,test_images,test_labels = load_data_series_siamese(x_size,y_size, data_path=os.path.join(path,'image_crop2/'),label_path=os.path.join(path,'lab_seriesbew.csv'),
                                                                                                                          image_s_path=os.path.join(path,'patient_s.csv'), uncentain_path=os.path.join(path,'uncentain.csv'),
                                                                                                                          validation_name=validation_name,test_name=test_name)
    
#     val_images = val_images[1]
#     test_images = test_images[1]

    
    train_generator, train_labels = DataGenerator_seires_siamese(x_size,y_size,data_path=os.path.join(path,'image_crop2/'),label_path=os.path.join(path,'lab_seriesbew.csv'),train_normal=train_normal,train_glaucoma=train_glaucoma)
#     train_generator = train_generator[1]

    train_labels = train_labels.astype(np.float)
    val_labels = val_labels.astype(np.float)
    test_labels = test_labels.astype(np.float)
    np.savetxt('train_labels.txt', np.reshape(train_labels,(len(train_labels),)))
    np.savetxt('val_labels.txt', np.reshape(val_labels,(len(val_labels),)))
    np.savetxt('test_labels.txt', np.reshape(test_labels,(len(test_labels),)))
#     test_labels_s = test_labels_s.astype(np.float)
#     test_labels_un = test_labels_un.astype(np.float)
    print('the shape of training image:', np.shape(train_generator))
    print('the number of positive pair of training data:', len(np.argwhere(train_labels==1)))
    print('the number of positive pair of test data:', len(np.argwhere(test_labels==1)))
    print('the number of positive pair of validation data:', len(np.argwhere(val_labels==1)))


# add glaucoma example for siamense network
    index_1=np.argwhere(train_labels==0)
    index_1 = np.reshape(index_1,(len(index_1),))
    index_2=np.argwhere(train_labels==1)
    index_2 = np.reshape(index_2,(len(index_2),))
    train_generator1 = [train_generator[0][index_1],train_generator[1][index_1]]
    train_labels1 = train_labels[index_1]
    train_generator2 = [train_generator[0][index_2],train_generator[1][index_2]]
    train_labels2 = train_labels[index_2]
    print(type(train_generator1))
    print(type(train_generator2))
    
    print('the shape of train_generator:', np.shape(train_generator))
    print('the shape of training label:', np.shape(train_labels))
    print(type(train_generator))
    print(type(train_labels))
    
    print('the shape of train_generator2:', np.shape(train_generator2))
    print('the shape of train_labels2:', np.shape(train_labels2))
    
    temp1 = copy.deepcopy(train_generator1)
    temp2 = copy.deepcopy(train_generator2)
    train_generator1 = np.concatenate((temp1[0],temp2[0],temp2[0]),axis=0)
    train_generator2 = np.concatenate((temp1[1],temp2[1],temp2[1]),axis=0)
    train_generator = [train_generator1,train_generator2]
    
    
#     train_generator = np.concatenate((train_generator1,train_generator2,train_generator2),axis=1)
    train_labels = np.concatenate((train_labels1,train_labels2,train_labels2),axis=0)
    
    print('the shape of train_generator:', np.shape(train_generator))
    print('the shape of train_labels:', np.shape(train_labels))
    
    print(type(train_generator))
    print(type(train_labels))
    
    train_generator1 =[]
    train_generator2 =[]  
    temp1 = []
    temp2 = []
    
   
   ## single input
#   
#    train_simense(train_generator, train_labels, val_images, val_labels, model, epochs, weights_path)
    train_simense_au(train_generator, train_labels, val_images, val_labels, model, epochs, weights_path)
#     train(train_generator, train_labels, val_images, val_labels, model, epochs, weights_path)
    
#     test_images = np.concatenate((test_images, val_images, train_generator), axis=0)
#     test_labels= np.concatenate((test_labels, val_labels, train_labels), axis=0)

    #single
   # test(test_images, test_labels, model, weights_path)
    
    
#     #simense
    test(test_images, test_labels, model, weights_path)


