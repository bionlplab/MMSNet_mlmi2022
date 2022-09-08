import os
import numpy as np
# from skimage.io import imread
import cv2
import copy
from skimage.transform import resize
import pandas as pd
def load_data_series_siamese(x_size,y_size,data_path,label_path,image_s_path,uncentain_path,validation_name,test_name):

    tmp = np.loadtxt(label_path, dtype=np.str, delimiter=",")
    # delete one image because we don't have the jpg image, 8252 is the position of this item and 1 is related to the title
    tmp = np.delete(tmp,8252+1, axis = 0)
    ran = tmp[:,0]
    lr = tmp[:,1]
    tracking = tmp[:,2]
    tmp1=tmp[:,3]
    year = tmp[:,4]
    ran = ran[1:len(ran)]
    lr = lr[1:len(lr)]
    tracking = tracking[1:len(tracking)]
    tmp1=tmp1[1:len(tmp1)]
    year = year[1:len(year)]
    
#     #generate ran and tracking numer for image with ending -s 
#     tmp_s = np.loadtxt(image_s_path, dtype=np.str, delimiter=",")
#     ran_s = tmp_s[:,1]
#     tracking_s = tmp_s[:,2]
#     ran_s = ran_s[1:len(ran_s)]
#     tracking_s = tracking_s[1:len(tracking_s)]
    
#     #generate ran and tracking numer for image with uncentain label 
#     tmp_un = np.loadtxt(uncentain_path, dtype=np.str, delimiter=",")
#     ran_un = tmp_un[:,0]
#     tracking_un = tmp_un[:,1]
#     ran_un = ran_un[1:len(ran_un)]
#     tracking_un = tracking_un[1:len(tracking_un)]
    
#     x_size = 331
#     y_size = 331
    val_images1 = np.ndarray((len(validation_name)*20, x_size, y_size,3))
    val_images2 = np.ndarray((len(validation_name)*20, x_size, y_size,3))
   # val_images = []
    val_labels = []
    le = 0
  
    for i in range(len(validation_name)):
        ind = np.argwhere((ran==validation_name[i][0]) & (lr==validation_name[i][1]))
        kk = 0
        # eye has non-glaucoma
        if len(np.argwhere(tmp1[ind].astype(float) == 1)) == 0:
            for j in range(len(ind)):
                if lr[int(ind[j])] == validation_name[i][1]:
                    data_paths = os.path.join(data_path, (ran[int(ind[j])] + '-'+ tracking[int(ind[j])] + '.jpg'))
                    IM = cv2.imread(data_paths)
                    if kk ==0:
                        validation_names_base = cv2.resize(IM, (x_size, y_size))
                        gt = tmp1[int(ind[j])]
                        kk = 1
#                         ind_start = np.append(ind_start,ll_index)
                    else:
                        val_images1[le] = validation_names_base
                        val_images2[le] = cv2.resize(IM, (x_size, y_size))
                        le += 1
                        val_labels = np.append(val_labels,0)
#                         if gt == tmp1[int(ind[j])]:
#                             val_labels = np.append(val_labels,1)
#                         else:
#                             val_labels = np.append(val_labels,0)
#                     ll_index += 1

        # eye has glaucoma, if the eye has glaucoma since baseline, skip.   
        if len(np.argwhere(tmp1[ind].astype(float) == 1)) != 0:
            le_num = len(np.argwhere(tmp1[ind].astype(float) == 0))
            if len(np.argwhere(tmp1[ind].astype(float) == 0)) < 2:
                kk = 0

            else:
                for j in range(le_num):           
                    if lr[int(ind[j])] == validation_name[i][1]:
                        data_paths = os.path.join(data_path, (ran[int(ind[j])] + '-'+ tracking[int(ind[j])] + '.jpg'))
                        IM = cv2.imread(data_paths)
                        if kk ==0:
                            validation_names_base = cv2.resize(IM, (x_size, y_size))
                            gt = tmp1[int(ind[j])]
                            kk = 1
#                         ind_start = np.append(ind_start,ll_index)
                        else:
                            val_images1[le] = validation_names_base
                            val_images2[le] = cv2.resize(IM, (x_size, y_size))
                            le += 1
                            if  int(float(year[int(ind[le_num])]) - float(year[int(ind[j])])) > 5:
                                val_labels = np.append(val_labels,0)
                            else:
                                val_labels = np.append(val_labels,1)                            
#                             val_labels = np.append(val_labels,0)
#                             if gt == tmp1[int(ind[j])]:
#                                 val_labels = np.append(val_labels,1)
#                             else:
#                                 val_labels = np.append(val_labels,0)
#                 if le_num < 5:
#                     val_labels[len(val_labels)-le_num:] = 1
#                 else:       
#                     val_labels[len(val_labels)-5:] = 1
    
    val_images1 = val_images1[0:le,:,:,:]
    val_images2 = val_images2[0:le,:,:,:]
    val_images = [val_images1,val_images2]
    
    test_images1 = np.ndarray((len(test_name)*20, x_size, y_size,3))
    test_images2 = np.ndarray((len(test_name)*20, x_size, y_size,3))
    #test_images = []
    test_labels = []
    le = 0
    
    for i in range(len(test_name)):
        ind = np.argwhere((ran==test_name[i][0]) & (lr==test_name[i][1]))
        kk = 0
        
         # eye has non-glaucoma
        if len(np.argwhere(tmp1[ind].astype(float) == 1)) == 0:
            for j in range(len(ind)):
                if lr[int(ind[j])] == test_name[i][1]:
                    data_paths = os.path.join(data_path, (ran[int(ind[j])] + '-'+ tracking[int(ind[j])] + '.jpg'))
                    IM = cv2.imread(data_paths)
                    if kk ==0:
                        test_names_base = cv2.resize(IM, (x_size, y_size))
                        gt = tmp1[int(ind[j])]
                        kk = 1
#                         ind_start = np.append(ind_start,ll_index)
                    else:
                        test_images1[le] = test_names_base
                        test_images2[le] = cv2.resize(IM, (x_size, y_size))
                        le += 1
                        test_labels = np.append(test_labels,0)
#                         if gt == tmp1[int(ind[j])]:
#                             val_labels = np.append(val_labels,1)
#                         else:
#                             val_labels = np.append(val_labels,0)
#                     ll_index += 1

        # eye has glaucoma, if the eye has glaucoma since baseline, skip.   
        if len(np.argwhere(tmp1[ind].astype(float) == 1)) != 0:
            le_num = len(np.argwhere(tmp1[ind].astype(float) == 0))
            if len(np.argwhere(tmp1[ind].astype(float) == 0)) < 2:
                kk = 0

            else:
                for j in range(le_num):           
                    if lr[int(ind[j])] == test_name[i][1]:
                        data_paths = os.path.join(data_path, (ran[int(ind[j])] + '-'+ tracking[int(ind[j])] + '.jpg'))
                        IM = cv2.imread(data_paths)
                        if kk ==0:
                            test_names_base = cv2.resize(IM, (x_size, y_size))
                            gt = tmp1[int(ind[j])]
                            kk = 1
#                         ind_start = np.append(ind_start,ll_index)
                        else:
                            test_images1[le] = test_names_base
                            test_images2[le] = cv2.resize(IM, (x_size, y_size))
                            le += 1
                            if int(float(year[int(ind[le_num])]) - float(year[int(ind[j])])) > 5:
                                test_labels = np.append(test_labels,0)
                            else:
                                test_labels = np.append(test_labels,1)
#                             test_labels = np.append(test_labels,0)
#                             if gt == tmp1[int(ind[j])]:
#                                 val_labels = np.append(val_labels,1)
#                             else:
#                                 val_labels = np.append(val_labels,0)
#                 if le_num < 5:
#                     test_labels[len(test_labels)-le_num:] = 1
#                 else:       
#                     test_labels[len(test_labels)-5:] = 1

    test_images1 = test_images1[0:le,:,:,:]
    test_images2 = test_images2[0:le,:,:,:]
    test_images = [test_images1,test_images2]
#     test_images = test_images[0:le,:,:,:]
    

#     file_name = np.reshape(file_name,(len(file_name),1))
#     list=file_name
#     column=['path']
#     lab=pd.DataFrame(columns=column,data=list)
#     lab.to_csv("fold_name5.csv",index=False)
    
    
                
   # return val_labels, test_labels
    return val_images,val_labels, test_images,test_labels