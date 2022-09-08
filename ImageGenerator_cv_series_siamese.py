import os
import numpy as np
# from skimage.io import imread
import cv2
import copy
import random
from skimage.transform import resize
def DataGenerator_seires_siamese(x_size,y_size,data_path,label_path,train_normal,train_glaucoma):
#     x_size = 331
#     y_size = 331
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
    
    glaucoma_images1 = np.ndarray((len(train_glaucoma)*20, x_size, y_size,3))
    glaucoma_images2 = np.ndarray((len(train_glaucoma)*20, x_size, y_size,3))
    glaucoma_labels = []
    le = 0
    for i in range(len(train_glaucoma)):        
        ind = np.argwhere((ran==train_glaucoma[i][0]) & (lr==train_glaucoma[i][1]))
        kk = 0
        # eye has non-glaucoma
        if len(np.argwhere(tmp1[ind].astype(float) == 1)) == 0:
            for j in range(len(ind)):
                if lr[int(ind[j])] == train_glaucoma[i][1]:
                    data_paths = os.path.join(data_path, (ran[int(ind[j])] + '-'+ tracking[int(ind[j])] + '.jpg'))
                    IM = cv2.imread(data_paths)
                    if kk ==0:
                        glaucoma_images_base = cv2.resize(IM, (x_size, y_size))
                        gt = tmp1[int(ind[j])]
                        kk = 1
#                         ind_start = np.append(ind_start,ll_index)
                    else:
                        glaucoma_images1[le] = glaucoma_images_base
                        glaucoma_images2[le] = cv2.resize(IM, (x_size, y_size))
                        le += 1
                        glaucoma_labels = np.append(glaucoma_labels,0)
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
                    if lr[int(ind[j])] == train_glaucoma[i][1]:
                        data_paths = os.path.join(data_path, (ran[int(ind[j])] + '-'+ tracking[int(ind[j])] + '.jpg'))
                        IM = cv2.imread(data_paths)
                        if kk ==0:
                            glaucoma_images_base = cv2.resize(IM, (x_size, y_size))
                            gt = tmp1[int(ind[j])]
                            kk = 1
#                         ind_start = np.append(ind_start,ll_index)
                        else:
                            glaucoma_images1[le] = glaucoma_images_base
                            glaucoma_images2[le] = cv2.resize(IM, (x_size, y_size))
                            le += 1
                            if int(float(year[int(ind[le_num])]) - float(year[int(ind[j])]))  > 5:
                                glaucoma_labels = np.append(glaucoma_labels,0)
                            else:
                                glaucoma_labels = np.append(glaucoma_labels,1)
#                             glaucoma_labels = np.append(glaucoma_labels,0)
#                             if gt == tmp1[int(ind[j])]:
#                                 val_labels = np.append(val_labels,1)
#                             else:
#                                 val_labels = np.append(val_labels,0)
#                 if le_num < 5:
#                     glaucoma_labels[len(glaucoma_labels)-le_num:] = 1
#                 else:       
#                     glaucoma_labels[len(glaucoma_labels)-5:] = 1
        

    glaucoma_images1 = glaucoma_images1[0:le,:,:,:]
    glaucoma_images2 = glaucoma_images2[0:le,:,:,:]

   # print('the length of glaucoma', len(glaucoma_images))
    
    normal_labels = []
    #sampling
#    non_ind_train = random.sample(range(0,len(train_normal)),len(train_glaucoma))
#    #non_ind_train = np.arange(len(train_glaucoma))
#   # print('the non_ind_train is', len(non_ind_train))
#    normal_images = np.ndarray((len(train_glaucoma)*20, 224, 224,3))
    
#    le = 0
#    for i in range(len(non_ind_train)):        
#        ind = np.argwhere(ran==train_normal[int(non_ind_train[i])][0])
#        for j in range(len(ind)):
#            if lr[int(ind[j])] == train_normal[int(non_ind_train[i])][1]:
#                data_paths = os.path.join(data_path, (ran[int(ind[j])] + '-'+ tracking[int(ind[j])] + '.jpg'))
#                IM = cv2.imread(data_paths)        
#                normal_images[i] = IM
#                le += 1
#                normal_labels = np.append(normal_labels,tmp1[int(ind[j])])
#           # continue
    
    normal_images1 = np.ndarray((len(train_normal)*20, x_size, y_size,3))
    normal_images2 = np.ndarray((len(train_normal)*20, x_size, y_size,3))
    le = 0
    for i in range(len(train_normal)):        
#         ind = np.argwhere(ran==train_normal[i][0])
        ind = np.argwhere((ran==train_normal[i][0]) & (lr==train_normal[i][1]))
        kk = 0
        for j in range(len(ind)):
            if lr[int(ind[j])] == train_normal[i][1]:
                data_paths = os.path.join(data_path, (ran[int(ind[j])] + '-'+ tracking[int(ind[j])] + '.jpg'))
                IM = cv2.imread(data_paths)
                if kk == 0:
                    normal_images_base = cv2.resize(IM, (x_size, y_size))
                    gt = tmp1[int(ind[j])]
                    kk = 1
                else:
                    normal_images1[le] = normal_images_base
                    normal_images2[le] = cv2.resize(IM, (x_size, y_size))
               # glaucoma_images[le] = resize(IM, (x_size, y_size, 3))
               # glaucoma_images[le] = IM
                    le += 1
                    normal_labels = np.append(normal_labels,0)
#                     # if the two image from the same class, labels = 1 else labels =0
#                     if gt == tmp1[int(ind[j])]:
#                         normal_labels = np.append(normal_labels,0)
#                     else:
#                         normal_labels = np.append(normal_labels,1)
                        
                    #                     # take the second image as the ground truth
#                     glaucoma_labels = np.append(glaucoma_labels,tmp1[int(ind[j])])
           # continue
    normal_images1 = normal_images1[0:le,:,:,:]   
    normal_images2 = normal_images2[0:le,:,:,:] 
    
    train_images1 = np.concatenate((normal_images1,glaucoma_images1),axis=0)
    train_images2 = np.concatenate((normal_images2,glaucoma_images2),axis=0)
    train_images = [train_images1,train_images2]
    train_labels = np.concatenate((normal_labels,glaucoma_labels),axis=0)
    
    return train_images, train_labels
    #return train_images
    