# copyright: @Trumpetertimes(Siu hamburger) github: https://github.com/Trumpetertimes/Sparse_Generation
# This work is supported by: This work was supported by the National Key R\&D program of China (2022ZD0119005), and National Key Laboratory of Spatial Intelligent Control Technology (HTKJ2024KL502027).
## 1.  Only need to use a small set of supervised annotation data to train a pre-model, use this pre-trained model to predict the pseudo labels on entire dataset.
## 2. "point_labels_URL" is the folder path to put the Point-annotation data.
## 3. "inferenced_labels_URL" is the folder path to put predicted pseudo labels.
## 4. "Sparse_Generation_save_URL" is the output derictory of Sparse Generation, which in yolo txt format for easy to demonstration.
## 5. "Final_save_URL" is the final output derictory of Sparse Generation.
## 6. "epochs" is to set the epochs for parameter updating.
## 7. "val_labels_URL" is the folder path to put the small amount supervised labels.
## 8. "inferenced_val_labels_URL" is to set the path which pseudo labels predicted from the small amount supervised pictures.
##  If your detector model output the COCO json or VOC format annotation, transforming them to yolo txt format.
##  The other initial parameters were already set.

import argparse
import random
import time
import cv2 as cv2
import os
import numpy as np
from tqdm import tqdm
from utils.general import print_args

np.set_printoptions(threshold=np.inf)
import math
import torch as torch

torch.set_printoptions(precision=None, threshold=400, edgeitems=None, linewidth=None, profile=None)


class Location:  # Each predicted label
    def __init__(self):
        self.x = 0
        self.y = 0
        self.h = 0
        self.w = 0
        self.conf = 0


class Heatmap:
    def __init__(self):
        # device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
        self.x = 0
        self.y = 0
        self.h = 0
        self.w = 0
        self.matrix = torch.ones([])
        self.conf = 0
        self.gradient_init = [0.1, 0.2, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.2, 0.1]
        self.matrix_h = 0
        self.matrix_w = 0
        self.radius = 0
        self.bias = 1
        self.gradient = 15
        self.class_name = 0
        # print(self.conf , ' ' , self.x , ' ' , self.y , ' ' , self.w , ' ' , self.h)
        # print(self.matrix)

    def get_heatmap_matrix(self, location):  # Initialize a tensor for a prediction box
        # device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
        self.class_name = location[0]
        self.x = location[1]
        self.y = location[2]
        self.w = location[3]
        self.h = location[4]
        # self.conf = location[5]
        self.x = self.x * self.bias  # Optional multiplication by deviation weight
        self.y = self.y * self.bias

        if int(self.w / 2) % 2 == 0:  # Specify the size of the two-dimensional tensor at intervals of 2 pixels and set it as an odd number.
            matrix_w = int(self.w / 2) + 1
        else:
            matrix_w = int(self.w / 2)
        if int(self.h / 2) % 2 == 0:
            matrix_h = int(self.h / 2) + 1
        else:
            matrix_h = int(self.h / 2)

        self.matrix = torch.ones([matrix_h, matrix_w])  # Specify the size of the two-dimensional tensor at intervals of 2 pixels and set it as an odd number.
        self.matrix_w = matrix_w
        self.matrix_h = matrix_h
        # print(self.conf,' ',self.x,' ',self.y,' ',self.w,' ',self.h)
        # print(self.matrix)
        return self

    def convert_function(self, x, y, w):  # Generate a tensor based on the predicted box size

        center_x = int(self.matrix_w / 2)
        center_y = int(self.matrix_h / 2)
        if center_y == 0:
            center_y = 1
        if center_x == 0:
            center_x = 1
        if center_y < center_x:
            scale_small = center_y
            scale_big = center_x
        else:
            scale_small = center_x
            scale_big = center_y
        
        distance = math.sqrt((center_x - y) * (center_x - y) + (center_y - x) * (center_y - x))

        if distance <= scale_small:
            if distance == 0:
                self.matrix[x, y] = 1
            if 0 < distance / scale_small <= 0.25:
                self.matrix[x, y] = 0.8
            if 0.25 < distance / scale_small <= 0.50:
                self.matrix[x, y] = 0.8
            if 0.50 < distance / scale_small <= 0.75:
                self.matrix[x, y] = 0.8
            if 0.75 < distance / scale_small <= 1.0:
                self.matrix[x, y] = w
            if 1 < distance / scale_small:
                self.matrix[x, y] = 0
        else:
            if distance == 0:
                self.matrix[x, y] = 1
            if 0 < distance / scale_big <= 0.25:
                self.matrix[x, y] = 0.8
            if 0.25 < distance / scale_big <= 0.50:
                self.matrix[x, y] = 0.8
            if 0.50 < distance / scale_big <= 0.75:
                self.matrix[x, y] = 0.8
            if 0.75 < distance / scale_big <= 1.0:
                self.matrix[x, y] = w
            if 1 < distance / scale_big:
                self.matrix[x, y] = 0
        # print(self.matrix)
        return self

    def generate_heatmap_matrix(self):

        return self


class Generation_Label:
    def __init__(self):
        self.centroid_x = 0
        self.centroid_y = 0
        self.left_Anchor = 0
        self.right_Anchor = 0


def Sigmoid_function(z):
    fz = 1 / (1 + math.exp(-z))
    return fz


class OverlapClasses_BigHeatmap_SinglePic:
    def __int__(self):
        self.class_num = 0
        self.class_list = []
        self.Big_HeatMap41PicList = []


class Big_HeatMap41Pic:
    def __init__(self):
        # device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
        self.class_name = 0
        self.pic_w = 640
        self.pic_h = 640
        self.bw = 640 / 2
        self.bh = 640 / 2
        self.num_heatmap = 0
        self.heatmaps = []
        self.sum_matrix = torch.zeros([int(self.bw), int(self.bh)])
        # print("self.background_matrix: ", self.background_matrix)

    def Get_Big_Heatmap(self):
        return self.heatmaps

    def Fusion_heatmaps2big1(self):
        return

    def Padding_heatmaps(self):  # Fill the tensor corresponding to each predicted box obtained to the same size as the image

        i = 0
        while i < len(self.heatmaps):
            left_pd = int((self.heatmaps[i].x - self.heatmaps[i].w / 2) / 2)
            right_pd = int((self.pic_w - (self.heatmaps[i].x + self.heatmaps[i].w / 2)) / 2)
            top_pd = int((self.heatmaps[i].y - self.heatmaps[i].h / 2) / 2)
            bottom_pd = int((self.pic_h - (self.heatmaps[i].y + self.heatmaps[i].h / 2)) / 2)
            pad = torch.nn.ZeroPad2d(padding=(left_pd, right_pd, top_pd, bottom_pd))
            tensor_heatmap = torch.tensor(self.heatmaps[i].matrix)

            self.heatmaps[i].matrix = pad(tensor_heatmap)
            # matrix_numpy = self.heatmaps[0].matrix.numpy()
            matrix_hei = len(self.heatmaps[i].matrix)  # Padding again to make the size 321 * 321
            matrix_wid = len(self.heatmaps[i].matrix[0])
            
            if matrix_hei < 320:
                gap_h = 320 - matrix_hei
                final_top_pd = torch.nn.ZeroPad2d(padding=(0, 0, gap_h, 0))
                self.heatmaps[i].matrix = final_top_pd(self.heatmaps[i].matrix)

            if matrix_wid < 320:
                gap_w = 320 - matrix_wid
                final_left_pd = torch.nn.ZeroPad2d(padding=(gap_w, 0, 0, 0))
                self.heatmaps[i].matrix = final_left_pd(self.heatmaps[i].matrix)

            if matrix_hei > 320:
                gap_h = 320 - matrix_hei
                final_top_pd = torch.nn.ZeroPad2d(padding=(0, 0, gap_h, 0))
                self.heatmaps[i].matrix = final_top_pd(self.heatmaps[i].matrix)
            if matrix_wid > 320:
                gap_w = 320 - matrix_wid
                final_left_pd = torch.nn.ZeroPad2d(padding=(gap_w, 0, 0, 0))

                self.heatmaps[i].matrix = final_left_pd(self.heatmaps[i].matrix)

            i += 1

        return self

    def Sum_heatmaps(self, class_id):
        save_path = "./Padding_matrix/Single_sum_matrix.txt"
        big_heatmaps = self.heatmaps
        j = 0
        sum = torch.zeros([int(self.bw), int(self.bh)])
        while j < len(big_heatmaps):
            if float(big_heatmaps[j].class_name) == float(class_id):
                sum = big_heatmaps[j].matrix
                break
            j += 1
        for i in range(j, len(big_heatmaps) - 1):
            if float(big_heatmaps[i + 1].class_name) == float(class_id):
                sum += big_heatmaps[i + 1].matrix
            # print(len(sum))
        self.sum_matrix = sum
        
        return sum

    def Get_MUL_maskANDheatmap(self, masked_labels_matrix):  # Multiply the sum of the obtained point label mask tensor and the corresponding predicted box heatmap tensor for each element
        masked_heatmaps_matrix = []
        for i in range(len(masked_labels_matrix)):
            # print(len(masked_labels_matrix))
            mid = self.sum_matrix.mul(masked_labels_matrix[i])
            masked_heatmaps_matrix.append(mid)
            # print(masked_heatmaps_matrix[i])

        return masked_heatmaps_matrix

    def box_Location_predict_j(self, j, M_x, R, sum, centroid_x, m_axis_x):
        while j < (centroid_x):
            sum += m_axis_x[j]
   
            if sum >= (M_x / 2) * R:
                break
            j += 1
        return j

    def box_Location_predict_k(self, centroid, check, flag, hit, k, M_y, R, sum, m_axis_x):

        while k > centroid:
            if flag==0:
                sum += m_axis_x[k - 1]
           
                if sum >= (M_y / 2) * R:
                    break
                k -= 1

        return k

    def Get_Centroid(self, masked_heatmaps_matrix, rough_labels, count, ave_anchors_width, ave_anchors_height, R,
                     id, num, sequence, Inference_labels_thiPic, w_list, maskArea_list_x,
                     maskArea_list_y):  # Obtain the centroid and border of the mask tensor corresponding to each marker point
        # print(rough_labels)
        # device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
        pred_labels_in1pic = torch.zeros([num, 5])
        # print(pred_labels_in1pic)
        rough_labels_x = 0
        rough_labels_y = 0
        rough_labels_bullseye = 0
        Area_list_num = -1

        for i in range(num):
            # Find the corresponding point_1abel center point for the point being processed at this time
            for seek in range(rough_labels_bullseye, len(rough_labels)):
                rough_labels_bullseye += 1
                if int(float(rough_labels[seek][0])) == int(float(id)):
                    Area_list_num += 1
                    rough_labels_x = float(rough_labels[seek][1]) / 2
                    rough_labels_y = float(rough_labels[seek][2]) / 2

                    break
            
            m_axis_x = torch.sum(masked_heatmaps_matrix[i], dim=0) 
            m_axis_y = torch.sum(masked_heatmaps_matrix[i], dim=1)
            # b=m_axis_y
            # a=m_axis_x
            # b=torch.reshape(b, [320,1])
            # a=a.numpy()
            # b = b.numpy()
            # save_path = "./Masked_axis_check/"+ str(num) + "_" + str(i) + "Y.txt"
            # with open(save_path, 'w') as f:
            #     np.savetxt(save_path, b, fmt='%0.1f')
            #     f.close()
            # save_path = "./Masked_axis_check/" + str(num) + "_" + str(i) + "X.txt"
            # with open(save_path, 'w') as f:
            #     np.savetxt(save_path, a, fmt='%0.1f')
            #     f.close()
            M_x = torch.sum(m_axis_x)
            # aircraft
            centroid_x = torch.sum(m_axis_x.mul(torch.arange(0, 320, 1))) / (M_x + 0.000001)  # Obtain the sum of the products of the corresponding elements on the column and their coordinates, divided by the total mass on the column
            if (centroid_x == 0):
                centroid_x = float(rough_labels_x)
            # print("x: ",centroid_x)
            pred_labels_in1pic[i][1] = centroid_x
            M_y = M_x
            centroid_y = torch.sum(m_axis_y * torch.arange(0, 320, 1)) / (M_y + 0.000001)
            # aircraft
            if (centroid_y == 0):
                centroid_y = float(rough_labels_y)
            # print("y: ",centroid_y)
            pred_labels_in1pic[i][0] = float(id)
            pred_labels_in1pic[i][2] = centroid_y
            # print('M_x',M_x)
            # print('M_y ',M_y)
            if M_x < 10:  # If this point marker does not have a heatmap tensor, specify its box

                pred_labels_in1pic[i][3] = Get_Perspective_Average_Distance(ave_anchors_width, 0,
                                                                            Inference_labels_thiPic, id, rough_labels_y*2,
                                                                            w_list,0)/2
            else:
                if centroid_x % 10 == 0:  # Take remainder
                    j = 0
                    sum = 0.0
                    while j < (centroid_x):
                        sum += m_axis_x[j]
                        # if count==2:
                        #     if sum >= M_x / 2 * (0.2-(math.log10(ave_anchors_lenth/16))/3):
                        #         break
                        # else:
                        #     if sum>=M_x/2*0.1:
                        #         break
                        if sum >= (M_x / 2) * R:
                            break
                        j += 1
                    # lleft=j
                    # # print("lleft",lleft)
                    k = int(len(m_axis_x)-1)
                    sum = 0
                    flag = 0
                    check = 0
                    hit = 0
                    k = self.box_Location_predict_k(centroid_x,check, flag,  hit, k, M_x, R, sum, m_axis_x)
                    # lright = k
                    # print("lright",lright)
                else:
                    j = 0
                    sum = 0.0
                    j = self.box_Location_predict_j(j, M_x, R, sum, centroid_x, m_axis_x)
                    # lleft=j
                    # print("lleft",lleft)
                    # print(centroid_x)
                    k = int(len(m_axis_x)-1)
                    # print('k: ',k)
                    sum = 0
                    flag=0
                    check=0
                    hit=0
                    k = self.box_Location_predict_k(centroid_x, check, flag, hit, k, M_x, R, sum, m_axis_x)
                    # lright=k
                    # print("lright",lright)
                if M_x == 0:  # If the marked point does not have a heatmap tensor, specify its box
                    pred_labels_in1pic[i][3] = Get_Perspective_Average_Distance(ave_anchors_width, 0,
                                                                                Inference_labels_thiPic, id,
                                                                                rough_labels_y*2, w_list,0) / 2

                if ((k - j) > 300) or (k - j) < 8:
                    pred_labels_in1pic[i][3] = Get_Perspective_Average_Distance(ave_anchors_width, 0,
                                                                                Inference_labels_thiPic, id,
                                                                                rough_labels_y*2, w_list,0)/2
                if ((k-j)/maskArea_list_x[Area_list_num]<0.1):
                    pred_labels_in1pic[i][3] = Get_Perspective_Average_Distance(ave_anchors_width, 0,
                                                                                Inference_labels_thiPic, id,
                                                                                rough_labels_y*2, w_list,0)/2
                else:
                    pred_labels_in1pic[i][3] = (k - j)
                # print(pred_labels_in1pic[i][2])

            if M_y <10:  

                pred_labels_in1pic[i][4] = Get_Perspective_Average_Distance(ave_anchors_height, 1,
                                                                            Inference_labels_thiPic, id, rough_labels_y*2,
                                                                            w_list,0)/2
            else:
                if centroid_y % 10 == 0:
                    j = 0
                    sum = 0.0
                    j = self.box_Location_predict_j(j, M_y, R, sum, centroid_y, m_axis_y)
                    k = int(len(m_axis_y)-1)
                    # ltop=j
                    # print("ltop",ltop)
                    sum = 0.0
                    flag = 0
                    check = 0
                    hit = 0

                    k = self.box_Location_predict_k(centroid_y, check, flag, hit, k, M_y, R, sum, m_axis_y)
                    # ldown = k
                    # print("ldown",ldown)
                else:

                    j = 0
                    sum = 0.0
                    j = self.box_Location_predict_j(j, M_x, R, sum, centroid_y, m_axis_y)
                    # ltop = j
                    # print("ltop",ltop)
                    k = int(len(m_axis_y)-1)


                    sum = 0.0
                    flag = 0
                    check = 0
                    hit = 0
                    k = self.box_Location_predict_k(centroid_y, check, flag, hit, k, M_y, R, sum, m_axis_y)
                    # ldown = k
                    # print("ldown",ldown)
                if M_y == 0:  # If this point marker does not have a heatmap tensor, specify its box
                    pred_labels_in1pic[i][4] = Get_Perspective_Average_Distance(ave_anchors_height, 1,
                                                                                Inference_labels_thiPic, id,
                                                                                rough_labels_y*2, w_list,0)/2

                if (k - j > 300) or (k - j) < 6:
                    pred_labels_in1pic[i][4] = Get_Perspective_Average_Distance(ave_anchors_height, 1,
                                                                                Inference_labels_thiPic, id,
                                                                                rough_labels_y*2, w_list,0)/2
                if ((k-j)/maskArea_list_y[Area_list_num]<0.1):     #If the instance prediction at this point is unreasonable, correct it to PDA
                    pred_labels_in1pic[i][4] = Get_Perspective_Average_Distance(ave_anchors_height, 1,
                                                                                Inference_labels_thiPic, id,
                                                                                rough_labels_y*2, w_list,0)/2
                else:
                    pred_labels_in1pic[i][4] = (k - j)
            if pred_labels_in1pic[i][3] > pred_labels_in1pic[i][4]:
                if pred_labels_in1pic[i][3] / pred_labels_in1pic[i][4] > 20:
                    pred_labels_in1pic[i][3] = pred_labels_in1pic[i][3] * 0.5
                if pred_labels_in1pic[i][3] / pred_labels_in1pic[i][4] > 20:
                    pred_labels_in1pic[i][3] = ave_anchors_width
                    pred_labels_in1pic[i][4] = ave_anchors_height
            else:
                if pred_labels_in1pic[i][4] / pred_labels_in1pic[i][3] > 20:
                    pred_labels_in1pic[i][4] = pred_labels_in1pic[i][4] * 0.5
                if pred_labels_in1pic[i][4] / pred_labels_in1pic[i][3] > 20:
                    pred_labels_in1pic[i][3] = ave_anchors_width
                    pred_labels_in1pic[i][4] = ave_anchors_height
            if pred_labels_in1pic[i][3] / ave_anchors_width > 20:
                pred_labels_in1pic[i][3] = pred_labels_in1pic[i][3] * 0.7
            if pred_labels_in1pic[i][3] / ave_anchors_width < 0.1:
                pred_labels_in1pic[i][3] = pred_labels_in1pic[i][3] * 1.3
            # print(pred_labels_in1pic[i][3])
            pred_labels_in1pic[i][0] = float(id)
            pred_labels_in1pic[i][1] = (random.uniform(-0.005, 0.005) + float(
                rough_labels[rough_labels_bullseye - 1][1]) / 2) / 320
            pred_labels_in1pic[i][2] = (random.uniform(-0.005, 0.005) + float(
                rough_labels[rough_labels_bullseye - 1][2]) / 2) / 320

            if pred_labels_in1pic[i][3]/2 > rough_labels_x:
                pred_labels_in1pic[i][3] = rough_labels_x*2
            if pred_labels_in1pic[i][4]/2 > rough_labels_y:
                pred_labels_in1pic[i][4] = rough_labels_y*2
            pred_labels_in1pic[i][3] = pred_labels_in1pic[i][3] / 320
            pred_labels_in1pic[i][4] = pred_labels_in1pic[i][4] / 320
        return pred_labels_in1pic

    def Get_Pred_Labels(self, centroids):

        return


class Fake_label:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.w = 0
        self.h = 0
        self.type = 0
        self.label_area = 50
        self.hand_label_locationea = {0, 0}

    def define_label_area(self):  # Define marker point mask area
        self.label_area = {50, 50}

    def get_single_hand_centre_label(self, hand_label_location):  # Get point markers
        self.hand_label_locationea = hand_label_location


class Heatmap_Matrix:  #
    def __init__(self):
        self.matrix = [15][15]

    def get_matrix_size(self, box_info):
        return

def Open_txt(URL):
    with open(URL, 'r', encoding='utf-8') as f:
        data = f.readlines()
        # print(data)
    f.close()
    return data


def Get_Inference_Labels(data, height, width):  # Obtain the inferred labels from a graph
    i = 0
    changed_data = []

    while i < len(data):
        # coordinate=list(data[i])
        # print(coordinate)
        coordinate = data[i]
        coordinate = coordinate.split(' ')

        coordinate[1] = float(coordinate[1]) * 640
        # print(coordinate[1])
        coordinate[2] = float(coordinate[2]) * 640
        coordinate[3] = float(coordinate[3]) * 640
        coordinate[4] = float(coordinate[4]) * 640

        global predict_labels_num
        predict_labels_num += 1
        changed_data.append(coordinate)

        i += 1

    # print(changed_data)
    return changed_data

def Write_val_Rough_Labels(labels, height, width):
    i = 0
    # print(labels)
    folder_path = "./Rough_val_labels_generated"
    save_path = "./Rough_val_labels_generated/1.txt"
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    with open(save_path, 'w') as f:
        while i < len(labels):
            # coordinate=list(data[i])
            # print(coordinate)
            changed_data = []
            coordinate = labels[i]
            coordinate = coordinate.split(' ')

            coordinate[1] = float(coordinate[1]) * 640
            # print(coordinate[1])
            coordinate[2] = float(coordinate[2]) * 640

            changed_data.append(float(coordinate[0]))

            changed_data.append(float(coordinate[1]))
            changed_data.append(float(coordinate[2]))
            i += 1
            # print(changed_data)
            for j in range(len(coordinate)):
                coordinate[j] = float(coordinate[j]) / 2

            # print("Point:   ",coordinate)           #point labels
            f.write(str(changed_data) + '\n')
    f.close()

    return


def Write_Rough_Labels(labels, height, width):
    i = 0
    # print(labels)
    folder_path = "./Rough_labels_generated"
    save_path = "./Rough_labels_generated/1.txt"
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    with open(save_path, 'w') as f:
        while i < len(labels):
            # coordinate=list(data[i])
            # print(coordinate)
            changed_data = []
            coordinate = labels[i]
            coordinate = coordinate.split(' ')

            coordinate[1] = float(coordinate[1]) * 640
            # print(coordinate[1])
            coordinate[2] = float(coordinate[2]) * 640

            changed_data.append(float(coordinate[0]))

            changed_data.append(float(coordinate[1]))
            changed_data.append(float(coordinate[2]))
            i += 1
            # print(changed_data)
            for j in range(len(coordinate)):
                coordinate[j] = float(coordinate[j]) / 2
            # print("Point:   ",coordinate)           #point labels
            f.write(str(changed_data) + '\n')
    f.close()

    return


def Get_Rough_Labels(Rough_label_URL):
    data = Open_txt(Rough_label_URL)
    i = 0
    # print(len(data))
    while i < len(data):
        data[i] = data[i].strip()
        data[i] = data[i].lstrip('[')
        data[i] = data[i].rstrip(']')
        data[i] = data[i].split(',')
        # print(data[i])
        # print(type(data[i][2]))
        i += 1

    return data


# Get_changed_coordinate(Open_txt(), 640, 640)

def Get_id_Point_label(point_labels, id):
    flag = 0
    for i in range(len(point_labels)):
        if point_labels[i][0] == id:
            flag += 1
            return point_labels[i]
    if flag == 0:
        print("Point label id: ", id, " Not found")
        return False


class Masked_Matrix:
    def __init__(self):
        self.maskArea_list_x = []
        self.maskArea_list_y = []
        self.masked_labels_matrix_list = []

    def Get_Mask_Rough_Labels_Matrix_COCO(self, labels, ave_width, ave_height,
                                          id, class_id_list, w_list,
                                          inference_labels):  # Cover the label of an image with a mask image, expand the center point of the label, and use a dynamic mask based on the predicted average pseudo label value
        # device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

        masked_labels_matrix_list = []

        for i in range(len(labels)):
            if int(float(labels[i][0])) == int(float(id)):

                ave_mask_width = ((Get_Perspective_Average_Distance(ave_width, 0, inference_labels,
                                                                    id, labels[i][2], w_list, 0)) / 1 *
                                  w_list[3])/2
                ave_mask_height = ((Get_Perspective_Average_Distance(ave_height, 1, inference_labels,
                                                                     id, labels[i][2], w_list, 0)) / 1 *
                                   w_list[3])/2
                # ave_mask_width=int(ave_width)
                # ave_mask_width = int(ave_mask_width)
                ave_mask_width=int(ave_mask_width)
                ave_mask_height = int(ave_mask_height)
                if int(ave_mask_width) % 2 == 0:
                    ave_mask_width = ave_mask_width + 0
                else:
                    ave_mask_width = ave_mask_width + 1
                if int(ave_mask_height) % 2 == 0:
                    ave_mask_height = ave_mask_height + 0
                else:
                    ave_mask_height = ave_mask_height + 1
                half_mask_lenth_width = int(ave_mask_width / 2)
                half_mask_lenth_height = int(ave_mask_height / 2)
                # if labels[i][0] == id:
                mask_area = torch.ones([int(ave_mask_height), int(ave_mask_width)])

                left_pad = int(float(labels[i][1]) / 2) - half_mask_lenth_width
                right_pad = 320 - (int((float(labels[i][1])) / 2) + half_mask_lenth_width)
                top_pad = int((float(labels[i][2])) / 2) - half_mask_lenth_height
                bottom_pad = 320 - (int((float(labels[i][2])) / 2) + half_mask_lenth_height)
                pad = torch.nn.ZeroPad2d(padding=(left_pad, right_pad, top_pad, bottom_pad))
                mask_area = pad(mask_area)

                matrix_hei = len(mask_area)  # Padding again to make the size 321 * 321
                matrix_wid = len(mask_area[0])
                
                if matrix_hei > 320:
                    gap_h = matrix_hei - 320
                    final_top_pd = torch.nn.ZeroPad2d(padding=(0, 0, -gap_h, 0))
                    mask_area = final_top_pd(mask_area)
                    # matrix_hei = len(mask_area)
                if matrix_wid > 320:
                    gap_w = matrix_wid - 320
                    final_left_pd = torch.nn.ZeroPad2d(padding=(-gap_w, 0, 0, 0))
                    mask_area = final_left_pd(mask_area)
                    # matrix_wid = len(mask_area)
                # b = mask_area.numpy()
                # save_path = "./mask_check/" + str(i) + ".txt"
                # with open(save_path, 'w') as f:
                #     np.savetxt(save_path, b, fmt='%0.1f')
                #     f.close()
                self.maskArea_list_x.append(int(ave_mask_width))
                self.maskArea_list_y.append(int(ave_mask_height))
                masked_labels_matrix_list.append(mask_area)

        return masked_labels_matrix_list


def Change_File_Name_In1directrory(URL):
    ALL_URL_Pic_To_Inference = os.listdir('./')

    return


def Get_Masked_bias_label(pseudo_labels, rough_labels):

    for i in range(len(rough_labels)):
        if len(rough_labels) == 0:
            pseudo_labels[i][0] = 0
            pseudo_labels[i][1] = 0
            pseudo_labels[i][2] = 0
            pseudo_labels[i][3] = 0
            pseudo_labels[i][4] = 0
        pseudo_labels[i][0] = rough_labels[i][0]
        pseudo_labels[i][1] = rough_labels[i][1]
        pseudo_labels[i][2] = rough_labels[i][2]

    # print(Max_conf_label)
    return pseudo_labels


def Get_Point_labels_from_labels(labels_URL, point_save_URL):
    path = labels_URL
    Point_labels_num = 0
    filenames = os.listdir(path)
    filenames.sort(key=lambda x: int(x[:-4].lstrip('1 ').lstrip('(').rstrip(')')))
    folder_path = point_save_URL
    Rough_labels_root_URL = labels_URL
    ALL_URL_Rough_labels = os.listdir(labels_URL)
    ALL_URL_Rough_labels.sort(key=lambda x: int(x[:-4].lstrip('1 ').lstrip('(').rstrip(')')))

    for i in range(len(filenames)):
        sum_rough_labels_lenth = 0.0
        save_path = folder_path + '1 (' + str(i + 1) + ')' + '.txt'
        file_path = path + filenames[i]
        label = Open_txt(file_path)
        rough_label = Open_txt(Rough_labels_root_URL + ALL_URL_Rough_labels[i])
        labels_in_onePic = []
        rough_labels_in_onePic = []
        for m in range(len(label)):
            new_label = label[m].split(' ')
            labels_in_onePic.append(new_label)
        for m in range(len(rough_label)):
            new2_label = rough_label[m].split(' ')
            sum_rough_labels_lenth += float(new2_label[3]) + float(new2_label[4])
            rough_labels_in_onePic.append(new2_label)
        # print(i,': ')
        ave_rough_labels_lenth = (sum_rough_labels_lenth / 4) / len(rough_label)
        print(ave_rough_labels_lenth)
        pseudo_labels = Get_Masked_bias_label(labels_in_onePic, rough_labels_in_onePic)

        randNum_x = random.uniform(-(ave_rough_labels_lenth * 0.2), (ave_rough_labels_lenth * 0.2))
        randNum_y = random.uniform(-(ave_rough_labels_lenth * 0.2), (ave_rough_labels_lenth * 0.2))
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)
        with open(save_path, 'w') as f:
            for j in range(len(pseudo_labels)):
                new_label = pseudo_labels[j]
                Point_labels_num += 1
                f.write(new_label[0] + ' ' + str(float(new_label[1]) + randNum_x) + ' ' + str(
                    float(new_label[2]) + randNum_y) + ' ' + new_label[3] + ' ' + new_label[4])


def Get_Final_PseudoBox_labels(Sparse_Generation_save_URL, point_labels_URL, Final_save_URL):
    path = Sparse_Generation_save_URL
    Point_labels_num = 0
    filenames = os.listdir(path)
    filenames.sort(key=lambda x: int(x[:-4]))
    folder_path = Final_save_URL  # 改3
    Rough_labels_root_URL = point_labels_URL
    ALL_URL_Rough_labels = os.listdir(point_labels_URL)  # Point tag label directory
    ALL_URL_Rough_labels.sort(key=lambda x: int(x[:-4]))

    for i in tqdm(range(len(filenames))):
        sum_rough_labels_lenth = 0.0
        save_path = folder_path + ALL_URL_Rough_labels[i]
        file_path = path + filenames[i]
        label = Open_txt(file_path)
        rough_label = Open_txt(Rough_labels_root_URL + ALL_URL_Rough_labels[i])
        labels_in_onePic = []
        rough_labels_in_onePic = []
        for m in range(len(label)):
            new_label = label[m].split(' ')
            labels_in_onePic.append(new_label)
        for m in range(len(rough_label)):
            new2_label = rough_label[m].split(' ')
            # sum_rough_labels_lenth += float(new2_label[3]) + float(new2_label[4])
            rough_labels_in_onePic.append(new2_label)
        # print(i,': ')
        # ave_rough_labels_lenth = (sum_rough_labels_lenth / 4) / len(rough_label)
        # print(ave_rough_labels_lenth)
        pseudo_labels = Get_Masked_bias_label(labels_in_onePic, rough_labels_in_onePic)

    
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)
        with open(save_path, 'w') as f:
            for j in range(len(pseudo_labels)):
                new_label = pseudo_labels[j]
                Point_labels_num += 1
                # f.write(new_label[0] + ' ' + str(float(new_label[1]) + randNum_x) + ' ' + str(
                #     float(new_label[2]) + randNum_y) + ' ' + new_label[3] + ' ' + new_label[4])
                if len(new_label) == 0:
                    f.write('')
                else:
                    f.write(new_label[0] + ' ' + str(float(new_label[1])) + ' ' + str(
                        float(new_label[2])) + ' ' + new_label[3] + ' ' + new_label[4])


def Update_w(w_list, val_labels_URL, inferenced_val_labels_URL):
    root_url_rough = val_labels_URL  
    root_url_selected = inferenced_val_labels_URL
    All_url_rough = os.listdir(root_url_rough)
    All_url_rough.sort(key=lambda x: int(x[:-4]))
    All_url_selected = os.listdir(root_url_selected)
    All_url_selected.sort(key=lambda x: int(x[:-4]))
    loss = 0.0
    lr = 0.2
    count_numOfLabels = 0
    for i in range(len(All_url_rough)):
        rough_labels_inThisPic = Open_txt(root_url_rough + All_url_rough[i])
        selected_labels_inThisPic = Open_txt(root_url_selected + All_url_selected[i])
        count_numOfLabels += len(rough_labels_inThisPic)
        for j in range(len(rough_labels_inThisPic)):
            rough_labels_inThisPic[j] = rough_labels_inThisPic[j].split(' ')
            selected_labels_inThisPic[j] = selected_labels_inThisPic[j].split(' ')
            loss += (float(selected_labels_inThisPic[j][3]) - float(rough_labels_inThisPic[j][3])) + (
                    float(selected_labels_inThisPic[j][4]) - float(rough_labels_inThisPic[j][4]))

    loss = loss / float(count_numOfLabels)
    print("loss: ", loss)

    w_list[0] += lr * math.tanh(loss)
    # w_list[1]-=lr*math.tanh(loss)*6
    # w_list[2]+=lr*math.tanh(loss)
    # w_list[3]+=lr*math.tanh(loss)
    w_list[1] += 0
    w_list[2] += 0
    w_list[3] += 0
    print("w1:", w_list[0], "  w2:", w_list[1], "  w3:", w_list[2], "  w4:", w_list[3])
    return w_list


predict_labels_num = 0


# Id_InClass_Judge(id,class_list) and Seek_Class_Num_SinglePic(PointLabel_SinglePic)：80 categories in COCO, determine how many categories there are in an image, and return the number of categories. Placed after Write-Rought_Labels
def Id_InClassList_Judge(id, class_list):
    i = 0
    while i < len(class_list):
        if id == class_list[i].class_name:
            class_list[i].num += 1
            return True
        i += 1
    return False


def Seek_Class_List_SinglePic(PointLabel_SinglePic):
    i = 1
    id_class_list = []
    if len(PointLabel_SinglePic) == 0:
        id_class_list.append(int(0))
        return id_class_list
    else:
        id = class_id()
        id.class_name = PointLabel_SinglePic[0][0]
        id.num += 1
        id_class_list.append(id)
    while i < len(PointLabel_SinglePic):
        if Id_InClassList_Judge(PointLabel_SinglePic[i][0], id_class_list):
            i += 1
            continue
        else:
            id = class_id()
            id.class_name = PointLabel_SinglePic[i][0]
            id.num += 1
            id_class_list.append(id)
        i += 1
    return id_class_list


class class_id:
    def __init__(self):
        self.class_name = 0
        self.anchor_height = 0
        self.anchor_width = 0
        self.mask_width = 0
        self.mask_height = 0
        self.num = 0


class COCO_Heatmap_SgPic:
    def __init__(self):
        self.id_List = []
        self.heatmap_List = []

    def Get_MUL_maskANDheatmap(self, masked_labels_matrix, id):  # Multiply the sum of the obtained point label mask tensor and the corresponding predicted box heatmap tensor for each element

        masked_heatmaps_matrix = []
        for i in range(len(masked_labels_matrix)):
            # print(len(masked_labels_matrix))
            mid = self.heatmap_List[id].mul(masked_labels_matrix[i])
            b=mid.numpy()
            save_path = "./masked_matrix_check/" + str(id) + "_" +str(i)+".txt"
            with open(save_path, 'w') as f:
                np.savetxt(save_path,b,fmt='%0.1f')
                f.close()
            masked_heatmaps_matrix.append(mid)
            # print(masked_heatmaps_matrix[i])

        return masked_heatmaps_matrix


def Set_NUll_InfObjct_Pic(Point_labels_SgPic, Sparse_Generation_save_URL, txt_name):
    labels_save_path = Sparse_Generation_save_URL
    save_path = labels_save_path + txt_name
    if not os.path.exists(labels_save_path):
        os.mkdir(labels_save_path)
    with open(save_path, 'w') as f:
        for i in range(len(Point_labels_SgPic)):
            if 640 * 0.22 - (640 - float(Point_labels_SgPic[i][1])) > 0:
                x_set = 640 * 0.22
            else:
                x_set = float(Point_labels_SgPic[i][1]) * 1
            if 640 * 0.22 - (640 - float(Point_labels_SgPic[i][2])) > 0:
                y_set = 640 * 0.22
            else:
                y_set = float(Point_labels_SgPic[i][2]) * 1
            if i == len(Point_labels_SgPic) - 1:
                f.write(str(int(float(Point_labels_SgPic[i][0]))) + ' ' + str(
                    float(Point_labels_SgPic[i][1]) / 640) + ' ' + str(float(
                    Point_labels_SgPic[i][2]) / 640) + ' ' + str(float(x_set)/640) + ' ' + str(float(y_set)/640))
            else:
                f.write(str(int(float(Point_labels_SgPic[i][0]))) + ' ' + str(
                    float(Point_labels_SgPic[i][1]) / 640) + ' ' + str(float(
                    Point_labels_SgPic[i][2]) / 640) + ' ' + str(float(x_set)/640) + ' ' + str(float(y_set)/640) + '\n')
    return


def Set_NUll_PointObjct_Pic(Sparse_Generation_save_URL, txt_name):
    labels_save_path = Sparse_Generation_save_URL
    save_path = labels_save_path + txt_name
    if not os.path.exists(labels_save_path):
        os.mkdir(labels_save_path)
    with open(save_path, 'w') as f:
        f.write('')
    return


def Get_Ave_Class_lenth(class_id_list, inference_labels, w_list):
    for i in range(len(class_id_list)):

        sum_box_width = 0  # Obtain the average width and height of the predicted boxes for each class in an image
        sum_box_height = 0
        num_of_id = 0
        for q in range(len(inference_labels)):
            if float(inference_labels[q][0]) == float(class_id_list[i].class_name):
                num_of_id += 1
                sum_box_width += float(inference_labels[q][3])
                sum_box_height += float(inference_labels[q][4])
        if num_of_id != 0:
            ave_anchors_width = ((sum_box_width / num_of_id))
            ave_anchors_height = ((sum_box_height / num_of_id))
            ave_mask_width = (ave_anchors_width / 0.75) * w_list[3]
            ave_mask_height = (ave_anchors_height / 0.75) * w_list[3]
            # ave_mask_lenth = (ave_mask_height + ave_mask_width) / 2
            class_id_list[i].anchor_width = ave_anchors_width
            class_id_list[i].anchor_height = ave_anchors_height
            class_id_list[i].mask_width = ave_mask_width
            class_id_list[i].mask_height = ave_mask_height
        else:
            class_id_list[i].anchor_width = 60
            class_id_list[i].anchor_height = 60
            class_id_list[i].mask_width = 70
            class_id_list[i].mask_height = 70

    return class_id_list


def Get_Init_Heatmaps(big_heatmap, inference_labels, w_list):
    m = 0
    while m < len(inference_labels):
        # print(m)
        heatmap1 = Heatmap()

        heatmap1 = heatmap1.get_heatmap_matrix(inference_labels[m])  # Initialize a heatmap

        matrix_x = heatmap1.matrix_w
        matrix_y = heatmap1.matrix_h
        # print('matrix_x: ',matrix_x,' matrix_y:',matrix_y)
        i = 0
        while i < matrix_y:
            j = 0
            while j < matrix_x:
                heatmap1 = heatmap1.convert_function(i, j, w_list[1])
                j += 1
            i += 1
        # print(heatmap1.matrix)
        big_heatmap.heatmaps.append(heatmap1)
        m += 1
    return big_heatmap


def Get_Perspective_Average_Distance(class_ave, xORy, Inference_labels_singlePic, class_id, this_point_label_y,
                                     w_list, MaskOrWidth):  # Perspective weighted average distance
    gap = 10
    ave_gap_up = 0
    gap_count_up = 0
    ave_gap_down = 0
    gap_count_down = 0
    inf_labels_id_list = []
    if MaskOrWidth == 0:
        for i in range(len(Inference_labels_singlePic)):
            if int(float(Inference_labels_singlePic[i][0])) == int(float(class_id)):
                inf_labels_id_list.append(float(Inference_labels_singlePic[i][2]))  # Perspective distance sorting
        inf_labels_id_list.sort()
        if len(inf_labels_id_list) != 0:
            distance_range = (float(inf_labels_id_list[len(inf_labels_id_list) - 1])) - float(inf_labels_id_list[0])
            if distance_range != 0:
                if xORy == 0:
                    for j in range(len(Inference_labels_singlePic)):
                        if int(float(Inference_labels_singlePic[j][0])) == int(float(class_id)):
                            if (float(Inference_labels_singlePic[j][2]) < (inf_labels_id_list[0] + gap)):
                                ave_gap_up += float(Inference_labels_singlePic[j][3])
                                gap_count_up += 1
                else:
                    for j in range(len(Inference_labels_singlePic)):
                        if int(float(Inference_labels_singlePic[j][0])) == int(float(class_id)):
                            if (float(Inference_labels_singlePic[j][2]) < (inf_labels_id_list[0] + gap)):
                                ave_gap_up += float(Inference_labels_singlePic[j][4])
                                gap_count_up += 1
                ave_gap_up = (ave_gap_up) / gap_count_up
                if xORy == 0:
                    for j in range(len(Inference_labels_singlePic)):
                        if float(Inference_labels_singlePic[j][0]) == float(class_id):
                            if (float(Inference_labels_singlePic[j][2]) > (
                                    float(inf_labels_id_list[len(inf_labels_id_list) - 1]) - gap)):
                                ave_gap_down += float(Inference_labels_singlePic[j][3])
                                gap_count_down += 1
                else:
                    for j in range(len(Inference_labels_singlePic)):
                        if float(Inference_labels_singlePic[j][0]) == float(class_id):
                            if (float(Inference_labels_singlePic[j][2]) > (
                                    float(inf_labels_id_list[len(inf_labels_id_list) - 1]) - gap)):
                                ave_gap_down += float(Inference_labels_singlePic[j][4])
                                gap_count_down += 1

                ave_gap_down = (ave_gap_down) / gap_count_down
                # class_ave = (ave_gap_up + ave_gap_down) / 2

                # gradient = (class_ave - ave_gap_up)*2 / distance_range  
                gradient = (ave_gap_down - ave_gap_up) / distance_range   # Perspective distance gradient
                bullseye = (float(inf_labels_id_list[0]) + float(inf_labels_id_list[len(inf_labels_id_list) - 1])) / 2

                ave_PDA = ((float(this_point_label_y) - inf_labels_id_list[0]) * gradient + ave_gap_up) * w_list[2]

                # ave_PDA=class_ave
                if ave_PDA < ((ave_PDA / class_ave) * 0.01):
                    ave_PDA = class_ave * 0.4
                if ave_PDA > (640):
                    ave_PDA = 600
            else:
                ave_PDA = class_ave * w_list[2]
        else:
            ave_PDA = class_ave * w_list[2]
            if ave_PDA==0:
                ave_PDA = 640 * 0.2

    if MaskOrWidth == 1:
        ave_PDA = class_ave * w_list[2]

    return ave_PDA


def Get_Random_PointLabels(Point_labels_SgPic, class_id_list):
    # for x in range(len(class_id_list)):
    #     for z in range(len(Point_labels_SgPic)):
    #         if Point_labels_SgPic[z][0] == class_id_list[x].class_name:
    # ave_PDA_width = Perspective_Average_Distance(class_id_list[x].anchor_width, Point_labels_SgPic,      #Obtain the average perspective distance
    #                                              class_id_list[x].class_name, Point_labels_SgPic[z])
    # ave_PDA_height = Perspective_Average_Distance(class_id_list[x].anchor_height, Point_labels_SgPic,
    #                                               class_id_list[x].class_name, Point_labels_SgPic[z])

    # Point_labels_SgPic[z][1] = float(Point_labels_SgPic[z][1])+random.uniform((-ave_PDA_width * 0.1),
    #                                           ave_PDA_width * 0.1)
    # Point_labels_SgPic[z][2]= float(Point_labels_SgPic[z][2])+random.uniform((-ave_PDA_height * 0.1),
    #                                           ave_PDA_height * 0.1)
    return Point_labels_SgPic


def Test(w_list, inferenced_labels_URL, Sparse_Generation_save_URL, point_labels_URL):
    round = 1
    epoch = 1
    ALL_URL_Labels_Inferenced = os.listdir(inferenced_labels_URL)
    ALL_URL_Labels_Inferenced.sort(key=lambda x: int(x[:-4]))  # Sort to read in order

    # Point label directory
    ALL_URL_Rough_labels = os.listdir(point_labels_URL)

    ALL_URL_Rough_labels.sort(key=lambda x: int(x[:-4]))

    labels_save_path = Sparse_Generation_save_URL
    while epoch < 2:
        # print("epoch :", epoch)

        count_Forinference_URL = 0
        for count in tqdm(range(len(ALL_URL_Rough_labels)), desc='Sparse with train set'):

            coco_Heatmap_sgpic = COCO_Heatmap_SgPic()
            URL_Inference = inferenced_labels_URL + ALL_URL_Labels_Inferenced[count_Forinference_URL]  # 改2
            URL_For_Genert_RoughLabels = point_labels_URL + ALL_URL_Rough_labels[count]
            global predict_labels_num
            inference_labels = Get_Inference_Labels(Open_txt(URL_Inference), 576, 640)

            URL_Rough_Label_Generated = './Rough_labels_generated/1.txt'  # Obtain the id_List in an image
            Write_Rough_Labels(Open_txt(URL_For_Genert_RoughLabels), 576, 640)
            Point_labels_SgPic = Get_Rough_Labels(URL_Rough_Label_Generated)
            class_id_list = Seek_Class_List_SinglePic(Point_labels_SgPic)
            if class_id_list[0] == 0:
                Set_NUll_PointObjct_Pic(Sparse_Generation_save_URL, ALL_URL_Rough_labels[count])
                if ALL_URL_Labels_Inferenced[count_Forinference_URL] == ALL_URL_Rough_labels[
                    count]:  # Is it predicted for the corresponding image, Otherwise, the predicted result for this image is empty
                    count_Forinference_URL += 1
                continue

            if ALL_URL_Labels_Inferenced[count_Forinference_URL] == ALL_URL_Rough_labels[count]:  # 
                count_Forinference_URL += 1

                class_id_list = Get_Ave_Class_lenth(class_id_list, inference_labels, w_list)  # Obtain the average predicted box length for each class in a graph
                # Point_labels_SgPic = Get_Random_PointLabels(Point_labels_SgPic,class_id_list)  

                coco_Heatmap_sgpic.id_List = class_id_list
                big_heatmap = Big_HeatMap41Pic()
                big_heatmap = Get_Init_Heatmaps(big_heatmap, inference_labels, w_list)  # Initialize all predicted boxes in this image as tensors

                big_heatmap = big_heatmap.Padding_heatmaps()  # All mapping tensors have been obtained, and the padding for each predicted box tensor in a single image is 321 * 321
                Sumed_Tensors_ID_List = []

                for h in range(len(class_id_list)):
                    temp_sum_matrix = big_heatmap.Sum_heatmaps(class_id_list[h].class_name)  # Sum up the predicted box tensors for each class in a single image
                    Sumed_Tensors_ID_List.append(temp_sum_matrix)
                coco_Heatmap_sgpic.heatmap_List = Sumed_Tensors_ID_List

                labels_List = []
                for i in range(len(class_id_list)):
                    masked_labels_matrixes = Masked_Matrix()
                    masked_labels_matrixes.masked_labels_matrix_list = masked_labels_matrixes.Get_Mask_Rough_Labels_Matrix_COCO(
                        Point_labels_SgPic,
                        class_id_list[i].anchor_width,
                        class_id_list[i].anchor_height,
                        class_id_list[
                            i].class_name, class_id_list, w_list,
                        inference_labels)  # Obtain the mask tensor for each marker point in an image

                    masked_heatmaps_matrix = coco_Heatmap_sgpic.Get_MUL_maskANDheatmap(
                        masked_labels_matrixes.masked_labels_matrix_list, i)
                    # Multiplying each mask tensor in an image with the heatmap tensor of the entire image yields the tensor corresponding to each point within the labeled range

                    labels_List.append(
                        big_heatmap.Get_Centroid(masked_heatmaps_matrix, Point_labels_SgPic, round,
                                                 class_id_list[i].anchor_width, class_id_list[i].anchor_height,
                                                 w_list[0], class_id_list[i].class_name, class_id_list[i].num, i,
                                                 inference_labels, w_list, masked_labels_matrixes.maskArea_list_x,
                                                 masked_labels_matrixes.maskArea_list_y))

                    labels_List[i] = labels_List[i].numpy()
                    for q in range(len(labels_List[i])):
                        labels_List[i][q][0] = int(labels_List[i][q][0])

                save_path = labels_save_path + ALL_URL_Rough_labels[count]
                if not os.path.exists(labels_save_path):
                    os.mkdir(labels_save_path)
                with open(save_path, 'w') as f:
                    for i in range(len(labels_List)):
                        for q in range(len(labels_List[i])):
                            if q == len(labels_List[i]) - 1 and i == len(labels_List) - 1:
                                f.write(str(int(labels_List[i][q][0])) + ' ' + str(
                                    format(float(labels_List[i][q][1]), '.6f')) + ' ' + str(
                                    format(float(labels_List[i][q][2]), '.6f')) + ' ' + str(
                                    format(float(labels_List[i][q][3]), '.6f')) + ' ' + str(
                                    format(float(labels_List[i][q][4]), '.6f')))
                            else:
                                f.write(str(int(labels_List[i][q][0])) + ' ' + str(
                                    format(float(labels_List[i][q][1]), '.6f')) + ' ' + str(
                                    format(float(labels_List[i][q][2]), '.6f')) + ' ' + str(
                                    format(float(labels_List[i][q][3]), '.6f')) + ' ' + str(
                                    format(float(labels_List[i][q][4]), '.6f')) + '\n')


            else:
                Set_NUll_InfObjct_Pic(Point_labels_SgPic, Sparse_Generation_save_URL, ALL_URL_Rough_labels[count])

            time.sleep(0.1)
        epoch += 1

        print()

    return w_list


def Get_Mask_Rough_Labels_Matrix(labels, ave_mask_lenth):  # Cover the label of an image with a mask image, simply mark the center point for expansion, and use a dynamic mask based on the predicted average pseudo label value
    masked_labels_matrix = [0] * len(labels)
    # print(masked_labels_matrix)
    ave_mask_lenth = int(ave_mask_lenth)
    if int(ave_mask_lenth) % 2 == 0:
        ave_mask_lenth = ave_mask_lenth + 0
    else:
        ave_mask_lenth = ave_mask_lenth + 1
    half_mask_lenth = int(ave_mask_lenth / 2)
    for i in range(len(labels)):
        mask_area = torch.ones([int(ave_mask_lenth), int(ave_mask_lenth)])
        left_pad = int((float(labels[i][1])) / 2) - half_mask_lenth
        right_pad = 320 - (int((float(labels[i][1])) / 2) + half_mask_lenth)
        top_pad = int((float(labels[i][2])) / 2) - half_mask_lenth
        bottom_pad = 320 - (int((float(labels[i][2])) / 2) + half_mask_lenth)
        pad = torch.nn.ZeroPad2d(padding=(left_pad, right_pad, top_pad, bottom_pad))
       
        mask_area = pad(mask_area)
        masked_labels_matrix[i] = mask_area

    return masked_labels_matrix


def Test_val(w_list, inferenced_labels_URL, Sparse_Generation_save_URL, point_labels_URL):
    # ave_anchors_lenth = 0
    round = 1
    epoch = 1

    ALL_URL_Labels_Inferenced = os.listdir(inferenced_labels_URL)

    ALL_URL_Labels_Inferenced.sort(key=lambda x: int(x[:-4]))  # Sort to read in order

    # ALL_URL_Rough_labels=os.listdir('./')    #Point label directory
    ALL_URL_Rough_labels = os.listdir(point_labels_URL)
    ALL_URL_Rough_labels.sort(key=lambda x: int(x[:-4]))
    # print(ALL_URL_Rough_labels)
    labels_save_path = Sparse_Generation_save_URL
    while epoch < 2:
        # print("epoch :", epoch)

        count_Forinference_URL = 0
        for count in tqdm(range(len(ALL_URL_Rough_labels)), desc='Evaluate with val set'):

            coco_Heatmap_sgpic = COCO_Heatmap_SgPic()
            # print()
            URL_Inference = inferenced_labels_URL + ALL_URL_Labels_Inferenced[count_Forinference_URL]
            # print(URL_Inference)
            URL_For_Genert_RoughLabels = point_labels_URL + ALL_URL_Rough_labels[count]
            # Rough_labels=Get_Rough_Labels(Open_txt(URL_Rough_Label))
            global predict_labels_num
            inference_labels = Get_Inference_Labels(Open_txt(URL_Inference), 576, 640)

            URL_Rough_Label_Generated = './Rough_labels_generated/1.txt' # Obtain the id_List in an image
            Write_Rough_Labels(Open_txt(URL_For_Genert_RoughLabels), 576, 640)
            Point_labels_SgPic = Get_Rough_Labels(URL_Rough_Label_Generated)
            class_id_list = Seek_Class_List_SinglePic(Point_labels_SgPic)
            if class_id_list[0] == 0:
                Set_NUll_PointObjct_Pic(Sparse_Generation_save_URL, ALL_URL_Rough_labels[count])
                continue
            # print(inference_labels[0][1])

            # print("processed_labels_num: ", predict_labels_num)
            if ALL_URL_Labels_Inferenced[count_Forinference_URL] == ALL_URL_Rough_labels[count]:  # Is it predicted for the corresponding image, Otherwise, the predicted result for this image is empty
                count_Forinference_URL += 1
                for i in range(len(class_id_list)):

                    sum_box_width = 0  # Obtain the average width and height of the predicted boxes for each class in an image
                    sum_box_height = 0
                    for q in range(len(inference_labels)):
                        if float(inference_labels[q][0]) == float(class_id_list[i].class_name):
                            sum_box_width += float(inference_labels[q][3])
                            sum_box_height += float(inference_labels[q][4])
                    ave_anchors_width = ((sum_box_width / len(inference_labels))) * w_list[2]
                    ave_anchors_height = ((sum_box_height / len(inference_labels))) * w_list[2]
                    ave_mask_width = (ave_anchors_width / 0.55) * w_list[3]
                    ave_mask_height = (ave_anchors_height / 0.55) * w_list[3]
                    ave_mask_lenth = (ave_mask_height + ave_mask_width) / 2
                    class_id_list[i].anchor_width = ave_anchors_width
                    class_id_list[i].anchor_height = ave_anchors_height
                    class_id_list[i].mask_width = ave_mask_width
                    class_id_list[i].mask_height = ave_mask_height
                coco_Heatmap_sgpic.id_List = class_id_list
                big_heatmap = Big_HeatMap41Pic()
                COCO_bigheatmap = OverlapClasses_BigHeatmap_SinglePic()
                m = 0

                while m < len(inference_labels):
                    # print(m)
                    heatmap1 = Heatmap()

                    heatmap1 = heatmap1.get_heatmap_matrix(inference_labels[m])  # Initialize a heatmap

                    matrix_x = heatmap1.matrix_w
                    matrix_y = heatmap1.matrix_h
                    # print('matrix_x: ',matrix_x,' matrix_y:',matrix_y)
                    i = 0
                    while i < matrix_y:
                        j = 0
                        while j < matrix_x:
                            heatmap1 = heatmap1.convert_function(i, j, w_list[1])
                            j += 1
                        i += 1
                    # print(heatmap1.matrix)
                    big_heatmap.heatmaps.append(heatmap1)
                    m += 1
                big_heatmap = big_heatmap.Padding_heatmaps()  # All mapping tensors have been obtained, and the padding for each predicted box tensor in a single image is 321 * 321
                # Point_labels_SgPic = Open_txt(URL_For_Genert_RoughLabels)
                # class_id_list=Seek_Class_List_SinglePic(Point_labels_SgPic)
                Sumed_Tensors_ID_List = []

                for h in range(len(class_id_list)):
                    temp_sum_matrix = big_heatmap.Sum_heatmaps(class_id_list[h].class_name)  # Sum up the predicted box tensors for each class in a single image
                    Sumed_Tensors_ID_List.append(temp_sum_matrix)
                coco_Heatmap_sgpic.heatmap_List = Sumed_Tensors_ID_List
        
                labels_List = []
                for i in range(len(class_id_list)):

                    # Rough_labels = Get_Rough_Labels(URL_Rough_Label_Generated)
                    masked_labels_matrixes = Masked_Matrix()
                    masked_labels_matrixes.masked_labels_matrix_list = masked_labels_matrixes.Get_Mask_Rough_Labels_Matrix_COCO(
                        Point_labels_SgPic, ave_mask_lenth,
                        class_id_list[
                            i].class_name)  # Obtain the mask tensor for each marker point in an image

                    masked_heatmaps_matrix = coco_Heatmap_sgpic.Get_MUL_maskANDheatmap(masked_labels_matrixes,
                                                                                       i)  # Multiplying each mask tensor in an image with the heatmap tensor of the entire image yields the tensor corresponding to each point within the labeled range
                    # print(masked_heatmaps_matrix)
                    labels_List.append(
                        big_heatmap.Get_Centroid(masked_heatmaps_matrix, Point_labels_SgPic, round,
                                                 class_id_list[i].anchor_width, class_id_list[i].anchor_height,
                                                 w_list[0], class_id_list[i].class_name))
                    labels_List[i] = labels_List[i].div(320)

                    labels_List[i] = labels_List[i].numpy()
                    for q in range(len(labels_List[i])):
                        labels_List[i][q][0] = int(labels_List[i][q][0])
                    # print(labels)

                save_path = labels_save_path + ALL_URL_Rough_labels[count]
                if not os.path.exists(labels_save_path):
                    os.mkdir(labels_save_path)
                with open(save_path, 'w') as f:
                    for i in range(len(labels_List)):
                        for q in range(len(labels_List[i])):
                            f.write(str(int(labels_List[i][q][0])) + ' ' + str(labels_List[i][q][1]) + ' ' + str(
                                labels_List[i][q][2]) + ' ' + str(labels_List[i][q][3]) + ' ' + str(
                                labels_List[i][q][4]) + '\n')
            else:
                Set_NUll_InfObjct_Pic(Point_labels_SgPic, Sparse_Generation_save_URL, ALL_URL_Rough_labels[count])

        time.sleep(0.1)

        epoch += 1

        print()

    return w_list

def TEST2():
    URL_Inference = './'
    URL_Gene_RoughLabels = './'
    URL_Rough_Label = './Rough_labels_generated/1.txt'
    Write_Rough_Labels(Open_txt(URL_Gene_RoughLabels), 640, 640)
    Rough_labels = Get_Rough_Labels(URL_Rough_Label)
    i = 0


def Test3():
    a = torch.zeros([3, 3])
    torch.count_zero(a)
    print(torch.is_nonzero(a))
    return


def Delet_last_number_in_labelsTXT():
    path = './/'
    filenames = os.listdir(path)
    filenames.sort(key=lambda x: int(x[:-4].lstrip('1 ').lstrip('(').rstrip(')')))
    folder_path = './/'

    for i in range(len(filenames)):
        save_path = folder_path + '1 (' + str(i + 1) + ')' + '.txt'
        file_path = path + filenames[i]
        label = Open_txt(file_path)

        if not os.path.exists(folder_path):
            os.mkdir(folder_path)
        with open(save_path, 'w') as f:
            for j in range(len(label)):
                new_label = label[j].split(' ')
                f.write(new_label[0] + ' ' + new_label[1] + ' ' + new_label[2] + ' ' + new_label[3] + ' ' + new_label[
                    4] + '\n')


def Get_Masked_label(pseudo_labels, rough_labels):
    # print(len(pseudo_labels))
    # print(len(rough_labels))
    expand_stride = float(15 / 640)
    # print(expand_stride)
    Max_conf_label = []
    for i in range(len(rough_labels)):
        masked_label = []
        for j in range(len(pseudo_labels)):

            if float(rough_labels[i][1]) <= float(pseudo_labels[j][1]) + expand_stride and float(
                    rough_labels[i][1]) >= float(pseudo_labels[j][1]) - expand_stride and float(
                rough_labels[i][2]) <= float(pseudo_labels[j][2]) + expand_stride and float(
                rough_labels[i][2]) >= float(pseudo_labels[j][2]) - expand_stride:
                masked_label.append(pseudo_labels[j])
        if (len(masked_label) > 0):
            max_conf_label = masked_label[0]
            # print((max_conf_label))
            for k in range(len(masked_label) - 1):


                if float(max_conf_label[5].rstrip('\n')) < float(masked_label[k + 1][5].rstrip('\n')):
                    global mid_max
                    mid_max = masked_label[k + 1]

                    # print(max_conf_label)
            Max_conf_label.append(mid_max)

    # print(Max_conf_label)
    return Max_conf_label


def Get_Max_Conf_labels():
    path = './/'
    filenames = os.listdir(path)
    filenames.sort(key=lambda x: int(x[:-4].lstrip('1 ').lstrip('(').rstrip(')')))
    folder_path = './/'
    Rough_labels_root_URL = './/'
    # ALL_URL_Rough_labels = os.listdir('./')  # Point label directory
    ALL_URL_Rough_labels = os.listdir('./')
    ALL_URL_Rough_labels.sort(key=lambda x: int(x[:-4].lstrip('1 ').lstrip('(').rstrip(')')))

    for i in range(len(filenames)):
        save_path = folder_path + '1 (' + str(i + 1) + ')' + '.txt'
        file_path = path + filenames[i]
        label = Open_txt(file_path)
        rough_label = Open_txt(Rough_labels_root_URL + ALL_URL_Rough_labels[i])
        labels_in_onePic = []
        rough_labels_in_onePic = []
        for m in range(len(label)):
            new_label = label[m].split(' ')
            labels_in_onePic.append(new_label)
        for m in range(len(rough_label)):
            new2_label = rough_label[m].split(' ')
            rough_labels_in_onePic.append(new2_label)
            
        max_labels = Get_Masked_label(labels_in_onePic, rough_labels_in_onePic)

        if not os.path.exists(folder_path):
            os.mkdir(folder_path)
        with open(save_path, 'w') as f:
            for j in range(len(max_labels)):
                new_label = max_labels[j]

                f.write(new_label[0] + ' ' + new_label[1] + ' ' + new_label[2] + ' ' + new_label[3] + ' ' + new_label[
                    4] + '\n')


def Begin_Update(epoch, w_list):
    for i in range(epoch):
        print("epoch: ", i + 1)
        w_list = Update_w(w_list)  # Update Parameters
    return w_list


def main(inferenced_labels_URL, Sparse_Generation_save_URL, val_labels_URL, inferenced_val_labels_URL,
         Sparse_generation_val_labels_save_URL, point_labels_URL, epochs, Final_save_URL):
    num = 0
    if num == 0:
        R_w1 = 0.02  # initial parameter
        staircase_w2 = 0.75
        avepseudobbox_lenth_w3 = 1.0
        masklenth_w4 = 1.0
        w_list = [R_w1, staircase_w2, avepseudobbox_lenth_w3, masklenth_w4]
    for i in range(epochs):
        w_list = Test(w_list, inferenced_labels_URL, Sparse_Generation_save_URL, point_labels_URL)
        # Test_val(w_list, inferenced_val_labels_URL, Sparse_generation_val_labels_save_URL, val_labels_URL)
        # w_list = Update_w(w_list, val_labels_URL, Sparse_generation_val_labels_save_URL)

    # Get_Final_PseudoBox_labels(Sparse_Generation_save_URL, point_labels_URL, Final_save_URL)
    print()
    print("finished")

    print("w1:", w_list[0], "  w2:", w_list[1], "  w3:", w_list[2], "  w4:", w_list[3])
    print()
    print("Sparse pseudo labels were saved in: ", Final_save_URL)


def parse_your_individual_data():
    parser = argparse.ArgumentParser()

    parser.add_argument('--inferenced_labels_URL', type=str,
                        default='.//',
                        help='inferenced labels path ')
    parser.add_argument('--Sparse_Generation_save_URL', type=str, default='.//',
                        help='Sparse Generation save path')
    parser.add_argument('--val_labels_URL', type=str, default='.//',
                        help='the small amount fully labeled annotations path')
    parser.add_argument('--inferenced_val_labels_URL', type=str,
                        default='.//',
                        help='labels path which inferenced from the small amount of data')
    parser.add_argument('--Sparse_generation_val_labels_save_URL', type=str,
                        default='.//', help='Sparse generation val labels save path')
    parser.add_argument('--point_labels_URL', type=str,
                        default='.//',
                        help='point labels URL')
    parser.add_argument('--epochs', type=int, default=1, help='epochs')
    parser.add_argument('--Final_save_URL', type=str, default='.//',
                        help='Final save path')


    individual = parser.parse_args()
    print_args(vars(individual))
    return individual


if __name__ == '__main__':
    individual = parse_your_individual_data()
    individual = individual
    main(**vars(individual))
