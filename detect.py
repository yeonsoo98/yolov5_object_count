import torch
from torch import nn
import yaml
import cv2
import numpy as np
from vidgear.gears import CamGear
from matplotlib import pyplot as plt
from IPython.display import Image, clear_output
import argparse
import os
import datetime
import sys
from PIL import ImageFont, ImageDraw, Image
import time
from pathlib import Path
from utils.plots import *
from utils.torch_utils import *
from utils.general import *
from utils.datasets import letterbox
import gdown
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort
import pymysql

# path 정리
path = "C:/Users/YongJun/Desktop/YOLO/1228_TNS/images"
model = torch.load('C:/Users/YongJun/Desktop/YOLO/YOLOv5s_1229.pt')
image_paths = sorted([os.path.join(path, f) for f in os.listdir(path) if f.endswith(".jpg") or f.endswith(".png")])
label_paths = sorted([os.path.join(path, f) for f in os.listdir(path) if f.endswith(".txt")])

# DB 연결
conn = pymysql.connect(
    host='localhost',
    user='root',
    password='013579',
    db='tns_db',
    charset='utf8'
)

# DB 저장 데이터 - 시작 시간 기록
start_time = time.time()

# 라벨링한 클래스
class_dict = {
    "OK": 0,
    "NG_Blur": 1,
    "NG_Scratch": 2,
}

ok_idx = class_dict['OK']
ng_blur_idx = class_dict['NG_Blur']
ng_scratch_idx = class_dict['NG_Scratch']

labels = []

# 웹캠 설정
cap1 = cv2.VideoCapture(0)  
cap2 = cv2.VideoCapture(1) 

# 웹캠 1번 탐지
ok_count1 = 0
ng_blur_count1 = 0
ng_scratch_count1 = 0

# 웹캠 2번 탐지
ok_count2 = 0
ng_blur_count2 = 0
ng_scratch_count2 = 0

# 최종 탐지 
ok_count = 0
ng_count = 0

# 추가 부분 
ok_detected1 = False
ok_detected2 = False
ng_detected1 = False
ng_detected2 = False

# YOLO 실행
while True:
    # 현재 시간 기록
    current_time = time.time()

    # 이전 프레임과의 시간 차이 계산
    elapsed_time = current_time - start_time

    # 이전 프레임의 시간 업데이트
    start_time = current_time    
    
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    
    labels1 = []
    labels2 = []

    results1 = model(frame1)
    results2 = model(frame2)

    detections1 = results1.xyxy[0]
    detections2 = results2.xyxy[0]
    
    
    # 각 객체에 대해 Loop를 돌며, Line을 지나갔는지 검사
    for detection in detections1:
        
    # 객체의 중심점 좌표 계산
        center_x = (detection[0] + detection[2]) / 2
        center_y = (detection[1] + detection[3]) / 2
        
        # 객체가 Line을 지나갔는지 검사 - Line 설정 (317 ~ 323)
        if center_x > 317 and center_x < 323:
            label = detection[5]
            labels1.append(label)
            
            if label == ok_idx:
                ok_count1 += 1
            elif label == ng_blur_idx:
                ng_blur_count1 += 1
            elif label == ng_scratch_idx:
                ng_scratch_count1 += 1
    
    # 각 객체에 대해 Loop를 돌며, Line을 지나갔는지 검사
    for detection in detections2:

    # 객체의 중심점 좌표 계산
        center_x = (detection[0] + detection[2]) / 2
        center_y = (detection[1] + detection[3]) / 2
    
        # 객체가 Line을 지나갔는지 검사 - Line 설정 (317 ~ 323)
        if center_x > 317 and center_x < 323:
            label = detection[5]
            labels2.append(label)
            
            if label == ok_idx:
                ok_count2 += 1
            elif label == ng_blur_idx:
                ng_blur_count2 += 1
            elif label == ng_scratch_idx:
                ng_scratch_count2 += 1
   
    # 추가 부분 
    if ok_idx in labels1 and ok_idx in labels2:
        ok_detected1 = True
        ok_detected2 = True

    if ok_detected1 and ok_detected2:
        ok_count += 1
        ok_detected1 = False
        ok_detected2 = False

    if ng_blur_idx in labels1 or ng_scratch_idx in labels1:
        ng_detected1 = True

    if ng_blur_idx in labels2 or ng_scratch_idx in labels2:
        ng_detected2 = True

    if ng_detected1 or ng_detected2:
        ng_count += 1
        ng_detected1 = False
        ng_detected2 = False
       
   
    # DB 연동
    cursor = conn.cursor()
    count = 0
    
    for detection in detections1:
        count += 1
        name = f"name{count}"
        
        # SQL 문을 통해서 MariaDB의 Table에 저장 - 제품 , 개수 , 상태 , confidence 등 추가 
        
        # (INSERT INTO tns (table name) id (table column ... ) VALUES (%s , %s , ...), (python_msg, python_msg))
        
        cursor.execute("INSERT INTO tns (id) VALUES (%s)", 
                       (ok_count1))
    conn.commit()
    
    
    # 동영상에서 나오는 cv2 의 text 정리 
    cv2.putText(frame1, f'OK: {ok_count}', 
        (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame1, f'NG_Blur: {ng_blur_count}', 
        (30, 55), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame1, f'NG_Scratch: {ng_scratch_count}', 
        (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    cv2.putText(frame2, f'OK: {ok_count2}', 
        (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame2, f'NG_Blur: {ng_blur_count2}', 
        (30, 55), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame2, f'NG_Scratch: {ng_scratch_count2}', 
        (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 0, 255), 2)
    
    cv2.line(frame1, (320, 0), (320, 640), (255, 0, 0), 2)
    cv2.line(frame2, (320, 0), (320, 640), (255, 0, 0), 2)
                
    cv2.imshow('TNS_CAP1', np.squeeze(results1.render()))
    cv2.imshow('TNS_CAP2', np.squeeze(results2.render()))

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) == ord("q"):
        break

# 종료시 리소스 해제
cap1.release()
cap2.release()
cv2.destroyAllWindows()
