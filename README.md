# yolov5_object_count


## object_count 및 db 연동

제품이 컨베이어 벨트를 지날 때 일정 line을 지날때 제품의 상태를 카운트


### YOLO 실행
양쪽에 프린팅이 되어있는 플라스틱 제품의 불량품 검출
웹캠을 2개 사용해서 제품의 양 옆면을 확인

1번, 2번 웹캠이 동시에 판별했을 때 제품의 결과를 하나로 확인해야함. 
1번 웹캠 탐지 , 2번 웹캠 탐지 , 최종 탐지 결과 

객체의 중심점 좌표를 계산하여 중심점이 제품이 일정 Line을 지났을 때 카운트 
1번 웹캠 , 2번 웹캠 동시 진행

### DB 연동
제품의 품질 및 공정 상태 파악을 위해 DB 연동

MariaDB 연결
tns table 에 데이터 적재 


![image](https://user-images.githubusercontent.com/77741178/226542695-a9b4490a-e550-4baf-912a-b2f61c9d7fc5.png)
