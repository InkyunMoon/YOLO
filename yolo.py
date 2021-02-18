import numpy as np
import argparse
import time
import cv2
import os

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
    help="path to input image")
ap.add_argument("-y", "--yolo", required=True,
    help="base path to YOLO directory")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
    help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
    help="threshold when applying non-maxima suppression")
args = vars(ap.parse_args())

# 우리의 YOLO 모델이 훈련한 COCO 데이터셋의 클래스 레이블을 가져온다.
labelsPath = os.path.sep.join([args['yolo'], 'coco.names'])
LABELS = open(labelsPath).read().strip().split("\n")

# 각각의 클래스 레이블을 나타내기 위한 색상을 초기화한다.
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
    dtype = 'uint8')

# 모델 cfg와 가중치가 담긴 디렉토리
weightsPath = os.path.sep.join([args['yolo'], 'yolov3.weights'])
configPath = os.path.sep.join([args['yolo'], 'yolov3.cfg'])

# COCO 데이터셋으로 훈련된 YOLO 모델을 읽어들인다.
print('[INFO] loading YOLO from disk...')
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# 인풋 이미지를 로드하고 차원 객체를 얻는다.
image = cv2.imread(args['image'])
(H, W) = image.shape[:2]

# YOLO에서 필요한 마지막 아웃풋 레이어만 가져온다.
ln = net.getLayerNames()
ln = [ln[i[0] -1] for i in net.getUnconnectedOutLayers()]

# 인풋 이미지로부터 Blob을 구성하여 YOLO로 전달한다. YOLO는 바운딩박스와 객체의 확률을 리턴한다.
blob = cv2.dnn.blobFromImage(image, 1/255.0,(416,416),
    swapRB=True, crop=False)
net.setInput(blob)
start = time.time()
layerOutputs = net.forward(ln)
end = time.time()

# YOLO의 시간정보를 보여준다.
print("[INFO] YOLO took {:.6f} seconds".format(end-start))

# 탐지된 바운딩 박스와 신뢰도와 클래스ID를 담을 리스트를 초기화한다.
boxes = []
confidences = []
classIDs = []

for output in layerOutputs:
    for detection in output:

        scores = detection[5:]
        classID = np.argmax(scores)
        confidence = scores[classID]

        if confidence > args["confidence"]:
        
            # YOLO는 바운딩박스의 중앙(x, y)값과 width, height를 리턴한다.
            box = detection[0:4] * np.array([W, H, W, H])
            (centerX, centerY, width, height) = box.astype("int")

            # centerX, Y정보를 이용해서 좌측 상단의 좌표를 얻는다.
            x = int(centerX - (width / 2))
            y = int(centerY - (height / 2))
            # 바운딩박스 좌표와 신뢰도, Class ID를 업데이트한다.
            boxes.append([x, y, int(width), int(height)])
            confidences.append(float(confidence))
            classIDs.append(classID)
# YOLO는 NMS를 적용하지 않으므로 직접적으로 적용해주도록 한다.
idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"], args["threshold"])
# 크게 겹치는 바운딩 박스 중 가장 신뢰도가 높은 것을 제외하고 삭제한다.

# 최소 하나의 박스가 있는 경우에만 진행
if len(idxs) > 0:
    for i in idxs.flatten():
        # 바운딩박스의 좌표를 추출
        (x, y) = (boxes[i][0], boxes[i][1])
        (w, h) = (boxes[i][2], boxes[i][3])

        # 사진위에 바운딩박스와 신뢰도를 추가한다.
        color = [int(c) for c in COLORS[classIDs[i]]]
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
        cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
		0.5, color, 2)

cv2.imshow("image", image)
cv2.waitKey(0)

