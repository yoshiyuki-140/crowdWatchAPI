
import cv2 
import numpy as np 

class FilePaths:
    class yoloPath:
        coco = "C:/Users/moyas/Documents/programsForMe/apps/clowdWatch/data/yoloFiles/coco.names"
        cfg = "C:/Users/moyas/Documents/programsForMe/apps/clowdWatch/data/yoloFiles/yolov3.cfg"
        weights = "C:/Users/moyas/Documents/programsForMe/apps/clowdWatch/data/yoloFiles/yolov3.weights"

    class destPaths:
        destPath = "C:/Users/moyas/Documents/programsForMe/apps/clowdWatch/data/dest/image.jpg"
        checkedImgPath = "C:/Users/moyas/Documents/programsForMe/apps/clowdWatch/data/dest/marked.jpg"

class Detector:
    def __init__(self):
        self.WHITE = [255,255,255]
        self.BLACK = [0,0,0]

    def loadImage(self,filepath:str):
        self.image = cv2.imread(filepath)

    def mask(self,leftTop:list,rightBottom:list):
        self.image[leftTop[1]:rightBottom[1],leftTop[0]:rightBottom[0]] = np.array(self.BLACK)

    def load_classes(self):
        self.classes = None
        with open(FilePaths.yoloPath.coco, 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]

    def loadConfigFile(self):
        self.net = cv2.dnn.readNet(FilePaths.yoloPath.weights, FilePaths.yoloPath.cfg)
        self.net.setInput(cv2.dnn.blobFromImage(self.image, 0.00392, (416,416), (0,0,0), True, crop=False))
        layer_names = self.net.getLayerNames()
        output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
        self.outs = self.net.forward(output_layers)

    def markToTheImage(self):
        self.class_ids = []
        self.confidences = []
        self.boxes = []
        Width = self.image.shape[1]
        Height = self.image.shape[0]
        for out in self.outs:
            for detection in out:
                scores = detection[5:]
                self.class_id = np.argmax(scores)
                confidence = scores[self.class_id]
                if confidence > 0.1:
                    center_x = int(detection[0] * Width)
                    center_y = int(detection[1] * Height)
                    w = int(detection[2] * Width)
                    h = int(detection[3] * Height)
                    x = center_x - w / 2
                    y = center_y - h / 2
                    self.class_ids.append(self.class_id)
                    self.confidences.append(float(confidence))
                    self.boxes.append([x, y, w, h])

    
    def checkThePeople(self):
        self.indices = cv2.dnn.NMSBoxes(self.boxes, self.confidences, 0.1, 0.1)

        for i in self.indices:
            box = self.boxes[i]
            if self.class_ids[i]==0:
                label = str(self.classes[self.class_id]) 
                cv2.rectangle(self.image, (round(box[0]),round(box[1])), (round(box[0]+box[2]),round(box[1]+box[3])), (0, 0, 255), 5)
                cv2.putText(self.image, label, (round(box[0])-10,round(box[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 4)

        self.imwrite(FilePaths.destPaths.checkedImgPath)

    def imwrite(self,filePath:str):
        cv2.imwrite(filePath,self.image)

    def howManyPeople(self):
        # ここちょっと不安
        cnt = 0
        for i in self.indices:
            if self.class_ids[i] == 0:
                cnt += 1
        return cnt

    def init(self):
        self.load_classes()
        self.loadConfigFile()
        self.markToTheImage()
        self.checkThePeople()

class PeopleNumberChecker:
    @staticmethod
    def get_current_people():
        detector = Detector()
        detector.loadImage(FilePaths.destPaths.destPath)
        detector.init()
        count = detector.howManyPeople()
        if count > 15:
            count = 15

        return count

if __name__ == "__main__":
    # デモ用のコードこれ実行したらカレントディレクトリにsample.pngが生成される
    detector = Detector()
    busy_image_path = "C:/Users/moyas/Documents/programsForMe/apps/clowdWatch/demo/busy_images/DSC_0403.JPG"
    detector.loadImage(busy_image_path)
    # マスクデータをつけるかどうか... 
    # detector.mask([0,0],[2248,1948])
    # detector.mask([1821,1948],[2248,2116])
    detector.init()
    detector.imwrite("sample.png")
    print(detector.indices)
    print(detector.howManyPeople())