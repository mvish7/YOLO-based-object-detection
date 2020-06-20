import cv2
import numpy as np


CAMERA_DEVICE_ID = 0
yolo_config_files = 'give suitable path here'
classes = yolo_config_files +'yolov3.txt'
config = yolo_config_files + '/yolov3.cfg'
weights = yolo_config_files + '/yolov3.weights'

# code to run this script from command line
'''
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True,
                help = 'path to input image')
ap.add_argument('-c', '--config', required=True,
                help = 'path to yolo config file')
ap.add_argument('-w', '--weights', required=True,
                help = 'path to yolo pre-trained weights')
ap.add_argument('-cl', '--classes', required=True,
                help = 'path to text file containing class names')
args = ap.parse_args()
'''


def get_output_layers(net):
    
    layer_names = net.getLayerNames()
    
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers


def draw_prediction(img, class_id, confidence, COLORS, x, y, x_plus_w, y_plus_h):

    label = str(classes[class_id])

    color = COLORS[class_id]

    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)

    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def object_detection():
    # to run with webcam uncomment following 2 line and uncomment the while loop as well
    # video_cap = cv2.VideoCapture(CAMERA_DEVICE_ID)
    # image = video_cap.read()

    image = cv2.imread('C:\\Users\\mvish\\PycharmProjects\\object_detection\\object-detection-opencv-master\\images\\test_img2.jpeg')
    cv2.imshow('original image', image)

    # while imgae.shape[1]>100:
    if image.shape[1] > 100:
        # ok, frame = video_cap.read()
        '''
        if not ok:
            logging.error("Could not read frame from camera. Stopping video capture.")
            break
        '''
        Width = image.shape[1]
        Height = image.shape[0]
        scale = 0.00392

        temp_classes = None

        with open(classes, 'r') as f:
            temp_classes = [line.strip() for line in f.readlines()]

        COLORS = np.random.uniform(0, 255, size=(len(temp_classes), 3))

        net = cv2.dnn.readNet(weights, config)

        blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)

        net.setInput(blob)

        outs = net.forward(get_output_layers(net))

        class_ids = []
        confidences = []
        boxes = []
        conf_threshold = 0.5
        nms_threshold = 0.4

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * Width)
                    center_y = int(detection[1] * Height)
                    w = int(detection[2] * Width)
                    h = int(detection[3] * Height)
                    x = center_x - w / 2
                    y = center_y - h / 2
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])

        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

        for i in indices:
            i = i[0]
            box = boxes[i]
            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]
            draw_prediction(image, class_ids[i], confidences[i], COLORS, round(x), round(y), round(x+w), round(y+h))

        cv2.imshow("objects detected", image)
        cv2.waitKey()
    
        cv2.imwrite("object-detection.jpg", image)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    # Release handle to the webcam
    # video_cap.release()
    # cv2.destroyAllWindows()


object_detection()
