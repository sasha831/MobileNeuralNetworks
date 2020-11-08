#####################################
# Project: Mobile Neural Networks   #
# Author: Alexander Goldenberg      #
# Email: ajgol11@student.monash.edu #
# Last Edited: 2nd November 2020    #
#####################################

# Import Libraries
import time
import serial
import argparse
import cv2
import numpy as np
import face_net
import pycuda.autoinit  # This is needed for initializing CUDA driver
from utils.ssd_classes import get_cls_dict
from utils.ssd import TrtSSD
from utils.camera import add_camera_args, Camera
from utils.display import open_window, set_display, show_fps
from utils.visualization import BBoxVisualization
from utils.mtcnn import TrtMtcnn

# This is to record the output of the NN for ;ater viewing
outNN = cv2.VideoWriter('outputNN.avi',cv2.VideoWriter_fourcc(*'DIVX'),30.0,(1280,720))

# Define Variables
WINDOW_NAME = 'SSD | MTCNN | FaceRecognition'
BBOX_COLOR = (0, 255, 0)  # green
INPUT_HW = (300, 300)
# Available Models. Recomended is ssd_mobilenet_v2_coco
SUPPORTED_MODELS = [
    'ssd_mobilenet_v1_coco',
    'ssd_mobilenet_v2_coco',
    'ssd_inception_v2_coco',
    'ssdlite_mobilenet_v2_coco',
]

# Open Serial communications to Arduino Mini
serial_port = serial.Serial(
    port="/dev/ttyTHS1",
    baudrate=115200,
    bytesize=serial.EIGHTBITS,
    parity=serial.PARITY_NONE,
    stopbits=serial.STOPBITS_ONE,
)
# Wait a second to let the port initialize
time.sleep(1)

# Preload known faces
sasha_image = face_recognition.load_image_file("/home/nvidia/Pictures/sasha.jpg")
sasha_face_encoding = face_recognition.face_encodings(sasha_image)[0]
#name_image = face_recognition.load_image_file("/home/nvidia/Pictures/name.jpg")
#name_face_encoding = face_recognition.face_encodings(name_image)[0]


known_face_encodings = [
    sasha_face_encoding,
    #"Name"
]

known_face_names = [
    "Sasha",
    #"Name"
]

# Argument Parser
# Input: image, video, usb, onboard
# Model: ssd_mobilenet_v1_coco, ssd_mobilenet_v2_coco,
def parse_args():
    """Parse input arguments."""
    desc = ('Capture and display live camera video, while doing '
            'real-time face detection with TrtMtcnn on Jetson '
            'Nano')
    parser = argparse.ArgumentParser(description=desc)
    parser = add_camera_args(parser)
    parser.add_argument('--minsize', type=int, default=40,
                        help='minsize (in pixels) for detection [40]')
    parser.add_argument('-m', '--model', type=str,
                        default='ssd_mobilenet_v1_coco',
                        choices=SUPPORTED_MODELS)
    args = parser.parse_args()
    return args

# Draw bounding boxes for faces
def show_faces(img, boxes, landmarks):
    for bb, ll in zip(boxes, landmarks):
        x1, y1, x2, y2 = int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])
        cv2.rectangle(img, (x1, y1), (x2, y2), BBOX_COLOR, 2)
        for j in range(5):
            cv2.circle(img, (int(ll[j]), int(ll[j+5])), 2, BBOX_COLOR, 2)
    return img

# Draw labels for faces
def show_labels(img, face_locations,face_names):
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(img, name, (left, bottom + 20), font, 1.0, (255, 255, 255), 1)
    return img
    
# Compare face encoding with known face encodings    
def match_faces(face_encodings):
    face_names = []
    for face_encoding in face_encodings:
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        # Or instead, use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
            
        face_names.append(name)
    return face_names

# Determine where to drive the robot and communicate this to the Arduino
def robot_drive(img, boxes):
    # Adjust robot direction based on person location
    midpoint = 0
    for box in boxes:
        midpoint = midpoint + box[4]
        
    midpoint = int(midpoint / len(boxes))
    height, width, channels = img.shape    
    centre = width/2
    
    # Draw line on screen
    cv2.line(img, (midpoint,0),(midpoint,height),(0,0,255), thickness=2)
    
    # Communicate Direction to Arduino
    # For simple communication leave as is
    # For more precise communication uncomment the two lines below
    if (midpoint > centre*1.2):
        output = "R"# + "{:.2f}".format(midpoint/centre - 1) + ";"
        serial_port.write(output.encode())
        print(output);
    elif (midpoint < centre*0.80):
        output = "L"# + "{:.2f}".format(1 - midpoint/centre) + ";"
        serial_port.write(output.encode())
        print(output);
    else:
        output = "F"
        serial_port.write(output.encode())
        print(output);

    return img

# Main Loop
def loop_and_detect(img, mtcnn, minsize, trt_ssd, conf_th, vis):
    # Continuously capture images from camera and do face detection
    fps = 0.0
    tic = time.time()
    # Get input image
    if img is not None:
        # Perform Human Detection
        boxes, confs, clss = trt_ssd.detect(img, conf_th)
        
        # Copy lists for parsing
        boxes2 = boxes.copy()
        confs2 = confs.copy()
        clss2 = clss.copy()
        midpointX = 0
        
        # Remove non-human and low confidence results
        for box, conf, cls in zip(boxes2,confs2,clss2):
            if conf < 0.6 or cls!=1 :
                boxes.remove(box)
                confs.remove(conf)
                clss.remove(cls)
            else:
                x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                midpointX = x1 + (x2-x1)/2
                size = (x2-x1)*(y2-y1)
                index = boxes.index(box)
                boxes[index] = list(boxes[index])
                boxes[index].append(int(midpointX))
                boxes[index].append(size)

        # Perform face detection
        dets, landmarks = mtcnn.detect(img, minsize=minsize)

        # Parse face locations into correct format
        i = 0
        face_locations = []
        for bb in dets:
            fx1, fy1, fx2, fy2 = int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])
            face_locations.insert(i, [fy1,fx2,fy2,fx1])
            i = i+1
        rgb_img = img[:, :, ::-1]
        
        # Pass faces into recognition neural net
        face_encodings = face_recognition.face_encodings(rgb_img, face_locations)
        face_names = match_faces(face_encodings)

        # Draw Faces and Landmarks
        img = vis.draw_bboxes(img, boxes, confs, clss)
        img = show_faces(img, dets, landmarks)
        img = show_labels(img,face_locations,face_names)
        
        # Calculate FPS
        toc = time.time()
        curr_fps = 1.0 / (toc - tic)
        tic=toc
        fps = curr_fps if fps == 0.0 else (fps*0.95 + curr_fps*0.05)
        print(fps)
        return (img,boxes,face_locations,face_names)

# Initial Function
def main():
    # Parse arguments and get input
    args = parse_args()
    cam = Camera(args)
    if not cam.isOpened():
        raise SystemExit('ERROR: failed to open camera!')

    # Create NN1 and NN2 models and load into memory
    cls_dict = get_cls_dict(args.model.split('_')[-1])
    trt_ssd = TrtSSD(args.model, INPUT_HW)
    mtcnn = TrtMtcnn()
    
    # Create Preview Window
    open_window(WINDOW_NAME, 'Camera Preview', cam.img_width, cam.img_height)
    vis = BBoxVisualization(cls_dict)

    # Enter Detection Mode
    while True:
        # Get Image
        img = cam.read()
        out.write(img)
        nn1_results = []
        # Run Neural Networks
        img, nn1_results, nn2_results, nn3_results = loop_and_detect(img, mtcnn, args.minsize, trt_ssd, conf_th=0.3, vis=vis)
        
        # Communicate to Arduino
        if(nn1_results != []):
            img = robot_drive(img, nn1_results)
        else:
            serial_port.write("N".encode())
            print("N")
        
        # Display and save output
        cv2.imshow(WINDOW_NAME, img)
        outNN.write(img)

        # User/Keyboard Input
        key = cv2.waitKey(1)
        if key == ord('q'):
            out.release()
            outNN.release()
            break
    
    # Clean up and exit
    cam.release()
    cv2.destroyAllWindows()
    serial_port.close()

# Call initial function
if __name__ == '__main__':
    main()
