#####################################
# Project: Mobile Neural Networks   #
# Author: Alexander Goldenberg      #
# Email: ajgol11@student.monash.edu #
# Last Edited: 25th September 2020  #
#####################################

# Import Libraries
import time
import argparse
import cv2
import numpy as np
import face_recognition
import pycuda.autoinit  # This is needed for initializing CUDA driver

from utils.ssd_classes import get_cls_dict
from utils.ssd import TrtSSD
from utils.camera import add_camera_args, Camera
from utils.display import open_window, set_display, show_fps
from utils.visualization import BBoxVisualization
from utils.mtcnn import TrtMtcnn

# Define Variables
WINDOW_NAME = 'SSD | MTCNN | FaceRecognition'
BBOX_COLOR = (0, 255, 0)  # green
INPUT_HW = (300, 300)
SUPPORTED_MODELS = [
    'ssd_mobilenet_v1_coco',
    'ssd_mobilenet_v1_egohands',
    'ssd_mobilenet_v2_coco',
    'ssd_mobilenet_v2_egohands',
    'ssd_inception_v2_coco',
    'ssdlite_mobilenet_v2_coco',
]

# Preload known faces
amy_image = face_recognition.load_image_file("/home/nvidia/Pictures/b99amy.jpg")
amy_face_encoding = face_recognition.face_encodings(amy_image)[0]
charles_image = face_recognition.load_image_file("/home/nvidia/Pictures/b99charles.jpg")
charles_face_encoding = face_recognition.face_encodings(charles_image)[0]
holt_image = face_recognition.load_image_file("/home/nvidia/Pictures/b99holt.jpg")
holt_face_encoding = face_recognition.face_encodings(holt_image)[0]
jake_image = face_recognition.load_image_file("/home/nvidia/Pictures/b99jake.jpg")
jake_face_encoding = face_recognition.face_encodings(jake_image)[0]
rosa_image = face_recognition.load_image_file("/home/nvidia/Pictures/b99rosa.jpg")
rosa_face_encoding = face_recognition.face_encodings(rosa_image)[0]
scully_image = face_recognition.load_image_file("/home/nvidia/Pictures/b99scully.jpg")
scully_face_encoding = face_recognition.face_encodings(scully_image)[0]
terry_image = face_recognition.load_image_file("/home/nvidia/Pictures/b99terry.jpg")
terry_face_encoding = face_recognition.face_encodings(terry_image)[0]
hitchcock_image = face_recognition.load_image_file("/home/nvidia/Pictures/b99hitchock.jpg")
hitchcock_face_encoding = face_recognition.face_encodings(hitchcock_image)[0]

known_face_encodings = [
    amy_face_encoding,
    charles_face_encoding,
    holt_face_encoding,
    jake_face_encoding,
    rosa_face_encoding,
    scully_face_encoding,
    terry_face_encoding,
    hitchcock_face_encoding
]

known_face_names = [
    "Amy",
    "Charles",
    "Holt",
    "Jake",
    "Rosa",
    "Scully",
    "Terry",
    "Hitchcock"
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
        #cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 255), 2)
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

# Main Loop
def loop_and_detect(img, mtcnn, minsize, trt_ssd, conf_th, vis):
    # Continuously capture images from camera and do face detection
    fps = 0.0
    
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
            if conf < 0.3 or cls!=1 :
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
        tic = time.time()
        # Perform face detection
        dets, landmarks = mtcnn.detect(img, minsize=minsize)
        toc = time.time()
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
        cv2.imwrite('stage1.jpg',img)
        img = show_faces(img, dets, landmarks)
        cv2.imwrite('stage2.jpg',img)
        img = show_labels(img,face_locations,face_names)
        cv2.imwrite('stage3.jpg',img)
        
        # Calculate FPS
        curr_fps = 1.0 / (toc - tic)
        # calculate an exponentially decaying average of fps number
        fps = curr_fps
        print(fps)
        return (img,boxes,face_locations,face_names)

# Initial Function
def main():
    # Parse arguments and get input
    args = parse_args()

    # Create NN1 and NN2 models
    cls_dict = get_cls_dict(args.model.split('_')[-1])
    trt_ssd = TrtSSD(args.model, INPUT_HW)
    mtcnn = TrtMtcnn()
    
    # Create Preview Window
    vis = BBoxVisualization(cls_dict)
    
    imageNum = 10

    # Enter Detection Mode
    while True:
        # Get Image
        imageName = "/home/nvidia/Pictures/test13.jpg"
        #imageName = "/media/E76F-73E0/Faces/1 (" + str(imageNum) + ").jpg"
        #imageName = "/home/nvidia/Pictures/Benchmarks/Pedestrians/PennPed000" + str(imageNum) + ".png"
        imageNum = imageNum + 1
        #print(imageName)
        img = cv2.imread(imageName)

        cv2.imshow(WINDOW_NAME,img)

        # Run Neural Networks
        img, nn1_results, nn2_results, nn3_results = loop_and_detect(img, mtcnn, args.minsize, trt_ssd, conf_th=0.3, vis=vis)

        # Display Results
        cv2.imshow(WINDOW_NAME, img)
        #cv2.waitKey(0)
        
        # User/Keyboard Input
        key = cv2.waitKey(1)
        if key == 27:  # ESC key: quit program
            break
    
    # Clean up and exit
    cv2.destroyAllWindows()
    serial_port.close()

# Call initial function
if __name__ == '__main__':
    main()
