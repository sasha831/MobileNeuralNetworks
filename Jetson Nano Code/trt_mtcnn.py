"""trt_mtcnn.py

This script demonstrates how to do real-time face detection with
Cython wrapped TensorRT optimized MTCNN engine.
"""
print("Begin")

import time
import argparse
import cv2
import numpy as np
import face_recognition

from utils.camera import add_camera_args, Camera
from utils.display import open_window, set_display, show_fps
from utils.mtcnn import TrtMtcnn

print("Imports Finished")

WINDOW_NAME = 'TrtMtcnnDemo'
BBOX_COLOR = (0, 255, 0)  # green

blackPanther_image = face_recognition.load_image_file("/home/nvidia/Pictures/Avengers/blackpanther.jpeg")
blackPanther_face_encoding = face_recognition.face_encodings(blackPanther_image)[0]

blackWidow_image = face_recognition.load_image_file("/home/nvidia/Pictures/Avengers/blackwidow.jpg")
blackWidow_face_encoding = face_recognition.face_encodings(blackWidow_image)[0]

bruceBanner_image = face_recognition.load_image_file("/home/nvidia/Pictures/Avengers/brucebanner.jpg")
bruceBanner_face_encoding = face_recognition.face_encodings(bruceBanner_image)[0]

bucky_image = face_recognition.load_image_file("/home/nvidia/Pictures/Avengers/bucky.jpg")
bucky_face_encoding = face_recognition.face_encodings(bucky_image)[0]

captainAmerica_image = face_recognition.load_image_file("/home/nvidia/Pictures/Avengers/captainamerica.jpg")
captainAmerica_face_encoding = face_recognition.face_encodings(captainAmerica_image)[0]

captainAmerica2_image = face_recognition.load_image_file("/home/nvidia/Pictures/Avengers/captainamerica2.jpeg")
captainAmerica2_face_encoding = face_recognition.face_encodings(captainAmerica2_image)[0]

drStrange_image = face_recognition.load_image_file("/home/nvidia/Pictures/Avengers/drstrange.png")
drStrange_face_encoding = face_recognition.face_encodings(drStrange_image)[0]

drStrange2_image = face_recognition.load_image_file("/home/nvidia/Pictures/Avengers/drstrange2.png")
drStrange2_face_encoding = face_recognition.face_encodings(drStrange2_image)[0]

#falcon_image = face_recognition.load_image_file("/home/nvidia/Pictures/Avengers/falcon.jpg")
#falcon_face_encoding = face_recognition.face_encodings(falcon_image)[0]

hulk_image = face_recognition.load_image_file("/home/nvidia/Pictures/Avengers/hulk.jpg")
hulk_face_encoding = face_recognition.face_encodings(hulk_image)[0]

loki_image = face_recognition.load_image_file("/home/nvidia/Pictures/Avengers/loki.png")
loki_face_encoding = face_recognition.face_encodings(loki_image)[0]

peterParker_image = face_recognition.load_image_file("/home/nvidia/Pictures/Avengers/peterparker.jpg")
peterParker_face_encoding = face_recognition.face_encodings(peterParker_image)[0]

scarletWitch_image = face_recognition.load_image_file("/home/nvidia/Pictures/Avengers/scarletwitch.jpg")
scarletWitch_face_encoding = face_recognition.face_encodings(scarletWitch_image)[0]

#spiderman_image = face_recognition.load_image_file("/home/nvidia/Pictures/Avengers/spiderman.jpg")
#spiderman_face_encoding = face_recognition.face_encodings(spiderman_image)[0]

thanos_image = face_recognition.load_image_file("/home/nvidia/Pictures/Avengers/thanos.jpg")
thanos_face_encoding = face_recognition.face_encodings(thanos_image)[0]

thor_image = face_recognition.load_image_file("/home/nvidia/Pictures/Avengers/thor.jpeg")
thor_face_encoding = face_recognition.face_encodings(thor_image)[0]

#thor2_image = face_recognition.load_image_file("/home/nvidia/Pictures/Avengers/thor2.jpg")
#thor2_face_encoding = face_recognition.face_encodings(thor2_image)[0]

tonyStark_image = face_recognition.load_image_file("/home/nvidia/Pictures/Avengers/tonystark.jpg")
tonyStark_face_encoding = face_recognition.face_encodings(tonyStark_image)[0]

#vision_image = face_recognition.load_image_file("/home/nvidia/Pictures/Avengers/vision.jpeg")
#vision_face_encoding = face_recognition.face_encodings(vision_image)[0]

vision2_image = face_recognition.load_image_file("/home/nvidia/Pictures/Avengers/vision2.jpeg")
vision2_face_encoding = face_recognition.face_encodings(vision2_image)[0]

wong_image = face_recognition.load_image_file("/home/nvidia/Pictures/Avengers/wong.jpg")
wong_face_encoding = face_recognition.face_encodings(wong_image)[0]

print("Faces Loaded")

known_face_encodings = [
    blackPanther_face_encoding,
    blackWidow_face_encoding,
    bruceBanner_face_encoding,
    bucky_face_encoding,
    captainAmerica_face_encoding,
    captainAmerica2_face_encoding,
    drStrange_face_encoding,
    drStrange2_face_encoding,
    #falcon_face_encoding,
    hulk_face_encoding,
    loki_face_encoding,
    peterParker_face_encoding,
    scarletWitch_face_encoding,
    #spiderman_face_encoding,
    thanos_face_encoding,
    thor_face_encoding,
    #thor2_face_encoding,
    tonyStark_face_encoding,
    #vision_face_encoding,
    vision2_face_encoding,
    wong_face_encoding
]

known_face_names = [
    "Black Panther",
    "Black Widow",
    "Bruce Banner",
    "Bucky Barnes",
    "Captain America",
    "Captain America",
    "Dr Strange",
    "Dr Strange",
    #"Falcon",
    "Hulk",
    "Loki",
    "Peter Parker",
    "Scarlet Witch",
    #"Spiderman",
    "Thanos",
    "Thor",
    #"Thor",
    "Tony Stark",
    #"Vision",
    "Vision",
    "Wong"
]

out = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc(*'DIVX'),30.0,(1920,1080))

def parse_args():
    """Parse input arguments."""
    desc = ('Capture and display live camera video, while doing '
            'real-time face detection with TrtMtcnn on Jetson '
            'Nano')
    parser = argparse.ArgumentParser(description=desc)
    parser = add_camera_args(parser)
    parser.add_argument('--minsize', type=int, default=40,
                        help='minsize (in pixels) for detection [40]')
    args = parser.parse_args()
    return args


def show_faces(img, boxes, landmarks):
    """Draw bounding boxes and face landmarks on image."""
    for bb, ll in zip(boxes, landmarks):
        x1, y1, x2, y2 = int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])
        cv2.rectangle(img, (x1, y1), (x2, y2), BBOX_COLOR, 2)
        for j in range(5):
            cv2.circle(img, (int(ll[j]), int(ll[j+5])), 2, BBOX_COLOR, 2)
    return img

def show_labels(img, face_locations,face_names):
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.putText(img, name, (left, bottom + 20), font, 1.0, (255, 255, 255), 1)
    return img
    
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

def loop_and_detect(cam, mtcnn, minsize):
    """Continuously capture images from camera and do face detection."""
    full_scrn = False
    fps = 0.0
    tic = time.time()
    while True:
        if cv2.getWindowProperty(WINDOW_NAME, 0) < 0:
            break
        img = cam.read()
        if img is not None:
            dets, landmarks = mtcnn.detect(img, minsize=minsize)

            # Parse face locations into correct format
            #i = 0
            #face_locations = []
            #for bb in dets:
            #    fx1, fy1, fx2, fy2 = int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])
            #    face_locations.insert(i, [fy1,fx2,fy2,fx1])
            #    i = i+1
            #rgb_img = img[:, :, ::-1]
            
            # Pass faces into recognition neural net
            #face_encodings = face_recognition.face_encodings(rgb_img, face_locations)
            
            #face_names = match_faces(face_encodings)
    
            # Draw Faces and Landmarks
            img = show_faces(img, dets, landmarks)
            #img = show_labels(img,face_locations,face_names)
            #cv2.imshow(WINDOW_NAME, img)
            out.write(img)
            
            toc = time.time()
            curr_fps = 1.0 / (toc - tic)
            # calculate an exponentially decaying average of fps number
            fps = curr_fps if fps == 0.0 else (fps*0.95 + curr_fps*0.05)
            print(fps)
            tic = toc
        key = cv2.waitKey(1)
        if key == 27:  # ESC key: quit program
            break
        elif key == ord('F') or key == ord('f'):  # Toggle fullscreen
            full_scrn = not full_scrn
            set_display(WINDOW_NAME, full_scrn)


def main():
    print("Began Code")
    args = parse_args()
    cam = Camera(args)
    if not cam.isOpened():
        raise SystemExit('ERROR: failed to open camera!')

    mtcnn = TrtMtcnn()

    open_window(
        WINDOW_NAME, 'Camera TensorRT MTCNN Demo for Jetson Nano',
        cam.img_width, cam.img_height)
    loop_and_detect(cam, mtcnn, args.minsize)

    cam.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
