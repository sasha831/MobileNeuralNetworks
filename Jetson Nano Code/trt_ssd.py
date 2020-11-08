"""trt_ssd.py

This script demonstrates how to do real-time object detection with
TensorRT optimized Single-Shot Multibox Detector (SSD) engine.
"""

import time
import argparse

import cv2
import pycuda.autoinit  # This is needed for initializing CUDA driver

from utils.ssd_classes import get_cls_dict
from utils.ssd import TrtSSD
from utils.camera import add_camera_args, Camera
from utils.display import open_window, set_display, show_fps
from utils.visualization import BBoxVisualization


WINDOW_NAME = 'TrtSsdDemo'
INPUT_HW = (300, 300)
SUPPORTED_MODELS = [
    'ssd_mobilenet_v1_coco',
    'ssd_mobilenet_v1_egohands',
    'ssd_mobilenet_v2_coco',
    'ssd_mobilenet_v2_egohands',
    'ssd_inception_v2_coco',
    'ssdlite_mobilenet_v2_coco',
]

out = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc(*'DIVX'),60.0,(1920,1080))

def parse_args():
    """Parse input arguments."""
    desc = ('Capture and display live camera video, while doing '
            'real-time object detection with TensorRT optimized '
            'SSD model on Jetson Nano')
    parser = argparse.ArgumentParser(description=desc)
    parser = add_camera_args(parser)
    parser.add_argument('-m', '--model', type=str,
                        default='ssd_mobilenet_v1_coco',
                        choices=SUPPORTED_MODELS)
    args = parser.parse_args()
    return args


def loop_and_detect(cam, trt_ssd, conf_th, vis):
    """Continuously capture images from camera and do object detection.

    # Arguments
      cam: the camera instance (video source).
      trt_ssd: the TRT SSD object detector instance.
      conf_th: confidence/score threshold for object detection.
      vis: for visualization.
    """
    full_scrn = False
    fps = 0.0
    tic = time.time()
    while True:
        if cv2.getWindowProperty(WINDOW_NAME, 0) < 0:
            break
        img = cam.read()
        if img is None:
            break
        
        # Perform Human Detection
        boxes, confs, clss = trt_ssd.detect(img, conf_th)
        
        # Copy lists for parsing
        boxes2 = boxes.copy()
        confs2 = confs.copy()
        clss2 = clss.copy()
        
        # Remove non-human and low confidence results
        for box, conf, cls in zip(boxes2,confs2,clss2):
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            size = (x2-x1)*(y2-y1)
            if cls!=1:
                boxes.remove(box)
                confs.remove(conf)
                clss.remove(cls)
        
        img = vis.draw_bboxes(img, boxes, confs, clss)
        #cv2.imshow(WINDOW_NAME, img)
        out.write(img)
        toc = time.time()
        curr_fps = 1.0 / (toc - tic)
        # calculate an exponentially decaying average of fps number
        fps = curr_fps if fps == 0.0 else (fps*0.95 + curr_fps*0.05)
        tic = toc
        print(fps)
        key = cv2.waitKey(1)
        if key == 27:  # ESC key: quit program
            break
        elif key == ord('F') or key == ord('f'):  # Toggle fullscreen
            full_scrn = not full_scrn
            set_display(WINDOW_NAME, full_scrn)


def main():
    args = parse_args()
    cam = Camera(args)
    if not cam.isOpened():
        raise SystemExit('ERROR: failed to open camera!')

    cls_dict = get_cls_dict(args.model.split('_')[-1])
    trt_ssd = TrtSSD(args.model, INPUT_HW)

    open_window(
        WINDOW_NAME, 'Camera TensorRT SSD Demo',
        cam.img_width, cam.img_height)
    vis = BBoxVisualization(cls_dict)
    loop_and_detect(cam, trt_ssd, conf_th=0.4, vis=vis)

    cam.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
