"""
This file is specifically used to test yolov5 models
This file shouldnot be used while doing github actions.
This file has been created just to see the performance of diana for object detection use case
"""
from dianaquantlib.models.yolov5 import train

# the main function takes opt as a input.
# opt is a dictionary
"""
Usage in CLI:
    $ python --m train.py --data=coco128.yaml --weights yolov5ns.pt --img 640 # from pretrained
    $ python train.py --data coco128.yaml --weights '' --cfg yolov5s.yaml --img 640 # from scratch
"""
opt = {
    "data": "coco128.yaml",
    "weights": "yolov5ns.pt",
    "img": 640
}
train.main(opt=opt)