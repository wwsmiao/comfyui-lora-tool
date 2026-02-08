import cv2
import numpy as np

CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)


def detect_faces(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(40, 40)
    )
    return faces


def crop_and_resize_face(
    image,
    face_box,
    target_w,
    target_h,
    expand_ratio=2.0
):
    """
    裁剪 + 等比缩放 + padding，不变形
    """
    img_h, img_w = image.shape[:2]
    x, y, w, h = face_box

    cx = x + w // 2
    cy = y + h // 2

    target_ratio = target_w / target_h

    crop_h = int(h * expand_ratio)
    crop_w = int(crop_h * target_ratio)

    if crop_w > img_w:
        crop_w = img_w
        crop_h = int(crop_w / target_ratio)

    x1 = max(0, cx - crop_w // 2)
    y1 = max(0, cy - crop_h // 2)
    x2 = min(img_w, x1 + crop_w)
    y2 = min(img_h, y1 + crop_h)

    crop = image[y1:y2, x1:x2]

    # 等比 resize
    ch, cw = crop.shape[:2]
    scale = min(target_w / cw, target_h / ch)
    new_w = int(cw * scale)
    new_h = int(ch * scale)

    resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # padding
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    px = (target_w - new_w) // 2
    py = (target_h - new_h) // 2
    canvas[py:py+new_h, px:px+new_w] = resized

    return canvas
