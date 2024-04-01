import cv2
import dlib
import numpy as np

detector = dlib.get_frontal_face_detector()


def face_locate(img):
    dets = detector(img, 0)
    if not dets:
        return None
    return max(dets, key=lambda det: (det.right() - det.left()) * (det.bottom() - det.top()))


predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')


def get_key_points(img, face_pos):
    landmark_shape = predictor(img, face_pos)
    key_points = []
    for i in range(68):
        pos = landmark_shape.part(i)
        key_points.append(np.array([pos.x, pos.y], dtype=np.float32))
    return key_points


def generate_points(key_points):
    def centre(index_array):
        return sum([key_points[i] for i in index_array]) / len(index_array)
    l_brow = [18, 19, 20, 21]
    r_brow = [22, 23, 24, 25]
    Chin_ = [6, 7, 8, 9, 10]
    nose_ = [29, 30]
    return centre(l_brow + r_brow), centre(Chin_), centre(nose_)


def generate_features(points):
    _browcentre, Chin_centre, nose_centre = points
    中线 = _browcentre - Chin_centre
    斜边 = _browcentre - nose_centre
    H_Rotate_amount = np.cross(中线, 斜边) / np.linalg.norm(中线)**2
    V_Rotate_amount = 中线 @ 斜边 / np.linalg.norm(中线)**2
    return np.array([H_Rotate_amount, V_Rotate_amount])


def draw(H_Rotate_amount, V_Rotate_amount):
    img = np.ones([512, 512], dtype=np.float32)
    face_length = 200
    centre = 256, 256
    l_eye = int(220 + H_Rotate_amount * face_length), int(249 + V_Rotate_amount * face_length)
    r_eye = int(292 + H_Rotate_amount * face_length), int(249 + V_Rotate_amount * face_length)
    mouth = int(256 + H_Rotate_amount * face_length / 2), int(310 + V_Rotate_amount * face_length / 2)
    cv2.circle(img, centre, 100, 0, 1)
    cv2.circle(img, l_eye, 15, 0, 1)
    cv2.circle(img, r_eye, 15, 0, 1)
    cv2.circle(img, mouth, 5, 0, 1)
    return img


def get_img_features(img):
    face_pos = face_locate(img)
    if not face_pos:
        cv2.imshow('self', img)
        cv2.waitKey(1)
        return None
    key_points = get_key_points(img, face_pos)
    # for i, (px, py) in enumerate(key_points):
    #     cv2.putText(img, str(i), (int(px),int(py)), cv2.FONT_HERSHEY_COMPLEX, 0.25, (255, 255, 255))
    points = generate_points(key_points)
    # for i, (px, py) in enumerate(points):
    #     cv2.putText(img, str(i), (int(px),int(py)), cv2.FONT_HERSHEY_COMPLEX, 0.25, (255, 255, 255))
    Rotate_amounts = generate_features(points)
    # cv2.putText(img, '%.3f' % 旋转量,
    #             (int(points[-1][0]), int(points[-1][1])), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255))
    cv2.imshow('self', img)
    return Rotate_amounts


if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    origin_features = get_img_features(cv2.imread('std_face.jpg'))
    features = origin_features - origin_features
    while True:
        ret, img = cap.read()
        # img = cv2.flip(img, 1)
        new_features = get_img_features(img)
        if new_features is not None:
            features = new_features - origin_features
        H_Rotate_amount, V_Rotate_amount = features
        cv2.imshow('Vtuber', draw(H_Rotate_amount, V_Rotate_amount))
        cv2.waitKey(1)
