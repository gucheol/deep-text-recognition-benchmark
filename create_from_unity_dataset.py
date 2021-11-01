import numpy as np
import math 
import PIL
import json
import cv2 
import os
import glob
import tqdm
import time


def four_point_transform(image, rect):
    (tl, tr, br, bl) = rect
    width = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))    
    height = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    dst = np.array([[0, 0],[width - 1, 0],[width - 1, height - 1],[0, height - 1]], dtype = "float32")
    # print(rect)
    # print(dst)
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (int(width), int(height)))
    return warped


def extract(src_path, target_path):
    sequence = 0
    for json_file_path in tqdm.tqdm([os.path.join(src_path, x) for x in os.listdir(src_path) if '.json' in x]):
        with open(json_file_path, 'r', encoding='utf8') as f:
            json_obj = json.load(f)
        rgb_file_path = json_file_path.replace('BoundingBox','rgb').replace('.json','.jpg')
        rotated_im = cv2.imread(rgb_file_path)
        for segment in json_obj['info_list'][0]['segment_list']:
            for word in segment['words']:
                transcription = word['transcription'].replace('\\','#').replace('/','#')
                rotated_points = np.array([[int(point['x']), int(point['y'])] for point in word['word_points']])
                # rotate img
                rotate_angle = json_obj['camera_angle']
                center = [rotated_im.shape[1],rotated_im.shape[0]]
                im, points = rotate_image(rotated_im, center, rotate_angle, rotated_points)
                align_box_points(points)
                cv2.imshow('rot_img',im)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                # warp transform and crop
                patch_im = four_point_transform(im, np.float32(points))
                cv2.imwrite(os.path.join(target_path, transcription + f'_{sequence}.jpg'), patch_im)
                sequence += 1
        time.sleep(1)

def compute_rotation_angle(box_pos, theta_threshold=1):
    delta_x = box_pos[1][0] - box_pos[0][0]
    delta_y = box_pos[1][1] - box_pos[0][1]

    if delta_x == 0 and delta_y < 0:
        return -90
    elif delta_x == 0 and delta_y > 0:
        return 90

    theta = math.atan(delta_y/delta_x) * 180 / math.pi

    if abs(theta) < theta_threshold and delta_x > 0:
        return 0
    elif abs(theta) < theta_threshold and delta_x < 0:
        return -180

    if theta > 0 and delta_x > 0:
        theta = theta
    elif theta > 0 > delta_x:
        theta = theta - 180
    elif theta < 0 < delta_x:
        theta = theta
    else:
        theta = 180+theta

    return theta

def align_box_points(box):
    """
    시계 방향으로 box point 정렬
    :param box: np.Array(4,2)
    :return: np.Array(4,2)
    """
    centroid = np.sum(box, axis=0) / 4
    theta = np.arctan2(box[:, 1] - centroid[1], box[:, 0] - centroid[0]) * 180 / np.pi
    indices = np.argsort(theta)
    aligned_box = box[indices]

    start_idx = aligned_box.sum(axis=1).argmin()
    aligned_box = np.roll(aligned_box, 4 - start_idx, 0)
    return aligned_box

def add_margin(box, width, height, margin=2):
    adjust_bounds = np.zeros_like(box)
    adjust_bounds[0][0] = max(box[0][0] - margin , 0) 
    adjust_bounds[0][1] = max(box[0][1] - margin , 0) 

    adjust_bounds[1][0] = min(box[1][0] + margin , width) 
    adjust_bounds[1][1] = max(box[1][1] - margin , 0) 

    adjust_bounds[2][0] = min(box[2][0] + margin , width) 
    adjust_bounds[2][1] = min(box[2][1] + margin , height) 

    adjust_bounds[3][0] = max(box[3][0] - margin , 0) 
    adjust_bounds[3][1] = min(box[3][1] + margin , height) 

    left = min(adjust_bounds[:, 0])
    right = max(adjust_bounds[:, 0])
    top_y = min(adjust_bounds[:, 1])
    bottom_y = max(adjust_bounds[:, 1])

    return np.array([[left, top_y], [right, top_y], [right, bottom_y], [left, bottom_y]])

def rotate_image(img, center, angle, bounds):
    if angle == 0:
        return img, add_margin(bounds, img.shape[1], img.shape[0])

    height, width = img.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D(tuple(center), angle, 1)
    bounding_box = np.array([[0, 0], [0, height], [width, 0], [width, height]])
    adjust_box = np.transpose(np.dot(rotation_matrix[:, :2], np.transpose(bounding_box))) + rotation_matrix[:, 2]

    min_x = np.min(adjust_box[:, 0])
    min_y = np.min(adjust_box[:, 1])
    max_x = np.max(adjust_box[:, 0])
    max_y = np.max(adjust_box[:, 1])

    bound_w = int(max_x - min_x)
    bound_h = int(max_y - min_y)

    rotation_matrix[0, 2] -= min_x
    rotation_matrix[1, 2] -= min_y
    rotated_img = cv2.warpAffine(img, rotation_matrix, (bound_w, bound_h))
    adjust_bounds = np.transpose(np.dot(rotation_matrix[:, :2], np.transpose(bounds))) + rotation_matrix[:, 2]

    adjust_bounds = align_box_points(adjust_bounds)
    adjust_height, adjust_width = rotated_img.shape[:2]
    adjust_bounds = add_margin(adjust_bounds, adjust_width, adjust_height)
    return rotated_img, np.int32(adjust_bounds)


if __name__=='__main__':
    src_path = '/home/gucheol/data/tests'
    target_path = '/home/gucheol/data/tests_crop'
    extract(src_path, target_path)