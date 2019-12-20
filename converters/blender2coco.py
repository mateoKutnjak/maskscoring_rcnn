import os
import sys
import random
import shutil
import argparse
import json
import imageio
import base64
import operator
import datetime
import numpy as np
import numpy.ma as ma
from skimage import measure
import pycocotools.mask
import cv2
import matplotlib.pyplot as plt
from itertools import groupby

sys.path.append(os.getcwd())

HOLE_CATEGORY_ID = -1


def plot_polygon(mask, polygons):
    plt.imshow(mask)

    for polygon in polygons:
        plt.scatter(polygon[0::2], polygon[1::2], s=2)
    plt.show()


"""
Method assumes that all masks are single-object. If more contours are
found on mask, rest of them are holes on the object (valve), and are
treated as separate object with unique ID == len(objects)+1.
"""
def polygon_format(args, mask, category_name, annotate_holes):
    contours = measure.find_contours(mask, 0.5)
    segmentation_list = []

    for contour in contours:
        contour = np.flip(contour, axis=1)
        segmentation = contour.ravel().tolist()
        segmentation_list.append(segmentation)

    return segmentation_list


def get_contours(mask, holes=False):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    ordered_contours = []

    for contour in contours:
        area = cv2.contourArea(contour)
        ordered_contours.append((contour, area))

    # Descending ordering
    ordered_contours = sorted(ordered_contours, key=lambda x: x[1], reverse=True)

    return ordered_contours[1:] if holes else [ordered_contours[0]]


def filter_mask(mask, category_id):
    if category_id not in [1, HOLE_CATEGORY_ID]:
        return mask

    ordered_contours = get_contours(mask, holes=category_id == HOLE_CATEGORY_ID)

    mask.fill(0)
    for contour, area in ordered_contours:
        cv2.fillPoly(mask, np.array([contour], dtype=np.int32), color=category_id)

    return mask


def fill_holes(mask):
    ordered_contours = get_contours(mask)

    mask.fill(0)
    cv2.fillPoly(mask, np.array([ordered_contours[0][0]], dtype=np.int32), color=HOLE_CATEGORY_ID)

    return mask


def create_example_annotation(args, id, subset, annotations, filename,
                              images_dir, category_name, category_id,
                              annotate_holes=False):

    example_num = filename.split('.')[0]

    annotations['images'].append({
        'file_name': 'COCO_{}2019_{}-{}'.format(subset, category_name, filename),
        'height': args.img_height,
        'width': args.img_width,
        'id': category_id * 100000 + int(example_num) if not annotate_holes else HOLE_CATEGORY_ID * 100000 + int(example_num)
    })

    shutil.copy(
        os.path.join(args.src, category_name, 'rgb', filename),
        os.path.join(images_dir, 'COCO_{}2019_{}-{}'.format(subset, category_name, filename))
    )

    mask = imageio.imread(os.path.join(args.src, category_name, 'mask', filename))
    mask = filter_mask(mask, HOLE_CATEGORY_ID if annotate_holes else category_id)

    rle_mask = pycocotools.mask.encode(np.asfortranarray(mask.astype(np.uint8)))

    annotations['annotations'].append({
        'id': id,
        'category_id': category_id if not annotate_holes else HOLE_CATEGORY_ID,
        'image_id': category_id * 100000 + int(example_num),
        'iscrowd': 0,
        'bbox': pycocotools.mask.toBbox(rle_mask).tolist(),
        'area': pycocotools.mask.area(rle_mask).tolist()
    })

    annotations['annotations'][-1]['segmentation'] = polygon_format(args, mask, category_name, annotate_holes)

    # plot_polygon(mask, polygons=annotations['annotations'][-1]['segmentation'])


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str)
    parser.add_argument('--dest', type=str, default='/custom_coco_dataset')
    parser.add_argument('--img-width', type=int, default=1280)
    parser.add_argument('--img-height', type=int, default=720)
    parser.add_argument('--split', type=float, default=0.95)
    args = parser.parse_args()

    images_dir = os.path.join(args.dest, 'images')
    annotations_dir = os.path.join(args.dest, 'annotations')

    if not os.path.exists(args.dest):
        os.makedirs(args.dest)
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
    if not os.path.exists(annotations_dir):
        os.makedirs(annotations_dir)

    with open(os.path.join(args.src, 'objects_info.json'), 'r') as f:
        objects_info = json.load(f)['objects']
    objects_info = sorted(objects_info.items(), key=operator.itemgetter(1))

    global HOLE_CATEGORY_ID
    HOLE_CATEGORY_ID = len(objects_info) + 1
    HOLE_ANNOTATION_CATEGORY = {'id': HOLE_CATEGORY_ID, 'name': 'hole'}

    train_annotations = {'categories': [], 'annotations': [], 'images': []}
    valid_annotations = {'categories': [], 'annotations': [], 'images': []}

    cntr = 0

    print("{}: Conversion started".format(datetime.datetime.now()))

    for category_name, category_id in objects_info:
        train_annotations['categories'].append({
            'id': category_id,
            'name': category_name
        })
        valid_annotations['categories'].append({
            'id': category_id,
            'name': category_name
        })

        filenames = os.listdir(os.path.join(args.src, category_name, 'rgb'))

        random.shuffle(filenames)
        train_filenames = filenames[:int(len(filenames) * args.split)]
        valid_filenames = filenames[int(len(filenames) * args.split):]

        for train_filename in train_filenames:
            cntr += 1
            create_example_annotation(args, id=cntr, subset='train', annotations=train_annotations,
                                      filename=train_filename, images_dir=images_dir,
                                      category_name=category_name, category_id=category_id)

            if category_name in ['valve']:
                cntr += 1
                create_example_annotation(args, id=cntr, subset='train', annotations=train_annotations,
                                          filename=train_filename, images_dir=images_dir,
                                          category_name=category_name, category_id=category_id, annotate_holes=True)

                if HOLE_ANNOTATION_CATEGORY not in train_annotations['categories']:
                        train_annotations['categories'].append(HOLE_ANNOTATION_CATEGORY)

        for valid_filename in valid_filenames:
            cntr += 1
            create_example_annotation(args, id=cntr, subset='valid', annotations=valid_annotations,
                                      filename=valid_filename, images_dir=images_dir,
                                      category_name=category_name, category_id=category_id)

            if category_name in ['valve']:
                cntr += 1
                create_example_annotation(args, id=cntr, subset='valid', annotations=valid_annotations,
                                          filename=valid_filename, images_dir=images_dir,
                                          category_name=category_name, category_id=category_id, annotate_holes=True)

                if HOLE_ANNOTATION_CATEGORY not in valid_annotations['categories']:
                    valid_annotations['categories'].append(HOLE_ANNOTATION_CATEGORY)

        print('{}: Object {} conversion finished'.format(datetime.datetime.now(), category_name))

    print('Writing to files...')
    with open(os.path.join(annotations_dir, 'train.json'), 'w') as train_json:
        json.dump(train_annotations, train_json)
    with open(os.path.join(annotations_dir, 'valid.json'), 'w') as valid_json:
        json.dump(valid_annotations, valid_json)