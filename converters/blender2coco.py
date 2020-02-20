import os
import sys
import random
import shutil
import argparse
import json
import imageio
import operator
import datetime
import numpy as np
from skimage import measure
import pycocotools.mask
import cv2
import matplotlib.pyplot as plt

sys.path.append(os.getcwd())

HOLE_CATEGORY_ID = -1


def plot_polygon(rgb, mask, annotation):
    f, ax = plt.subplots(1, 2)

    ax[0].imshow(rgb)
    ax[1].imshow(mask)

    ax[0].set_title(str(annotation['category_id']))
    ax[1].set_title(str(annotation['category_id']))

    for polygon in annotation['segmentation']:
        ax[0].scatter(polygon[0::2], polygon[1::2], s=2)
        ax[1].scatter(polygon[0::2], polygon[1::2], s=2)
    plt.show()

    import pdb
    pdb.set_trace()


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
    if category_id != HOLE_CATEGORY_ID:
        mask[mask != category_id] = 0

    ordered_contours = get_contours(mask, holes=category_id == HOLE_CATEGORY_ID)

    # Hardcoded for valve holes:
    #   Valve has largest contour of whole object and the rest of contours on mask
    #   are contours of 3 or more holes.
    mask.fill(0)
    for contour, area in ordered_contours:
        cv2.fillPoly(mask, np.array([contour], dtype=np.int32), color=category_id)

    return mask


def fill_holes(mask):
    ordered_contours = get_contours(mask)

    mask.fill(0)
    cv2.fillPoly(mask, np.array([ordered_contours[0][0]], dtype=np.int32), color=HOLE_CATEGORY_ID)

    return mask


def create_example_annotation(args, id, annotations, filename,
                              images_dir, category_name, category_id,
                              annotate_holes=False, dir_index=None, rgb=None, classes=None):

    example_num = filename.split('.')[0]

    if classes.index(category_id) == 0:
        annotations['images'].append({
            'file_name': '{}-{}'.format(category_name, filename),
            'height': args.img_height,
            'width': args.img_width,
            # Unnecessary doubling of image indices
            'id': dir_index * 100000 + int(example_num) # if not annotate_holes else HOLE_CATEGORY_ID * 100000 + int(example_num)
        })

    shutil.copy(
        os.path.join(args.src, category_name, 'rgb', filename),
        os.path.join(images_dir, '{}-{}'.format(category_name, filename))
    )

    mask = imageio.imread(os.path.join(args.src, category_name, 'mask', filename))
    mask = filter_mask(mask, category_id)

    rle_mask = pycocotools.mask.encode(np.asfortranarray(mask.astype(np.uint8)))

    annotations['annotations'].append({
        'id': id,
        'category_id': category_id,
        'image_id': dir_index * 100000 + int(example_num), # if not annotate_holes else HOLE_CATEGORY_ID * 100000 + int(example_num),
        'iscrowd': 0,
        'bbox': pycocotools.mask.toBbox(rle_mask).tolist(),
        'area': pycocotools.mask.area(rle_mask).tolist()
    })

    annotations['annotations'][-1]['segmentation'] = polygon_format(args, mask, category_name, annotate_holes)

    # if annotations['annotations'][-1]['segmentation'].__len__() not in [1, 3]:
    #     import pdb
    #     pdb.set_trace()
    if annotations['annotations'][-1]['segmentation'].__len__() not in [1, 3] and annotate_holes:
        plot_polygon(rgb, mask, annotation=annotations['annotations'][-1])


def plot_coco_annotation(rgb_filename, annotation):
    rgb = imageio.imread(rgb_filename)
    bbox = list(map(int, annotation['bbox']))
    rgb = cv2.rectangle(rgb, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (255, 0, 0), 2)
    plt.imshow(rgb)

    for polygon in annotation['segmentation']:
        plt.scatter(polygon[0::2], polygon[1::2], s=2)

    plt.suptitle('ID={}, Category={}, Area={}'.format(annotation['id'], annotation['category_id'], annotation['area']), fontsize=20)
    plt.show()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str)
    parser.add_argument('--dest', type=str, default='/custom_coco_dataset')
    parser.add_argument('--img-width', type=int, default=1280)
    parser.add_argument('--img-height', type=int, default=720)
    parser.add_argument('--split', type=float, default=0.95)
    opt = parser.parse_args()

    images_dir = os.path.join(opt.dest, 'images')
    annotations_dir = os.path.join(opt.dest, 'annotations')

    if not os.path.exists(opt.dest):
        os.makedirs(opt.dest)
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
    if not os.path.exists(annotations_dir):
        os.makedirs(annotations_dir)

    with open(os.path.join(opt.src, 'objects_info.json'), 'r') as f:
        objects_info = json.load(f)['objects']
        objects_info = {int(v): k for k, v in objects_info.items()}
    # objects_info = sorted(objects_info.items(), key=operator.itemgetter(1))

    global HOLE_CATEGORY_ID
    HOLE_CATEGORY_ID = len(objects_info) + 1
    HOLE_ANNOTATION_CATEGORY = {'id': HOLE_CATEGORY_ID, 'name': 'hole'}

    train_annotations = {'categories': [], 'annotations': [], 'images': []}
    valid_annotations = {'categories': [], 'annotations': [], 'images': []}

    cntr = 0

    print("{}: Conversion started".format(datetime.datetime.now()))

    for k, v in objects_info.items():
        category_data = {'id': k, 'name': v}

        train_annotations['categories'].append(category_data)
        valid_annotations['categories'].append(category_data)
    train_annotations['categories'].append(HOLE_ANNOTATION_CATEGORY)
    valid_annotations['categories'].append(HOLE_ANNOTATION_CATEGORY)

    obj_dir_ctr = 0

    for base_path, obj_dirs, _ in os.walk(opt.src):
        for obj_dir in obj_dirs:
            obj_dir_ctr += 1
            obj_dir_abs = os.path.join(base_path, obj_dir)

            for filename in os.listdir(os.path.join(obj_dir_abs, 'mask')):
                stripped_filename = filename.split('.')[0]
                mode = 'train' if np.random.sample() < opt.split else 'valid'

                rgb = imageio.imread(os.path.join(obj_dir_abs, 'rgb', filename))
                mask = imageio.imread(os.path.join(obj_dir_abs, 'mask', filename))
                classes = list(map(int, np.unique(mask)))

                if 0 in classes:
                    classes.remove(0)

                if obj_dir in ['valve']:
                    classes.append(HOLE_CATEGORY_ID)

                for c in classes:
                    cntr += 1

                    create_example_annotation(opt, id=cntr, filename=filename, images_dir=images_dir,
                                              annotations=train_annotations if mode == 'train' else valid_annotations,
                                              category_name=obj_dir, category_id=c, dir_index=obj_dir_ctr, rgb=rgb, classes=classes)

                    # if obj_dir in ['valve']:
                    #     cntr += 1
                    #     create_example_annotation(opt, id=cntr, filename=filename, images_dir=images_dir,
                    #                               annotations=train_annotations if mode == 'train' else valid_annotations,
                    #                               category_name=obj_dir, category_id=c, annotate_holes=True, dir_index=obj_dir_ctr, rgb=rgb, classes=classes)
            print('{}: Object {} conversion finished'.format(datetime.datetime.now(), obj_dir))
        break
    print('{}: Writing to files...'.format(datetime.datetime.now()))

    with open(os.path.join(annotations_dir, 'train.json'), 'w') as train_json:
        json.dump(train_annotations, train_json)
    with open(os.path.join(annotations_dir, 'valid.json'), 'w') as valid_json:
        json.dump(valid_annotations, valid_json)