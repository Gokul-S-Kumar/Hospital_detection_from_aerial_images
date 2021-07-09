# Importing packages
import os
import cv2
import shutil
import numpy as np
from lxml import etree
from tqdm import tqdm
import shutil

# Base dirs for images, annotations and masks
image_base_dir = '../input/images/'
label_base_dir = '../input/annotations/'
output_base_dir = '../input/masks/'
mask_bitness = 24
height, width = 768, 768
scale_factor = 1

# Func for parsing the CVAT annotation file
def parse_anno_file(city):
    cvat_xml = label_base_dir + city + '/annotations.xml'
    root = etree.parse(cvat_xml).getroot()
    anno = []
    image_name_attr = './/image'
    for image_tag in root.iterfind(image_name_attr):
        image = {}
        for key, value in image_tag.items():
            image[key] = value
        image['shapes'] = []
        for poly_tag in image_tag.iter('polygon'):
            polygon = {}
            for key, value in poly_tag.items():
                polygon[key] = value
            image['shapes'].append(polygon)
        name = image['name']
        # Transferring images having annotations to trainable folder and the ones not having annotations to non-trainable folder.
        if image['shapes'] == []:
            shutil.move(image_base_dir + city + '/{}'.format(name), image_base_dir + city + '/non_trainable/' + name)
        else:
            shutil.move(image_base_dir + city + '/{}'.format(name), image_base_dir + city + '/trainable/' + name)
        anno.append(image)
    return anno

def create_mask_file(width, height, bitness, background, shapes, scale_factor):
    mask = np.full((height, width, bitness // 8), background, dtype=np.uint8)
    for shape in shapes:
        points = [tuple(map(float, p.split(','))) for p in shape['points'].split(';')]
        points = np.array([(int(p[0]), int(p[1])) for p in points])
        points = points*scale_factor
        points = points.astype(int)
        mask = cv2.drawContours(mask, [points], -1, color=(255, 255, 255), thickness=1)
        mask = cv2.fillPoly(mask, [points], color=(255, 255, 255))
    return mask


if __name__ == '__main__':
    city_lists = os.listdir(label_base_dir)
    for i in tqdm(range(len(city_lists))):
        city = city_lists[i]
        # Creating seperate folders for trainable and non-trainable images
        # Refer to Readme for more details
        if not os.path.exists(image_base_dir + city + '/trainable'):
            os.makedirs(image_base_dir + city + '/trainable')
        if not os.path.exists(image_base_dir + city + '/non_trainable'):
            os.makedirs(image_base_dir + city + '/non_trainable')
        annotations = parse_anno_file(city)
        if not os.path.exists(output_base_dir + city):
            os.makedirs(output_base_dir + city)
        
        background = []
        is_first_image = True
        for image in annotations:
            if image['shapes'] != []:
                background = np.zeros((height, width, 3), np.uint8)
                background = create_mask_file(width, height, mask_bitness, background, image['shapes'], scale_factor)
                output_path = output_base_dir + city + '/{}'.format(image['name'])
                cv2.imwrite(output_path, background)
        
        
        