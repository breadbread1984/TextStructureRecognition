#!/usr/bin/python3

from os.path import join;
import json;
import cv2;
import tensorflow as tf;

def load_boundings(anno_path, imdir = '.'):

  with open(anno_path, 'r') as f:
    via = json.load(f);
  retval = list();
  for fid in via['_via_img_metadata']:
    fn = join(imdir, via['_via_img_metadata'][fid]['filename']);
    img = cv2.imread(fn);
    if img is None:
      print('cannot open image %s' % (fn));
      continue;
    for region in via['_via_img_metadata'][fid]['regions']:
      if region['shape_attributes']['name'] != 'rect':
        print('extraction of %s regions not yet implemented!' % region['shape_attributes']['name']);
        continue;
      x = region['shape_attributes']['x']
      y = region['shape_attributes']['y']
      w = region['shape_attributes']['width']
      h = region['shape_attributes']['height']
      left = int(max(0, x));
      top = int(max(0, y));
      right = int(min(img.shape[1], x + w));
      bottom = int(min(img.shape[0], y + h));
      crop = img[top:bottom, left:right, :];
      retval.append({'img': crop, 'pos': (left, top, right, bottom)});
  return retval;

if __name__ == "__main__":

  bounds = load_boundings('/mnt/c/Users/Lenovo/Downloads/reading order example/before.json', '/mnt/c/Users/Lenovo/Downloads/reading order example/');
  for bound in bounds:
    cv2.imshow('img', bound['img']);
    cv2.waitKey();
