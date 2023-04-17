import os
import sys
import argparse

import h5py
import numpy as np
from PIL import Image

parser = argparse.ArgumentParser(
  description='Transform h5py to png'
)
parser.add_argument('--set-name', type=str, default='',
                   help='LO19 | LN83 | LN84 | LG36 | L498')
args = parser.parse_args()

set_dir = args.set_name
for dirpath, dirnames, filenames in os.walk(set_dir):
  for filename in filenames:
    if '2000.h5' in filename:
      db_name = os.path.join(dirpath, filename)
db = h5py.File(db_name,'r')

root_dir = os.path.join(set_dir, 'images')
if not os.path.isdir(root_dir):
  os.mkdir(root_dir)

id = 0
for key in db['data'].keys():
  img_datas = db['data'][key]['images']
  for i in range(img_datas.shape[0]):
    id += 1
    img = img_datas[i]
    img[img < 0] = 0
    pil_img = Image.fromarray(img, mode='I')
    pil_img = pil_img.resize((960, 960), Image.Resampling.BILINEAR) #Needed for Pillow 10
    img_name = '{:05d}.png'.format(id)
    img_path = os.path.join(root_dir, img_name)
    pil_img.save(img_path)
