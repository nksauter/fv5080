from dials.array_family import flex
from libtbx import easy_pickle
import sys

data = easy_pickle.load(sys.argv[1])
img_numbers = data['bbox'].parts()[4]
min_refls = int(sys.argv[2])

for img_number in sorted(set(img_numbers)):
  refls = data.select(img_numbers == img_number)
  print "%05d % 5d %s"%(img_number+1, len(refls), "Hit" if len(refls) >= min_refls else "Miss")
