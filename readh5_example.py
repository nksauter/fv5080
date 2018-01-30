import h5py, sys
from matplotlib import pyplot as plt
import numpy as np
path = sys.argv[1]

f = h5py.File(path, 'r')

for key in f['data']:
  chunk = f['data'][key]
  images = chunk['images']
  for i in xrange(images.len()):
    img = images[i]
    distance = chunk['distance'][i]
    wavelength = chunk['wavelength'][i]
    timestamp = chunk['timestamp'][i]
    print timestamp, wavelength, distance, img.shape, np.mean(img)
    plt.imshow(img)
    plt.show()
    #break
  break

