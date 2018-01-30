from __future__ import division, print_function
from dxtbx.format.Format import Format
from dxtbx.format.FormatHDF5 import FormatHDF5
from dxtbx.format.FormatStill import FormatStill

class FormatHDF5ImageDictionary(FormatHDF5, FormatStill):

  def __init__(self, image_file, **kwargs):
    assert(self.understand(image_file))
    FormatHDF5.__init__(self, image_file, **kwargs)

  @staticmethod
  def understand(image_file):
    import h5py
    h5_handle = h5py.File(image_file, 'r')
    fv5080 = 'metadata' in h5_handle and \
      'idh5_version' in h5_handle['metadata'] and \
      h5_handle['metadata/idh5_version'].value == 'idh5_v1.0'
    if fv5080: print("Understood H5 format for FV5080 paper.")
    return fv5080

  def _start(self):
    import h5py
    self._h5_handle = h5py.File(self.get_image_file(), 'r')

    self.indices = {}
    counter = 0
    for chunkname in self._h5_handle['data'].keys():
      n = self._h5_handle['data/%s/images'%chunkname].len()
      self.indices[chunkname] = (counter, counter + n)
      counter += n
    self.n_images = counter

  def _get_chunk_and_index(self, index):
    for chunkname in self.indices:
      if index >= self.indices[chunkname][0] and index < self.indices[chunkname][1]:
        return chunkname, index - self.indices[chunkname][0]
    raise IndexError("Index out of range: %d"%index)

  def _detector(self, index = None):
    '''Return a model for a simple detector'''
    if index is None: index = 0
    chunkname, chunkindex = self._get_chunk_and_index(index)
    return self._detector_factory.simple(
        sensor = 'PAD',
        distance = self._h5_handle['data/%s/distance'%chunkname][chunkindex],
        beam_centre = (self._h5_handle['metadata']['BEAM_CENTER_X'].value,
                       self._h5_handle['metadata']['BEAM_CENTER_Y'].value),
        fast_direction = '+x',
        slow_direction = '-y',
        pixel_size = (self._h5_handle['metadata']['PIXEL_SIZE'].value,
                      self._h5_handle['metadata']['PIXEL_SIZE'].value),
        image_size = (self._h5_handle['metadata']['SIZE1'].value,
                      self._h5_handle['metadata']['SIZE2'].value),
        trusted_range = (self._h5_handle['metadata']['MIN_TRUSTED_VALUE'].value,
                         self._h5_handle['metadata']['SATURATED_VALUE'].value),
        mask = [])  # a list of dead rectangles

  def get_mask(self, index=None, goniometer=None):
    '''Creates a mask merging untrusted pixels with active areas.'''
    if index is None: index = 0

    from scitbx.array_family import flex
    # get effective active area coordinates
    tiling = flex.int(self._h5_handle['metadata']['ACTIVE_AREAS'].value)
    if tiling is None or len(tiling) == 0:
      return None

    n_tiles = len(tiling) // 4
    if n_tiles <= 1:
      return None

    peripheral_margin = 1
    for i in xrange(n_tiles):
      tiling[4 * i + 0] += peripheral_margin
      tiling[4 * i + 1] += peripheral_margin
      tiling[4 * i + 2] -= peripheral_margin
      tiling[4 * i + 3] -= peripheral_margin

    # get the raw data to get the size of the mask
    data = self.get_raw_data(index)

    # set the mask to the same dimensions as the data
    mask = flex.bool(flex.grid(data.focus()))

    # set active areas to True so they are not masked
    for i in xrange(n_tiles):
      x1,y1,x2,y2=tiling[4*i:(4*i)+4]
      sub_array = flex.bool(flex.grid(x2-x1,y2-y1),True)
      mask.matrix_paste_block_in_place(sub_array,x1,y1)

    # create untrusted pixel mask
    detector = self.get_detector()
    assert len(detector) == 1
    trusted_mask = detector[0].get_trusted_range_mask(data)

    # returns merged untrusted pixels and active areas using bitwise AND (pixels are accepted
    # if they are inside of the active areas AND inside of the trusted range)
    return (mask & trusted_mask,)

  def _beam(self, index = None):
    '''Return a simple model for the beam.'''
    if index is None: index = 0
    chunkname, chunkindex = self._get_chunk_and_index(index)
    return self._beam_factory.simple(self._h5_handle['data/%s/wavelength'%chunkname][chunkindex])

  def get_detector(self, index=None):
    return self._detector(index)

  def get_beam(self, index=None):
    return self._beam(index)

  def get_raw_data(self, index = None):
    from scitbx.array_family import flex
    if index is None: index = 0
    chunkname, chunkindex = self._get_chunk_and_index(index)
    import numpy as np 
    return flex.int(self._h5_handle['data/%s/images'%chunkname][chunkindex].astype(np.int32))

  def get_num_images(self):
    return self.n_images

  def get_image_file(self, index=None):
    return Format.get_image_file(self)

if __name__ == '__main__':
  import sys
  for arg in sys.argv[1:]:
    print (FormatRawData.understand(arg))
