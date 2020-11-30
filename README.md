# fv5080
### Code related to Journal of Synchrotron Radiation paper FV5080.

Tsung-Wei Ke, Aaron S. Brewster, Stella X. Yu, Daniela Ushizima, Chao Yang, and Nicholas K. Sauter (2018).
A Convolutional Neural Network-Based Screening Tool for X-ray Serial Crystallography.  *J. Synchrotron Rad.* **25**, 655-670.
https://doi.org/10.1107/S1600577518004873.

Explanation:
Raw data (five datasets) may be downloaded from the Coherent X-ray Imaging Data Bank at the following URL:
http://cxidb.org/id-76.html
Please note that each file is large (about 13 GB).

Data are written in HDF5 format, 2000 images per dataset.
Image data may be viewed with the program `dials.image_viewer` as follows:
1) Download DIALS from https://dials.github.io
2) Work on your local machine – slow X-connections will stall.
3) Install the image format class with `dxtbx.install_format -u https://raw.githubusercontent.com/nksauter/fv5080/master/FormatHDF5ImageDictionary.py`
4) Use the command `dials.image_viewer <*.h5 file>`
