
from utils import download_file


class PascalVOC(object):
    def __init__(self, path, download=False):
        self.path = path

        if download:
            download_file(('http://host.robots.ox.ac.uk/pascal/VOC/'
                           'voc2012/VOCtrainval_11-May-2012.tar'),
                          'pascal_voc.tar')
            download_file(('http://www.eecs.berkeley.edu/Research/'
                           'Projects/CS/vision/grouping/'
                           'semantic_contours/benchmark.tgz'),
                          'additional_contours.tgz')
