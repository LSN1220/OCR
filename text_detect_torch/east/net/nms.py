import numpy
from text_detect_torch.east import cfg


def should_merge(region, i, j):
    neighbor = {(i, j - 1)}
    return not region.isdisjoint(neighbor)

def region_neighbor(region_set):
    region_pixels = np.array(list)
