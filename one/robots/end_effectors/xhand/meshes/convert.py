import os
import numpy as np
import one.utils.math as oum
from one.geom.loader import _load_stl, _save_stl

if __name__ == '__main__':
    '''
    author: weiwei
    date: 20201207osaka
    '''
    rotmat = oum.rotmat_from_euler(oum.pi, 0, 0)
    for root, dirs, files in os.walk("."):
        for file in files:
            if file.lower().endswith(".stl"):  # ignore case
                vs, fs = _load_stl(file)
                vs = (rotmat @ vs.T).T.astype(np.float32)
                _save_stl(vs, fs, file[:-4] + "_.stl")
    # # mt.convert_to_stl("base_link.stl", "base_link.stl", scale_ratio=.001)
    # mt.convert_to_stl("inward_left_finger_link.stl", "inward_left_finger_link.stl", scale_ratio=.001)
    # mt.convert_to_stl("inward_right_finger_link.stl", "inward_right_finger_link.stl", scale_ratio=.001)
