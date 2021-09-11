import numpy as np
from paper3.txt_data import L2Txt
import glob

l2_files = sorted(glob.glob('/media/kyle/Samsung_T5/l2txt/orbit03400/*3453*'))
l2_file = L2Txt(l2_files[5])

o3 = l2_file.o3[:120, :]
print(np.amin(o3))
