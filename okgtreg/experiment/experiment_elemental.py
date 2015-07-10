__author__ = 'panc'

import El as el
import numpy as np
from ctypes import *
import sys


m_np = np.matrix(np.random.randn(10000).reshape((100, 100)))
m_np = m_np * m_np.T
m_np_buffer = np.getbuffer(m_np)
# np.frombuffer(m_np_buffer)

pointer, read_only_flag = m_np.__array_interface__['data']
p = POINTER(c_double).from_address(pointer)
# p

m_el = el.Matrix()
m_el.Attach(100, 100, p,100)

m_el.Get(0,0) # TODO: Segmentation fault

type(m_el)

sys.getsizeof(m_el) # TODO: always 64

# m_el.Viewing()
m_el.ToNumPy() # TODO: Segmentation fault
el.HermitianEig(el.LOWER, m_el, 1) # TODO: Segmentation fault

