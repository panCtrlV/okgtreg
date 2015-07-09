__author__ = 'panc'

import El as el
import numpy as np

m = el.Matrix()
m.Set(1, 1, 1.)
m.Set(1, 2, 2.)
m.Get(1,1)
m.Get(0,1)
m.__setattr__('matrix', [[1,2,3],[3,2,1]])
# el.MakeReal(1, m)
# el.MakeSymmetric(1, m)
el.Display(m)
(w, z) = el.HermitianEig(1, m, 1)
el.Display(w)
el.Display(z)

el.Walsh(m, 8)
m.Get(1,1)

