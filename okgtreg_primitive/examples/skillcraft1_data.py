__author__ = 'panc'

from okgtreg_primitive.okgtreg import *
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.interpolate import spline
from scipy import interpolate

# load data as a Pandas DataFrames
scdata = pd.read_csv('okgtreg_primitive/data/SkillCraft1_Dataset.csv')

# Prepare data for OKGT
y = scdata['LeagueIndex']
y = np.matrix(y).T

xnames = [# 'Age',
            # 'HoursPerWeek', 'TotalHours',
            'APM', 'SelectByHotkeys', 'AssignToHotkeys', 'UniqueHotkeys',
            'MinimapAttacks', 'MinimapRightClicks',
            'NumberOfPACs', 'GapBetweenPACs', 'ActionLatency', 'ActionsInPAC',
            'TotalMapExplored',
            'WorkersMade',
            'UniqueUnitsMade', 'ComplexUnitsMade',
            'ComplexAbilitiesUsed']
x = scdata[xnames]
x = np.matrix(x)
# x.shape

# Kernel names and parameters
n,p = x.shape
kname = 'Gaussian'
kparam = dict(sigma=0.5)

# Construct okgt object
scokgt = OKGTReg_Nystroem(x, y, [kname]*p, [kname], [kparam]*p, [kparam], nComponents=50)

start = time.time()
scokgt.TrainOKGT_Nystroem()
end = time.time()

print 'Sample size:', n
print 'Group number', p
print 'Training time:', end - start, 'sec'

# plot
plt.scatter(y, scokgt.g)

# plot each x ~ f(x)
def plotf(ind, type='d', negate=False):
    xplot = x[:,ind]
    fplot = scokgt.f[:,ind]

    if negate:
        fplot = -fplot

    if type=='d':
        plt.scatter(xplot, fplot)
    else:
        xplot = np.array(xplot.T).squeeze()
        fplot = np.array(fplot.T).squeeze()
        order = np.argsort(xplot)
        xplot = xplot[order]
        fplot = fplot[order]
        plt.plot(xplot, fplot)

    plt.title(xnames[ind])
    return

ind=8
xplot = np.array(x[:,ind]).T.squeeze()
order = np.argsort(xplot)
xplot = xplot[order]
xplot = np.unique(xplot)

fplot = -np.array(scokgt.f[:,ind]).T.squeeze()
fplot = fplot[order]
plt.plot(xplot, fplot)


tck = interpolate.splrep(xplot, fplot, s=0)
tck

xnew = np.linspace(xplot.min(), xplot.max(), 100)
ynew = interpolate.splev(xnew, tck, der=0)
plt.plot(xnew, ynew)


plotf(14, 'd', 1)

# g(y) ~ sum(f(x))
plt.scatter(scokgt.f.sum(axis=1), scokgt.g)
plt.scatter(x.sum(axis=1), y)