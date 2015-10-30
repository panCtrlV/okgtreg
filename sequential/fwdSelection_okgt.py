# This script implemented the forward procedure for determining the group structure for OKGT.

# TODO: set a stopping criterion to avoid adding all variables. variable selection.

# The following defines the Group Structure class, which represents the state of the current 
# group structure.
class GrpStructure:
	def __init__(self, grp):
		'''
			Constructor.

			**Input**
				grp: list of lists, e.g. [[1,3], [2], [4,5]], indicating the grouping for a given number of variables.

			##Output**

		'''
		self.grp = grp 
		self.nGrp = len(grp)

		# @property
		# def grp(self):
		# 	self._grp = grp
		# @grp.setter
		# def grp(self):
		# 	self._grp = grp
		# 	self._nGrp = len(self._grp)

		# @property
		# def nGrp(self):
		# 	return self._nGrp
		# @grp.setter
		# def nGrp(self):

	def __getitem__(self, index):
		'''
			This is the function to override to overload the [] operator for a class.
			Ref: http://stackoverflow.com/questions/1957780/how-to-overload-operator

			**Output**
				list, `index`-th element of `GrpStructure.grp`, i.e. a list of variable indeces for the 
				`index`-th group in the current model. 
		'''
		return self.grp[index]

	def addNew(self, xInd, grpInd=None):
		'''
			Add a new `xInd`-th variable to the `grpInd`-th group in the exisitng group structure `self.grp`.
			If `self.grp == []`, then a new group is created.
			If `self.grp` is not empty, but `grpInd == None` a new group is added at the end.

			TODO: prevent illegal arguments, for example:
					1) `grpInd` > existing number of groups
		'''
		if not self.grp: # if self.grp is empty
			self.grp.append([xInd])
			self.nGrp = len(self.grp) # update number of groups
		elif not [grpInd]: # if self.grp is not empty but `grpInd` is None or 0
			self.grp.append([xInd]) 
			self.nGrp = len(self.grp) # update the number of groups
		else:
			self.grp[grpInd].append(xInd)
			
		return self

	def appendNew(self, xInd):
		'''
			Add a new variable as a group at the end of the existing group structure.
			If `self.grp` is empty, i.e. `[]`, a new group is created.
		'''
		self.grp.append([xInd])
		self.nGrp += 1
		return self


# Now define the forward procedure for group structure determination.
## Simulate data
## Model:
##	Y = \exp( sin(2 * \pi * X_1) + X_2^2 + X_3 * X_6 + | (X_4 + X_5) * X_7 | ) 
import numpy as np 
import copy

np.random.seed(10)
n = 500; p = 7
X = np.matrix(np.random.uniform(-1, 1, (n, p)))
y = np.exp( np.sin(2. * np.pi * X[:,0]) + np.power(X[:,1], 2) +  np.multiply(X[:,2], X[:, 5]) + 
		np.abs(np.multiply(X[:,3]+X[:,4], X[:,6])) )

# Given the current group structure, add a new X in each group
def includeNewX_intoGroup(grpStr, xInd, kernFun, maxGrpSize=3, eps=1e-3, tol=1e-6):
	'''
		The new X is only allowed to be added to one of the existing groups 
		(i.e. cannot be added as a new group).
		This function determines which group the new X should be added by running
		`okgt_ace` (TODO: maybe faster using sequential okgt) for each group where 
		the new X is included.
		The new group structure is decided by the smallest MSE.

		If the existing group structure is empty, a new group is created.
		If the existing group structure contains only one group, the new variable is added,
		provided that its size is not > 3.

		Note: currently, we use the same RKSH for all groups.

		**Input**
			grpStr: GrpStructure type, the existing group structure.
			xInd: integer, `xInd`-th predictor is to be added.

		**Output**
			grpStr_best: GrpStructure type, updated group structure. If the new variable has to be added 
					as a new group, return `None`.

		** Give up the idea of using recursion... **
	'''
	# always add the new X as or in the first group
	if not grpStr.grp: # if there is no variable in the existing group structure.
		# return grpStr.addNew(xInd)
		return None
	elif grpStr.nGrp == 1: # if there is only one group now
		if len(grpStr[0]) < maxGrpSize: # if the group size has not reached the maximum
			return grpStr.addNew(xInd, grpInd = 0)
		else:
			# return grpStr.appendNew(xInd)
			return None
	else:
		r2Max = 0.
		r2Max_ind = 0
		g = None
		f = None
		grpStr_best = None
		for i in range(0, grpStr.nGrp):
			if len(grpStr[i]) >= maxGrpSize:
				continue
			else:
				grpStrIn = copy.deepcopy(grpStr) 
				# use deep copy, otherwise using assignment `=` just make a new binding.
				# Ref: https://docs.python.org/2/library/copy.html
				grpStrIn.addNew(xInd, grpInd = i)
				yKern = kernFun
				xKern = [kernFun] * grpStrIn.nGrp
				r2In, gIn, fIn = okgt_ace(X, y, yKern, xKern, grpStrIn.grp, eps, tol)
				# TODO: `yKern`, `xKern` are passed from outside.
				if r2In > r2Max:
					r2Max = r2
					r2Max_ind = i
					g = gIn
					f = fIn
					grpStr_best = grpStrIn
		if not grpStr_best: # if all groups are of full capacity
			# grpStr_best = copy.deepcopy(grpStr.appendNew(xInd)) # then create a new group
			return None
		else:
			return grpStr_best

# Given the current state of the group structure, where all of the group have sizes reaching the maximum,
# include the new variable as a new group at the end of the group structure.
def includeNewX_newGroup(grpStr, xInd):
	'''
		Include the new variable as a new group in the existing group structure.
		`grpStr` is well-formed, i.e. not empty.

		**Output**
			new GrpStructure type object, which the new variable is added as a new group at the end of the list.
	'''
	return grpStr.appendNew(xInd)

# Wrapper of the above two functions.
# Given the current state of the group structure, include a new variable
def includeNewX(grpStr, xInd, kernFun, maxGrpSize=3, eps=1e-3, tol=1e-6, verbose=False):
	'''
		Given the current group structure, update it by including a new variable 
		if there are more variable(s) available.

		There are two possibilities:
			1) The new variable is added in an existing group, if the group size is smaller than 
				a specified limit;
			2) The new variable is added as a new group.

		**Input**
			grpStr: GrpStructure type, the current specification of the group structure.
			xInd: integer, index of the variable to be added.
			maxGrpSize: integer, the maximum number of variables allowed in a single group.
			eps: numeric, the regularization parameter for matrix inversion.
			tol: numeric, the tolerance for the convergence of the OKGT ACE-type algorithm.

		**Output**
			grpStr_best: GrpStructure, the updated group structure. 
	'''
	yKern = kernFun

	grpStr_new_1 = includeNewX_intoGroup(copy.deepcopy(grpStr), xInd, kernFun, maxGrpSize, eps, tol)
		# Try to include the new variable in an existing group, 
		# if it cannnot be added in an existing group, `None` is returned.
		# Must be a new copy of `grpStr`, otherwise the original `grpStr` will be modified.
	if grpStr_new_1 != None: # if the return is not `None`
		xKern = [kernFun] * grpStr_new_1.nGrp
		r2_1, g_1, f_1 = okgt_ace(X, y, yKern, xKern, grpStr_new_1.grp, eps, tol)
		print str(grpStr_new_1.grp) + " -> " + str(r2_1)
		grpStr_new_2 = includeNewX_newGroup(copy.deepcopy(grpStr), xInd) 
			# Try the possibility that the new variable is added as a new group.
			# Must be a new copy of `grpStr`, otherwise the original `grpStr` will be modified.
		xKern = [kernFun] * grpStr_new_2.nGrp
		r2_2, g_2, f_2 = okgt_ace(X, y, yKern, xKern, grpStr_new_2.grp, eps, tol)
		print str(grpStr_new_2.grp) + " -> " + str(r2_2)
		if r2_1 > r2_2:
			if verbose:
				print "    --> " + str(grpStr_new_1.grp)
			return grpStr_new_1
		else:
			if verbose:
				print "    --> " + str(grpStr_new_2.grp)
			return grpStr_new_2
	else:
		grpStr_new_2 = includeNewX_newGroup(grpStr, xInd)
		if verbose:
				print "    --> " + str(grpStr_new_2.grp)
		return grpStr_new_2

def includeNewX_all(grpStr, xIndPool, kernFun, maxGrpSize=3, eps=1e-3, tol=1e-6, verbose=False):
	if not xIndPool: # if empty
		return (grpStr, None)
	else:
		bestGroup = None
		newX = None
		yKern = kernFun
		r2 = 0.
		for i in range(0, len(xIndPool)):
			if verbose: 
				print "Try X" + str(xIndPool[i]).strip('[]') + ":"
			grpStrIn = copy.deepcopy(grpStr)
			group = includeNewX(grpStrIn, xIndPool[i], kernFun, maxGrpSize, eps, tol, verbose)
			xKern = [kernFun] * group.nGrp
			r2_new, g, f = okgt_ace(X, y, yKern, xKern, group.grp, eps, tol)
			# TODO: `(r2_new, g, f)` may have already been calculated while determining `group` (two lines above).
			#		If this is the case, this line is redundent. 
			#		If a new variable is included in the existing model as a new group, then `(r2_new, g, f)` are not
			#		computed yet, the previous line is necessary. But we can add calling `okgt_ace` in `includeNewX_newGroup`.
			if verbose:
				print "  r2 = " + str(r2_new)
			if r2_new > r2:
				bestGroup = group
				r2 = r2_new
				newX = xIndPool[i]
			else:
				pass
		return (bestGroup, newX)

def fwdSelection_okgt(X, y, kernFun, maxGrpSize=3, eps=1e-3, tol=1e-6, verbose=True):
	'''
		Forward selection for OKGT group structure determination.

		**Input**
			X: numpy matrix.
			y: numpy matrix (column matrix).
			kernFun: a callable function, a return from `KOT.ConstructKernelFns`.
				
				Note: currently, we assume all groups are equipped with the same RKHS.

			maxGrpSize: integer, the maximum number of variables in a single group.
			eps: numeric, the regularization coefficient for matrix inversion.
			tol: numeric, tolerance for the convergence of OKGT ACE-type algorithm.

		**Output**
			grp: GrpStructure type, encapsulate the final determination of the group structure.
	'''
	n, p = X.shape

	# grpStr = GrpStructure(grp = []) # initialize the group structure, nothing there yet.
	# xIndPool = range(0, p)
	grpStr = GrpStructure(grp = [[0]])
	xIndPool = range(1, p)

	while xIndPool:
		if verbose:
			print "-> Current Model: " + str(grpStr.grp)
			print "-> Available X: " + str(xIndPool).strip('[]')
		grpStr, newX = includeNewX_all(grpStr, xIndPool, kernFun, maxGrpSize, eps, tol, verbose)
		xIndPool.remove(newX)

	return grpStr
