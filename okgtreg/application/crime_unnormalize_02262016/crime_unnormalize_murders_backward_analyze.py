__author__ = 'panc'

'''
Process the pkl files in the "crime_unnormalize_murders_backward" folder.
'''

import glob
import pickle
import matplotlib.pyplot as plt


class MurderBackwardFolderProcessor(object):
    def __init__(self, print_file_path=False):
        crime_folder = "okgtreg/application/crime_unnormalize_02262016"
        pkl_folder = "crime_unnormalize_murders_backward"
        self.res_list = []
        for i, file_path in enumerate(glob.glob(crime_folder + '/' + pkl_folder + '/*.pkl')):
            if print_file_path:
                print "[%d]" % (i + 1), file_path
            with open(file_path, 'rb') as f:
                self.res_list.append(pickle.load(f))

    def testError(self):
        # (mu, alpha) corresponding to the test error
        param_error_dict = {}
        for res in self.res_list:
            for k, v in res['test'].items():
                param_error_dict[k] = v
        return param_error_dict

    def minTestError(self):
        return min(self.testError().values())

    def groupStructureFromTrain(self):
        param_gstruct_dict = {}
        for res in self.res_list:
            for k, v in res['train'].iteritems():
                param_gstruct_dict[k] = v['group'].__str__()
        return param_gstruct_dict

    def groupStructureWithMinTestError(self):
        gstruct_list = []
        for k, v in self.testError().items():
            if v == self.minTestError():
                gstruct_list.append(self.groupStructureFromTrain()[k])
        return set(gstruct_list)


if __name__ == '__main__':
    ########################
    # Explore one pkl file #
    ########################
    crime_folder = "okgtreg/application/crime_unnormalize_02262016"
    pkl_folder = "crime_unnormalize_murders_backward"

    sample_pkl_file = "crime_unnormalize_murders_backward_mu1-alpha1-20160228-230835-006836.pkl"
    with open(crime_folder + '/' + pkl_folder + '/' + sample_pkl_file, 'rb') as f:
        res = pickle.load(f)

    res.keys()
    res['train']  # (mu, alpha) : dict_returned_by_backwardPartition
    res['test']

    ########################################
    # Process all .pkl files in the folder #
    ########################################
    FolderProcessor = MurderBackwardFolderProcessor(True)

    # test errors
    test_errors_dict = FolderProcessor.testError()
    sorted_errors = [v for k, v in sorted(test_errors_dict.items())]
    plt.scatter(range(1, 51), sorted_errors)
    plt.plot(range(1, 51), sorted_errors, 'k')

    # group structures from training phase
    gstruct_dict = FolderProcessor.groupStructureFromTrain()
    ## unique group structures
    set(gstruct_dict.values())

    # Best group structure from validation
    best_gstruct_str = list(FolderProcessor.groupStructureWithMinTestError())[0]

    ############################################
    # Estimation with the best group structure #
    ############################################
    import numpy as np
    from okgtreg.Kernel import Kernel
    from okgtreg.application.crime_unnormalize_02262016.utility import readCleanDataForMurders
    from okgtreg.OKGTReg import OKGTReg2
    from okgtreg.Group import Group

    data = readCleanDataForMurders()
    kernel = Kernel('gaussian', sigma=0.5)

    # okgt = OKGTReg2(data, kernel=kernel, group=Group(group_struct_string=best_gstruct_str))
    # fit = okgt._train_lr(data.y)
    # print fit['r2']
    #
    # # Plot
    # j=22; plt.scatter(data.X[:,j], fit['f'][:,j], linewidth=0, s=15)

    # Normalize data first
    for i in range(data.p):
        data.X[:, i] = data.X[:, i] - np.mean(data.X[:, i])
        data.X[:, i] = data.X[:, i] / np.std(data.X[:, i])

    data.y = data.y - np.mean(data.y)
    data.y = data.y / np.std(data.y)

    # Fit the normalized data
    okgt_normalize = OKGTReg2(data,
                              kernel=kernel,
                              group=Group(group_struct_string=best_gstruct_str))
    fit_normalize = okgt_normalize._train_lr(data.y)
    print fit_normalize['r2']

    # j = 5; plt.scatter(data.X[:, j], fit_normalize['f'][:, j], linewidth=0, s=15)

    # ===
    # Plot all transformations
    # ===
    ncol = 6
    nrow = 4
    fig, axarr = plt.subplots(nrow, ncol)
    for i in range(nrow):
        for j in range(ncol):
            idx = i * ncol + j
            print "plot transformation", idx+1
            if idx < 23:
                # plot transformed data
                axarr[i, j].scatter(data.X[:, idx], fit_normalize['f'][:, idx], linewidth=0, s=15)
                axarr[i, j].set_title(''.join([r'$f($', data.xnames[idx], r'$)$']), fontsize=10)
                # plot function curve
                xnew = np.linspace(min(data.X[:, idx]), max(data.X[:, idx]), 100)[:, np.newaxis]
                xnewfit = fit_normalize['f_call'][idx + 1](xnew)
                axarr[i, j].plot(xnew, xnewfit, color='red', linewidth=1)
                axarr[i,j].tick_params(labelsize=10)
            else:
                break
    fig.tight_layout(pad=0, w_pad=0, h_pad=0)
    fig.show()

    # ===
    # Selected plots
    # ===
    idx_list = [4, 12]
    fig, axarr = plt.subplots(1, 2, figsize=(6, 2.5), sharex=False, sharey=False)
    for i in range(2):
        idx = idx_list[i]
        # plot transformation
        axarr[i].scatter(data.X[:, idx], fit_normalize['f'][:, idx], linewidth=0, s=15)
        axarr[i].set_title(''.join([r'$f($', data.xnames[idx], r'$)$']))
        # === plot function curve ===
        xnew = np.linspace(min(data.X[:, idx]), max(data.X[:, idx]), 100)[:, np.newaxis]
        xnewfit = fit_normalize['f_call'][idx + 1](xnew)
        axarr[i].plot(xnew, xnewfit, color='red', linewidth=1)
        # === add text box ===
        # Reference: placing text boxes on plots
        #   http://matplotlib.org/users/recipes.html#placing-text-boxes
        # Reference: Text properties and layout
        #   http://matplotlib.org/users/text_props.html
        # props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        # axarr[i].text(0.05, 0.15, data.xnames[idx], transform=axarr[i].transAxes,
        #               fontsize=14, verticalalignment='top', bbox=props)
        axarr[i].tick_params(labelsize=10)
    fig.tight_layout()