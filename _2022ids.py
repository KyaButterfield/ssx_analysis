import numpy as np
from numpy import pi, sqrt
from matplotlib import pyplot as plt
import _2022ssxreadin as ssxr
import seaborn as sns
import scipy.optimize as opt

def ids_routine(shotname):

    ids_data = ssxr.ids_data(shotname)
    data = ids_data.data
    time = ids_data.time



    """data is in ids_data.data, in 16 lists, one for each bin. In each of these is
       every data point over the recording time. """

    data_us_averaged = []
    for bin in range(16):

        # SMOOTHING DATA
        data[bin] = ssxr.lowpass_gauss_smooth(data[bin], time, sigma = .35)

        microseconds = []

        for microsecond in range(len(data[bin])//10):
            bit = np.average(data[bin][10*microsecond:10*microsecond+10])
            microseconds.append(bit)

        data_us_averaged.append(microseconds)

    bins = np.linspace(1, 16, 16)
    temps = []

    stdevs = []
    means = []

    '''
    for i in range(len(data_us_averaged[0])):
        curve = []
        for bin in range(16):
            curve.append(data_us_averaged[bin][i])

        params, err = opt.curve_fit(gaussian, bins, curve,
                                    p0 = (6, max(curve), 2.0, 0.1))

        # generate temperature based on params returned
        # add temperature to temps list

        stdevs.append(params[2])
        means.append(params[0])

    plt.title('Standard Deviations')
    plt.plot(stdevs, 'x')
    plt.show()

    plt.title('Averages')
    plt.plot(means, 'x')
    plt.show()
    '''

    Plot = False
    if Plot:
        sns.set(rc={'axes.facecolor': '#f0f0f7',
                    'axes.grid': False,
                    'text.color': '.05',
                    'ytick.left': True,
                    'axes.axisbelow': True,
                    'xtick.bottom': True})

        plot16, axs = plt.subplots(4, 4, sharex=True, sharey=True)
        plot16.set_figwidth(10)
        plot16.set_figheight(6)
        plot16.suptitle('IDS Shot %s' % shotname)

        for bin in range(16):

            if bin % 4 == 0:
                col = 0
            elif bin % 4 == 1:
                col = 1
            elif bin % 4 == 2:
                col = 2
            elif bin % 4 == 3:
                col = 3
            if bin < 4:
                row = 0
            elif bin < 8:
                row = 1
            elif bin < 12:
                row = 2
            else:
                row = 3
            axs[row, col].plot(data_us_averaged[bin])
            axs[row, col].set_title('Bin %d' % (bin + 1))
        plt.show()

    return data_us_averaged


def gaussian(bins, mean, amplitude, width, offset):
    return amplitude * (width / (2 * np.pi)) * 1 / ((bins - mean) ** 2 + (0.5 * width) ** 2) + offset



if __name__ == '__main__':
    for run in range(26,27):
        data_us_averaged = ids_routine('052417r%d' % run)
        '''
        points = []
        bins = np.linspace(1, 16, 16)
        for bin in range(len(bins)):
            for point in [102]:
                point = data_us_averaged[bin][point]
                points.append(point)
                plt.plot(bin+1, point, 'x', color = '#674ea7')
    
        params, err = opt.curve_fit(gaussian,
                                    bins,
                                    points,
                                    p0 = (6.0, max(points), 2.0, 0.1))
    
        chalk_full_bins = np.linspace(1, 16, 100)
        
        '''

        bins = np.linspace(1, 16, 16)
        chalk_full_bins = np.linspace(1, 16, 100)  # just to make a smoother Gaussian
        timept = 100

        mean      = []
        amplitude = []
        width     = []
        offset    = []

        for i in range(len(data_us_averaged[0])):
            points = []
            for bin in range(len(bins)):
                point = data_us_averaged[bin][i]
                points.append(point)  # now points is made of blocks of 16
                if i == timept:
                    plt.plot(bin + 1, point, 'x', color='#674ea7')

            params, err = opt.curve_fit(gaussian, bins, points, maxfev = 5000,
                                        p0 = (6.0, max(points), 2.0, 0.1))
            mean.append(params[0])
            amplitude.append(params[1])
            width.append(params[2])
            offset.append(params[3])

        fitted_points = gaussian(chalk_full_bins, mean[timept], amplitude[timept],
                                 width[timept], offset[timept])

        plt.plot(chalk_full_bins, fitted_points, ':', color='#C46161')
        plt.show()

        plt.plot(mean)
        plt.title('Mean')
        plt.show()
        plt.plot(width)
        plt.title('Width')
        plt.show()
        plt.plot(amplitude)
        plt.title('Amplitude')
        plt.show()
        print(amplitude.index(max(amplitude)))







    # need to generate an array of points for each index of the array (range 420)
    # then we need to do a gaussian fit on each point and return the parameters to the respective arrays
    # this way param_0[100], param_1[100], and param_2[100] can all be parameters that correspond to the same
    # index of 420.


    # we do some relating of the gaussian function to the maxwellina fucntion. This tells us that the temperatire
    # is related to the standard deviation and that