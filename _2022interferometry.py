import _2022ssxreadin as ssxr
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

global scope_globe
scope_globe = 1
global channel_a_globe
channel_a_globe = 3
global channel_b_globe
channel_b_globe = 4

def hene_calibration(calib_shotname, scope=scope_globe, channel_a=channel_a_globe,
                     channel_b=channel_b_globe,sigma=0.01,
                     Plotting_Raw=False,Plotting_Phasecorrected=False,
                     verbose=True, saveFigs=False):
    """
    hene_calibration reads data from the appropriate scope to calibrate
    the interferometer. Ultimately the function returns a list of 3 pieces
    of information, which are detailed below.

    :param: calib_shotname: MMDDYYr# (str) shotname of the calibration shot
    :param: scope(str): the ocillisope used for interferometry (2022: scope 1)
    :param: Plotting_Raw(bool): if True, shows plot of the raw data
    :param: Plotting_Phasecorrected(bool): if True, shows plot of corrected data
    :param: verbose(bool): if True, prints calib_info once produced
    :param: saveFigs(bool): if True, saves the figures that are shown

    :return: calib_info(list) -- contains the following:

            env(list): 2 components.
            the envelope is the maximum signal reached by the calibration
            shot minus the minimum signal. Because both channels can be
            collecting different amounts of light (depending on the
            orientation of the photodetectors), the envelope is bound to be
            different for each channel. The first number refers to the env of
            the channel we call sin and the second to cos.

            offset(list): 2 components
            using a similar organization scheme to the env, the offset
            is the minimum voltage of each channel.

            phase_error(float):
            imperfections in setting up the interferometer will result in some
            phase error between the two signals. Phase error refers to the
            deviation from pi/2 in radians of the signal's phases.
        """

    scope_data = ssxr.scope_data(calib_shotname, scope)

    sin = eval('scope_data.ch' + str(channel_a) + '[50:]')
    cos = eval('scope_data.ch' + str(channel_b) + '[50:]')
    time = scope_data.time[50:]

    sns.set(rc={'font.family': ['Avenir', 'Trattatelo'],
                'axes.facecolor': '#f0f0f7',
                'axes.grid': False,
                'text.color': '.05',
                'ytick.left': True,
                'xtick.bottom': True,
                'axes.axisbelow': True
                })
    if Plotting_Raw:
        print("plotting raw calibration signals for %s..." % scope_data.shotname)
        plt.plot(time, sin, '#65b9cb', label="sine")
        plt.plot(time, cos, '#296082', label="cosine")
        plt.legend(loc='lower right')
        plt.title("Raw Calibration Shot: %s" % scope_data.shotname)
        plt.xlabel("Time ($\mu s$)")
        plt.ylabel("Voltage ($V$)")
        plt.title("Raw Calibration: %s" % scope_data.shotname)
        if saveFigs:
            plt.savefig('Raw_%s'% scope_data.shotname, dpi=700)
        plt.show()

    sin = ssxr.lowpass_gauss_smooth(sin, time, sigma)  # a typical sigma is ~ 0.01
    cos = ssxr.lowpass_gauss_smooth(cos, time, sigma)

    env = [max(sin) - min(sin), max(cos) - min(cos)]
    offset = [min(sin), min(cos)]

    cos = (cos - offset[1]) / env[1]
    sin = (sin - offset[0]) / env[0]

    x_ym_list = []  # array of  xvals around the ymax pt
    weights_list = []
    for i in range(len(cos)):
        if abs(sin[i]) > 0.99:
            x_ym_list.append(cos[i])
            weights_list.append(np.abs((0.99 - sin[i]) / 0.01))

    x_ym = np.dot(x_ym_list, weights_list) / sum(weights_list)
    # use cosine when sin = 1 to find phase error
    phase_error = np.arccos((x_ym - 0.5) * 2) - np.pi / 2

    cos_shifted = ((cos * 2 - 1) / np.cos(phase_error) + (sin * 2 - 1) * np.tan(phase_error)) * 0.5 + 0.5
    # Justification for this cos_shifted in the documentation for the density calculation

    if Plotting_Phasecorrected:
        print("plotting normalized calibration signals for %s, shifted to pi/2 out of phase..." % scope_data.shotname)
        plt.plot(time, cos, '#296082', label="Cosine, Original")
        plt.plot(time, sin, '#65b9cb', label="Sin")
        plt.plot(time, cos_shifted, '#5835b1', label="Cosine, Shifted")
        plt.legend(loc='lower right')
        plt.title("Sine and Cosine, Shifting to be $\\frac{\pi}{2}$ out of phase")
        plt.xlabel("Time ($\mu s$)")
        plt.ylabel("Normalized Voltage (V)")
        plt.title("Phase Corrected Calibration: %s" %scope_data.shotname)
        if saveFigs:
            plt.savefig('Corrected_%s' % scope_data.shotname, dpi=700)
        plt.show()

    calib_info = [env, offset, phase_error]

    if verbose:
        print()
        print('--------------------------------------------------------------------')
        print('* CALIBRATION INFO (rounded): %s \n' % scope_data.shotname)
        print('Envelope Sine: \t', round(env[0], 5), 'Volts\tEnvelope Cosine:\t', round(env[1], 5), 'Volts')
        print('Offset Sine: \t', round(offset[0], 5), 'Volts\tOffset Cosine: \t\t', round(offset[1], 5), 'Volts')
        print('\nPhase Error: ', round(phase_error, 5), 'Radians')
        print('--------------------------------------------------------------------')
        print()

    return calib_info

def hene_density_routine(shotname, calib_info, scope=scope_globe,
                         channel_a=channel_a_globe, channel_b=channel_b_globe,
                         sigma=.75, path=0.16, linfit=True, showPlot=True,
                         savePlot=False):
    """

    :param shotname:
    :param calib_info:
    :param scope:
    :param channel_a:
    :param channel_b:
    :param sigma:
    :param path:
    :param linfit:
    :param showPlot:
    :param savePlot:
    :return:
    """

    print("analyzing %s..." % shotname)

    scope_data = ssxr.scope_data(shotname, scope)
    scope_data.sigma = sigma

    # removing the first and potentially erroneous bits of data
    sin = eval('scope_data.ch' + str(channel_a) + '[50:]')
    cos = eval('scope_data.ch' + str(channel_b) + '[50:]')
    time = scope_data.time[50:]

    cos = ssxr.lowpass_gauss_smooth(cos, time, sigma)
    sin = ssxr.lowpass_gauss_smooth(sin, time, sigma)

    env, offset, phase_error = calib_info[0], calib_info[1], calib_info[2]

    sin = (sin - offset[0]) / env[0]
    cos = (cos - offset[1]) / env[1]

    # Correction for phase error (from calibration):
    cos = ((cos * 2 - 1) / np.cos(phase_error) + (sin * 2 - 1) * np.tan(phase_error)) * 0.5 + 0.5
    cos = 2 * cos - 1
    sin = 2 * sin - 1

    """Initialize tri-plot here: """
    sns.set(rc={'font.family': ['Avenir', 'Trattatelo'],
                'axes.facecolor': '#f0f0f7',
                'axes.grid': False,
                'text.color': '.05',
                'ytick.left': True,
                'axes.axisbelow': True
                })

    tridensity, (phi_with_jumps, phi_sans_jumps, density_plot) = plt.subplots(3, sharex=True)
    density_plot.set_xlabel("time ($\mu s$)")
    tridensity.set_figwidth(10)
    tridensity.set_figheight(6)

    phi = np.arctan(cos / sin)
    # seting phi_0 to zero
    phi = phi - np.mean(phi[:10])

    """Adding phi as first subplot"""
    phi_with_jumps.plot(time, np.zeros(len(time)), 'grey', linewidth=0.8)
    phi_with_jumps.plot(time, phi,label='Jumps Included', color='#44668f')

    # take our jumps in the arctan function:
    for i in range(len(phi)):
        if i < len(phi) - 1:
            if phi[i + 1] - phi[i] > 3:
                phi[i + 1:] = phi[i + 1:] - np.pi
            if phi[i + 1] - phi[i] < -3:
                phi[i + 1:] = phi[i + 1:] + np.pi

    phi_sans_jumps.plot(time, np.zeros(len(time)), 'grey', linewidth=0.8)
    phi_sans_jumps.plot(time, phi,label='Jumps Removed',color='#44668f')

    if linfit:
        for i in range(len(time)):
            if time[i] > 30:
                time_index = i
                break

        ti = time[:time_index]
        fit = np.polyfit(ti, ssxr.convolution_smooth(phi, length=15, iterations=10)[:time_index], deg=1)
        mechanical_phi = time * fit[0] + fit[1]

        """Adding linear fit (from t_0 to t=25) to 2nd subplot"""
        phi_sans_jumps.plot(time, mechanical_phi, 'r:', label='Linear Fit')

        phi = phi - mechanical_phi

    lamb = 632.8e-9  # wavelength
    e = 1.6022e-19  # elementary charge
    c = 3e8  # speed o' light in a vacuum
    eps0 = 8.854e-12  # permittivity of free space in a vacuum
    L = path  # length of scene beam that passes through plasma
    me = 9.1093e-31  # electron mass

    """See derivation of density in documentation, attached."""
    density = (4 * np.pi * c ** 2 * eps0 * me * phi) / (lamb * e ** 2 * L)

    density_plot.plot(time, np.zeros(len(time)), 'grey', linewidth=0.8)
    density_plot.plot(time, density,color='#44668f')
    density_plot.set_ylabel('Density')
    tridensity.suptitle('%s: From Phase to Density' %shotname,fontweight="bold")
    if savePlot:
        tridensity.savefig('%sDensityProcess'%scope_data.shotname,dpi=500)
    if showPlot:
        tridensity.show()

    scope_data.densityplot = tridensity
    scope_data.cos = cos
    scope_data.sin = sin
    scope_data.time = time
    scope_data.phi = phi
    scope_data.density = density

    return scope_data

def data_finder(shotname):
    """
    The set() command can be used to collapse a list into a list of only unique
    values. If the length of this collapsed list is > 10, we know we have data
    rather than bit noise.

    This function will plot
    """

    for scope_number in range(1, 4):
        scope_data = ssxr.scope_data(shotname, scope=scope_number)
        scope_data.fetch_scope_data()
        sig1, sig2, sig3, sig4, time = scope_data.ch1, scope_data.ch2, scope_data.ch3, scope_data.ch4, scope_data.time

        Plot = False  # default
        if len(set(sig1)) > 10:
            plt.plot(time, sig1, label='ch1')
            Plot = True
        if len(set(sig2)) > 10:
            plt.plot(time, sig2, label='ch2')
            Plot = True
        if len(set(sig3)) > 10:
            plt.plot(time, sig3, label='ch3')
            Plot = True
        if len(set(sig4)) > 10:
            plt.plot(time, sig4, label='ch4')
            Plot = True

        if not Plot:
            continue

        plt.title("Run %s, Scope %s" % (shotname, scope_number))
        plt.legend(loc='lower right')
        plt.show()

    return


if __name__ == "__main__":
    """This only runs if you're running this file, but not if you are importing it somewhere else! 
       It can be handy while you're testing code."""

    calib_info = hene_calibration('060222r4', Plotting_Raw=False)
    hene_density_routine('052417r6', calib_info, showPlot= True)



