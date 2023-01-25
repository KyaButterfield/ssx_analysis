import numpy as np
from numpy import pi, sqrt

from matplotlib import pyplot as plt
import _2022ssxreadin as ssxr
from scipy.integrate import cumulative_trapezoid
import seaborn as sns
import scipy as sp

global d_ax
d_ax = 0.03625  # axial distance between probe stalks 1 & 2

global d_r
d_r = 0.038  # radial distance between probes 1 & 2, 2 & 3, et cetera

sns.set(rc={'axes.facecolor': '#ffffff',
            'axes.grid': False,
            'text.color': '.05',
            'ytick.left': True,
            'axes.axisbelow': True})


def helmholtz_current():

    num_measurements = 8192  # same for each channel
    frequency = 65e6  # Hz, 2022
    total_time = num_measurements / frequency  # this is total time in seconds
    total_time *= 1e6  # now total time is in microseconds (us)

    dtacq_time = np.linspace(2, total_time + 2, num_measurements)

    # TODO: figure out current, scale (run specific)
    current = []
    # information about current is located on scope 3, channel 2, as of 2022
    for run in [20, 22, 24]:
        shotname = '102021r' + str(run)
        current_from_scope  = ssxr.scope_data(shotname, scope=3)
        calibration_current = ssxr.adjust(current_from_scope.ch2, current_from_scope.time, dtacq_time)
        calibration_current = np.mean(calibration_current[-200:-30])
        current.append(calibration_current)

    return current



def B_calibration(shotdate=102021, shots=[20, 22, 25]):
    """
    This function takes the date that calibration shots were taken and the
    shot numbers for shots in the axial, angular, and axial directions, and
    produces calibration matrices for each of the 16 probes.

    :param shotdate: date on which calibration was taken. in 2022 we are using
        calibration shots from 102021.
    :param shots: list [shot r, shot t, shot z]

    :return:
    """

    B_calib = []
    I = []

    # read in data for each of the shots, and collect values for the current
    # through the coil over time, which is in scope 3 ch 2 as of 2022.
    for shotnumber in shots:
        shotname = '%dr%d' % (shotdate, shotnumber)
        mag_data = B_routine(shotname, Calibrate=False)
        B_calib.append(mag_data.B)
        current = ssxr.scope_data(shotname, scope=3)
        I.append(list(current.ch2))
        # this is the current through the Helmholtz coil during this time

    dtacq_time, scope_time = mag_data.time, current.time
    for shot in range(3):
        I[shot] = ssxr.adjust(I[shot], scope_time, dtacq_time)
        # adjust() accounts for the difference in timebases between the
        # oscilloscopes and the dtacq

    calib_matrices = []

    """ Now, B_calib is a 3 element list, and each list contains 16
        lists representing all 16 probes. I is a list of 3 lists which
        are the current through the coil over time for each of the
        3 shots with which we concern ourselves. 

        The for loop just below runs 16 times, and adds one calibration
        matrix to the calib_matrices list. In order 1-16, obviously."""

    for probe in range(16):  # for each probe...

        list_measuredB, list_idealB = [], []  # resetting

        for shot in range(3):  # for each of the three shots we'll use for calibration...

            """ This little if-block is deciding which direction the Helmholtz
                coil is oriented.the way the shots list comes in, the runs are
                in the order of rtz, so this block parses that into axes for
                the Helmholtz analytical calculation. It is set up now(2022)
                for the shots taken on 102021. """

            if shot == 0:
                direction = 'north-south'
            elif shot == 1:
                direction = 'up-down'
            elif shot == 2:
                if shots[0] in [24, 25]:
                    direction = 'east-west-Door'
                elif shots[0] in [26, 27]:
                    direction = 'east-west-Window'

            measuredB = B_calib[shot][probe]  # at every time point
            current = I[shot]

            """ Here, 'measuredB' is a 3-element list for a specific probe, 
                and each element is a list of B values at every time step for
                a particular direction (r, theta, z). The 'current' list
                is the current values over time. Now we take a specific time. 
                The timerange should be around the place where the change
                in I is the greatest, and can be edited. 

                For the 102021 calibration shots, there are 8191 data points. """

            timerange = '[4000:4020]'  # variable 2022!

            avgBr = np.mean(eval('measuredB[0]' + timerange))  # r component
            avgBt = np.mean(eval('measuredB[1]' + timerange))  # theta component
            avgBz = np.mean(eval('measuredB[2]' + timerange))  # z component
            avgcurrent = np.mean(eval('current' + timerange))

            measuredB = np.array([avgBr, avgBt, avgBz])
            # now this is averaged around a specific time point

            idealB = kya_helmholtz(direction, pos[probe], avgcurrent)
            """ Using kya_helmholtz() to calculate the actual magnetic field
                based on the current and probe position. This will be
                compared to the measured values. """

            list_measuredB.append(measuredB)
            list_idealB.append(idealB)

        """ Back in the outer for loop, which iterates once for each probe. 
            We'll now perform the linear algebra outlined in the documentation
            in order to compute the calibration matrix for each probe. """

        Bm = np.array(list_measuredB)  # B_measured matrix
        Bh = np.array(list_idealB)  # B_helmholtz matrix

        Bm = np.matrix.transpose(Bm)
        Bh = np.matrix.transpose(Bh)
        """ Now Bm and Bh are the matrices to be operated on like outlined
            in the documentation."""

        Bm = np.linalg.inv(Bm)
        C = np.matmul(Bh, Bm)

        """ With this, C is the calibration matrix for this probe, so it will
            be added to the list of calibration matrices and we'll go back
            to the top of the loop to catch the next probe! """

        calib_matrices.append(C)

    return calib_matrices


# TODO: comment for our matrix generators.
def B_measured_matrices():
    # RADIAL HELMHOLTZ SHOTS
    B_20 = []
    run = 20
    mag_data = B_routine('102021r%d' % run, Calibrate=False)
    for axis in range(3):
        for probe in range(16):
            B_20.append(np.average(mag_data.B[probe][axis][-200:-30]))


    B_21 = []
    run = 21
    mag_data = B_routine('102021r%d' % run, Calibrate=False)
    for axis in range(3):
        for probe in range(16):
            B_21.append(np.average(mag_data.B[probe][axis][-200:-30]))

    B_Rad_average = (np.array(B_20) + np.array(B_21)) / 2
    B_Rad_r = B_Rad_average[:16]
    B_Rad_t = B_Rad_average[16:32]
    B_Rad_z = B_Rad_average[32:48]

    # ANGULAR HELMHOLTZ SHOTS
    B_22 = []
    run = 22
    mag_data = B_routine('102021r%d' % run, Calibrate=False)
    for axis in range(3):
        for probe in range(16):
            B_22.append(np.average(mag_data.B[probe][axis][-200:-30]))

    B_23 = []

    run = 23
    mag_data = B_routine('102021r%d' % run, Calibrate=False)
    for axis in range(3):
        for probe in range(16):
            B_23.append(np.average(mag_data.B[probe][axis][-200:-30]))

    B_Ang_average = (np.array(B_22) + np.array(B_23)) / 2
    B_Ang_r = B_Ang_average[:16]
    B_Ang_t = B_Ang_average[16:32]
    B_Ang_z = B_Ang_average[32:48]

    # AXIAL HELMHOLTZ SHOTS (first eight probe locations)
    B_24 = []
    run = 24
    mag_data = B_routine('102021r%d' % run, Calibrate=False)
    for axis in range(3):
        for probe in range(16):
            B_24.append(np.average(mag_data.B[probe][axis][-200:-30]))

    B_25 = []
    run = 25
    mag_data = B_routine('102021r%d' % run, Calibrate=False)
    for axis in range(3):
        for probe in range(16):
            B_25.append(np.average(mag_data.B[probe][axis][-200:-30]))

    B_Axi1_average = (np.array(B_24) + np.array(B_25)) / 2
    '''
    # AXIAL HELMHOLTZ SHOTS (second eight probe locations)
    B_26 = []
    run = 26
    mag_data = B_routine('102021r%d' % run, Calibrate=False)
    for axis in range(3):
        for probe in range(8, 16):
            B_26.append(np.average(mag_data.B[probe][axis][-200:-30]))

    B_27 = []
    run = 27
    mag_data = B_routine('102021r%d' % run, Calibrate=False)
    for axis in range(3):
        for probe in range(8, 16):
            B_27.append(np.average(mag_data.B[probe][axis][-200:-30]))

    B_Axi2_average = (np.array(B_26) + np.array(B_27)) / 2
    
    B_Axi_r = np.concatenate([B_Axi1_average[:8], B_Axi2_average[:8]])
    B_Axi_t = np.concatenate([B_Axi1_average[8:16], B_Axi2_average[8:16]])
    B_Axi_z = np.concatenate([B_Axi1_average[16:], B_Axi2_average[16:]])
    #'''

    B_Axi_r = B_Axi1_average[:16]
    B_Axi_t = B_Axi1_average[16:32]
    B_Axi_z = B_Axi1_average[32:]

    B_Rad_Vect = []
    B_Ang_Vect = []
    B_Axi_Vect = []

    for probe in range(16):
        B_Rad_Vect.append([B_Rad_r[probe], B_Rad_t[probe], B_Rad_z[probe]])
        B_Ang_Vect.append([B_Ang_r[probe], B_Ang_t[probe], B_Ang_z[probe]])
        B_Axi_Vect.append([B_Axi_r[probe], B_Axi_t[probe], B_Axi_z[probe]])

    measured_matrices = []
    for probe in range(16):
        B_mtx_m = np.array([B_Rad_Vect[probe], B_Ang_Vect[probe], B_Axi_Vect[probe]])
        B_mtx_m = np.matrix.transpose(B_mtx_m)
        measured_matrices.append(B_mtx_m)
    for probe in range (1):
        print('\nprobe %d measured matrix:\n' % (probe+1), str(B_mtx_m))

    return measured_matrices


def B_helmholtz_matrices():

    current = helmholtz_current()

    helmholtz_matrices = []
    for probe in range(16):
        Radshot = kya_helmholtz('north-south',      pos[probe], -current[0])
        Angshot = kya_helmholtz('up-down',          pos[probe], -current[1])
        Axishot = kya_helmholtz('east-west-Window', pos[probe], current[2])

        B_mtx_helm = np.array([Radshot, Angshot, Axishot])
        B_mtx_helm = np.matrix.transpose(B_mtx_helm)

        helmholtz_matrices.append(B_mtx_helm)

    for probe in range (1):
        print('\nprobe %d helmholtz matrix:\n' % (probe+1), str(B_mtx_helm))

    return helmholtz_matrices


def Calibration_matrices():
    measured_matrices = B_measured_matrices()
    helmholtz_matrices = B_helmholtz_matrices()

    Calib_matrices = []
    for probe in range(16):
        inverted = np.linalg.inv(measured_matrices[probe])
        calib = np.matmul(helmholtz_matrices[probe], inverted)

        Calib_matrices.append(calib)

    for probe in range (1):
        print('\nprobe %d calibration matrix:\n' % (probe+1), str(Calib_matrices[probe]))

    return Calib_matrices


def B_routine(shotname, Calibrate=True):
    """
    This function uses the mag_data() class in ssxreadin.py to read in the
    data for a given shot and uses the cumulative_trapezoid integration
    method from scipy to calculate magnetic field from Emf.

    :param shotname: (str) MMDDYYr# ion-style shotname
    :param Calibrate: (bool) if True, uses calibration matrix to calibrate
            B to Helmholtz shots. Otherwise, nothing happens
    :return:
        mag_data, which has mag_data.B, where the magnetic field data is stored. """

    positions()
    mag_data = ssxr.mag_data(shotname)  # (mag_data is a class in _2022ssxreadin.py)
    V = mag_data.data  # (.data is an attribute of mag_data

    """ At this point, V is a list of 3 arrays, each containing 16
        arrays. The three outer arrays correspond to three spacial directions:
        radial, angular, and axial (in that order: r, theta, z). The 16 arrays 
        within them correspond to each probe, 1-16. Note that indexing must refer
        to the number below whichever component we're accessing; for instance,
        the theta component of probe 5 is: mag_data.data[1][4]. That component
        and all of the other ones in similar positions are arrays of data. 

        The data is voltage measured across each coil contained within the 
        probe. Faraday's law tells us that Emf = dB/dt, so we'll integrate Emf to 
        find B at each time point, at each probe, in 3 spacial directions. """

    time = mag_data.time  # for ease :)
    for t in time:
        if t >= 0:
            t0_index = list(time).index(t)
            # will report the index of t which crosses t = 0
            t0_index = 15
            break

    r = [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []]
    theta = [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []]
    z = [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []]

    B = [r, theta, z]

    """ Given this empty B list, the following 'for' loop does the integration
        to get us from Emf to magnetic field, and fills this B list with
         values of magnetic field. 

         Before integration, any offset, found before t=0, is subtracted off. 
         The cumulative_trapazoid() function returns one less data point than 
         is receives in input, so the time list needs to be shortened by 1. """

    for axis in range(3):
        for probe in range(16):
            to_analyze = ssxr.lowpass_gauss_smooth(V[axis][probe][5:], time, sigma=.2)  # with smoothing
            offset = float(np.average(to_analyze[:25]))
            to_analyze = to_analyze - offset
            B[axis][probe] = cumulative_trapezoid(y=to_analyze, x=time[5:])

    mag_data.time = time[5:-1]
    """ Now B has 3 components, each representing an orthogonal direction. For our 
        purposes going forward, it is preferable to have B organized by
        probe instead of by direction, so this next block does that reorganization."""

    B_reorganized = []
    for i in range(16):
        B_reorganized.append(np.array([B[0][i], B[1][i], B[2][i]]))
    B = B_reorganized

    """ Now B is a 16 element list. Each element represents one probe, 
        and has 3 components, each representing one direction, in the 
        order (r, theta, z). It is so far uncalibrated, so we'll save this
        uncalibrated data to mag_data, and then go ahead and calibrate if
        Calibrate=True. When B_routine() is called from B_calibration(), 
        Calibrate=False, and the if statement is skipped, so the 
        uncalibrated data is what is sent back to B_calibration(). 

        Calibration is matrix multiplication with previously computed
        matrices, which were computed using B_calibration(). np.matmul()
        is matrix multiplication. """

    mag_data.uncalibB = B

    if Calibrate:
        Calib_matrices = Calibration_matrices()
        # this calls the calibration function (B_calibration) above
        for probe in range(16):
            B[probe] = np.matmul(Calib_matrices[probe], B[probe])
        for probe in range(1):
            print('hello here i am')
            print(B[probe])

    mag_data.B = B

    """ And then we save B to mag_data and return mag_data! """

    return mag_data


def positions():
    p1 = [0 * d_r, 0, -2 * d_ax]
    p2 = [1 * d_r, 0, -2 * d_ax]
    p3 = [2 * d_r, 0, -2 * d_ax]
    p4 = [3 * d_r, 0, -2 * d_ax]

    p5 = [0 * d_r, 0, -1 * d_ax]
    p6 = [1 * d_r, 0, -1 * d_ax]
    p7 = [2 * d_r, 0, -1 * d_ax]
    p8 = [3 * d_r, 0, -1 * d_ax]

    p9 = [0 * d_r, 0, 1 * d_ax]
    p10 = [1 * d_r, 0, 1 * d_ax]
    p11 = [2 * d_r, 0, 1 * d_ax]
    p12 = [3 * d_r, 0, 1 * d_ax]

    p13 = [0 * d_r, 0, 2 * d_ax]
    p14 = [1 * d_r, 0, 2 * d_ax]
    p15 = [2 * d_r, 0, 2 * d_ax]
    p16 = [3 * d_r, 0, 2 * d_ax]

    global pos
    pos = np.array([p1, p2, p3, p4,
                    p5, p6, p7, p8,
                    p9, p10, p11, p12,
                    p13, p14, p15, p16])
    return


def vector_plot(mag_data, field=False, scalor=1e7, o=[0, 0, 0], savePlot=False, opacity=1):
    """At specific time."""

    d_ax = 0.03625  # axial distance between probe stalks 1 & 2
    d_r = 0.038

    sns.set(rc={'axes.facecolor': '#ffffff',
                'axes.grid': False,
                'text.color': '.05',
                'ytick.left': True,
                'axes.axisbelow': True})
    ax = plt.axes()
    plt.xlabel('radial (meters)')
    plt.ylabel('axial (meters)')

    positions()
    if not field:
        B = mag_data.B
        plt.title(mag_data.shotname + " Bird's Eye")
    else:
        B = mag_data
        plt.title('Helmholtz Shot')

    # initiate plot:

    plt.plot([0, 4 * 0.038], [0.03625, 0.03625], 'r:')
    plt.plot([0, 4 * 0.038], [-0.03625, -0.03625], 'r:')
    plt.plot([0, 4 * 0.038], [2 * 0.03625, 2 * 0.03625], 'r:')
    plt.plot([0, 4 * 0.038], [-2 * 0.03625, -2 * 0.03625], 'r:')

    # time_range = '[-1:-20:-1]'
    time_range = '[4000:4020]'
    for probe in range(16):
        if (probe) == 5:
            # continue
            pass
        plt.plot(pos[probe][0], pos[probe][2], 'ro')

        Br = B[probe][0]  # isolated r component
        Bz = B[probe][2]

        if not field:
            Br = (np.mean(eval('Br' + time_range)))  # average around specific time
            Bz = (np.mean(eval('Bz' + time_range)))

        x1 = pos[probe][0]
        y1 = pos[probe][2]
        x2 = Br * scalor
        y2 = Bz * scalor

        ax.arrow(x1, y1, x2, y2, color='#44668f', alpha=opacity)

    plt.plot(o[0], o[2], 'x', color='red', markersize=10)
    if savePlot:
        plt.savefig('plot_%s' % mag_data.shotname, dpi=700)
        pass
    plt.show()
    return


def kya_helmholtz(direction, rtz, current, a=.1584, turns=2, ):
    """
    TODO: finish documentation here
    :param direction:
    :param rtz: location in SSX coords where Helmholtz field is to be calculated
    :param current:
    :param a: radius of the coils
    :param turns: number of turns on coil
    :return:
    """
    if current == 0:
        print("Your Helmholtz calculation used zero current.")
        return [0, 0, 0]

    if a == 0:
        to_print = "Your Helmholtz calculation didn't work. "
        to_print += "You specified a radius of zero for the coils, "
        to_print += "which is nonsensical."
        print(to_print)
        return None

    """Depending on the direction the Helmholtz coil was facing and the
       origin around which it was centered, adjustments need to be made. In 
       2022 we are using calibration shots in three different directions, 
       around four different origins. Because the Helmholtz coil didn't fit
       in the z direction centered at the center of the probes, there
       were separate shots around the left and right probes. The if
       statements here account for these differences, and their counter-
       parts at the end of this function put the resultant field back
       into the coordinate system we use in SSX. """

    if direction == 'east-west-Door':
        origin = [1.5 * d_r, 0, 2 * d_ax]  # positive z component for origin
        r = rtz[0] - origin[0]
        z = rtz[2] - origin[2]
        I = -current * turns  # checked. good.

    elif direction == 'east-west-Window':
        origin = [1.5 * d_r, 0, -2 * d_ax]  # negative z component for origin
        r = rtz[0] - origin[0]
        z = rtz[2] - origin[2]
        I = -current * turns  # also checked. also good
        """For east-west shots, the axis of the Helmholtz coil is the
           same as the axis of SSX, so no changes need to be made."""

    elif direction == 'north-south':
        origin = [1.5 * d_r, 0, 0]
        # list(rtz).reverse()
        r = rtz[0] - origin[0]
        z = rtz[2] - origin[2]
        I = current * turns
        """For north-west shots, z and r get swapped, thus reverse()."""

    elif direction == 'up-down':
        origin = [1.5 * d_r, 0, 0]
        r = rtz[0] - origin[0]
        z = rtz[2] - origin[2]
        r = sqrt(r ** 2 + z ** 2)  # sqrt(x^2 + y^2)
        theta = np.arctan(z / r)  # arctan(y/x)
        z = 0
        I = -current * turns

    if r < 0:
        NegativeRadius = True  # this gets accounted for towards
        r = abs(r)  # the end of the function
    else:
        NegativeRadius = False

    uo = 4E-7 * pi
    # uo is the permeability constant, with units of H/m
    Bo = uo * I / (2 * a)
    # Bo is the B-field at the origin(center) of the Helmholtz coil.
    # This is a physics 8 homework problem.

    alpha = r / a
    beta = z / a
    if r != 0:
        gamma = z / r
    Q = (1 + alpha) ** 2 + beta ** 2
    k = sqrt(4 * alpha / Q)

    K = sp.special.ellipk(k ** 2)
    E = sp.special.ellipe(k ** 2)

    # first find radial component
    if (r == 0) or (z == 0):
        Br = 0  # on axis, there is no radial component of Helmholtz field
    else:
        Br = Bo * gamma * (E * ((1 + alpha ** 2 + beta ** 2) / (Q - 4 * alpha)) - K) / pi / sqrt(Q)

    # now find axial component
    if r == 0:
        if z == 0:
            Bz = Bo  # at origin of Helmholtz coil
        else:
            Bz = (uo * I * a ** 2) / 2 / (a ** 2 + z ** 2) ** (3 / 2)  # on axis, Bz is simple
    else:
        Bz = Bo * (E * ((1 - alpha ** 2 - beta ** 2) / (Q - 4 * alpha)) + K) / pi / sqrt(Q)

    if NegativeRadius:
        Br = -Br

    if (direction == 'east-west-Door') or (direction == 'east-west-Window'):
        Bt = 0
    elif direction == 'north-south':
        Br, Bt, Bz = Bz, 0, Br  # swap, like above
        # Br, Bt, Bz = Br, 0, Bz  # no swap
    elif direction == 'up-down':
        Bt = Bz
        Bz = Br * np.cos(theta)
        Br = Br * np.sin(theta)

    return [Br, Bt, Bz]


def plot_helmholtz_pretty(scalor=8e7, opacity=1, savePlot=False):
    sns.set(rc={'axes.facecolor': '#ffffff',
                'axes.grid': False,
                'text.color': '.05',
                'ytick.left': True,
                'axes.axisbelow': True})
    fig, axes = plt.subplots(3, sharex=True)
    fig.suptitle('\n\n\n\n\nHelmholtz Shots Analytical Solution')
    fig.set_figwidth(10)
    fig.set_figheight(20)
    directions = ['north-south', 'east-west-Window', 'east-west-Door']
    for probe in range(16):
        for direction in directions:
            if direction == 'north-south':
                axis = axes[0]
                o = [1.5 * d_r, 0]
                axis.set_ylabel('North-South Shot')
            elif direction == 'east-west-Window':
                axis = axes[2]
                o = [1.5 * d_r, -2 * d_ax]
                axis.set_ylabel('East-West Shot (Window Side)')
                axis.set_xlabel('radial direction (meters)')
            elif direction == 'east-west-Door':
                axis = axes[1]
                o = [1.5 * d_r, 2 * d_ax]
                axis.set_ylabel('East-West Shot(Door Side)')
            B = kya_helmholtz(direction, rtz=pos[probe], current=2.7e-5)
            axis.plot([0, 3.5 * 0.038], [0.03625, 0.03625], 'r:')
            axis.plot([0, 3.5 * 0.038], [-0.03625, -0.03625], 'r:')
            axis.plot([0, 3.5 * 0.038], [2 * 0.03625, 2 * 0.03625], 'r:')
            axis.plot([0, 3.5 * 0.038], [-2 * 0.03625, -2 * 0.03625], 'r:')

            axis.plot(pos[probe][0], pos[probe][2], 'ro')

            Br = B[0]  # isolated r component
            Bz = B[2]

            x1 = pos[probe][0]
            y1 = pos[probe][2]
            x2 = Br * scalor
            y2 = Bz * scalor
            # print(Br)

            # x2=0.1
            # y2 = 0.1

            axis.arrow(x1, y1, x2, y2, color='#44668f', alpha=opacity, linewidth=0.5)
            axis.plot(o[0], o[1], 'rx', markersize=10)
    if savePlot:
        plt.savefig('Helmholtz_analytical_soln', dpi=700)
    fig.show()
    return


def plottingvoltageonsameplot(runrange, proberange):
    for run in runrange:
        fig, axs = plt.subplots(3, sharex=True)

        axs[0].set_title('Radial Direction')
        axs[1].set_title('Angular Direction')
        axs[2].set_title('Axial Direction')
        for probe in proberange:
            mag_data = B_routine('102021r%s' % run, Calibrate=False)

            axs[0].plot([0, 130], [0, 0], 'gray', linewidth=0.75)
            axs[0].plot(mag_data.time, mag_data.B[probe][0][:])
            # axs[0].plot([4, 4], [-500, 500])

            axs[1].plot([0, 130], [0, 0], 'gray', linewidth=0.75)
            axs[1].plot(mag_data.time, mag_data.B[probe][1][:])
            # axs[1].plot([4, 4], [-500, 500])

            axs[2].plot([0, 130], [0, 0], 'gray', linewidth=0.75)
            axs[2].plot(mag_data.time, mag_data.B[probe][2][:])
            # axs[2].plot([4, 4], [-0.05,0.05])

        fig.suptitle('Shot 102021r%s Probes by Component' % (run))
        plt.show()


def plotquiverofprobes(i, j, k, color, quiver_scale):
    """
    :param i: negative radial component
    :param j: negative axial component
    :param k: positive angular component
    :return:
    """
    if color == 2:
        color = '#2D4057'
        opacity = 1
    if color == 1:
        color = '#C46161'
        opacity = 0.4

    colors = ['#efada5','#d98c6c','#b76447','#8e4820','#f4f0c6','#ded491','#d5cb58','#c6be18',
             '#e8ebb1','#c4d283','#9ab34c','#6d9a1c','#a2dbe0','#6dbdcf','#4096ba','#246a8e']

    sns.set(rc={'axes.facecolor': '#ffffff',
                'axes.grid': False,
                'text.color': '.05',
                'ytick.left': True,
                'axes.axisbelow': True})

    ax.set_ylim(bottom=-0.125, top=0.125)

    # x, y, z = np.meshgrid()
    x = []
    y = []
    z = []
    for probe in range(16):
        x.append(-pos[probe][0])  # negative radial
        y.append(-pos[probe][2])  # negative axial
        z.append(pos[probe][1])  # positive angular

    # x y z are lists that indicate the 3d position of each vector tail
    # i j k are lists that contain the component parts of each vector at respective x y z locations

    # ax.quiver(x[:], y[:], z[:], i[:], j[:], k[:], length = 0.025, normalize=True, color=color, alpha=opacity)

    # ax.quiver(x[:8], y[:8], z[:8], i[:8], j[:8], k[:8], length=3500, normalize=False, color='#296082', alpha=opacity)
    # ax.quiver(x[8:], y[8:], z[8:], i[8:], j[8:], k[8:], length=3500, normalize=False, color='#965577', alpha=opacity)
    for probe in range(16):
        ax.quiver(x[probe], y[probe], z[probe], i[probe], j[probe], k[probe], length = quiver_scale, normalize=False, color=colors[probe], alpha = opacity)

    ax.plot([0, -0.15], [2 * d_ax, 2 * d_ax], [0, 0], 'r:')
    ax.plot([0, -0.15], [1 * d_ax, 1 * d_ax], [0, 0], 'r:')
    ax.plot([0, -0.15], [-2 * d_ax, -2 * d_ax], [0, 0], 'r:')

    ax.plot([0, -0.15], [-1 * d_ax, -1 * d_ax], [0, 0], 'r:')

    # ax.plot(np.array([-1.5*d_r]), np.array([0]), np.array([0]), 'rx')
    # ax.plot(np.array([-1.5 * d_r]), np.array([-2*d_ax]), np.array([0]), 'rx')

    for probe in range(16):
        ax.plot(-pos[probe][0], -pos[probe][2], pos[probe][1], 'r.')
    ax.set_xlabel('x: -radial')
    ax.set_ylabel('y: -axial')
    ax.set_zlabel('z: angular')
    ax.set_title('102021r20 Snapshot (Normalized)')


positions()

if __name__ == '__main__':

    # direction = 'north-south'        ~ this is the radial helmholtz shot
    # direction = 'up-down' /          ~ this is the negative angular helmholtz shot
    # direction = 'east-west-Window'   ~ this is the negative axial direction, 1st two stalks
    # direction = 'east-west-Door'     ~ this is the negative axial direction, 2nd two stalks
    ax = plt.figure().add_subplot(projection='3d')
    Calib_matrices = Calibration_matrices()

    # TODO: figure out scale so normalize is never necessary

    current = helmholtz_current()
    length = 45000

    for run in [20]:

        B_run = []
        mag_data = B_routine('102021r%d' % run, Calibrate=False)
        for axis in range(3):
            for probe in range(16):
                B_run.append(np.average(mag_data.B[probe][axis][-200:-30]))

        B_run_r = B_run[:16]
        B_run_t = B_run[16:32]
        B_run_z = B_run[32:48]

        corrected_vectors = []
        for probe in range(16):
            B_run_Vect = [B_run_r[probe], B_run_t[probe], B_run_z[probe]]
            corrected_vectors.append(np.matmul(Calib_matrices[probe], B_run_Vect))


        i, j, k = [], [], []
        for probe in range(16):
            i.append(- corrected_vectors[probe][0])
            j.append(- corrected_vectors[probe][2])
            k.append(corrected_vectors[probe][1])\


        plotquiverofprobes(i, j, k, color=2, quiver_scale=length)


        i, j, k = [], [], []
        for probe in range(16):
            B_helmholtz = kya_helmholtz('north-south', pos[probe], current=-current[2])
            i.append(- B_helmholtz[0])
            j.append(- B_helmholtz[2])
            k.append(B_helmholtz[1])

        plotquiverofprobes(i, j, k, color=1, quiver_scale=length)

        plt.show()
'''
    run = '20'
    x = ssxr.scope_data('102021r%s' % run, scope=3)
    current = x.ch2
    current_smoothed = ssxr.lowpass_gauss_smooth(current, x.time)
    plt.plot(current)
    plt.show()
'''


