import gzip
import os
import matplotlib.pyplot as plt
import scipy as sp
import numpy as np
import seaborn as sns
import scipy.interpolate as spinterp
from scipy.ndimage import gaussian_filter
import tarfile
import struct



# TODO: create an imports function

"""
    Links to imported modules (if you run on pycharm, you can easily add modules in interpreter settings (recommended)),
    or if preferred, you can use the pip download command. 
    gzip:        part of your standard library!  
    os:          part of your standard library! 
    matplotlib:  https://matplotlib.org/    pip command: pip3 install matplotlib
    numpy:       https://numpy.org/         pip command: pip install numpy
    scipy:       https://scipy.org/         pip command: -m pip install scipy

"""

# Notes and Updates
# this code is equipped to deal with txt or gz files as of 05/12/22. Must incorporate TZG files before full use.

"""
    One of the essential components to SSX data analysis is your base directory.
    Your base directory is unique to you and should include a specific file path
    that will lead your computer to the folder that contains all the SSX 
    experiment data (this file should be labeled 'data'). The folder which 
    contains your 'data' folder, should also contain all relevant SSX code 
    (things like ssxreadin.py or ssxanalysis.py). It is essential that after your 
    base directory, your folders are organized in the same way as found on ion: 
    your_base_directory >> data >> yyyy >> mmddyy >>> specific scope/mag/ids file

    Example base directory for Windows:
    basedir = r'C:\\Users\\kyabo\\OneDrive\\Desktop\\SSX2\\SSX_python_master\\'

    Example base directory for Mac:
    basedir = r'/Users/aylacimen/Documents/'

    The if statement block below the basedir assignment identifies whether you 
    are working on a Mac or on Windows. The difference in slashes permeates 
    through ALL readin code in the form of the variable 's'.
 """

# basedir = r'C:\\Users\\kyabo\\OneDrive\\Desktop\\SSX2\\SSX_python-master\\'
# basedir = r'/Users/aylacimen/Documents/SSX_code/'

basedir = os.getcwd() # this can be your base directory if your data folder is contained in the same greater folder
                      # as this code (ssxreadin). Otherwise, create your base directory manually as detailed above.

# setting variable 's' and checking base directory:
if '\\' in basedir:
    s = '\\'
elif '/' in basedir:
    s = '/'
else:
    raise ValueError("investigate your basedir assignment!")


#################
# --> CLASSES <--#
#################

class ssx_data(object):

    shotname = ''
    xlabel = 'Time (s)'
    ylabel = 'Signal (Arb. Units)'

    def __init__(self, shotname, scope=None):
        """
        Initializes SSX data object.

        inputs:
            shotname(str) - MMDDYYr# where we are analyzing the '#'th run of the day MMDDYY ('r' is a character that is
                            always present between date and run number)
            scope(int) - not necessary in this class, but defined in scope_data where interferometry is analyzed
        """
        self.info(shotname) # defines self.shotname, self.month, self.date, self.year (##), self.fullyear (20##),
                            # self.run, all STRINGS

    def __str__(self):
        return  # not yet assigned to a particular function

    def info(self, shotname):
        """
        Uses shotname to decifer the following, all strings
            - run year
            - run month
            - run day
            - run number
        """
        self.shotname = shotname                  # Should be something like '110321r11'
        self.fullyear = '20' + self.shotname[4:6] # Produces something like '11032021r11' should we need it
        self.run = self.shotname[7:]              # Identifies the run number
        self.date = self.shotname[0:6]            # Identifies the full date

    def path_to_date(self):
        """
        You've already constructed a base directory at the top of this document. Using knowledge of ion's formatting,
        this function will build the directory that will take your computer from basedir to a particular shot of
        interest

        Product:
            self.path(str) ~ entire path up until '-scope' or '-mag'
        """
        self.path = '%s%sdata%s%s%s%s%s%s-' % (basedir, s, s, self.fullyear, s, self.date, s, self.shotname)
        # This is illegible... ends up producing something like: 'basedir/data/2021/110321/110321r11'
        return self.path


class scope_data(ssx_data):
    def __init__(self, shotname, scope):
        """
        Sets up scope data class for specific shot
        inputs:
        :param: shotname(str)- MMDDYYr# where # is the run number
        :param: scope(int)- scope number
        """
        self.info(shotname)
        self.scope = str(scope)
        self.path_to_date()
        # self.scopepath = '%sscope%s' %(self.path, self.scope)
        self.scopepath = self.path + 'scope%s' % self.scope

        self.fetch_scope_data()

    def fetch_scope_data(self):
        """
        Acquires the raw data found in the particular scope file that fetch_scope_data is called to fetch. Depending on
        how labview and the scopes are programmed, the columns contain be a variety of data. When this code was written,
        the interferometer was hooked up to scope 1, and connected to channels 3 and 4, which correspond to columns
        4 and 5 of the data file. Note: labels at the top of the columns are not reliable in the least. This function
        organizes the raw data in the following way:

            self.time(list) - list of time values from document (column 1)
            ---essentially a numerical list of what time each voltage measurement was taken

            self.ch1(list) - (column 2) could be list voltage measurements, or other data/nonsense

            self.ch2(list) - (column 3) could be list voltage measurements, or other data/nonsense

            self.ch3(list) - (column 4) same -- summer 2022 this has voltage measurements for scope 1.

            self.ch4(list) - (column 5) -- summer 2022 this has the other voltage measurements for scope 1.
        """

        self.filename = self.scopepath + '.txt.gz'  # files are usually zipped, so we'll try that first
        try:
            '''
                this try statement will be attempting to grab the file from your data folder in the case that it was 
                downloaded from ion as a .txt.gz file. 
            '''
            file = gzip.open(self.filename)
            lines_encoded = file.readlines()
            ''' lines_encoded is a List of 'byte' objects, each representing one line of the document being read'''
            file.close()

            lines = []
            '''now we'll decode the 'byte' objects into strings. if the document has N lines, the lines list
               here will contain N strings, each representing a line. '''
            for l in lines_encoded:
                lines.append(l.decode())

        except:
            '''
                If the file was not collected as a .txt.gz and was instead just a .txt file, this except statement will
                be engaged and the .txt file will be collected from you data folder.  
            '''
            self.filename = self.scopepath + '.txt'  # removes '.gz' from the end of the filename
            file = open(self.filename)
            lines = file.readlines()
            file.close()
        ''' At this point, 'lines' is  a list of strings, each representing one line of the document. Lines with 
        non-data information (titles, for instance) start with a hashtag, so we'll take those out and isolate data '''

        data = []
        for l in lines:
            if l[0] != '#':  # if this line contains data...

                data.append(l)  # add it to the data list.
            else:
                header = l
        '''
        'data' at this point is an organization of ...
        '''

        self.time = []
        self.ch1 = []
        self.ch2 = []
        self.ch3 = []
        self.ch4 = []
        for l in data:
            l = l.split('\t')  # splits up the string by tab(\t) seperations: by columns
            self.time.append(float(l[0]))
            self.ch1.append(float(l[1]))
            self.ch2.append(float(l[2]))
            self.ch3.append(float(l[3]))
            self.ch4.append(float(l[4]))
        self.time = np.array(self.time)
        self.ch1 = np.array(self.ch1)
        self.ch2 = np.array(self.ch2)
        self.ch3 = np.array(self.ch3)
        self.ch4 = np.array(self.ch4)
        return self.time, self.ch1, self.ch2, self.ch3, self.ch4
        # usually it makes sense to not return anything but the option seems potentially helpful


class mag_data(ssx_data):
    """This is the read - in class for the data from the dtacq, including
    channels that are rerouted to some of the scopes. After the init function
    runs, the following attributes are available:

    self.shotname(str) - example: 110321r11
    self.fullyear(str) - example: 2021
    self.run(str) - example: 11
    self.date(str) - example: 110321
    self.path(str) - example: [basedir]/data/2021/110321r11
    self.path_to_mags(str) - example: [basedir]/data/2021/110321r11mag
    self.data(list): - 3 components, representing the three directions (order:
                        r, theta, z), with 16 lists inside of each of them -
                        one for each probe. They are in order from probe 1 to
                        probe 16, and contain the list of data for that
                        direction at that probe location over time.
    """

    def __init__(self, shotname):
        """
        :param shotname: (string) MMDDYYr# The ion-style shotname

        """
        self.info(shotname)  # collects basic info about the shot
        self.path_to_date()  # creates the path to the folder with that shot

        self.path_to_mags = self.path + 'mag'
        self.fetch_mag_data()  # collects data and deals with anomalies

    def fetch_mag_data(self):
        """
        This method does the work to read in the data for magnetics, and
        accounts for weirdness detailed in the documentation:
            - strange labeling systems
            - dead channels, rerouted through scopes
            - dead channels not rerouted
        By the end of this function, self.data contains all of the data
        from the input shot; its shape and information is detailed above
        in the class description.
        :return:
        """
        self.data = []

        for mag in [1, 2, 3]:
            filename = self.path_to_mags + '%s.tgz' % mag
            t = tarfile.open(filename)  # opens the zipped mag folder
            names = t.getnames()  # identifies each file in folder

            dats = []  # referred to as such because of the file type
            for name in names:
                if os.path.splitext(name)[1] == '.dat':
                    dats.append(name)  # gathers only .dat files
            dats.reverse()  # puts files in order (ch01 to ch32)
            files_in_mag_folder = []
            for fn in dats:  # for each file ...
                if int(fn[-5]) % 2 == 1:  # data is only on odd channels 2022
                    myf = t.getmember(fn)  # identifies the file individually
                    fp = t.extractfile(fn)  # gathers info from file
                    s = fp.read()  # reads lines of info from file
                    a = np.array(struct.unpack('h' * int((myf.size / 2)), s))
                    fp.close()  # unpacks and closes file
                    files_in_mag_folder.append(a)
            self.data.append(files_in_mag_folder)

        num_measurements = len(self.data[0][0])  # same for each channel
        frequency = 65e6  # Hz, 2022
        total_time = num_measurements / frequency  # this is total time in seconds
        total_time *= 1e6  # now total time is in microseconds (us)

        self.time = np.linspace(2, total_time +2, num_measurements)
        # Recording starts at 2us, t=0 is when the shot begins
        # Particular to 2022!!

        """ After this, mag_data is a list containing 3 arrays: each corres-
            ponding to a spacial direction. In 2022 when this is being written,
            mag1 corresponds to B field in the radial direction, mag2 to the 
            theta direction, and mag3 to z direction. In each of these arrays 
            data is associated with one channel -- they are in order from 
            'channel' 1 to 31, only odd numbers. This is our 16 probes.

            Also as of 2022, there are five dead channels. Our documentation 
            details the nuances of the relationships found in the code which 
            can't be made explicit here. Mag:2 Probe:13 is dead as well as 
            Mag:3  Probes:1,2,9,11. These probes are wired through the scopes, 
            so we need to now go find the data there! I'm sorry if my hard 
            coding causes strife down the line. """

        rerouted_data = scope_data(self.shotname, scope=2)
        scope_time = rerouted_data.time

        """ The adjust function interpolates over the data from the scopes to 
            match it to the timescale of the d-tacq. Before we can use it, we 
            need to chop the data from the scopes so it occurs over the same time
            frame as the data from the d-tacq. """

        # ACCOUNTING FOR DEAD CHANNELS, REROUTED THROUGH SCOPES
        self.data[1][12] = adjust(rerouted_data.ch4, scope_time, self.time)  # 'mag 2 ch 25 probe 13'
        self.data[2][0] = adjust(rerouted_data.ch1, scope_time, self.time)  # 'mag 3 ch 1 probe 1'
        # self.mag_data[2][1] = 'mag 3 ch 3 probe 2'             this one is currently not being rerouted
        self.data[2][8] = adjust(rerouted_data.ch2, scope_time, self.time)  # 'mag 3 ch 17 probe 9'
        self.data[2][10] = adjust(rerouted_data.ch3, scope_time, self.time)  # 'mag 3 ch 21 probe 11'

        """ The other detail is that the hardware labeling of channels is
            different from the way they are labeled on the files from which we
            read in data. These relationships also in the documentation, and
            are resolved below. 

            Specifically: in contrast to the theta and axial components, the
            radial components of the magnetic field are not read in in order
            of probe number, which must be accounted for. """

        # ACCOUNTING FOR DISCREPANCIES IN CHANNEL LABELS:
        reordered_data = []
        for probe in range(7, -1, -1):
            reordered_data.append(self.data[0][probe])
        for probe in range(15, 7, -1):
            reordered_data.append(self.data[0][probe])
        self.data[0] = reordered_data

        # DEAD CHANNEL INTERPOLATION:
        if int(self.fullyear) == 2021:
            """ When we ran in 2021, mag3(z) probe 2 (ch 3) was dead, and we
                didn't know until the beginning of summer 2022, when we 
                rerouted it through the scopes. To make up for this in the
                2021 shots, we'll interpolate the axial components of the 
                surrounding probes. Probes 1, 3, and 6 are the nearest probes. 
                """

            for i in range(len(self.data[2][2])):
                # this for loop is for each data point in the dead channel
                probe1 = self.data[2][0][i]
                probe3 = self.data[2][2][i]
                probe6 = self.data[2][5][i]
                self.data[2][1][i] = np.mean([probe1, probe3, probe6])

        """ And with that, we have fetched the mag data! """

        return self.data


class ids_data(ssx_data):
    """

    """

    def __init__(self, shotname):
        self.info(shotname)  # collects basic info about the shot
        self.path_to_date()  # creates the path to the folder with that shot

        self.path_to_mags = self.path + 'ids'
        self.fetch_ids_data()

    def fetch_ids_data(self):
        filename = self.path_to_mags + '.tgz'
        # print(filename)
        t = tarfile.open(filename)  # opens the zipped mag folder
        names = t.getnames()  # identifies each file in folder

        dats = []  # referred to as such because of the file type
        for name in names:
            if os.path.splitext(name)[1] == '.dat':
                dats.append(name)  # gathers only .dat files
        dats.reverse()  # puts files in order (ch01 to ch16)
        files_in_mag_folder = []
        for fn in dats:  # for each file ...
            myf = t.getmember(fn)  # identifies the file individually
            fp = t.extractfile(fn)  # gathers info from file
            s = fp.read()  # reads lines of info from file
            a = np.array(struct.unpack('h' * int((myf.size / 2)), s))
            fp.close()  # unpacks and closes file
            files_in_mag_folder.append(a)

        self.data = files_in_mag_folder

        num_measurements = len(self.data[0])  # same for each channel
        frequency = 10e6  # Hz, 2022
        total_time = num_measurements / frequency  # this is total time in seconds
        total_time *= 1e6  # now total time is in microseconds (us)

        self.time = np.linspace(-2, total_time - 2, num_measurements)

        """ Now, files_in_mag_folder is a list with 16 elements, each another
            list. Each of these 16 lists is for one bin, and contains data 
            points over the recording time. """

        '''
        sns.set(rc={'axes.facecolor': '#f0f0f7',
                    'axes.grid': False,
                    'text.color': '.05',
                    'ytick.left': True,
                    'axes.axisbelow': True,
                    'xtick.bottom': True})

        plot16, axs = plt.subplots(4, 4, sharex=True)
        plot16.set_figwidth(10)
        plot16.set_figheight(6)
        plot16.suptitle('IDS Shot %s' % self.shotname)

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
            axs[row, col].plot(files_in_mag_folder[bin][:])
            axs[row, col].set_title('Bin %d' % (bin + 1))

        plt.show()
        '''

# FUNCTIONS

def convolution_smooth(signal, length, window='flat', iterations=1):
    """
    A function that iteritively smooths data using one of np.hanning, np.hamming, np.bartlett,
    or np.blackman.
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the beginning and end part of the output signal.

    :param: signal (list / array) input signal that is to be smoothed.
    :param: length (odd int) the dimension of the smoothing window, should be an odd integer.
    :param: window (string) the type of window. Options include: 'flat', 'hanning', 'hamming',
            'bartlett', 'blackman'. Flat will produce a moving average.
    :param: iterations (int) the number of times the smoothing will be iterated.

    :return: smoothed (list) the signal that has been iteratively smoothed.
    """
    # Error catching
    signal = np.array(signal)
    if signal.ndim != 1:
        raise ValueError("convolution_smooth only accepts 1-dimensional arrays.")
    if signal.size < length:
        raise ValueError("Window size is too big! Input signal vector must be bigger than window size.")
    if length <= 1:
        return signal
    if length % 2 != 1:
        raise ValueError("Window length must be an odd integer.")
    if window not in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window must be one of 'flat', 'hanning', 'hamming', 'bartlett', or 'blackman'")

    # Now the function begins
    for i in range(iterations):
        s = np.r_[signal[length - 1: 0: -1], signal, signal[-1: -length: -1]]

        if window == 'flat':  # moving average case
            w = np.ones(length, dtype='d')
        else:
            w = eval('np.' + window + '(length)')

        signal = np.convolve(w / w.sum(), s, mode='valid')
        signal = signal[int((length - 1) / 2): int(-(length) / 2)]

    smoothed = signal
    return smoothed


def lowpass_gauss_smooth(data, time, sigma=0.01):
    """
    This function uses scipy's lowpass gaussian_filter function to smooth out
    the high frequency wiggles in the input curve.

    :param: data (list) the list of data points of the noisy signal to be smoothed
    :param: time (list) the list of time indices associated with the data of interest
    :param: cutoff (float) cutoff frequency for this lowpass filter

    :returns: smooth_data (list) a list of smoothed data!
    """

    dt = time[1] - time[0]
    g_kernel = np.round((1 / (2 * np.pi * sigma * dt)))
    smooth_data = gaussian_filter(data, g_kernel)

    return smooth_data


def normalize(sinusoid, envelope=None, offset=None, Calibrating=True):
    """
    This function normalizes a sinusoidal curve by subtracting off any
    offset and dividing by the envelope. If Calibrating, no envelope or
    offset inputs are necessary, and the sinusoid will be normalized
    between -1 and 1.

    :param envelope: double the amplitude of calibration sinusoid
    :param offset: minimum value of calibration sinusoid
    :param Calibrating: if Calibrating is True, there are no inputs necessary
                and the curve will be normalized between -1 and 1
    :return: a list containing the normalized curve, according to inputs
    """
    if Calibrating:
        offset = min(sinusoid)
        envelope = max(sinusoid) - offset

        sinusoid = (np.array(sinusoid) - offset) / envelope
        # now sinusoid is an np array with domain [0 - 1]
        sinusoid = (2 * sinusoid) - 1
        # now the sinusoid is an np array with domain [-1 - 1]

    else:
        sinusoid = (np.array(sinusoid) - offset) / envelope
        sinusoid = (2 * sinusoid) - 1

    return sinusoid


def adjust(scope_data, scope_time, dtacq_time):
    """
    This function takes data from a scope and interpolates it over the time
    base of the dtacq. It is specifically for probes where the dtacq channel
    has gone dead, so the probe is rerouted through one of the scopes.

    :param scope_data: data from the scope (len: L)
    :param scope_time: time points from the scope (len: L)
    :param dtacq_time: time points from the dtacq (len: X)

    :return: list containing data interpolated over dtacq_time
    """

    num_pts = len(scope_time)  # L

    for i in range(num_pts):
        if scope_time[i] >= dtacq_time[0]:
            t0 = i
            break
    for i in range(num_pts - 1, 0, -1):
        if scope_time[i] <= dtacq_time[-1]:
            tf = i
            break

    """ The scopes start recording before the dtacq and finish afterward, so 
        these two for loops indentify the indices where the dtacq starts and
        stops recording.

        Now we will chop off all of the scope_data taken before and after the 
        dtacq was running. """

    scope_time = scope_time[t0: tf]
    scope_data = scope_data[t0: tf]

    srep = spinterp.splrep(scope_time, scope_data)
    scope_data_interpolated = spinterp.splev(dtacq_time, srep)
    # These two functions are fairly convoluted in their process, but the
    # below plotting block that is commented out can be uncommented in
    # order to see the effects of the interpolation.

    '''
    plt.plot(scope_time, scope_data, 'b', label='before')
    plt.plot(dtacq_time, scope_data_interpolated, 'r:', label='after')
    plt.legend()
    plt.show()
    #'''

    return scope_data_interpolated


if __name__ == "__main__":
    """This only runs if you're running this file, but not if you are importing it somewhere else! 
       It can be handy while you're testing code."""

    # x = ids_data('102021r20')

    y = mag_data('102021r21')

