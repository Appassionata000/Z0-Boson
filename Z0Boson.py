 """
________________TITLE________________
PHYS20161 - Assignment 2 - Z0 Bozon
-------------------------------------
This is a programme that calculates the mass and width of
the Z0 boson using cross-section data at various energies.
Fitted curve and contour plots are provided and automatically
saved if the program is successfully run.
Additionally, other properties such as lifetime are also determined.

Last Updated: 15/12/2021
author: Zhiyu Liu
"""

import sys
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import scipy.constants as pc
import pandas as pd


# constants
SIGNIFICANT_DIGITS = 3
PARTIAL_WIDTH = 83.91E-3
# unit converting from GeV to nb
GEV2NB = 0.3894E6

# constants for hillclimbing function
STEP_START = 1
TOLERANCE = 0.001
MAX_FAILED_ATTEMPTS = 5

# functions

def cross_section_func(energy, boson_mass, boson_width):
    """
    Returns cross section , float
    calculated using the formula below
    Args:
    energy : float
    boson_mass : float
    boson_width : float
    """
    # calculate the squares in advance just to make the formula conciser
    mass_squared = boson_mass**2
    energy_squared = energy**2

    fraction_1 = (12 * np.pi / mass_squared)
    fraction_2 = energy_squared / ((mass_squared - energy_squared)**2 +
                                   mass_squared * boson_width**2)

    return fraction_1 * fraction_2 * (PARTIAL_WIDTH**2) * GEV2NB

def read_in_data():
    """
    Returns unfiltered data
    Reading values from two files
    Args:
        None
    """
    unfiltered = np.empty((0, 3))

    for index in ['1', '2']:

        file_name = 'z_boson_data_{}.csv'.format(index)

        try:
            raw = np.genfromtxt(file_name, delimiter=',', comments='%')

        except OSError:
            print('Unable to open file.')
            sys.exit()

        # exclude non-numeric data
        for _, raw in enumerate(raw):

            if not np.isnan(np.sum(raw)):

                unfiltered = np.vstack((unfiltered, raw))

        # for i in range(len(raw)):

        #     if not np.isnan(np.sum(raw[i])):

        #         unfiltered = np.vstack((unfiltered, raw[i]))

    # sort data by energy small to big
    unfiltered = unfiltered[unfiltered[:, 0].argsort()]

    return unfiltered

def filter_data(array):
    """
    Filter data, concentrating on data with physical problem
    and obvious outliers
    The data will be deleted using an auxiliary function: delete_by_indices()
    Parameters
    ----------
    array : numpy array
        raw data to be filtered
    Returns
    -------
    array : numpy array
        filtered data
    abandoned_data : list
        discarded data list
    """

    # discard nonpositive cross section data
    problem_indices_1 = np.where(array[:, 1] <= 0)[0]
    array, abandoned_list_1 = delete_by_indices(array, problem_indices_1,
                                                1, 'physical')
    # discard nonpositive uncertainty data
    problem_indices_2 = np.where(array[:, 2] <= 0)[0]
    array, abandoned_list_2 = delete_by_indices(array, problem_indices_2,
                                                2, 'physical')

    # filter notable large outliers based on 3-sigma rule
    mean = np.average(array[:, 1])
    std = np.std(array[:, 1])

    problem_indices_3 = np.where(np.abs(array[:, 1] - mean) > 3 * std)[0]
    array, abandoned_list_3 = delete_by_indices(array, problem_indices_3,
                                                1, 'outlier')

    abandoned_data = abandoned_list_1 + abandoned_list_2 + abandoned_list_3

    return (array, abandoned_data)

def filter_data_2(array, params):
    """
    Filter data, concentrating on outliers
    Determine outliers using 3-sigma rule as the curve
    is expected to be a Gaussian
    The data will be deleted using an auxiliary function: delete_by_indices()
    Parameters
    ----------
    array : numpy array
        Data to be filtered
    params : list
        fitted parameters gained from roughly filtered data
    Returns
    -------
    array : numpy array
        filtered data
    abandoned_data : list
        discarded data list
    problem_indices : list
        discarded data index list
        useful in future
    """

    problem_indices = np.empty((0, 0), dtype=int)

    energy, exp_cs = array[:, 0], array[:, 1]

    theoretical_cs = cross_section_func(energy, params[0], params[1])

    sigma_cs = np.std(exp_cs - theoretical_cs)

    # applying 3-sigma rulle
    problem_indices = np.where(np.abs(exp_cs - theoretical_cs) >
                               3 * sigma_cs)[0]

    array, abandoned_data = delete_by_indices(array, problem_indices,
                                              1, 'outlier')

    return (array, abandoned_data, problem_indices)

def delete_by_indices(array, delete_list, column, reason):
    """
    Auxiliary function for filter_data() & filter_data_()
    Delete elements in an array according to the index from a list
    Add notation to the discarded data
    Parameters
    ----------
    array : numpy array
        array whose elements are to be deleted
    delete_list : list
        elements are indices guiding the deletion
    column : int
        discarded data column number
    reason : string
        discard reason

    Returns
    -------
    array : numpy array
        data after deleting certain elements
    abandoned_list : list
        discarded data with notation and reasons

    """
    abandoned_list = [None] * len(delete_list)

    for i, delete_index in enumerate(delete_list):

        abandoned_list[i] = array[delete_index-i].tolist()

        abandoned_list[i].append(reason)

        abandoned_list[i][column] = '*' + str(abandoned_list[i][column]) + '*'

        array = np.delete(array, delete_index-i, 0)

        # print('problem data with index {} is removed'.format(problems[i]))

        delete_list = delete_list - 1

    return (array, abandoned_list)


def chi_squared(a_parameter, b_parameter, input_data):
    """
    Returns chi squared for a pre defined function depenedent on one
    variable, x, with two parameters, a & b.
    data is a 2D array composed of rows of [x values, f(x) values and
    uncertainties]
    Parameters
    ----------
    a_parameter : float
    b_parameter : float
    input_data : numpy array

    Returns
    -------
    chi_square : float

    """

    chi_square = 0

    for entry in input_data:
        chi_square += (((cross_section_func(entry[0], a_parameter, b_parameter)
                         - entry[1]) / entry[2])**2)

    return chi_square

def hill_climbing(input_data, step=STEP_START):

    """
    [COPY FROM BLACKBOARD, ORIGINALLY WRITTEN BY Prof.Lloyd]
    Runs Hill Climbing algorithm in 2D with a varying step size.
    Args:
        input_data array([float, float, float])
        step kwarg (float)
    Returns:
        np.array([minimum_chi_squared (float),
                  [a_fit, b_fit] ([float, float]),
                  counter (int),
                  success (bool)])
    """
    # Guess starting values
    a_fit = 92
    b_fit = 2.5
    minimum_chi_squared = chi_squared(a_fit, b_fit, input_data)
    difference = 1
    counter = 0

    # Count how many times we fail to find a better value after drecreasing the
    # step size. If it matches MAX_FAILED_ATTEMPTS, then exit loop and
    # set success = 0.
    timeout = 0
    success = 1

    # Look around current best fit to find better value

    while difference > TOLERANCE:
        counter += 1

        # Save current best values for comparison later
        a_test = a_fit
        b_test = b_fit

        for i in np.arange(-1, 2, 1):
            for j in np.arange(-1, 2, 1):

                test_chi_squared = chi_squared(a_fit + i * step, b_fit
                                               + j * step, input_data)

                if test_chi_squared < minimum_chi_squared:

                    timeout = 0

                    # If better solution found update parameters:

                    difference = np.abs(minimum_chi_squared - test_chi_squared)
                    minimum_chi_squared = test_chi_squared

                    a_fit += i * step
                    b_fit += j * step

        if a_fit == a_test and b_fit == b_test:
            # If we have failed to find better values then reduce the step size
            step = step * 0.1
            timeout += 1
            if timeout == MAX_FAILED_ATTEMPTS:
                success = 0
                print('Failed to reach desired accuracy with a step size of'
                      ' {:g}'.format(step))
                break

    return [minimum_chi_squared, [a_fit, b_fit], counter, success]

def plot_fit_new(data_r, data_f, fit_params_1, fit_params_2, problem):
    """
    Plot two fitted curves and compare them to the data
    One is fitted using roughly filtered data
    And another is fitted using twice filtered data
    Parameters
    ----------
    data_r : numpy array
        roughly filter data
    data_f : numpy array
        twice filtered data
    fit_params_1 : float
        mass
    fit_params_2 : float
        width
    problem : array of int64
        discarded data indices

    Returns
    -------
    None.

    """
    energy, exp_cs, uncertainties = data_f[:, 0], data_f[:, 1], data_f[:, 2]

    fit_fig = plt.figure(figsize=(10, 4))
    rough_fit_plot = fit_fig.add_subplot(121)
    filtered_fit_plot = fit_fig.add_subplot(122)

    rough_fit_plot.plot(data_r[:, 0], cross_section_func(data_r[:, 0],
                                                         fit_params_1[0],
                                                         fit_params_1[1]),
                        color='teal', linewidth=3,
                        label='Fitted curve')

    # Treating discarded data separately, showing them in a notable way
    problem_data = np.empty((0, 3))
    for _, index in enumerate(problem):

        problem_data = np.vstack((problem_data,
                                  data_r[index]))

    # plot data points that will be discarded in 2nd filter
    rough_fit_plot.errorbar(problem_data[:, 0], problem_data[:, 1],
                            yerr=problem_data[:, 2],
                            fmt='o', color='mediumvioletred', ecolor='purple',
                            label='outliers')

    # plot data points that are not discarded
    rough_fit_plot.errorbar(energy, exp_cs, yerr=uncertainties,
                            fmt='o', color='#00A087CC', ecolor='#4DBBD5CC',
                            alpha=0.5, label='Data')

    rough_fit_plot.set_xlabel(r'energy $E$', fontsize=14)
    rough_fit_plot.set_ylabel(r'cross section $\sigma$', fontsize=14)
    rough_fit_plot.set_title('Roughly filtered data fitting curve',
                             fontsize=14)

    rough_fit_plot.grid()
    rough_fit_plot.legend()

    # Second plot, with twice filtered data
    filtered_fit_plot.plot(energy, cross_section_func(energy, fit_params_2[0],
                                                      fit_params_2[1]),
                           color='teal', linewidth=3, label='Fitted curve')

    filtered_fit_plot.errorbar(energy, exp_cs, yerr=uncertainties,
                               fmt='o', color='#00A087CC', ecolor='#4DBBD5CC',
                               alpha=0.5, label='Data')

    filtered_fit_plot.set_xlabel(r'energy $E$', fontsize=14)
    filtered_fit_plot.set_ylabel(r'cross section $\sigma$', fontsize=14)
    filtered_fit_plot.set_title('Twice filtered data fitting curve',
                                fontsize=14)

    filtered_fit_plot.grid()
    filtered_fit_plot.legend()

    fit_fig.subplots_adjust(bottom=0.15)

    # Save figure
    plt.savefig('fitted curve.png', dpi=300)

    plt.show()

def mesh_arrays(x_array, y_array):
    """
    [COPY FROM BLACKBOARD, ORIGINALLY WRITTEN BY Prof.Lloyd]
    Returns two meshed arrays of size len(x_array)
    by len(y_array)
    x_array array[floats]
    y_array array[floats]
    """
    x_array_mesh = np.empty((0, len(x_array)))

    for _ in y_array:  # PyLint accepts _ as an uncalled variable.
        x_array_mesh = np.vstack((x_array_mesh, x_array))

    y_array_mesh = np.empty((0, len(y_array)))

    for dummy_element in x_array:  # PyLint accepts dummy_anything as well.
        y_array_mesh = np.vstack((y_array_mesh, y_array))

    y_array_mesh = np.transpose(y_array_mesh)

    return x_array_mesh, y_array_mesh

def contour_plot_new(fit, data):
    """
    Plot chi squared contour to parameters
    Two contours are plotted. A general one
    and one to help determine uncertainties
    Uncertainties are also calculated in this function
    Parameters
    ----------
    fit : list
        return value of hill_climbing()
    data : numpy array
    a : float
        geometric parameters for contour
    b : float
        geometric parameters for contour
    num : int
        number of contours in a figure

    Returns
    -------
    The uncertaintes are calculated in this function
    since these values can be obtained directly from the contours
    sigma_mass : float
        uncertainty of mass
    sigma_width : float
        uncertainty of width

    """
    min_chi_squared, fit_params = fit[0], fit[1]

    # setting up axis limits
    mass = np.linspace(fit_params[0] - 0.08,
                       fit_params[0] + 0.08, 500)
    width = np.linspace(fit_params[1] - 0.1,
                        fit_params[1] + 0.1, 500)

    mass_mesh, width_mesh = mesh_arrays(mass, width)

    fig = plt.figure(figsize=(12, 4))

    # Adjusting ratio of two subplots
    plots = gridspec.GridSpec(1, 2, width_ratios=[0.7, 1])
    params_contour_plot = fig.add_subplot(plots[0])
    sigma_contour_plot = fig.add_subplot(plots[1])

    # Adjust margins
    plt.subplots_adjust(left=None, bottom=None,
                        right=None, top=None,
                        wspace=0.4, hspace=None)

    # Plot first(left) contour
    # The purpose of defining chi2value is simply to shorten the length of line
    chi2value = chi_squared(mass_mesh, width_mesh, data)
    contour_plot = params_contour_plot.contour(mass_mesh, width_mesh,
                                               chi2value, 10)
    # Plot minimum chi squared point
    params_contour_plot.scatter(fit_params[0], fit_params[1], label='minimum')

    params_contour_plot.clabel(contour_plot, inline=1, fontsize=10)

    params_contour_plot.set_title(r'$\chi^2$ contours against parameters.',
                                  fontsize=14)
    params_contour_plot.set_xlabel(r'mass $m_{\mathrm{Z}}$', color='black')
    params_contour_plot.set_ylabel(r'width $\Gamma_{\mathrm{Z}}$',
                                   color='black')

    params_contour_plot.tick_params(axis='x', width=1, color='k',
                                    labelsize=10, labelcolor='k')
    params_contour_plot.tick_params(axis='y', width=1, color='k',
                                    labelsize=10, labelcolor='k')

    params_contour_plot.legend()


    # Plot minimum chi squared point
    sigma_contour_plot.scatter(fit_params[0], fit_params[1], label='Minimum')

    # Plot level one of second(right) contour
    sigma_1 = sigma_contour_plot.contour(mass_mesh, width_mesh,
                                         chi_squared(mass_mesh,
                                                     width_mesh, data),
                                         levels=[min_chi_squared + 1.00],
                                         linestyles='dashed',
                                         colors='k')

    sigma_contour_plot.set_title(r'$\chi^2$ contours against parameters.',
                                 fontsize=14)
    sigma_contour_plot.set_xlabel(r'mass $m_{\mathrm{Z}}$', color='black')
    sigma_contour_plot.set_ylabel(r'width $\Gamma_{\mathrm{Z}}$',
                                  color='black')

    # Set levels for more contours
    chi_squared_levels = (min_chi_squared + 2.30,
                          min_chi_squared + 5.99,
                          min_chi_squared + 9.21)

    # Plot more levels of second(right) contour
    sigmas = sigma_contour_plot.contour(mass_mesh, width_mesh,
                                        chi_squared(mass_mesh,
                                                    width_mesh, data),
                                        levels=chi_squared_levels)

    # sigma_contour_plot.clabel(sigma_1, inline=1, fontsize=8)
    sigma_contour_plot.clabel(sigmas, inline=1, fontsize=8, fmt='%.1f')

    # Set labels
    labels = ['Minimum',
              r'$\chi^2_{{\mathrm{{min.}}}}+1.00$',
              r'$\chi^2_{{\mathrm{{min.}}}}+2.30$',
              r'$\chi^2_{{\mathrm{{min.}}}}+5.99$',
              r'$\chi^2_{{\mathrm{{min.}}}}+9.21$']

    for index, label in enumerate(labels):
        sigma_contour_plot.collections[index].set_label(label)

    # Obtain coordinates of the contour
    dat0 = sigma_1.allsegs[0][0]

    # Find limits of first level of the contour
    index_1 = np.argmax(dat0[:, 1])
    index_2 = np.argmin(dat0[:, 1])
    index_3 = np.argmin(dat0[:, 0])
    index_4 = np.argmax(dat0[:, 0])

    # Determine uncertainties
    width_uncertainty = (dat0[:, 1][index_1] - dat0[:, 1][index_2]) / 2
    mass_uncertainty = (dat0[:, 0][index_4] - dat0[:, 0][index_3]) / 2

    # Plot limit points
    indices = [index_1, index_2, index_3, index_4]

    for index, point_index in enumerate(indices):

        sigma_contour_plot.scatter(dat0[:, 0][point_index],
                                   dat0[:, 1][point_index],
                                   label='point {}'.format(index+1))

    # Adjusting figure to fit the axes size
    dat1 = sigmas.allsegs[2][0]

    border_top = np.max(dat1[:, 1])
    border_down = np.min(dat1[:, 1])
    border_right = np.max(dat1[:, 0])
    border_left = np.min(dat1[:, 0])

    sigma_contour_plot.axis([border_left-0.01, border_right+0.01,
                             border_top+0.01, border_down-0.01,])

    # Set legend
    box = sigma_contour_plot.get_position()
    sigma_contour_plot.set_position([box.x0, box.y0, box.width * 0.7,
                                     box.height])
    sigma_contour_plot.legend(loc='center left', bbox_to_anchor=(1, 0.5),
                              fontsize=10)

    # Save figure
    plt.savefig('contour.png', dpi=300)

    plt.show()

    return (mass_uncertainty, width_uncertainty)

def show_abandoned_list(array):
    """
    Display the discarded data

    Parameters
    ----------
    array : list

    Returns
    -------
    None.

    """
    print('┌----------------------------------------------------------┐')
    print('|                    DISCARDED DATA                        |')
    print('└----------------------------------------------------------┘')
    header = ['energy', '  cross section',
              '  uncertainty', '        reason']
    abandoned_df = pd.DataFrame(array, columns=header)
    abandoned_df.index = np.arange(1, len(abandoned_df) + 1)
    print(abandoned_df)
    print('----------------------------------------')
    print('Problem data are surrounded by asterisks')

def calculate(array, length):
    """
    Do calculations of chi squared, reduced chi squared and lifetime

    Parameters
    ----------
    array : list
        return value of hill_climbing()
    length : int
        number of data, to calculate reduced chi squared

    Returns
    -------
    results : numpy array

    """
    min_chi_sqaured, fit_params = array[0], array[1]
    fitted_mass, fitted_width = fit_params

    min_chi_reduced = min_chi_sqaured / (length - 1)
    lifetime = (pc.hbar / pc.e) / (fitted_width * 1E9)

    # convert lifetime(float) to scitific notation
    rounded_lifetime = round(lifetime, SIGNIFICANT_DIGITS -
                             int(math.floor(math.log10(abs(lifetime)))) - 1)

    results = np.array([float(min_chi_sqaured), float(min_chi_reduced),
                        float(fitted_mass), float(fitted_width),
                        rounded_lifetime])
    return results

def show_results(results1, results2, sigmas):
    """
    Display results
    Parameters
    ----------
    results1 : numpy array
        results from first fit
    results2 : numpy array
        results from second fit
    sigmas : numpy array
        uncertainties

    Returns
    -------
    None.

    """

    print('------------------------------------------------------')
    print('|                  FINAL  RESULTS                    |')
    print('└----------------------------------------------------┘')
    print("                   first filtering      second filtering")
    print("min chi sqaured         {:.3f}".format(results1[0])
          + spacef(results1[0])*' ' + "{:.3f} ".format(results2[0]))
    print("reduced chi squared     {:.3f}".format(results1[1])
          + spacef(results1[1])*' ' + "{:.3f} ".format(results2[1]))
    print("fitted mass             {:.2f}GeV/c^2".format(results1[2])
          + (spacef(results1[2])-6)*' '  + "{:.3f}GeV/c^2 ".
          format(results2[2]))
    print("fitted width            {:.3f}GeV".format(results1[3])
          + (spacef(results1[3])-3)*' '  + "{:.3f}GeV ".format(results2[3]))
    print("lifetime                {:.3g}s".format(results1[4])
          + (spaceg(results1[4])-1)*' '  + "{:.3g}s ".format(results2[4]))

    print('--------------------UNCERTAINTIES--------------------')
    print('mass                    {:.3g}'.format(sigmas[0]))
    print('width                   {:.3g}'.format(sigmas[1]))
    print('lifetime                {:.3g}'.format(sigmas[2]))
    print('------------------------------------------------------')

def spacef(number):
    """
    Auxiliary function to format tables
    calculate how many spaces are needed to make table elements aligned
    Parameters
    ----------
    number : float

    Returns
    -------
    int
        number of spaces

    """
    formatted_number = '{:.3f}'.format(number)

    return 19 - len(str(formatted_number))

def spaceg(number):
    """
    Auxiliary function to format tables
    calculate how many spaces are needed to make table elements aligned
    Parameters
    ----------
    number : float

    Returns
    -------
    int
        number of spaces

    """
    formatted_number = '{:.3g}'.format(number)

    return 19 - len(str(formatted_number))

def preface():
    """
    Introduction to the program

    Returns
    -------
    None.

    """
    print('----------------------------------------------------------------')
    print("|  This is a program wrote to determine the mass and width of  |")
    print("|  Z boson using the cross-section at different energies.      |")
    print('----------------------------------------------------------------')
    print("|  The programme can perform a variety of calculations;        |")
    print("|  please select the one that best suits your needs.           |")
    print('----------------------------------------------------------------')
    print('                                            Zhiyu Liu 2021.12.15')
    print('\n')

    # input("Press Enter to begin reading in data...")


preface()

# read DATA from several files(.csv)
# use capital variable names to avoid clash with local variables
DATA = read_in_data()

print('\nEfforts to read in data have been successful')

# archive the abandoned data
total_abandoned = []

# rough filter the data
rough_filtered_data, abandoned = filter_data(DATA)

total_abandoned += abandoned

# fit for the first time, a Gaussian distribution is expected
# return [minimum_chi_squared, [mass_fit, width_fit], counter, success]
first_fit = hill_climbing(rough_filtered_data, 1)

# store results and do calculations
# FIT_PARAMS_1 => [fitted mass, fitted width]

min_chi_sqaured_1, FIT_PARAMS_1 = first_fit[0], first_fit[1]
fitted_mass_1, fitted_width_1 = FIT_PARAMS_1

RESULTS1 = calculate(first_fit, len(rough_filtered_data))


# filter for the 2nd time and find the remainning outliers
DATA = filter_data_2(rough_filtered_data, FIT_PARAMS_1)[0]
abandoned = filter_data_2(rough_filtered_data, FIT_PARAMS_1)[1]
abandoned_indices = filter_data_2(rough_filtered_data, FIT_PARAMS_1)[2]

total_abandoned += abandoned

show_abandoned_list(total_abandoned)

# fit for the second time with final data
second_fit = hill_climbing(DATA, 1)

# store results and do calculations
# FIT_PARAMS_2 => [fitted mass, fitted width]
FIT_PARAMS_2, min_chi_sqaured_2 = second_fit[1], second_fit[0]
min_chi_reduced_2 = min_chi_sqaured_2 / (len(DATA) - 1)
liftetime = (pc.hbar / pc.e) / (FIT_PARAMS_2[1] * 1E9)

RESULTS2 = calculate(second_fit, len(DATA))

# plot the fit curve and compare it with rough filtered DATA
plot_fit_new(rough_filtered_data, DATA, FIT_PARAMS_1,
             FIT_PARAMS_2, abandoned_indices)

# plot chi squared contour against parameters and determine uncertainties
sigma_mass, sigma_width = contour_plot_new(second_fit, DATA)

# uncertainty unchanged under reciprocation
sigma_lifetime = sigma_width
sigma = np.array([sigma_mass, sigma_width, sigma_lifetime])

show_results(RESULTS1, RESULTS2, sigma)
