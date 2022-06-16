import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from matplotlib.patches import Rectangle


def get_rectangle_points(width, height, color='blue'):
    '''
    Given width and height return rectangle
    center at (0, 0) with distance width/2 at x axis and distance height/2 in y axis.
    The purpose is to allow drawing a rectangle.
    '''
    rect = Rectangle(xy=(-width / 2, -height / 2), width=width, height=height, fill=False, color=color)
    return rect


def hypothesis_r(points):
    '''
    Compute the hypothesis distance at the x axis and at the y axis given a set of points.
    Returns the maximum distances form the origin in each axis only from the positive-labeled points.
    '''
    # Get positive points.
    inner_points_mask = (points[:, 2] == 1)
    inner_points = points[inner_points_mask]

    if len(inner_points) == 0:
        return -0.001, -0.001

    # compute distance in each axis.
    x = max([abs(x_i) for x_i in inner_points[:, 0]])
    y = max([abs(y_i) for y_i in inner_points[:, 1]])
    return x, y


def compute_sample_complexity(ε, δ):
    '''
    Computes the sample complexity needed for the concentric rectangle concept
    class given ε and δ.
    '''
    return int(np.ceil(np.log(2/δ) * (2 / ε)))


def show_experiment_view(hypothesis_xy, data, xy, experiment_title, ax, experiment_ε):
    '''
    Shows experiment instance space with the concept c, hypothesis h and training data D.
    '''
    rectangle = get_rectangle_points(2 * xy[0], 2 * xy[1])
    rectangle_h = get_rectangle_points(-2 * hypothesis_xy[0], -2 * hypothesis_xy[1], color='orange')
    ax.scatter(data[:, 0], data[:, 1], c=data[:, 2])
    ax.add_patch(rectangle)
    ax.add_patch(rectangle_h)
    ax.legend([rectangle, rectangle_h], ['c', 'h'])

    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_xlabel('${x}_{1}$', fontdict = {'fontsize' : 20})
    ax.set_ylabel('${x}_{2}$', fontdict = {'fontsize' : 20})
    if experiment_title == '':
        ax.set_title('X, ' + '$error_{π}(h=L(D),c)$ ' + '= {:.3f}'.format(experiment_ε), fontdict = {'fontsize' : 20})
    else:
        ax.set_title('X, ' + experiment_title + ', ' + '$error_{π}(h=L(D),c)$ ' + '= {:.3f}'.format(experiment_ε), fontdict = {'fontsize' : 20})
    ax.legend([rectangle, rectangle_h], ['c', 'h'], loc='upper left')


def run_experiment(sample_complexity, experiment_title, show_experiment, ax, return_data=False, A=1.0, B=0.5):
    '''
    Run single experiment of concept c with width  = 2 * A  and height = 2 * B = 1 .
    '''
    # Change name to sample_size
    # Set Radius
    xy = (A, B)
    # Draw Data
    points = np.random.uniform(low=-2, high=2, size=(sample_complexity, 2))
    labels = np.array([1 if (-xy[0] <= x <= xy[0] and -xy[1] <= y <= xy[1]) else 0 for x, y in zip(points[:, 0], points[:, 1])])
    data = np.c_[points, labels]
    # Compute hypothesis radius.
    h_xy = hypothesis_r(data)
    error_rate = ((xy[0]*2 * xy[1]*2) - (h_xy[0]*2 * h_xy[1]*2)) / 16
    if show_experiment:
        show_experiment_view(h_xy, data, xy, experiment_title, ax, error_rate)
    # Compute error rate.
    if return_data:
        return data
    else:
        return error_rate


def run_experiments(ε, δ, NUM_EXPERIMENTS, NUM_EXPERIMENTS_TO_PLOT, sample_complexity, show=True, A=1.0, B=0.5):
    '''
    Run couple of experiments for "Experiments and visualization" part.
    '''
    # np.random.seed(120)
    # r = np.random.rand()
    sample_complexities = [sample_complexity]
    amount_of_errors_larger_then_ε = 0
    fig, exp_axs = plt.subplots(nrows=int(NUM_EXPERIMENTS_TO_PLOT / 2), ncols=2,
                                figsize=(14, int(NUM_EXPERIMENTS_TO_PLOT / 2) * 7))
    errors = []
    for _ in range(NUM_EXPERIMENTS):
        experiment_title = f'sample complexity = {sample_complexity}, desired 1 - δ = {δ}, \n desired ε = {ε}'
        show_experiment = _ < NUM_EXPERIMENTS_TO_PLOT
        ax = None
        if show_experiment:
            ax = exp_axs[int(_ / 2), _ % 2]
        error_rate = run_experiment(sample_complexity, experiment_title, show_experiment,
                                    ax, return_data=False, A=A, B=B)
        if error_rate > ε:
            amount_of_errors_larger_then_ε += 1
        errors.append(error_rate)
    approximated_δ = amount_of_errors_larger_then_ε / NUM_EXPERIMENTS
    if show:
        st.pyplot(plt)
        plt.clf()
        st.info(f'General view of all 10k experiments.')
        errors = np.array(errors)
        x = np.array([i for i in range(NUM_EXPERIMENTS)])
        plt.figure(figsize=(20, 20))
        plt.scatter(x[errors > ε], errors[errors > ε], c='red',
                    label=f' Amount of experiments with ' + r'$error_{π}(h=L(D),c)$' + f' > {ε}: {x[errors > ε].shape[0]}.', s=1.5)
        plt.scatter(x[errors <= ε], errors[errors <= ε], c='green',
                    label=f'Amount of experiments with ' + r'$error_{π}(h=L(D),c)$' + f'≤ {ε}: {x[errors <= ε].shape[0]}.', s=1.5)
        plt.plot(x, [ε for i in range(x.shape[0])], label='desired ε')
        plt.text(0.02, 0.85, horizontalalignment='left', verticalalignment='top', transform=plt.gca().transAxes,
                 s=f'Sample Complexity: {sample_complexities[0]}. \n' +
                   f'Empirical 1 - δ: {1 - approximated_δ}. \n' +
                   r'Average $error_{π}(h=L(D),c)$' + ': {:.4f} \n'.format(errors.mean()),
                 bbox=dict(facecolor='white', alpha=0.5), fontsize=25)
        plt.xlabel("Data set number", fontsize=18)
        plt.ylabel(r'$error_{π}(h=L(D),c)$', fontsize=18)
        plt.legend(prop={'size': 25}, loc='upper left')
        st.pyplot(plt)
        plt.clf()
    return approximated_δ
