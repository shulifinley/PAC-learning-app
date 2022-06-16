import numpy as np
import matplotlib.pyplot as plt
import streamlit as st


def get_circle_points(r):
    '''
    Given a radius returns 100 points evenly ordered in the circle with
    center (0, 0) and radius r. The purpose is to allow drawing a circle.
    '''
    theta = np.linspace(0, 2 * np.pi, 100)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y


def hypothesis_r(points):
    '''
    Compute the hypothesis radius given a set of points.
    Returns the distance of the farthest positive point form the origin.
    '''
    # Get positive points.
    inner_points_mask = (points[:, 2] == 1)
    inner_points = points[inner_points_mask]

    if len(inner_points) == 0:
        return 0.001

    # compute radius. For running time we will only square root the max value.
    r = inner_points[:, 0] ** 2 + inner_points[:, 1] ** 2
    max_r = np.max(r)
    max_r = np.sqrt(max_r)

    return max_r


def graph_annulus(r, points):
    x, y = get_circle_points(r)
    hypothesis = hypothesis(points)
    width = c - hypothesis



def compute_sample_complexity(ε, δ):
    '''
    Computes the sample complexity needed for the concentric circle concept
    class given ε and δ.
    '''
    return int(np.ceil((np.log(1/δ) / ε)))
    # return int(np.ceil((np.log(δ) / np.log(1 - ε))))


def show_experiment_view(hypothesis_radius, data, r, experiment_title, ax, experiment_ε):
    '''
    Shows experiment instance space with the concept c, hypothesis h and training data D.
    '''
    x, y = get_circle_points(r)
    x_hr, y_hr = get_circle_points(hypothesis_radius)
    ax.scatter(data[:, 0], data[:, 1], c=data[:, 2])
    ax.plot(x, y, '-', label='c boundary', c='blue')
    ax.plot(x_hr, y_hr, '-', label='h boundary', c='orange')
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_xlabel('${x}_{1}$', fontdict = {'fontsize' : 20})
    ax.set_ylabel('${x}_{2}$', fontdict = {'fontsize' : 20})
    if experiment_title == '':
        ax.set_title('$X$, ' + '$error_{π}(h=L(D),c)$ ' + '= {:.3f}'.format(experiment_ε), fontdict = {'fontsize' : 20})
    else:
        ax.set_title('$X$, ' + experiment_title + ', ' + '$error_{π}(h=L(D),c)$ ' + '= {:.3f}'.format(experiment_ε), fontdict = {'fontsize' : 20})
    ax.legend(loc='upper left')


def run_experiment(sample_complexity, experiment_title, show_experiment, ax, return_data=False, r=1):
    '''
    Run single experiment with (Concept) Radius = 1.
    '''
    # Change name to sample_size
    # Set Radius
    # r = 1
    # Draw Data
    points = np.random.uniform(low=-2, high=2, size=(sample_complexity, 2))
    labels = points[:, 0] ** 2 + points[:, 1] ** 2 < r ** 2
    data = np.c_[points, labels]
    # Compute hypothesis radius.
    h_radius = hypothesis_r(data)
    error_rate = (np.pi * r ** 2 - np.pi * h_radius ** 2) / 16.0
    if show_experiment:
        show_experiment_view(h_radius, data, r, experiment_title, ax, error_rate)
    # Compute error rate.
    if return_data:
        return data
    else:
        return error_rate


def run_experiment_proof(sample_complexity, experiment_title, show_experiment, ax, return_data=False, r=1):
    '''
    Run single experiment with (Concept) Radius = 1.
    '''
    # Change name to sample_size
    # Set Radius
    # r = 1
    # Draw Data
    points = np.random.uniform(low=-2, high=2, size=(sample_complexity, 2))
    labels = points[:, 0] ** 2 + points[:, 1] ** 2 < r ** 2
    data = np.c_[points, labels]
    # Compute hypothesis radius.
    h_radius = hypothesis_r(data)
    error_rate = (np.pi * r ** 2 - np.pi * h_radius ** 2) / 16.0
    if show_experiment:
        show_experiment_view(h_radius, data, r, experiment_title, ax, error_rate)
    # Compute error rate.
    if return_data:
        return data, r - h_radius
    else:
        return error_rate



def run_experiments(ε, δ, NUM_EXPERIMENTS, NUM_EXPERIMENTS_TO_PLOT, sample_complexity, show=True, r=1):
    '''
    Run couple of experiments for "Experiments and visualization" part.
    '''
    # np.random.seed(120)
    # r = np.random.rand()
    sample_complexities = [sample_complexity]
    amount_of_errors_larger_then_ε = 0
    fig, exp_axs = plt.subplots(nrows=int(NUM_EXPERIMENTS_TO_PLOT / 2), ncols=2,
                                figsize=(14, int(NUM_EXPERIMENTS_TO_PLOT / 2) * 7))
    fig.tight_layout(pad=8.0)

    errors = []
    for _ in range(NUM_EXPERIMENTS):
        experiment_title = ''
        show_experiment = _ < NUM_EXPERIMENTS_TO_PLOT
        ax = None
        if show_experiment:
            ax = exp_axs[int(_ / 2), _ % 2]
        error_rate = run_experiment(sample_complexity, experiment_title, show_experiment, ax, return_data=False, r=r)
        if error_rate > ε:
            amount_of_errors_larger_then_ε += 1
        errors.append(error_rate)
    approximated_δ = amount_of_errors_larger_then_ε / NUM_EXPERIMENTS
    
    if show:
        st.pyplot(plt)
        plt.clf()
        st.subheader(f'All 10k experiments with $m={sample_complexity}$ samples')
        st.write('(May take a few moments to load below)')
        errors = np.array(errors)
        x = np.array([i for i in range(NUM_EXPERIMENTS)])
        plt.figure(figsize=(20, 20))

        plt.scatter(x[errors > ε], errors[errors > ε], c='red',
                    label=f'experiments with ' + r'$error_{π}(h=L(D),c)$' + f' > {ε} ({x[errors > ε].shape[0]} experiments)', s=1.5)
        plt.scatter(x[errors <= ε], errors[errors <= ε], c='green',
                    label=f'experiments with ' + r'$error_{π}(h=L(D),c)$' + f'≤ {ε} ({x[errors <= ε].shape[0]} experiments)', s=1.5)
        plt.plot(x, [ε for i in range(x.shape[0])], label='desired ε')
        plt.text(0.02, 0.85, horizontalalignment='left', verticalalignment='top', transform=plt.gca().transAxes,
                 s=f'sample complexity: {sample_complexities[0]} \n' +
                 f'empirical 1 - δ: {1 - approximated_δ} \n' +
                 r'average $error_{π}(h=L(D),c)$' + ': {:.4f} \n'.format(errors.mean()),
                 bbox=dict(facecolor='white', alpha=0.5), fontsize=25)
        plt.xlabel("dataset index", fontsize=25)
        plt.xticks(fontsize=20)
        plt.ylabel(r'$error_{π}(h=L(D),c)$', fontsize=25)
        plt.yticks(fontsize=20)
        plt.legend(prop={'size': 25}, loc='upper left')
        st.pyplot(plt)
        plt.clf()
    return approximated_δ



def run_experiments_comparison(ε, δ, NUM_EXPERIMENTS, NUM_EXPERIMENTS_TO_PLOT, sample_complexity, show=True, r=1):
    '''
    Run couple of experiments for "Experiments and visualization" part.
    '''
    # np.random.seed(120)
    # r = np.random.rand()
    sample_complexities = [sample_complexity]
    amount_of_errors_larger_then_ε = 0


    errors = []
    for _ in range(NUM_EXPERIMENTS):
        experiment_title = ''
        show_experiment = False
        ax = None

        error_rate = run_experiment(sample_complexity, experiment_title, show_experiment, ax, return_data=False, r=r)
        if error_rate > ε:
            amount_of_errors_larger_then_ε += 1
        errors.append(error_rate)
    approximated_δ = amount_of_errors_larger_then_ε / NUM_EXPERIMENTS
    
    
    return approximated_δ
