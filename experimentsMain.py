import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import auxiliaryCircles
import introductionMain
import random
from PIL import Image
from math import floor

add_footer="""
<style>
footer:after{
    content: "Shuli Finley, Ilai Genish, and Zohar Yakhini"
    display:block;
    position:relative;
    color:grey;
    padding:5px;
    top:3px;
}

</style>
"""


def show_overview_topic():

    st.markdown(add_footer, unsafe_allow_html=True)


    st.header('Overview')
    st.info('Consider the concept $c=c(r=1)$.  \n'
            'Fix error and confidence parameters ($\epsilon$ and $1-\delta$) using the sliders below, and observe the calculated sufficient number of samples, $m$.')
    r=1
    eps_interactive = st.slider("Desired error:", 0.0, 0.2, 0.05, 0.01)
    delt_interactive = st.slider("Desired confidence:", 0.0, 1.0, 0.95, 0.01)
    
    sample_complexity = auxiliaryCircles.compute_sample_complexity(eps_interactive, 1 - delt_interactive)

    sample_complexities = [floor(0.5 * sample_complexity),
                           floor(0.75 * sample_complexity),
                           sample_complexity,
                           floor(1.5 * sample_complexity),
                           floor(2 * sample_complexity)]

    st.write('Sufficient samples to achieve $\epsilon$ and $\delta$ as fixed above: ')
    st.write(f'$m=m(\epsilon, \delta)={sample_complexity}$')
    
    st.subheader('In the following sections...')
    st.info('In the following sections, we will compare the performance of $L$ on varying dataset sizes.  \n'
            'In other words, we will explore the effects of $m$ on the performance of our learner, given $\epsilon$ and $\delta$.  \n\n'
            'We present three scenarios, corresponding to the subsections in the menu to the left:  \n'
            '-less than $m$ samples  \n'
            '-$m$ samples  \n'
            '-more than $m$ samples  \n\n'
            'For each scenario, we draw a total of 10k datasets and run our learner $L$ on each one. '
            'We randomly select 6 of the 10k experiments and plot the data and resulting $L(D)=h$, along with the observed error on each dataset.  \n\n'
            'At the bottom of each section, we plot the observed error of all 10k experiments, and observe the percentage of experiments that is achieved the fixed $\epsilon$ value.  \n' 
            'How does this percentage compare to $\delta$?' )

    # global params
    params = {
        'm': sample_complexity,
        'sample_complexities': sample_complexities,
        'epsilon': eps_interactive,
        'delta': delt_interactive
    }
    # st.write(params)
    return params



def show_insufficient_topic(params):
    m = params['m']
    curr_m = params['sample_complexities'][0]
    epsilon = params['epsilon']
    delta = params['delta']
    st.write('Fixed parameters from sliders:')
    st.write(f'$\epsilon: {epsilon}$  \n $\delta: {delta}$  \n sufficient $m(\epsilon, \delta): {m}$')

    st.header('Insufficient Samples')
    st.info('In this section, we run the experiment with 10k different datasets $X$, each with **half** the number of samples that we computed to be sufficient in the previous section, and plot 6 randomly selected experiments out of the 10k.')
    st.subheader(f'6 random experiments with $m = {curr_m}$ samples  \n')

    approx_delta = auxiliaryCircles.run_experiments(epsilon, delta, experiments, separation, curr_m, show=True, r=r)
                                                     
    


def show_sufficient_topic(params):
    m = params['m']
    curr_m = m
    epsilon = params['epsilon']
    delta = params['delta']
    st.write('Fixed parameters from sliders:')
    st.write(f'$\epsilon: {epsilon}$  \n $\delta: {delta}$  \n sufficient $m(\epsilon, \delta): {m}$')

    st.header('Sufficient Samples')
    st.info('In this section, we run the experiment with 10k different datasets $X$, each with **exactly** the number of samples that we computed to be sufficient in the previous section, and plot 6 randomly selected experiments out of the 10k.')
    st.subheader(f'6 random experiments with $m = {curr_m}$ samples  \n')

    approx_delta = auxiliaryCircles.run_experiments(epsilon, delta, experiments, separation, curr_m, show=True, r=r)


def show_more_topic(params):
    m = params['m']
    curr_m = params['sample_complexities'][4]
    epsilon = params['epsilon']
    delta = params['delta']
    st.write('Fixed parameters from sliders:')
    st.write(f'$\epsilon: {epsilon}$  \n $\delta: {delta}$  \n sufficient $m(\epsilon, \delta): {m}$')

    st.header('More Than Sufficient Samples')
    st.info('In this section, we run the experiment with 10k different datasets $X$, each with **twice** the number of samples that we computed to be sufficient in the previous section, and plot 6 randomly selected experiments out of the 10k.')
    st.subheader(f'6 random experiments with $m = {curr_m}$ samples  \n')

    approx_delta = auxiliaryCircles.run_experiments(epsilon, delta, experiments, separation, curr_m, show=True, r=r)

def show_comparison_topic(params):
    m = params['m']
    sample_complexities = params['sample_complexities']
    epsilon = params['epsilon']
    delta = params['delta']
    approximated_δs = []

    experiments = 10000
    separation = 6

    st.write('Fixed parameters from sliders:')
    st.write(f'$\epsilon: {epsilon}$  \n $\delta: {delta}$  \n sufficient $m(\epsilon, \delta): {m}$')
    
    st.header('Comparison')

    st.info('In the previous sections, we observed $L$\'s performance with $m=30, 60, 120$ with 10k independent experiments each.  \n\n'
            'In this section, we run 10k experiments for the values $m = m(\epsilon, \delta) * [0.5, 0.75, 1, 1.75, 2] $, and plot the average empirical $1- \delta$ against $m$.  \n'
            'Practically, $1-\delta$ is the percentage of times that the learner acheived the desired $\epsilon$ value out of the 10k experiments.')

    for i, sample_amount in enumerate(sample_complexities):
        show = False
        approximated_δ = auxiliaryCircles.run_experiments_comparison(epsilon, delta, experiments,
                                                          separation, sample_amount, show=show, r=r)
        approximated_δs.append(approximated_δ)
    
    
    st.subheader('Empirical confidence versus $m$')
    plt.scatter(sample_complexities, 1-np.array(approximated_δs),
                label='$m$ ≠ $m(\epsilon, \delta$)', s=120, color='blue')
    
    plt.scatter(sample_complexities[len(sample_complexities) // 2], 1-approximated_δs[len(approximated_δs) // 2],
                label='$m = m(\epsilon, \delta$)', s=120, color='red')
    # plt.figsize((10,10))
    plt.xlabel("dataset size", fontsize=10)
    plt.ylabel('empirical 1-δ', fontsize=10)
    plt.legend(loc='upper left')
    st.pyplot(plt)
    plt.clf()

    st.subheader('Food for thought')
    st.info('Based on what you\'ve seen in these experiments, how tight is our bound on the sample complexity (the calculation of $m$)?  \n'
            'How do you explain the difference?')
    st.subheader('Concluding remarks')
    st.info('In this application, we demonstrated PAC learning on clean circles in $\mathbb{R}^2$ without errors. '
            'This is the first and simplest example of PAC learning theory.  \n'
            'In the real world, determining sample complexity is a much more complex issue, and many factors are involved.')


def main():
    
    topics = ['Overview', 'Insufficient Samples', 'Sufficient Samples', 'More Than Sufficient Samples', 'Comparison']

    placeholder = st.empty()
    isclick = placeholder.button('Next')

    # next = st.button('Next', )
    # if next:
    for i in range(len(topics)-1):
        if isclick:
            if st.session_state['radio_option'] == topics[i]:
                st.session_state.radio_option = topics[i+1]
                if i == len(topics)-2 :
                    if isclick:
                        placeholder.empty()
                break
            


    current_topic = st.sidebar.radio('Experiments and Visualizations subtopics:',
                                     topics, key='radio_option')

    st.title("Experiments and Visualizations")
    
    # st.info('In this section we demonstrate that the concept class of concentric (centered at the origin) circles is efficiently PAC-learnable.  \n\n')
    # st.info(#'In the first subtopic, we define the instance space. \n\n  '
    #         'In the various subtopics, the slidebar below can be used to fix the radius of a concentric circle from our concept space.')

    global r
    r=1 # default value until the slidebar is introduced in "Generating Datasets section"
    global params
    global experiments
    experiments = 10000
    global separation
    separation = 6
    if current_topic == topics[0]:
        params = show_overview_topic()
    if current_topic == topics[1]:
        show_insufficient_topic(params)
    if current_topic == topics[2]:
        show_sufficient_topic(params)
    if current_topic == topics[3]:
        show_more_topic(params)
    if current_topic == topics[4]:
        show_comparison_topic(params)
    

    

if __name__ == '__main__':
    main()
