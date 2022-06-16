import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import auxiliaryCircles
import introductionMain
import random
from PIL import Image
from math import floor


def show_circles_home_topic():
    '''
    Show the Probably Approximately Correct (PAC) framework topic.
    '''
    st.header('Overview')
    st.info('In this section we demonstrate that the concept class of concentric (centered at the origin) circles is efficiently PAC-learnable.  \n\n')
    st.info('We instantiate a full PAC learning instance:  \n'
        '-instance space  \n'
        '-concept  space  \n'
        '-hypothesis space  \n'
        '-training data  \n'
        '-describe a learner, and prove its consistency and efficiency  \n\n'
        'There is also an interactive section in which you can adjust different parameters to observe their effects. ')


def show_instance_space(ax, xlim=(-2, 2), ylim=(-2, 2)):
    '''
    Show the instance space only.
    '''
    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(ylim[0], ylim[1])
    ax.set_title('$D$ drawn from $(X, \pi)$', fontdict = {'fontsize' : 20})
    ax.set_xlabel(r'${x}_{1}$', fontdict = {'fontsize' : 20})
    ax.set_ylabel('${x}_{2}$', fontdict = {'fontsize' : 20})


def show_instance_space_topic():
    '''
    Show the instance space topic.
    '''
    st.header('Instance Space')
    st.info('We will consider training data $D^{(m)}$ of size $m$, drawn from the instance space $(X, \pi)$, where:  \n'
        ' •  $X$ consists of all the points within the square $[-2,2]$ x $[-2,2]$  \n'
        ' •  $\pi$ is the uniform distribution \n\n'
        'More formally,  \n'
        '$(X, \pi) = \{(x_1, x_2) \t{ | } x_1, x_2$ ~ Uniform$(-2,2)$, independently$\}$') # + '  \n '
            # (? r' we are sampling from a uniform distribution on the square' + '  \n ')
            # r'{$[{x}_{1},{x}_{2}] \text{ | } -2 \leq {x}_{1},{x}_{2} \leq 2$} ⊆ ${R}^{2}$ e.g: ${x}_{1}\text{, }{x}_{2}$~𝑈𝑛𝑖𝑓𝑜𝑟𝑚(−2, 2).' + '  \n'
            # 'We call this distribution π.')
    st.info('Below we have an example dataset $D^{(m)}$ drawn from instance space $(X, \pi)$ with $m=200$, independently.  \n\n'
            'The instances will be labelled by the concept $c$, which we introduce in the following section.')
    fig, ax = plt.subplots(num=1, ncols=1, nrows=1, figsize=(10, 10))
    show_instance_space(ax, xlim=(-3, 3), ylim=(-3, 3))
    points = np.random.uniform(low=-2, high=2, size=(200, 2))
    ax.scatter(x=points[:, 0], y=points[:, 1])
    st.pyplot(fig)


def show_concept_space_and_hypothesis_space_topic(r):
    '''
    Show the concept space and hypotheses space topic.
    '''
    random.seed(10)
    st.header('Concept Space')
    fig1, ax1 = plt.subplots(num=2, ncols=1, nrows=1, figsize=(10, 10))
    fig2, ax2 = plt.subplots(num=3, ncols=1, nrows=1, figsize=(10, 10))

    st.info(
            'We now define the concept space $C$, the space of concentric circles:  \n  \n'
            # 'Since we have finite amount of instances in our training set and since often X is infinite the Concept c '
            # 'is usually unknown in real world applications.  \n  \n')
			'Given the fixed radius $r \leq 2$ (as $X$ is defined over the range $[-2,2]$), a concept $c \in C$ is defined by:  \n\n'
            '$c = c(r) = \{(x_1, x_2) \t{ | }$ $x_1^2 + x_2^2 \leq r^2\}$  \n\n'
            'Note that $c$ includes the points on the circle itself as well as all interior points of the circle.  \n\n'
            'Now we can define:  \n\n'
            '$C = \{c(r)\}_{r \leq 2}$.   \n\n'
            'Note that $c(r) \in 2^X$ and $C \subseteq 2^X$.'
            )
            # 'Let\'s take a look at an concentric circles concept space C with simple example where c∈C is circle with '
            # f'r = {r} -  \n'
            # r'c = {$v = [{x}_{1},{x}_{2}] \text{ | } {{x}_{1}}^{2} + {{x}_{2}}^{2} \leq { }' + '{:.2f}'.format(r**2) + '$}' + '  \n'
            # 'or in other terms -  \n'
            # r'c = {${y}_{i}\text{ | }{y}_{i}=1\text{ if }{{x}_{1}}^{2}+{{x}_{2}}^{2}\leq { }' + '{:.2f}'.format(r**2) + r' \text{ else }{y}_{i}=0$}. ' + '  \n  \n'
            # r'Notice that C = {c|c={$v = [{x}_{1},{x}_{2}]\text{ | }{{x}_{1}}^{2}+{{x}_{2}}^{2}\leq{r}^{2}$} s.t $r \leq 2$}.')
    # st.error('above, we say we define concept space C but end up defining concept c')
    st.info('Below we have an example concept $c$ with $r=1$.')
    x, y = auxiliaryCircles.get_circle_points(r)
    show_instance_space(ax1)
    ax1.set_title('Concept $c$, with $r=1$', fontdict = {'fontsize' : 20})
    ax1.plot(x, y, label='c boundary', color='blue')
    ax1.legend(loc='upper left')
    st.pyplot(fig1)
    
    st.header('Hypothesis Space')
    st.info('We will consider a hypothesis space $H$ which is the same as $C$.  \n' 
        'Note that this implies that $H$ is consistent with $C$.')
    
    show_instance_space(ax2)
    x2, y2 = auxiliaryCircles.get_circle_points(r=1.8)
    x3, y3 = auxiliaryCircles.get_circle_points(r=1.3)
    x4, y4 = auxiliaryCircles.get_circle_points(r=0.7)
    x5, y5 = auxiliaryCircles.get_circle_points(r=0.4)

    ax2.set_title('Multiple hypotheses $h$ with varying $r$s', fontdict = {'fontsize' : 20})
    ax2.plot(x2, y2, label='$h_1$ boundary, $r_1$ = 1.8')#, color='orange')
    ax2.plot(x3, y3, label='$h_2$ boundary, $r_2$ = 1.3')#, color='orange')
    ax2.plot(x, y, label='$h_3$ boundary, $r_3$ = 1', color='orange')
    ax2.plot(x4, y4, label='$h_4$ boundary, $r_4$ = 0.7')#, color='orange')
    ax2.plot(x5, y5, label='$h_5$ boundary, $r_5$ = 0.4')#, color='orange')

    ax2.legend(loc='upper left')
    st.pyplot(fig2)


def show_training_data_topic(r):
    st.header('Training Data')
    st.info(
        'When a learning algorithm learns a concept it operates on data.  \n'
        'Our training data sets will have the form $D=[V,Y]$ where:  \n'
        '1. $V$ is the matrix whose rows are the vectors representing instances, $v_1, v_2, ..., v_m$  \n'
        '2. $Y$ is the vector of corresponding labels, as determined by $c \in C$.')
    st.info(
        'As $X$ in our case is the set of points in the square as previously described (see Instance Space section), '
        'therefore our vectors $v_i$ are two-dimensional vectors.  \n\n'
        'Observe sample training data $D$ below, and note that the blue \'$c$ boundary\' doesn\'t really appear as part of the training set, '
        'but we show it for context.'
        )

    
    fig1, ax1 = plt.subplots(num=4, ncols=1, nrows=1, figsize=(10, 10))
    points = np.random.uniform(low=-2, high=2, size=(15, 2))
    labels = points[:, 0] ** 2 + points[:, 1] ** 2 < r ** 2
    data = np.c_[points, labels]
    x, y = auxiliaryCircles.get_circle_points(r)
    show_instance_space(ax1)
    ax1.plot(x, y, label='c boundary', color='blue')
    ax1.scatter(points[labels == 1, 0], points[labels == 1, 1], label = 'label (y) = 1 (in c)', color='orange')
    ax1.scatter(points[labels == 0, 0], points[labels == 0, 1], label = 'label (y) = 0 (not in c)', color='purple')
    ax1.legend(loc='upper left')
    st.pyplot(fig1)
    
    df = pd.DataFrame(data=data, columns=['x1', 'x2', 'Y'])
    df1 = np.trunc(100 * df) / 100
    df1 = df1.astype(str)
    st.dataframe(data=df1, width=1000, height=500)

def show_hypotheses_space_topic(r):
    '''
    Show the hypotheses space topic.
    '''
    st.title('Hypotheses Space')
    st.info('Here we define H to be also a concentric circles or in other terms H is consistent.')
    fig, ax = plt.subplots(num=4, ncols=1, nrows=1, figsize=(10, 10))
    random.seed(10)
    points = np.random.uniform(low=-2, high=2, size=(15, 2))
    labels = points[:, 0] ** 2 + points[:, 1] ** 2 < r ** 2
    data = np.c_[points, labels]
    show_instance_space(ax)
    r_h = auxiliaryCircles.hypothesis_r(data)
    x_h, y_h = auxiliaryCircles.get_circle_points(r_h)
    ax.plot(x_h, y_h, label='h boundary', color='orange')
    ax.scatter(points[labels == 1, 0], points[labels == 1, 1], color='orange')
    ax.scatter(points[labels == 0, 0], points[labels == 0, 1], color='purple')
    ax.legend(loc='upper left')
    st.pyplot(fig)


def show_consistent_learner_topic(r):
    '''
    Show the proposing consistent learner (learning algorithm) topic.
    '''
    st.header('Consistent Learner (learning algorithm)')
    fig, ax = plt.subplots(num=5, ncols=1, nrows=1, figsize=(10, 10))
    st.info('We want to propose a learning algorithm $L$ and show that $L$ is a consistent learner.  \n'
            '(See the [Consistent Learner](#consistent-learner) section to review the definition of a consistent ).')
    # st.markdown("Instance Space](#instance-space)", unsafe_allow_html=True)
    st.info('$L$ fits a hypothesis (from $H$) to the training set '
            'by choosing $r$ to be the distance between the origin (the center of the circle), '
            'and the furthest data point in $D$ with label 1 (i.e., that the point belongs to the concept $c$).')
    st.info('In this case it should be clear why $L$ is a consistent learner.  \n'
            'All positive points in $D$ are inside $h$, and all negative points are outside of $h$,  \n\n'
            'It\'s also clear that the time complexity is polynomial in terms of the number of samples: finding a maximum distance is linear in the number of points .')
    st.info('The figure below allows you to draw training data $D$ with $m = 20$ and observe the resulting output $h=L(D)$, '  
            'while trying to learn the concept $c=c(1)$  \n'
            'The title also shows the resulting error of $h$.')
    auxiliaryCircles.run_experiment(20, '', True, ax, r)
    st.pyplot(fig)
    st.button("Draw another example")
    

def show_generating_data_sets_topic(r):
    '''
    Show the generating data sets topic.
    '''
    st.header('Generating  Datasets')
    # st.info('The slidebar below can be used to fix the radius of a concentric circle from our concept space.')

    fig, ax = plt.subplots(num=6, ncols=1, nrows=1, figsize=(10, 10))
    st.info('Now that we understand our instance space $X$, the concept $c \in C$ (which in practice we don\'t know) '
            'and we proposed a learning algorithm $L$, \n'
            'let\'s generate some training datasets to visualize the effects of the size $m$ of training set $D$, on learning $c$.  \n\n')
    st.info('With the slidebar below you can choose the radius $r$ to defind the concept $c$, and the number of instances $m$ that you want to sample.  \n\n'
            'In the figure below you can see the impact of $m$ on $h$ and on the observed error.')
    # with st.expander("Expander"):
    #     r = st.slider("Radius r:", 0.1, 2.0, 1.0, 0.01)
    r = st.slider("Concept radius r:", 0.1, 2.0, 1.0, 0.01)
    # st.write("--")
    samples_amount = st.slider("Training dataset size m:", 1, 100, 20, 1)
    show_instance_space(ax)
    data = auxiliaryCircles.run_experiment(samples_amount, '', True, ax, True, r=r)
    st.pyplot(fig)
    st.button(f"Draw another example with the same samples number ({samples_amount}).")
    df = pd.DataFrame(data=data, columns=['x1', 'x2', 'Y'])
    df = np.trunc(100 * df) / 100
    df = df.astype(str)
    st.dataframe(data=df, width=None, height=None)


def show_circles_concept_is_efficiently_pac_learnable_topic():
    '''
    Show the concept class of concentric circles is efficiently PAC-learnable topic.
    '''
    st.header('The concept class of concentric circles is efficiently PAC-learnable')
    st.info('PAC-learnability statement:  \n'
            'Given $\epsilon>0$ and $\delta>0$, then with sufficiently many instances $m$,  \n'
            'we can guarantee that our proposed learner $L$ will produce $h=L(D)$ such that  \n'
            '$P(error_{π}(h=L(D),c) > \epsilon) < \delta$ (actual error not training/test error).  \n\n'
            'To show efficient PAC-learnability, we also need to show that:  \n'
            '-the above required number $m$ is polynomial in $\dfrac{1}{\epsilon}$ and $\dfrac{1}{\delta}$  \n'
            '-that the algorithm $L$ is polynomial in $m$ (the input size)  \n\n'
            'As for the second point of efficient PAC-learnability:  \n'
            'Note that $L$ seeks the furthest label-1 instance from the origin and fixes the hypothesis circle at that radius. '
            'This process entails simply computing distances from all the data points to the origin to find the maximal distance, which is clearly linear in $m$.')


            # '\n\nRemember! we need to show that with the right amount of instances, m, in the training set D, where ' 
            # 'm need to be polynomial in terms of 1/ε and 1/δ, we can guarantee that our proposed learner L will '
            # 'produce $error_{π}(h=L(D),c)$ (actual error not training/test set error!) '
            # 'smaller then ε with probability of at least 1-δ, L also need to be efficient '
            # '(polynomial time and space complexity in terms of m).  \n  \n' 
            # 'we split the proof to two cases in the following section.')
    # st.warning('Notice that we keep our proposed consistent learner L with the polynomial time complexity   '
    #            ' in terms of the instance amount in the training set D (e.g m).'
    #            'and we only left to show that m is polynomial in terms of 1/ε and 1/δ and to generalize '
    #            'the proof to any π (not only uniform).')
    st.info('In the next section we address the correctness of the learner $L$.')


def show_correctness_proof_topic(r):
    st.header('Proof, Correctness')
    st.info('To show correctness of $L$, we must show that $L$ is a consistent learner (consistent with the concept it\'s trying to learn, $c$).  \n\n'
        'We show that for all points in the training data, $p$, and for $L(D)=h$, $h(p)=c(p)$.')
    st.info(
        'Let $\epsilon >0, \delta >0$.  \n'
        'Consider $c \in C$, and denote the radius of $c$ by $r$.  \n\n'
        'By definition of $c$, for all interior points $p_{in}$, $c(p_{in})=1$,  \n'
        'and for all exterior points $p_{out}$, $c(p_{out})=0$.  \n\n'
        'Let $p^*$ be the furthest point from the origin in $c$, i.e., $c(p^*)=1$,  \n'
        'and let $r^*$ be the distance of $d^*$ from the origin.  \n\n'
        'Recall that then $L(D)=h^*$ is the circle with radius $r^*$.  \n\n'
        'Therefore, for all interior points of $c$ given in the training data, $p_{in}$, $h^*(p_{in}) = 1$  \n'
        'and for all exterior points of $c$ given in the training data, $p_{out}$, $h^*(p_{out}) = 0$.  \n\n'
        '$\Rightarrow h^*$ is consistent with $c$.'
        )
    st.info('In the next section we address the polynomiality $m$.')


def show_how_many_samples_are_enough_topic(r):
    '''
    Show the how many samples are enough topic.
    '''
    st.header('Proof, Sample Complexity (old)')
    st.info(r'Case 1:' + r'Consider training data D. '
            r'Assume that D visits the annulus created between c and ${c}^{ε}$ defined below, thus h contains ${c}^{ε}$ '
            '(${r}^{ε} \leq r$), so the probability of the annulus created between c and h is less than ε '
            r'(since L is consistent learner)')
    image = Image.open('Images/circles-proof-image.png')
    st.image(image, caption='', use_column_width=True)
    st.info('Case 2: Otherwise, what is the probability of missing the annulus created between c (${r}^{*}$) and ${c}^{ε}$ '
            '(${r}^{ε}$), with m training examples?' + '  \n'
            r'P({$D∈{X}^{M}:error_{π}(h=L(D),c) \geq ε$}) $= {(1 - ε)}^m  \leq exp(-εm)$' + '  \n'
            r'So with sample size $m ≥ \dfrac {ln(\dfrac {1} {δ})} {ε}$ which is '
            r'polynomial in terms of 1/ε and 1/δ, we get ' + '  \n  \n'
            r'$exp(-εm) \leq exp( -ln(\dfrac{1} {δ}))  = exp(ln(δ)) = δ$' + '  \n'
            '• So if the probability of the annulus is very small, the error it incurs is also small' + '  \n'
            '• With enough examples, it is very unlikely to miss the annulus')
    st.info('Notice that in the circles case -  \n' 
            r' $error_{π}(h=L(D),c) <= \dfrac {c-area} {X-area}$' + '  \n'
            f'And in the case of our c (r={r}) -  \n'
            r'$error_{π}(h=L(D),c) = ε \leq \dfrac {\pi*' + '{:.2f}'.format(r) + r'} {16.0}$')

def show_complexity_proof_topic(r):
    st.header('Proof, Sample Complexity')
    st.info(
        'We want to show that:  \n'
        'Given the desired parameters $\epsilon$ and $\delta$, the number of training samples $m$ that is required to guarantee the desired error and confidence, is polynomial in $\dfrac{1}{\epsilon}>0$ and $\dfrac{1}{\delta}>0$.  \n'

        )

    st.info(
        'Let $\epsilon >0, \delta >0$.  \n'
        'Consider $c \in C$, and denote the radius of $c$ by $r$.  \n\n'
        'Now consider the circle $c^{(\epsilon)}$ of radius $r^{(\epsilon)} < r$, '
        'that is formed by moving inward from $c$ until the shaded annulus (the ring between $c$ and $c^{(\epsilon)}$) satisfies $\pi (annulus) = \epsilon$ (see figure).  \n\n'
        'More formally:  \n' 
        'For $s<r$ define the annulus $A_s$ as follows.  \n'
        '$A_s = \{(x_1, x_2) | s \leq d((x_1, x_2), (0,0)) \leq r\}$.  \n\n'
        'Now, $r^{(\epsilon)}$ is defined as  \n'
        '$r^{(\epsilon)} = inf\{ s | \pi(A_s) \leq \epsilon \}$  \n\n'
        'We further use the notation $A^{(\epsilon)} = A_{r^{(\epsilon)}}$. (This is the shaded annulus in Figure 1 below.)'
        )
    # st.write('figure 1')
    fig1 = Image.open('Images/fig1.png')
    st.image(fig1, caption='Figure 1.')
    # fig, ax = plt.subplots(num=6, ncols=1, nrows=1, figsize=(10, 10))
    # r = 1.
    # samples_amount = 20
    # show_instance_space(ax)
    # data, width = auxiliaryCircles.run_experiment_proof(samples_amount, '', True, ax, True, r=r)
    # st.pyplot(fig)

    st.info(
        'Now consider training data $D^{(m)}$, where $|D^{(m)}|=m$.  \n\n'
        'We have two cases:  \n'
        '(1) no points in the training data fall in the annulus, let\'s call this the bad case.  \n'
        '$\Rightarrow$ happens with probability $(1-\epsilon)^m \leq exp^{(- \epsilon m)}$  \n'
        'since for a single point $d \in D^{(m)}$, we have $\pi(d ∉ A^{(\epsilon)}) = 1 - \epsilon $, and the points are independent.  \n\n'
        '(2) there exists a point in the training set that falls in the annulus $A^{(\epsilon)}$, let\'s call this the good case.  \n'
        '$\Rightarrow$ happens with probability $1-(1-\epsilon)^m$ '
        )
    # st.write('figure 2.1 - bad case, figure 2.2 - good case')
    fig2_1 = Image.open('Images/fig2-1.png')
    st.image(fig2_1, caption='Figure 2.1. Bad case.')
    fig2_2 = Image.open('Images/fig2-2.png')
    st.image(fig2_2, caption='Figure 2.2. Good case.')

    st.info(
        'Let\'s explore the good case: at least one of the training points fell in $A^{(\epsilon)}$.  \n'
        'Why do we call this case \'good\'?  \n\n'
        'Because in this case, we have:  \n\n'
        '$err(L(D^{(m)}), c) = \pi(h(D^{(m)}) \Delta c)$, by error definition from the general set up.  \n\n'
        'However,  \n\n'
        '$\pi(h(D^{(m)}) \Delta c) \leq \pi(A^{(\epsilon)})$  \n'
        'because the symmetric difference, in this case, is contained in $A^{(\epsilon)}$ (see Figure 3).  \n\n'
        'And therefore, since $\pi(A^{(\epsilon)}) = \epsilon$ by definition, we now have:  \n\n'
        '$err(L(D^{(m)}), c) \leq \epsilon$'
        # '$\Rightarrow$  $= \pi(c \Delta h(D^{(m)}))$ (?)  \n\n'
        # '$\Rightarrow$  $\leq \pi(A^{(\epsilon)})$, as we\'re in the good case, $c^{(\epsilon)} \subseteq h(D^{(m)})$  \n\n'
        # '$\Rightarrow$  $= \epsilon$, by definition of $A^{(\epsilon)}$  \n\n'
        # 'So, we conclude that in the good case, $err(L(D^{(m)}), c) \leq \epsilon$.'
        )
    # st.write('figure 2')


    # st.info(
    #     'Now, the bad case: none of the training points fell in $A^{(\epsilon)}$.  \n'
    #     'We now estimate the probability of $D^{(m)}$. \n\n'
    #     'Formally, what is $\pi(D^{(m)} \cap A^{(\epsilon)} = \emptyset)$?  \n\n'
    #     # 'For a single point $d \in D^{(m)}$, we have $\pi(x ∉ A^{(\epsilon)}) = 1 - \epsilon$ .  \n\n'
    #     'Since $D^{(m)}$ consists of $m$ independent drawings from $\pi$, (all of which are not in $A^{(\epsilon)}$ in the bad case), we have:  \n\n'
    #     '$\pi(D^{(m)}$ is the bad case$) = (1 - \epsilon)^m \leq exp^{(- \epsilon m)}$'
    #     )
    # st.write('figure 3')
    fig3 = Image.open('Images/fig3.png')
    st.image(fig3, caption='Figure 3.')

    st.info(
        'Finally, assume we can tune $m$ to be so that:  \n'
        '$(**)$ $\pi(D^{(m)}$ is the bad case$)< \delta$  \n\n'
        'We then have:  \n\n'
        '$\pi(err(h(D^{(m)}, c) \leq \epsilon))$  \n\n'
        '$\geq \pi(D^{(m)}$ is the good case$)$  \n\n'
        '$= 1 - \pi(D^{(m)}$ is the bad case$)$  \n\n'
        '$> 1 - \delta$  \n\n'
        'To see what $m$ needs to be in order for $(**)$ to hold:  \n\n'
        '$\pi(D^{(m)}$ is the bad case$) \leq exp^{(- \epsilon m)} < \delta$  \n\n'
        '$\Rightarrow \epsilon m > ln({1 \over \delta})$  \n\n'
        '$\Rightarrow m > {1 \over \epsilon} ln({1 \over \delta})$  \n\n'
        'QED'
        )

def show_experiments_and_visualization_topic(r):
    '''
    Show the experiments and visualization topic.
    '''
    st.header('Experiments and Visualizations')
    st.info('select desired error and probability (ε and 1 - δ).')
    ε_interactive = st.slider("Desired ε:", 0.0, 0.2, 0.05, 0.01)
    δ_interactive = st.slider("Desired 1 - δ:", 0.0, 1.0, 0.95, 0.01)
    experiments = 10000
    separation = 6
    approximated_δs = []
    sample_complexity = auxiliaryCircles.compute_sample_complexity(ε_interactive, 1 - δ_interactive)
    sample_complexities = [floor(0.5 * sample_complexity),
                           floor(0.75 * sample_complexity),
                           sample_complexity,
                           floor(1.5 * sample_complexity),
                           floor(2 * sample_complexity)]
    st.info(f'Using the calculation from previous section we determined that m={sample_complexity} is the '
            'sufficient for your requirements.')
    for i, sample_amount in enumerate(sample_complexities):
        show = False
        if i % 2 == 0:
            show = True
            st.info(f'Here is what happening when we run 10k experiments with m={sample_amount}.')
            st.title(f'10K experiments with {sample_amount} samples  \n'
                     f'sample complexity = sufficient samples amount = {sample_complexity}.')
            st.info(f'From the 10k experiments we are visualizing 6.')
        approximated_δ = auxiliaryCircles.run_experiments(ε_interactive, δ_interactive, experiments,
                                                          separation, sample_amount, show=show, r=r)
        approximated_δs.append(approximated_δ)
    st.info(f'For each m ∈ {sample_complexities} we did 10k experiments and we showed you only the results for m ∈ {[sample_amount for i, sample_amount in enumerate(sample_complexities) if i%2==0]}, '
            f'below you can look how the different m values effect on the empirical 1-δ. (sufficient m is {sample_complexity})')
    plt.scatter(sample_complexities, 1-np.array(approximated_δs),
                label='10K experiments mean empirical confidence for each m', s=120, color='blue')
    plt.scatter(sample_complexities[len(sample_complexities) // 2], 1-approximated_δs[len(approximated_δs) // 2],
                label='10K experiments mean empirical confidence for sufficient m', s=120, color='red')

    plt.xlabel("Data set size", fontsize=18)
    plt.ylabel('Empirical 1-δ', fontsize=18)
    plt.legend(fontsize=18, loc='upper left')
    st.pyplot(plt)
    plt.clf()


def main():

    # option_names = ['Instance Space',
    #               'Concept and Hypothesis Spaces',
    #               'Training Data',
    #               'Consistent Learner (learning algorithm)',
    #               'Generating Datasets',
    #               'The concept class of concentric circles is efficiently PAC-learnable',
    #               'Proof (new)',
    #               'Experiments and Visualization']
    # output_container = st.empty()
    placeholder = st.empty()
    isclick = placeholder.button('Next')

    # next = st.button('Next', )
    # if next:
    if isclick:
        if st.session_state['radio_option'] == 'Concentric Circles Overview':
            st.session_state.radio_option = 'Instance Space'
        elif st.session_state['radio_option'] == 'Instance Space':
            st.session_state.radio_option = 'Concept and Hypothesis Spaces'
        elif st.session_state['radio_option'] == 'Concept and Hypothesis Spaces':
            st.session_state.radio_option = 'Training Data'
        elif st.session_state['radio_option'] == 'Training Data':
            st.session_state.radio_option = 'Consistent Learner (learning algorithm)'
        elif st.session_state['radio_option'] == 'Consistent Learner (learning algorithm)':
            st.session_state.radio_option = 'Generating Datasets'
        elif st.session_state['radio_option'] == 'Generating Datasets':
            st.session_state.radio_option = 'The concept class of concentric circles is efficiently PAC-learnable'
        elif st.session_state['radio_option'] == 'The concept class of concentric circles is efficiently PAC-learnable':
            st.session_state.radio_option = 'Proof, Correctness'
        elif st.session_state['radio_option'] == 'Proof, Correctness':
            st.session_state.radio_option = 'Proof, Sample Complexity'
        # elif st.session_state['radio_option'] == 'Proof, Sample Complexity':
        #     st.session_state.radio_option = 'Experiments and Visualization'
            if isclick:
                placeholder.empty()
            


    current_topic = st.sidebar.radio('Concentric Circles subtopics:',
                                     ['Concentric Circles Overview',
                                      'Instance Space',
                                      'Concept and Hypothesis Spaces',
                                      'Training Data',
                                      # 'Hypotheses Space',
                                      'Consistent Learner (learning algorithm)',
                                      'Generating Datasets',
                                      'The concept class of concentric circles is efficiently PAC-learnable',
                                      'Proof, Correctness',
                                      # 'Proof, Sample Complexity (old)',
                                      'Proof, Sample Complexity',],key='radio_option')

                                      # 'Experiments and Visualization'], key='radio_option')

    st.title("Concentric circles in $\mathbb{R}^2$ are PAC-learnable")
    # st.info('In this section we demonstrate that the concept class of concentric (centered at the origin) circles is efficiently PAC-learnable.  \n\n')
    # st.info(#'In the first subtopic, we define the instance space. \n\n  '
    #         'In the various subtopics, the slidebar below can be used to fix the radius of a concentric circle from our concept space.')

    r=1 # default value until the slidebar is introduced in "Generating Datasets section"
    if current_topic == 'Concentric Circles Overview':
        show_circles_home_topic()
    if current_topic == 'Instance Space':
        show_instance_space_topic()
    if current_topic == 'Concept and Hypothesis Spaces':
        show_concept_space_and_hypothesis_space_topic(r)
    if current_topic == 'Training Data':
        show_training_data_topic(r)
    if current_topic == 'Hypotheses Space':
        show_hypotheses_space_topic(r)
    if current_topic == 'Consistent Learner (learning algorithm)':
        show_consistent_learner_topic(r)
    if current_topic == 'Generating Datasets':
        show_generating_data_sets_topic(r)
    if current_topic == 'The concept class of concentric circles is efficiently PAC-learnable':
        show_circles_concept_is_efficiently_pac_learnable_topic()
    if current_topic == 'Proof, Correctness':
        show_correctness_proof_topic(r)
    # if current_topic == 'Proof, Sample Complexity (old)':
    #     show_how_many_samples_are_enough_topic(r)
    if current_topic == 'Proof, Sample Complexity':
        show_complexity_proof_topic(r)
    # if current_topic == "Experiments and Visualization":
        # show_experiments_and_visualization_topic(r)

    

if __name__ == '__main__':
    main()
