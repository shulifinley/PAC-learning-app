import streamlit as st
import pickle as pkle
import os.path

def show_intro_home():
    '''
    Show the Probably Approximately Correct (PAC) framework topic.
    '''
    st.header('Introduction Overview')
    st.info('In this section we give definitions, explain the general PAC learning framework, and give examples of PAC learning scenarios.  \n'
        'The PAC learning framework is a purely theoretical framework for analyzing sample complexity in the context of learning.')

    st.info('For a precise and detailed discussion of PAC learning, see Shai Shalev-Shwartz\'s and Shai Ben-David\'s [Understanding Machine Learning: From Theory to Algorithms](http://www.cs.huji.ac.il/~shais/UnderstandingMachineLearning), '
            'and Valiant\'s paper which first introduces PAC learning, [A Theory of the Learnable](https://people.mpi-inf.mpg.de/~mehlhorn/SeminarEvolvability/ValiantLearnable.pdf), '
            'and consequent book, [Probably Approximately Correct](https://web.archive.org/web/20170228150047/http://www.probablyapproximatelycorrect.com/).')

    st.info('On the left-hand side of this application, you will find two sections of topics to navigate to. The top section is a list of general topics, and the bottom section is a list of subtopics corresponding to the selected general topic.  \n\n'
    		'Work your way through the application by covering all the subtopics for each general topic, preferably by order of appearance, while referring back as needed.	 \n\n'
            'Enjoy!')

def show_pac_framework_topic():
    '''
    Show the Probably Approximately Correct (PAC) framework topic.
    '''
    st.header('Probably Approximately Correct (PAC) Framework')
    st.subheader('The PAC learning framework involves:')
    st.info('1. A bound on the error in learning (specified by $\epsilon > 0$ error)   \n'
            '2. Probability guarantees on the error in learning (specified by $1-\delta < 1$ certainty)  \n \n'
    		
    		'Given the above specified requirements, we investigate the following resources:   \n \n'
    		
    		'3. Size of training set (sample complexity polynomial in ${1}\over{\epsilon}$, ${1}\over{\delta}$)   \n'
            '4. Time / space of learning (learnability in time polynomial in the size of the training set)  \n')


def show_general_set_up_topic():
    '''
    Show the general set up topic.
    '''
    st.header('General Setup')
    st.subheader('We work within the following fixed setup:')
    st.info('• Probability space $(X, \pi)$ over instance space $X$ and fixed distribution $\pi$  \n'
            '• Concept space $C \subseteq 2^{X}$  \n'
            '• Hypotheses space $H \subseteq 2^{X}$  \n')
    st.subheader('What we want to learn:')
    st.info('Assume we want to learn a concept $c \in C$, within an error bound $\epsilon > 0$, and with a confidence parameter $\delta$.  \n\n'
    		'We consider data drawn from $X$ according to $\pi$, denoted by $D$.  \n'
    		'• $D$ consists of a finite set of points $d_i$ from $X$, labeled as to whether $d_i \in c$ or not.  \n\n'
    		'A learning algorithm (the learner) will output a hypothesis $L(D) = h \in H$. \n\n'
            'The aim is to evaluate the size of the dataset $D$ for which the following is true:  \n'
            '$$\pi(err(c,L(D))>\epsilon) < \delta$$  \n\n'
            'Or in words, the probability of getting more than $\epsilon$ error while learning the concept $c$ with learner $L$ on data $D$ is upper bound by $\delta$.   \n\n'
            'In this formulation, $\pi$ stands for an independent product of the distribution $\pi$ which governs the distribution of $D$, and $err(A,B) = \pi(A \Delta B)$ for any subsets $A, B \in 2^X$.  \n\n'
            'In our case, we are interested in $err(c,h) = \pi(c \Delta h) = $ the probability of observing the points included in $c$ but not in $h$, and points included in $h$ but not in $c$, which are by definition points of error.')


def show_consistency_topic():
    '''
    Show the consistency topic.
    '''
    st.header('Consistency')
    st.subheader('Consistent hypothesis space')
    st.info('A hypothesis space $H$ is $consistent$ with respect to a concept space $C$ if $C \subseteq H$.  \n'
    		'In other words, for any concept we try to learn, $c \in C$, there exists $h \in H$ such that $c=h$.')
    st.subheader('Consistent hypothesis')
    st.info('A hypothesis $h \in H$ is $consistent$ with respect to a concept $c \in C$ and training data $D$ if $∀$ instances $d_i \in D$, $h(d_i) = c(d_i)$.  \n'
    		'Meaning, the output of the hypothesis on each training data point equals the true label.  \n\n'
            'Note that we use the following notation to describe the labels of the data points:  \n\n' 
    		'For data points $d \in X$ and subset $A \in 2^X$ (remember, a concept $c$ and hypothesis $h$ are simply elements of the powerset of $X$),  \n'
    		'$A(d) = 1$ if and only if $d \in A$. Otherwise, $A(d)=0$.')
    st.subheader('Consistent learner')
    st.info('A learning algorithm $L$ using a hypothesis space $H$ to learn concepts from concept space $C$, and operating on training data from $(X,\pi)$ '
            'is said to be a $consistent$ $learner$,  \n'
            'if for any training data $D$ and for any $c \in C$, the output $h = L(D)$ is consistent with respect to $c$ and $D$.')


def show_pac_learnability_topic():
    '''
        Show the PAC Learnability topic.
    '''
    st.header('PAC Learnability')
    st.subheader('Probably Approximately Correct')
    st.info('Consider a space $C$ of possible target concepts defined over the instance space $X$ and a learning algorithm $L$ using hypothesis space $H$.  \n\n'
            '$C$ is $PAC-learnable$ by $L$ using $H$ if  \n' 
            '-for all $0 < \epsilon$,  \n'
            '-for all $0 < \delta$,  \n'
            '-for all $c \in C$,  \n'
            '-for all distributions $\pi$ over $X$,  \n'
            'the following holds:  \n\n'
            'With data drawn independently according to $\pi$, $L$ will output a hypothesis $h \in H$,   \n'
            'with probability of at least $1-\delta$ (probably),  \n'
            'such that $err(h=L(D),c) < \epsilon$ (approximately).  \n\n'
            '$L$ operates in time and sample complexity that is polynomial in ${1 \over ε}$, ${1 \over \delta}$ (efficiently PAC-learnable).  \n')


def show_pac_learnability_using_consistent_H_topic():
    '''
    Show the checking for PAC Learnability of a Concept Class 𝐶 Using a Consistent 𝐻 topic.
    '''
    st.header('PAC learnability of $C \subseteq H$')
    st.info('To prove that a concept space $C$ is efficiently PAC learnable from a hypothesis space $H$ such that $C \subseteq H$, we can do the following:  \n'
            '1. Find (define) a consistent learner.  \n'
            '2. Prove that the sample complexity, $m(\epsilon, \delta)$, is polynomial in ${1 \over ε}$, ${1 \over \delta}$.  \n'
            '3. Show that each step in the learning process is polynomial in the number of training samples $m$.  \n')


def show_what_next_topic():
    '''
    Show the what next topic.
    '''

    st.header('What\'s next?')
    st.info('The next section allows you to explore some cases of PAC learnable concept spaces. At this point we provide only concentric circles, and will expand in the future. \n\n  '
            'Select the next general topic in the menu to continue.  \n\n'
            'Enjoy!')


def main():


    placeholder = st.empty()
    isclick = placeholder.button('Next')
    if isclick:
        if st.session_state['radio_option'] == 'Introduction Overview':
            st.session_state.radio_option = 'Probably Approximately Correct (PAC) Framework'
        elif st.session_state['radio_option'] == 'Probably Approximately Correct (PAC) Framework':
            st.session_state.radio_option = 'General Setup'
        elif st.session_state['radio_option'] == 'General Setup':
            st.session_state.radio_option = 'Consistency'
        elif st.session_state['radio_option'] == 'Consistency':
            st.session_state.radio_option = 'PAC Learnability'
        elif st.session_state['radio_option'] == 'PAC Learnability':
            st.session_state.radio_option = 'PAC Learnability of C ⊆ H'
        elif st.session_state['radio_option'] == 'PAC Learnability of C ⊆ H':
            st.session_state.radio_option = 'What\'s next?'
            if isclick:
                placeholder.empty()

    current_topic = st.sidebar.radio('Introduction Subtopics:',
                                    ['Introduction Overview',
                                      'Probably Approximately Correct (PAC) Framework',
                                      'General Setup',
                                      'Consistency',
                                      'PAC Learnability',
                                      'PAC Learnability of C ⊆ H',
                                      'What\'s next?'], key='radio_option')

    
    st.title("Welcome to the PAC Learning Learnware")
      
    # st.info('In this section we give definitions, explain the general PAC learning framework, and give examples of PAC learning scenarios.  \n')
    
    if current_topic == 'Introduction Overview':
        show_intro_home()
    if current_topic == 'Probably Approximately Correct (PAC) Framework':
        show_pac_framework_topic()
    if current_topic == 'General Setup':
        show_general_set_up_topic()
    if current_topic == 'Consistency':
        show_consistency_topic()
    if current_topic == 'PAC Learnability':
        show_pac_learnability_topic()
    if current_topic == 'PAC Learnability of C ⊆ H':
        show_pac_learnability_using_consistent_H_topic()
    if current_topic == 'What\'s next?':
        show_what_next_topic()


if __name__ == '__main__':
    main()
