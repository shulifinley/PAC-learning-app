import streamlit as st
import circlesMain
# import rectanglesMain
import introductionMain
import experimentsMain
# plt.rcParams['figure.figsize'] = [10, 5]

add_footer="""
<style>

footer{
    visibility:visible;
}
footer:after{
    content: 'Shuli Finley, Ilai Genish, and Zohar Yakhini';
    display:block;
    position:relative;
    color:grey;
    # padding:5px;
    top:3px;
}

</style>
"""

def main():
    
    st.markdown(add_footer, unsafe_allow_html=True)

    app_page = st.sidebar.radio('General Topics:',
                                ['Introduction',
                                 'Concentric Circles',
                                 # 'Concentric Rectangles',
                                 'Experiments and Visualizations'])

    if app_page == 'Introduction':
        introductionMain.main()

    if app_page == 'Concentric Circles':
        circlesMain.main()

    # if app_page == 'Concentric Rectangles':
    #     rectanglesMain.main()

    if app_page == 'Experiments and Visualizations':
        experimentsMain.main()

if __name__ == '__main__':
    main()