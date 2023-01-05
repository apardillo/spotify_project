import streamlit as st
from streamlit_option_menu import option_menu
from _3_graphs import Graphs
import streamlit.components.v1 as components

g=Graphs()


st.set_page_config(layout="wide")



with st.sidebar:
    selection= option_menu(
        menu_title="I Belong to the Top",
        options=["Introduction",
                "Mainstay Artists",
                "Forecasting",
                "Collaborators"], 
        icons=["file-music","vinyl-fill","cash","person-check"],
        menu_icon="file-arrow-up",
        styles={
        #"container": {"padding": "0!important", "background-color": "#fafafa"},
        "container": {"padding": "0!important", "background-color": "#fafafa"},
        #"icon": {"color": "orange", "font-size": "25px"}, 
        "nav-link": {"font-size": "15px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},  ##eee
        #"nav-link-selected": {"background-color": "#eee"},    
        }
       # orientation="horizontal",
        
    )


if selection == "Introduction":





    st.image("pictures/hero_page_4.png", use_column_width=True,caption="From: www.nme.com, Credit:Press" )
    # col1, col2, col3 = st.columns([0.2,0.6,0.2], gap="small")
    # with col1:
    #     st.write("")
    # with col2:
    #     st.image("pictures/hero_page.png", use_column_width=True,caption="From: www.nme.com, Credit:Press")
    # with col3:
    #     st.write("")

    st.markdown("""---""")


    col1, col2 = st.columns([0.6,0.4], gap="large")
    with col1:
        st.markdown("<h1 style='text-align: center; font-size: 30px;color: black;'> \
            Background of the Band-<br> I Belong to The Zoo </h1>" 
            , unsafe_allow_html=True)  
    with col2:  
        st.markdown("Filipino indie rock band that became active on 2014. Became popular, mainly because of their hits 'Sana' and 'Balang Araw' back in 2018. See some of their singles/albums below: __(Source: Youtube/Apple Music)__"
        , unsafe_allow_html=True) 
    

    st.markdown("")
    st.markdown("")


    from PIL import Image
    col1, col2 , col3= st.columns([0.33,0.33,0.33], gap="small")
    with col1:
        st.image("pictures/gallery_1.jpg", use_column_width=True)
        st.image("pictures/gallery_5.jpg", use_column_width=True)

    with col2:  
        st.image("pictures/gallery_2.jpg", use_column_width=True)
        st.image("pictures/gallery_3.jpg", use_column_width=True)   
    with col3:  
        st.image("pictures/gallery_4.jpg", use_column_width=True) 
        st.image("pictures/gallery_6.jpg", use_column_width=True)

    st.markdown("""---""")

    col1, col2 = st.columns([0.3,0.6], gap="large")
    with col1:


        st.markdown("<h1 style='text-align: left; font-size: 25px;color: black;'> \
            However, they faced <br>Recent Struggles After <br>their Initial Hits </h1>" 
            , unsafe_allow_html=True)  

        st.markdown(
        """
        <ul>
            <li style="margin-left: 20px">Streams peaked in 2019</li>
            <li style="margin-left: 20px">No hits since 'Sana' and 'Balang Araw'</li>
            <li style="margin-left: 20px">Sharp decrease in monthly streams and average position</li>
        </ul>
        """,
        unsafe_allow_html=True)

        st.markdown(
        """
        <em>Data Source for all Charts: Spotify API</em> 
        """,
        unsafe_allow_html=True
        )

    with col2:  
        g.plot_1("I Belong to the Zoo")

    st.markdown("""---""")

    col1, col2 = st.columns([0.6,0.3], gap="large")

    with col1:  
        g.plot_2()
    with col2:


        st.markdown("<h1 style='text-align: left; font-size: 25px;color: black;'> \
            Looking back at their hits,<br>this was their formula<br>for success</h1>" 
            , unsafe_allow_html=True)  

        st.markdown("<h1 style='text-align: left; font-size: 20px;color: black;'> \
            Their previous hits were:" 
            , unsafe_allow_html=True)  

        st.markdown(
        """
        <ul>
            <li style="margin-left: 20px">Higher Tempo</li>
            <li style="margin-left: 20px">More Acoustic</li>
            <li style="margin-left: 20px">More Danceable</li>
        </ul>
        """,
        unsafe_allow_html=True)

        st.markdown(
        """
        <em>Compared to their other songs</em> 
        """,
        unsafe_allow_html=True
        )

    st.markdown("""---""")
    st.image("pictures/objectives.png")



    

elif selection == "Mainstay Artists":
    st.markdown("<h1 style='text-align: center; font-size: 55px;color: black;'> \
        Studying Mainstay Artists </h1>" 
        , unsafe_allow_html=True)  


    st.markdown("<h3 style='text-align: center; font-size: 20px;color: black;'> \
            <em>Learning from Artists<br>Dominating Today's Charts</em> </h3>"
            , unsafe_allow_html=True)  

    st.markdown("""---""")
    col1, col2 = st.columns([0.3,0.6], gap="large")

    with col1:  
        st.image("pictures/mainstay_1.png")
    with col2:
        st.image("pictures/mainstay_2.png",caption="Picture Sources: Youtube, Spotify, Manila Standard, @benandbenmusic Instagram,GMA Network")
    st.markdown("""---""")
    col1, col2 = st.columns([0.6,0.3], gap="small")


    col1, col2 = st.columns([0.6,0.3], gap="small")
    with col1:  
        g.plot_3()


    with col2:
        st.markdown("")
        st.markdown("")
        st.markdown("")
        st.markdown("<h1 style='text-align: left; font-size: 25px;color: black;'> \
            Stacked bar chart comparing<br>the streams of the 5 mainstays with other OPM artists that made the charts</h1>" 
            , unsafe_allow_html=True)  

        st.markdown(
        """
        <ul>
            <li style="margin-left: 20px">They comprise 50% of total streams from all OPM artists.</li>
            <li style="margin-left: 20px">Just shows how dominant these 5 mainstay artists are.</li>
            <li style="margin-left: 20px">The Big3 started their rise in April 2021 and have dominanted the charts since.</li>
        </ul>
        """,
        unsafe_allow_html=True)
        
    st.markdown("""---""")
    
    #Select box
    option = st.selectbox(
    'Select Mainstay Artist to see more Info',
    ('Zack Tabudlo', 'Arthur Nery', 'Adie', 'Ben&Ben', 'NOBITA'))


    col1, col2 = st.columns([0.3,0.6], gap="large")

    with col1:
        if option=="Zack Tabudlo":  
            st.image("pictures/zack.png", use_column_width=True)
        elif option=="Arthur Nery":  
            st.image("pictures/arthur.png", use_column_width=True)
        elif option=="Adie":  
            st.image("pictures/adie.png", use_column_width=True)
        elif option=="Ben&Ben":  
            st.image("pictures/bb.png", use_column_width=True)
        elif option=="NOBITA":  
            st.image("pictures/nobita.png", use_column_width=True)

    with col2:
        if option=="Zack Tabudlo":  
            g.plot_1("Zack Tabudlo")
        elif option=="Arthur Nery":  
            g.plot_1("Arthur Nery")
        elif option=="Adie":  
            g.plot_1("Adie")
        elif option=="Ben&Ben":  
            g.plot_1("Ben&Ben")
        elif option=="NOBITA":  
            g.plot_1("NOBITA") 


    st.markdown("")
    if option=="Zack Tabudlo":  
        g.plot_heatmap("Zack Tabudlo",10, 5,1,g.custom_colors_red)
    elif option=="Arthur Nery":  
        g.plot_heatmap("Arthur Nery",14, 5,1,g.custom_colors_red)
    elif option=="Adie":  
        g.plot_heatmap("Adie",10, 3,1,g.custom_colors_red)
    elif option=="Ben&Ben":  
        g.plot_heatmap("Ben&Ben",20, 12,1,g.custom_colors_red)
    elif option=="NOBITA":  
        g.plot_heatmap("NOBITA",10, 1,2.5,g.custom_colors_red)


    st.markdown("""---""")
    #col1, col2 = st.columns([0.6,0.3], gap="small")
    g.plot_4()
    st.markdown("")
    st.image("pictures/mainstay_comparison.png", use_column_width=True)
    st.markdown("""---""")
    
elif selection == "Forecasting":
    st.markdown("<h1 style='text-align: center; font-size: 55px;color: black;'> \
    Forecasting Future Music Trends </h1>" 
    , unsafe_allow_html=True)  


    st.markdown("<h3 style='text-align: center; font-size: 20px;color: black;'> \
            <em>Which Genre will<br>Dominate Tomorow's Charts?</em> </h3>"
            , unsafe_allow_html=True)  

    st.markdown("""---""")
    st.image("pictures/best_model.png", use_column_width=True)
    st.markdown("""---""")
    col1, col2 = st.columns([0.3,0.7], gap="small")
    with col1:

        st.markdown("")
        st.markdown("")
        st.markdown("")
        st.markdown("")
        st.markdown("")
        st.markdown("<h1 style='text-align: left; font-size: 25px;color: black;'> \
            OPM Genre Classification-<br> Hits the Past 5 years</h1>" 
            , unsafe_allow_html=True)  

        st.markdown(
        """
        <ul>
            <li style="margin-left: 20px">Mostly Acoustic and Rock Tracks</li>
            <li style="margin-left: 20px">A few rap and jazz tracks</li>
            <li style="margin-left: 20px">Almost no Dance and Reggae tracks</li>
        </ul>
        """,
        unsafe_allow_html=True)

        st.markdown(
        """
        <em>Normalized by Month</em> 
        """,
        unsafe_allow_html=True
        )

    with col2:  
        #g.plot_1("I Belong to the Zoo")

        g.genre_plot1()

    st.markdown("""---""")
    st.markdown("<h1 style='text-align: center; font-size: 35px;color: black;'> \
    Genre Prediction- Training, Testing and Forecasting</h1>", unsafe_allow_html=True )
    st.markdown("")
    option = st.selectbox(
    'Select Genre',
    ('opm acoustic', 'opm rock', 'opm rap', 'opm jazz'))

    if option=="opm acoustic":  
        g.forecasting("opm acoustic")
    if option=="opm rock":  
        g.forecasting("opm rock")
    if option=="opm rap":  
        g.forecasting("opm rap") 
    if option=="opm jazz":  
        g.forecasting("opm jazz")

    st.image("pictures/reco_forecasting.png", use_column_width=True )

elif selection == "Collaborators":
    st.markdown("<h1 style='text-align: center; font-size: 55px;color: black;'> \
    Finding Collaborators</h1>" 
    , unsafe_allow_html=True)  


    st.markdown("<h3 style='text-align: center; font-size: 20px;color: black;'> \
            <em>Using Similarity Measures to find Compatible Artists</em> </h3>"
            , unsafe_allow_html=True)  
    st.text("")
    st.markdown("""---""")
    #st.text("")
    st.image("pictures/collabs_process.png", use_column_width=True)
    st.markdown("""---""")
    st.image("pictures/collabs_genre.png", use_column_width=True)
    st.markdown("""---""")
    option = st.selectbox(
    'Select Genre to View Possible Collaborators',
    ('Acoustic', 'Jazz', 'Rock'))

    if option=="Acoustic":  
        st.image("pictures/acoustic_artists.png", use_column_width=True)
    if option=="Jazz":  
        st.image("pictures/jazz_artists.png", use_column_width=True)
    if option=="Rock":  
        st.image("pictures/rock_artists.png", use_column_width=True)

    st.markdown("""---""")
    st.markdown("<h1 style='text-align: center; font-size: 40px;color: black;'> \
    Recommender Engine/Playlist of Similar Songs</h1>",unsafe_allow_html=True )

    col1, col2,col3 = st.columns([0.2,0.6,0.2], gap="small")
    with col1:
        pass

    with col2:
        components.iframe("https://open.spotify.com/embed/playlist/20R3vDgunfzviJmCbJsh3e", height=600 ,scrolling=True)
    with col3:
        pass

