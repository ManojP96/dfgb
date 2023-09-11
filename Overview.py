import streamlit as st
from utilities import (set_header, 
                       initialize_data,
                       load_local_css,
                       create_channel_summary,
                       create_contribution_pie,
                       create_contribuion_stacked_plot,
                       create_channel_spends_sales_plot,
                       format_numbers,
                       channel_name_formating,
                       load_authenticator)
import plotly.graph_objects as go
import streamlit_authenticator as stauth
import yaml
from yaml import SafeLoader
import time

st.set_page_config(layout='wide')
load_local_css('styles.css')
set_header()

for k, v in st.session_state.items():
    print('printing')
    print(k)
    if k not in ['logout', 'login','config'] and not k.startswith('FormSubmitter'):
        st.session_state[k] = v

authenticator = st.session_state.get('authenticator')

if authenticator is None:
    authenticator = load_authenticator()
    
name, authentication_status, username = authenticator.login('Login', 'main')
auth_status = st.session_state['authentication_status']

if auth_status:
    authenticator.logout('Logout', 'main')
    
    is_state_initiaized = st.session_state.get('initialized',False)
    if not is_state_initiaized:
        initialize_data()
    scenario = st.session_state['scenario']
    raw_df = st.session_state['raw_df']
    st.header('Overview of previous spends')


    columns = st.columns((1,1,3))

    with columns[0]:
        st.metric(label = 'Spends', value=format_numbers(float(scenario.actual_total_spends)))
    print(f"##################### {scenario.actual_total_sales} ##################")
    with columns[1]:
        st.metric(label = 'Prospects', value=format_numbers(float(scenario.actual_total_sales),include_indicator=False))


    actual_summary_df = create_channel_summary(scenario)
    actual_summary_df['Channel'] = actual_summary_df['Channel'].apply(channel_name_formating) 

    columns = st.columns((2,1))
    with columns[0]:
        with st.expander('Channel wise overview'):
            st.markdown(actual_summary_df.style.set_table_styles(
            [{
                'selector': 'th',
                'props': [('background-color', '#15C39A')]
            },
                {
                'selector' : 'tr:nth-child(even)',
                'props' : [('background-color', '#15C39A4D')]
                }]).hide_index().to_html(), unsafe_allow_html=True)
            
    st.markdown("<hr>",unsafe_allow_html=True)
    ##############################

    st.plotly_chart(create_contribution_pie(scenario),use_container_width=True)
    st.markdown("<hr>",unsafe_allow_html=True)



    ################################3
    st.plotly_chart(create_contribuion_stacked_plot(scenario),use_container_width=True)
    st.markdown("<hr>",unsafe_allow_html=True)
    #######################################

    selected_channel_name = st.selectbox('Channel', st.session_state['channels_list'] + ['non media'], format_func=channel_name_formating)
    selected_channel = scenario.channels.get(selected_channel_name,None)

    st.plotly_chart(create_channel_spends_sales_plot(selected_channel), use_container_width=True)

    st.markdown("<hr>",unsafe_allow_html=True)

elif auth_status == False:
    st.error('Username/Password is incorrect')
    
if auth_status != True:
    try:
        username_forgot_pw, email_forgot_password, random_password = authenticator.forgot_password('Forgot password')
        if username_forgot_pw:
            st.success('New password sent securely')
            # Random password to be transferred to user securely
        elif username_forgot_pw == False:
            st.error('Username not found')
    except Exception as e:
        st.error(e)