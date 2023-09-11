import streamlit as st
import plotly.express as px
import numpy as np
import plotly.graph_objects as go
from utilities import channel_name_formating, load_authenticator, initialize_data
from sklearn.metrics import r2_score
from collections import OrderedDict
from classes import class_from_dict,class_to_dict
import pickle
import json

for k, v in st.session_state.items():
    if k not in ['logout', 'login','config'] and not k.startswith('FormSubmitter'):
        st.session_state[k] = v

def s_curve(x,K,b,a,x0):
    return K / (1 + b*np.exp(-a*(x-x0)))

def save_scenario(scenario_name):
    """
    Save the current scenario with the mentioned name in the session state
    
    Parameters
    ----------
    scenario_name
        Name of the scenario to be saved
    """
    if 'saved_scenarios' not in st.session_state:
        st.session_state = OrderedDict()
        
    #st.session_state['saved_scenarios'][scenario_name] = st.session_state['scenario'].save()
    st.session_state['saved_scenarios'][scenario_name] = class_to_dict(st.session_state['scenario'])
    st.session_state['scenario_input'] = ""
    print(type(st.session_state['saved_scenarios']))
    with open('../saved_scenarios.pkl', 'wb') as f:
        pickle.dump(st.session_state['saved_scenarios'],f)


def reset_curve_parameters():
    del st.session_state['K']
    del st.session_state['b']
    del st.session_state['a']
    del st.session_state['x0']
    
def update_response_curve():
    # st.session_state['rcs'][selected_channel_name]['K'] = st.session_state['K']
    # st.session_state['rcs'][selected_channel_name]['b'] = st.session_state['b']
    # st.session_state['rcs'][selected_channel_name]['a'] = st.session_state['a']
    # st.session_state['rcs'][selected_channel_name]['x0'] = st.session_state['x0']
    # rcs = st.session_state['rcs']
    _channel_class = st.session_state['scenario'].channels[selected_channel_name]
    _channel_class.update_response_curves({
                           'K'  : st.session_state['K'], 
                           'b'  : st.session_state['b'], 
                           'a'  : st.session_state['a'],
                           'x0' : st.session_state['x0']})
    

authenticator = st.session_state.get('authenticator')
if authenticator is None:
    authenticator = load_authenticator()

name, authentication_status, username = authenticator.login('Login', 'main')
auth_status = st.session_state.get('authentication_status')

if auth_status == True:
    is_state_initiaized = st.session_state.get('initialized',False)
    if not is_state_initiaized:
        print("Scenario page state reloaded")
        initialize_data()

    st.subheader("Build response curves")
        
    channels_list = st.session_state['channels_list']
    selected_channel_name = st.selectbox('Channel', st.session_state['channels_list'] + ['Others'], format_func=channel_name_formating,on_change=reset_curve_parameters)

    rcs = {}
    for channel_name in channels_list:
        rcs[channel_name] = st.session_state['scenario'].channels[channel_name].response_curve_params
    # rcs = st.session_state['rcs']


    if 'K' not in st.session_state:
        st.session_state['K'] = rcs[selected_channel_name]['K']
    if 'b' not in st.session_state:
        st.session_state['b'] = rcs[selected_channel_name]['b']
    if 'a' not in st.session_state:
        st.session_state['a'] = rcs[selected_channel_name]['a']
    if 'x0' not in st.session_state:
        st.session_state['x0'] = rcs[selected_channel_name]['x0']
        
    x = st.session_state['actual_input_df'][selected_channel_name].values
    y = st.session_state['actual_contribution_df'][selected_channel_name].values

    power = (np.ceil(np.log(x.max()) / np.log(10) )- 3)

    # fig = px.scatter(x, s_curve(x/10**power,
    #                             st.session_state['K'],
    #                             st.session_state['b'],
    #                             st.session_state['a'],
    #                             st.session_state['x0']))

    fig = px.scatter(x=x, y=y)
    fig.add_trace(go.Scatter(x=sorted(x), y=s_curve(sorted(x)/10**power,st.session_state['K'],
                                        st.session_state['b'],
                                        st.session_state['a'],
                                        st.session_state['x0']),
                            line=dict(color='red')))

    fig.update_layout(title_text="Response Curve",showlegend=False)
    fig.update_annotations(font_size=10)
    fig.update_xaxes(title='Spends')
    fig.update_yaxes(title='Prospects')

    st.plotly_chart(fig,use_container_width=True)

    r2 = r2_score(y, s_curve(x / 10**power, 
                            st.session_state['K'],
                            st.session_state['b'],
                            st.session_state['a'],
                            st.session_state['x0']))

    st.metric('R2',round(r2,2))
    columns = st.columns(4)

    with columns[0]:
        st.number_input('K',key='K',format="%0.5f")
    with columns[1]:
        st.number_input('b',key='b',format="%0.5f")
    with columns[2]:
        st.number_input('a',key='a',step=0.0001,format="%0.5f")
    with columns[3]:
        st.number_input('x0',key='x0',format="%0.5f")
        

    st.button('Update parameters',on_click=update_response_curve)
    st.button('Reset parameters',on_click=reset_curve_parameters)
    scenario_name = st.text_input('Scenario name', key='scenario_input',placeholder='Scenario name',label_visibility='collapsed')
    st.button('Save', on_click=lambda  : save_scenario(scenario_name),disabled=len(st.session_state['scenario_input']) == 0)

    file_name = st.text_input('rcs download file name', key='file_name_input',placeholder='file name',label_visibility='collapsed')
    st.download_button(
                        label="Download response curves",
                        data=json.dumps(rcs),
                        file_name=f"{file_name}.json",
                        mime="application/json",
                        disabled= len(file_name) == 0,
                    )