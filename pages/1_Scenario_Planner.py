import streamlit as st
from numerize.numerize import numerize
import numpy as np
from functools import partial
from collections import OrderedDict
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from utilities import format_numbers,load_local_css,set_header,initialize_data,load_authenticator,send_email,channel_name_formating
from classes import class_from_dict,class_to_dict
import pickle
import streamlit_authenticator as stauth
import yaml
from yaml import SafeLoader
import re

st.set_page_config(layout='wide')
load_local_css('styles.css')
set_header()

for k, v in st.session_state.items():
    if k not in ['logout', 'login','config'] and not k.startswith('FormSubmitter'):
        st.session_state[k] = v
# ======================================================== #
# ======================= Functions ====================== #
# ======================================================== #


def optimize():
    """
    Optimize the spends for the sales    
    """
    
    channel_list = [key for key,value in st.session_state['optimization_channels'].items() if value]
    print('channel_list')
    print(channel_list)
    print('@@@@@@@@')
    if len(channel_list) > 0 :
        scenario = st.session_state['scenario']
        result = st.session_state['scenario'].optimize(st.session_state['total_spends_change'],channel_list)
        for channel_name, modified_spends in result:
            st.session_state[channel_name] = numerize(modified_spends * scenario.channels[channel_name].conversion_rate,1)
            prev_spends = st.session_state['scenario'].channels[channel_name].actual_total_spends 
            st.session_state[f'{channel_name}_change'] = round(100*(modified_spends - prev_spends) / prev_spends,2)
            

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
        
def update_all_spends():
    """
    Updates spends for all the channels with the given overall spends change
    """
    percent_change = st.session_state['total_spends_change']
    for channel_name in st.session_state['channels_list']:
        channel = st.session_state['scenario'].channels[channel_name]
        current_spends =  channel.actual_total_spends
        modified_spends = (1 + percent_change/100) * current_spends
        st.session_state['scenario'].update(channel_name, modified_spends)
        st.session_state[channel_name] = numerize(modified_spends*channel.conversion_rate,1)
        st.session_state[f'{channel_name}_change'] = percent_change

def extract_number_for_string(string_input):
    string_input = string_input.upper()
    if string_input.endswith('K'):
        return float(string_input[:-1])*10**3
    elif string_input.endswith('M'):
        return float(string_input[:-1])*10**6
    elif string_input.endswith('B'):
        return float(string_input[:-1])*10**9
    
def validate_input(string_input):
    pattern = r'\d+\.?\d*[K|M|B]$'
    match = re.match(pattern, string_input)
    if match is None:
        return False
    return True

def update_data_by_percent(channel_name):
    prev_spends = st.session_state['scenario'].channels[channel_name].actual_total_spends * st.session_state['scenario'].channels[channel_name].conversion_rate
    modified_spends = prev_spends * (1 + st.session_state[f'{channel_name}_change']/100)
    st.session_state[channel_name] = numerize(modified_spends,1)
    st.session_state['scenario'].update(channel_name, modified_spends/st.session_state['scenario'].channels[channel_name].conversion_rate)
    
def update_data(channel_name):
    """
    Updates the spends for the given channel
    """
    
    if validate_input(st.session_state[channel_name]):
        modified_spends = extract_number_for_string(st.session_state[channel_name])
        prev_spends = st.session_state['scenario'].channels[channel_name].actual_total_spends * st.session_state['scenario'].channels[channel_name].conversion_rate
        st.session_state[f'{channel_name}_change'] = round(100*(modified_spends - prev_spends) / prev_spends,2)
        st.session_state['scenario'].update(channel_name, modified_spends/st.session_state['scenario'].channels[channel_name].conversion_rate)
    # st.session_state['scenario'].update(channel_name, modified_spends)
    # else:
    #     try:
    #         modified_spends = float(st.session_state[channel_name])
    #         prev_spends = st.session_state['scenario'].channels[channel_name].actual_total_spends * st.session_state['scenario'].channels[channel_name].conversion_rate
    #         st.session_state[f'{channel_name}_change'] = round(100*(modified_spends - prev_spends) / prev_spends,2)
    #         st.session_state['scenario'].update(channel_name, modified_spends/st.session_state['scenario'].channels[channel_name].conversion_rate)
    #         st.session_state[f'{channel_name}'] = numerize(modified_spends,1)
    #     except ValueError:
    #         st.write('Invalid input')

def select_channel_for_optimization(channel_name):
    """
    Marks the given channel for optimization
    """
    st.session_state['optimization_channels'][channel_name] = st.session_state[f'{channel_name}_selected']

def select_all_channels_for_optimization():
    """
    Marks all the channel for optimization
    """
    for channel_name in st.session_state['optimization_channels'].keys():
        st.session_state[f'{channel_name}_selected' ] = st.session_state['optimze_all_channels']
        st.session_state['optimization_channels'][channel_name] = st.session_state['optimze_all_channels']

def update_penalty():
    """
    Updates the penalty flag for sales calculation
    """
    st.session_state['scenario'].update_penalty(st.session_state['apply_penalty'])
  
def reset_scenario():
    # print(st.session_state['default_scenario_dict'])
    # st.session_state['scenario']  = class_from_dict(st.session_state['default_scenario_dict'])
    # for channel in st.session_state['scenario'].channels.values():
    #     st.session_state[channel.name] = float(channel.actual_total_spends * channel.conversion_rate)
    initialize_data()
    for channel_name in  st.session_state['channels_list']:
        st.session_state[f'{channel_name}_selected'] = False
        st.session_state[f'{channel_name}_change'] = 0
    st.session_state['optimze_all_channels'] = False
    
  
def s_curve(x,K,b,a,x0):
    return K / (1 + b*np.exp(-a*(x-x0)))

@st.cache
def plot_response_curves():
    cols=4
    rcs = st.session_state['rcs']
    shapes = []
    fig = make_subplots(rows=6, cols=cols,subplot_titles=channels_list)
    for i in range(0, len(channels_list)):
        col = channels_list[i]
        x = st.session_state['actual_df'][col].values
        spends = x.sum()
        power = (np.ceil(np.log(x.max()) / np.log(10) )- 3)
        x = np.linspace(0,3*x.max(),200)

        K = rcs[col]['K']
        b = rcs[col]['b']
        a = rcs[col]['a']
        x0 = rcs[col]['x0']
        
        y = s_curve(x/10**power,K,b,a,x0)
        roi = y/x
        marginal_roi = a * (y)*(1-y/K)
        fig.add_trace(
            go.Scatter(x=52*x*st.session_state['scenario'].channels[col].conversion_rate, 
                       y=52*y,
                       name=col, 
                       customdata = np.stack((roi, marginal_roi),axis=-1),
                       hovertemplate="Spend:%{x:$.2s}<br>Sale:%{y:$.2s}<br>ROI:%{customdata[0]:.3f}<br>MROI:%{customdata[1]:.3f}"),
            row=1+(i)//cols , col=i%cols + 1
        )
        
        fig.add_trace(go.Scatter(x=[spends*st.session_state['scenario'].channels[col].conversion_rate], 
                                 y=[52*s_curve(spends/(10**power*52),K,b,a,x0)],
                                 name=col,
                                 legendgroup=col,
                                 showlegend=False,
                                 marker=dict(color=['black'])),
                      row=1+(i)//cols , col=i%cols + 1)
        
        shapes.append(go.layout.Shape(type="line", 
                                      x0=0, 
                                      y0=52*s_curve(spends/(10**power*52),K,b,a,x0), 
                                      x1=spends*st.session_state['scenario'].channels[col].conversion_rate, 
                                      y1=52*s_curve(spends/(10**power*52),K,b,a,x0), 
                                      line_width=1, 
                                      line_dash="dash", 
                                      line_color="black",
                                      xref= f'x{i+1}',
                                      yref= f'y{i+1}'))
        
        shapes.append(go.layout.Shape(type="line", 
                                      x0=spends*st.session_state['scenario'].channels[col].conversion_rate, 
                                      y0=0, 
                                      x1=spends*st.session_state['scenario'].channels[col].conversion_rate, 
                                      y1=52*s_curve(spends/(10**power*52),K,b,a,x0), 
                                      line_width=1, 
                                      line_dash="dash", 
                                      line_color="black",
                                      xref= f'x{i+1}',
                                      yref= f'y{i+1}'))
        
        
        
    fig.update_layout(height=1500, width=1000, title_text="Response Curves",showlegend=False,shapes=shapes)
    fig.update_annotations(font_size=10)
    fig.update_xaxes(title='Spends')
    fig.update_yaxes(title='Prospects')
    return fig



# ======================================================== #
# ==================== HTML Components =================== #
# ======================================================== #

def generate_spending_header(heading):
    return st.markdown(f"""<h2 class="spends-header">{heading}</h2>""",unsafe_allow_html=True)


# ======================================================== #
# =================== Session variables ================== #
# ======================================================== #

with open('config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)
    st.session_state['config'] = config
    
authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['preauthorized']
)
st.session_state['authenticator'] = authenticator
name, authentication_status, username = authenticator.login('Login', 'main')
auth_status = st.session_state.get('authentication_status')
if auth_status == True:
    authenticator.logout('Logout', 'main')
    is_state_initiaized = st.session_state.get('initialized',False)
    if not is_state_initiaized:
        initialize_data()


    channels_list = st.session_state['channels_list']
        

    # ======================================================== #
    # ========================== UI ========================== #
    # ======================================================== #
    
    print(list(st.session_state.keys()))

    st.header('Simulation')
    main_header = st.columns((2,2))
    sub_header = st.columns((1,1,1,1))
    _scenario = st.session_state['scenario']

    with main_header[0]:
        st.subheader('Actual')

    with main_header[-1]:
        st.subheader('Simulated')

    with sub_header[0]:
        st.metric(label = 'Spends', value=format_numbers(_scenario.actual_total_spends))

    with sub_header[1]:
        st.metric(label = 'Prospects', value=format_numbers(float(_scenario.actual_total_sales),include_indicator=False))
        
    with sub_header[2]:
        st.metric(label = 'Spends', 
                    value=format_numbers(_scenario.modified_total_spends),
                    delta=numerize(_scenario.delta_spends,1))
        
    with sub_header[3]:
        st.metric(label = 'Prospects', 
                    value=format_numbers(float(_scenario.actual_total_sales),include_indicator=False), 
                    delta=numerize(_scenario.delta_sales,1))



    with st.expander("Channel Spends Simulator"):
        _columns = st.columns((2,4,1,1))
        with _columns[0]:
            st.checkbox(label='Optimize all Channels',
                            key=f'optimze_all_channels',
                            value=False,
                            on_change=select_all_channels_for_optimization,
                            )
            st.number_input('Percent change of total spends',
                            key=f'total_spends_change',
                            step= 1,
                            on_change=update_all_spends)
        with _columns[2]:
            st.button('Optimize',on_click=optimize)
        with _columns[3]:
            st.button('Reset',on_click=reset_scenario)
            
        
        
        st.markdown("""<hr class="spends-heading-seperator">""", unsafe_allow_html=True)
        _columns = st.columns((2.5,2,1.5,1.5,1))
        with _columns[0]:
            generate_spending_header('Channel')
        with _columns[1]:
            generate_spending_header('Spends Input')
        with _columns[2]:
            generate_spending_header('Spends')
        with _columns[3]:
            generate_spending_header('Prospects')
        with _columns[4]:
            generate_spending_header('Optimize')
            
        st.markdown("""<hr class="spends-heading-seperator">""", unsafe_allow_html=True)
        
        
        for i,channel_name in enumerate(channels_list):
            _channel_class = st.session_state['scenario'].channels[channel_name]
            _columns = st.columns((2.5,1.5,1.5,1.5,1))
            with _columns[0]:
                st.write(channel_name_formating(channel_name))
            with _columns[1]:
                channel_bounds = _channel_class.bounds
                channel_spends = float(_channel_class.actual_total_spends )
                min_value = float((1+channel_bounds[0]/100) * channel_spends )
                max_value = float((1+channel_bounds[1]/100) * channel_spends )
                #print(st.session_state[channel_name])
                spend_input = st.text_input(channel_name,
                              key=channel_name,
                              label_visibility='collapsed',
                              on_change=partial(update_data,channel_name))
                if not validate_input(spend_input):
                    st.error('Invalid input')
                    
                st.number_input('Percent change',
                            key=f'{channel_name}_change',
                            step= 1,
                            on_change=partial(update_data_by_percent,channel_name))
            
            with _columns[2]:
                # spends
                current_channel_spends = float(_channel_class.modified_total_spends *  _channel_class.conversion_rate)
                actual_channel_spends = float(_channel_class.actual_total_spends * _channel_class.conversion_rate)
                spends_delta = float(_channel_class.delta_spends * _channel_class.conversion_rate)
                ## REMOVE 
                st.metric('Spends',
                            format_numbers(current_channel_spends),
                            delta=numerize(spends_delta,1),
                            label_visibility='collapsed')

            with _columns[3]:
                # sales
                current_channel_sales = float(_channel_class.modified_total_sales)
                actual_channel_sales = float(_channel_class.actual_total_sales)
                sales_delta = float(_channel_class.delta_sales)
                st.metric('Prospects',
                            format_numbers(current_channel_sales,include_indicator=False),
                            delta=numerize(sales_delta,1),
                            label_visibility='collapsed')
                
            with _columns[4]:
                
                st.checkbox(label='select for optimization',
                            key=f'{channel_name}_selected',
                            value=False,
                            on_change=partial(select_channel_for_optimization,channel_name),
                            label_visibility='collapsed')


            st.markdown("""<hr class="spends-child-seperator">""",unsafe_allow_html=True)


    with st.expander("See Response Curves"):
        fig = plot_response_curves()
        st.plotly_chart(fig,use_container_width=True)

    _columns = st.columns(2)
    with _columns[0]:
        st.subheader('Save Scenario')
        scenario_name = st.text_input('Scenario name', key='scenario_input',placeholder='Scenario name',label_visibility='collapsed')
        st.button('Save', on_click=lambda  : save_scenario(scenario_name),disabled=len(st.session_state['scenario_input']) == 0)

elif auth_status == False:
    st.error('Username/Password is incorrect')
    
if auth_status != True:
    try:
        username_forgot_pw, email_forgot_password, random_password = authenticator.forgot_password('Forgot password')
        if username_forgot_pw:
            st.session_state['config']['credentials']['usernames'][username_forgot_pw]['password'] = stauth.Hasher([random_password]).generate()[0]
            send_email(email_forgot_password, random_password)
            st.success('New password sent securely')
            # Random password to be transferred to user securely
        elif username_forgot_pw == False:
            st.error('Username not found')
    except Exception as e:
        st.error(e)