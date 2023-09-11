import streamlit as st

# Define the pages of your app
def page1():
  st.write("This is page 1")

def page2():
  st.write("This is page 2")

# Create a sidebar menu
st.sidebar.header("Menu")
button1 = st.sidebar.button("Page 1")
button2 = st.sidebar.button("Page 2")

# Display the selected page
if button1:
  page1()
elif button2:
  page2()