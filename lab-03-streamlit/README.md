# Streamlit

Streamlit is a library that turns Python scripts into rich web applications with dynamic advanced controls. The library is flexible and allows you to create your own components to extend the rich set of built-in components. The best use for the Streamlit library is rapid prototyping of applications that use statistical models.

To install the library, simply issue the command

```bash
pip install streamlit
```

# Basic components

Create a separate directory and place the `helloworld.py` file in it. Place the following code in the file:


```python
import streamlit as st

st.title('Hello world app')
st.header('This is my first Streamlit app')

st.write('Hello, world!')
```

On the command line, navigate to the directory with the file and start the server with the command

```bash
streamlit run helloword.py
```

Open the specified URL in your browser.

Calling the `st.write()` function is redundant because by default Streamlit sends every variable it encounters in the script to this function. See what happens when you add the following line to the file:


```python
_str = 'Hello, universe!'
_str
```

If you want to add some formatted content in a simple and quick way, the easiest way is to use markdown. Add the following line to your document:


```python
st.markdown('> Streamlit is awsome!')
st.markdown('*So is Mikołaj Morzy*')
st.markdown("[Don't click on me](https://theuselessweb.com/)")
```

Similarly, you can attach a fragment of LaTex to the app:


```python
st.latex('\LaTeX: e^{i\pi}+1=0')
```

The Streamlit library is specifically designed to work with data stored in `pandas` dataframes. Import the library, load the data file and display its contents in the application.

```python
import pandas as pd

df = pd.read_csv("titanic.csv")

df
```

Instead of using the dynamic display of the `DataFrame` object, you can also display it statically using `st.table`:

```python
st.table(df.head())
```

Streamlit has a dedicated component for displaying tabular data with the ability to dynamically adjust the column list. Replace the above line with:

```python
cols = ["Name", "Sex", "Age"]

df_multi = st.multiselect("Columns", df.columns.tolist(), default=cols)
st.dataframe(df[df_multi])
```

The data that is already placed in the `DataFrame` object can be used for visualization. Let's prepare the data in such a way that we can display the age distribution of the Titanic's passengers.

```python
age_distribution = df.Age.dropna().value_counts()

st.bar_chart(age_distribution)
```

If you want to enable the conditional display of a part of your application, you can easily do so thanks to the `st.checkbox` component. Replace the last two lines with the code below:


```python
if st.checkbox('Show age distribution?'):
    age_distribution = df.Age.dropna().value_counts()

    st.bar_chart(age_distribution)
```

Another option for modifying the way data is displayed is to use the `st.selectbox` component to select one option among a list of options. Add the following code to the file:

```python
display_sex = st.selectbox('Select sex to display', df.Sex.unique())

st.dataframe(df[df.Sex == display_sex])
```

A large number of controls in the main data display panel may adversely affect the readability of the application. Any component can be automatically moved to the sidebar by replacing the call to `st.component` with `st.sidebar.component`. Try moving the gender selection list to display to the sidebar.

The code displayed in the main panel does not have to cover the entire width of the panel. The panel can be divided into any number of columns using the `st.columns` component.

Place the code below in the file and observe how it works.

```python
left_column, right_column = st.columns(2)

button_clicked = left_column.button('Click me!')

if button_clicked:
    right_column.write("Thank you!")
```

Particularly long descriptions can be placed in the `st.expander` component (you might need to install `lorem` with `pip install lorem`):

```python
import lorem

expander = st.expander("Lorem ipsum")
expander.write(lorem.paragraph())
```

If the script contains a long computation, its progress can be easily reported using the `st.progress` component. Analyze the example below, pay special attention to the use of the `st.empty` component used as a _placeholder_.



```python
import time

'Here we begin a long computation'

current_iteration = st.empty()
progress_bar = st.progress(0)

for i, _ in enumerate(range(100)):
    current_iteration.text(f'Iteration {i+1}')
    progress_bar.progress(i+1)
    time.sleep(0.1)
    
'Finally, computation is completed.'
```

# Advanced components

In the example below, we will generate fictitious Uber orders in Poznań and get acquainted with slightly more advanced methods of data visualization. Create a new file `uber_app.py` and place the following code in it:

```python
import streamlit as st
import pandas as pd
import numpy as np
import random

from datetime import datetime, timedelta

st.title('Uber pickups in Poznan')

def generate_data(n_rows: int = 100) -> pd.DataFrame:
    end = datetime.now()
    start = end - timedelta(days=100)
    
    dates = [
        random.random() * (end - start) + start 
        for _ in range(n_rows)
    ]
    
    lat = np.random.uniform(52.36, 52.46, n_rows)
    lon = np.random.uniform(16.85, 17.00, n_rows)
    
    return pd.DataFrame({
        'date': dates,
        'lat': lat,
        'lon': lon
    })

data = generate_data(n_rows=500)

data
```

The `generate_data()` function is an example of a function whose results could be cached to speed up script execution. Streamlit executes the entire script when any change occurs, so speeding up execution of such script fragments is of great importance.

- change the number of points generated to: 200, 1000, 10000 and observe the time needed to load the page
- decorate the function with the `@st.cache` decorator and compare the page loading time

Currently, the number of generated data points is hard-coded to 500. Let's add a text field that allows the user to enter this number themselves. Modify the above code so that you can pass the desired number of points to the script. Use the [number_input] component for this purpose(https://docs.streamlit.io/library/api-reference/widgets/st.number_input)

```python
def generate_data(...):
    ...
    
n_rows = st.number_input(
    label='How many points to generate?',
    min_value=1,
    max_value=100000,
    value=1000
)

data = generate_data(n_rows=n_rows)
```

Let's add the option to display raw data next to the map

```python
if st.checkbox("Show raw data?"):
    st.subheader('Raw data')
    st.dataframe(data)
```

Streamlit allows you to include virtually all types of charts in your application. Its upports charts generated by Matplotlib, Altair, Bokeh, GraphViz and many others. In the next step, we will determine the histograms of orders with an hourly precision and display them.

```python
hist_vals = np.histogram(data.date.dt.hour, bins=24)[0]

st.bar_chart(hist_vals)
```

A convenient way to generate a filter for numeric data is the `st.slider` component. We will now allow users to narrow down the hours to display.


```python
hour_filter = st.slider('hour', 0, 23, 17)

df_filtered = data[data.date.dt.hour == hour_filter]

st.subheader(f'Map of all uber pickups at {hour_filter}:00')

st.map(df_filtered)
```

Currently, the `st.slider` component allows only one value to be selected. If we want to use a range of hours to generate a map, we can use the `st.select_slider` component ([link to API](https://docs.streamlit.io/library/api-reference/widgets/st.select_slider)).


```python
min_hour, max_hour = st.select_slider(
    label='hours', 
    options=range(24),
    value=(7,16)

filter_idx = (data.date.dt.hour >= min_hour) & (data.date.dt.hour <= max_hour)
df_filtered = data[filter_idx]

st.subheader(f'Map of all uber pickups between {min_hour}:00 and {max_hour}:00')

st.map(df_filtered)
```

Previously, all components responded immediately to a change, which made it impossible to report multiple variables at the same time. The solution is to use the `st.form` component ([link to API](https://docs.streamlit.io/library/api-reference/control-flow/st.form))


```python
with st.form(key='my_form'):
    name_input = st.text_input('Name:')
    dob_input = st.date_input('Date of birth:')
    weight_input = st.number_input('Weight (kg):')
    height_input = st.number_input('Height (cm):')
    
    submit_btn = st.form_submit_button('Compute BMI')
    
if submit_btn:
    bmi = weight_input / (height_input/100)**2
    st.write(f'Hello {name_input}, your BMI={bmi:.2%f}')
```

We have already encountered the need to use a _placeholder_ to provide space for inserting data to an earlier place in the application script. The following example shows how to handle such components.

```python
st.text('First line')

empty_text = st.empty()
empty_chart = st.empty()

st.text('Fourth line')

empty_text.text('Second line')

empty_chart.line_chart(np.random.randn(50,2))
```

As a reward for reaching this point in the tutorial, add the following command at the end of the script:

```python
st.balloons()
```

# Independent task

Load the [Boston Apartments](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_boston.html) dataset from the `scikit-learn` library. 
You might need to install `sklearn 1.1.3` as the dataset has been removed for ethical concerns in `1.2`: `pip install scikit-learn==1.1.3`
Then prepare an analysis that will include:

- displaying the data set
- ability to display only the properties that are located on the river (checkbox)
- a filter allowing you to indicate the tax scope
- a chart showing the distribution of average house values
- a simple regression model (e.g. decision tree) that determines the value of the house based on a selected subset of 3 parameters
- a form accepting the selected parameters and displaying a prediction of the house value

---

# Creating your own components (advanced)

Streamlit is a very flexible environment in which we can create our own component relatively easily. However, this will require both front-end and back-end development.

First, clone the following repository [https://github.com/streamlit/component-template](https://github.com/streamlit/component-template) locally and install the necessary npm packages and start the webpack server.


```
git clone https://github.com/streamlit/component-template

cd component-template/template/my_component/frontend

npm install

npm run start
```

Go to the directory `component-template/template/my_component` and modify the file `__init__.py` . Place the code below in it:

```python
import os
import streamlit as st
import streamlit.components.v1 as components

st.title('My component example')

_component_func = components.declare_component(
    "my_component",
    url="http://localhost:3001",
)

def my_component(start, key=None):
    component_value = _component_func(start=start, key=key, default=100)

    return component_value

counter = my_component(10)

st.markdown(f"You have {counter} clicks left!")
```

Open a new terminal and run the component:

```bash
streamlit run component-template/template/my_component/__init__.py
```

Then go to the directory `component-template/template/my component/frontend/src/MyComponent.tsx` and place the following code:


```python
import {
  Streamlit,
  StreamlitComponentBase,
  withStreamlitConnection,
} from "streamlit-component-lib"
import React, { ReactNode } from "react"

interface State {
  counter: number
}

class MyComponent extends StreamlitComponentBase<State> {
  public state = { counter: this.props.args['start'] }

  public render = (): ReactNode => {

    return (
      <span>
        Clicks remaining: {this.state.counter} &nbsp;
        <button
          onClick={this.onClicked}
        >
          Click me!
        </button>
      </span>
    )
  }

  private onClicked = (): void => {
    if (this.state.counter > 0) {
        this.setState(
        prevState => ({ counter: prevState.counter - 1 }),
        () => Streamlit.setComponentValue(this.state.counter)
        )
    } 
  }
}

export default withStreamlitConnection(MyComponent)
```

