from collections import namedtuple
import altair as alt
import math
import pandas as pd
import streamlit as st
import plotly.figure_factory as ff
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder
from st_aggrid.shared import GridUpdateMode
"""
# 信用風險

"""

st.info('This is a purely informational message', icon="ℹ️")

# st.markdown(
#     """
#     <style>
#     h1 {
#         font-size: 20rem !important;
#     }
#     </style>
#     # Head
#     """,
#     unsafe_allow_html=True,
# )
# with st.echo():

cola, colb = st.columns([2, 1])
total_points = st.slider("Number of points in spiral", 1, 5000, 2000)
num_turns = st.slider("Number of turns in spiral", 1, 100, 9)
Point = namedtuple('Point', 'x y')
data = []

points_per_turn = total_points / num_turns

for curr_point_num in range(total_points):
    curr_turn, i = divmod(curr_point_num, points_per_turn)
    angle = (curr_turn + 1) * 2 * math.pi * i / points_per_turn
    radius = curr_point_num / total_points
    x = radius * math.cos(angle)
    y = radius * math.sin(angle)
    data.append(Point(x, y))
df = pd.DataFrame(data)

# with cola:
#     Point = namedtuple('Point', 'x y')
#     data = []

#     points_per_turn = total_points / num_turns

#     for curr_point_num in range(total_points):
#         curr_turn, i = divmod(curr_point_num, points_per_turn)
#         angle = (curr_turn + 1) * 2 * math.pi * i / points_per_turn
#         radius = curr_point_num / total_points
#         x = radius * math.cos(angle)
#         y = radius * math.sin(angle)
#         data.append(Point(x, y))

#     # st.altair_chart(alt.Chart(pd.DataFrame(data), height=500, width=500)
#     #     .mark_circle(color='#0068c9', opacity=0.5)
#     #     .encode(x='x:Q', y='y:Q'))

#     df = pd.DataFrame(data)
#     fig = px.scatter(df,x="x",y="y",
#                     width=200, height=500)
#     fig.update_traces(opacity=.4)
#     tab1, tab2 = st.tabs(["Streamlit theme (default)", "Plotly native theme"])

#     with tab1:
#         # Use the Streamlit theme.
#         # This is the default. So you can also omit the theme argument.
#         st.plotly_chart(fig, theme="streamlit", use_container_width=True)
#     with tab2:
#         # Use the native Plotly theme.
#         st.plotly_chart(fig, theme=None, use_container_width=True)


def aggrid_interactive_table(df: pd.DataFrame):
    """Creates an st-aggrid interactive table based on a dataframe.

    Args:
        df (pd.DataFrame]): Source dataframe

    Returns:
        dict: The selected row
    """
    options = GridOptionsBuilder.from_dataframe(
        df, enableRowGroup=True, enableValue=True, enablePivot=True
    )

    options.configure_side_bar()

    options.configure_selection("single")
    selection = AgGrid(
        df,
        enable_enterprise_modules=True,
        gridOptions=options.build(),
        # theme="dark",
        update_mode=GridUpdateMode.MODEL_CHANGED,
        allow_unsafe_jscode=True,
    )

    return selection

with cola:
    iris = pd.read_csv("/Users/chenkuiming/Downloads/Credit_score/train.csv", nrows=10)
    selection = aggrid_interactive_table(df=iris)

    if selection:
        st.write("You selected:")
        try:
            st.json(dict(Name=selection["selected_rows"][0]["Name"]))
        except:
            st.json([])

with colb:


    style = """
    <style>
        .container { 
        height: 500px;
        position: relative;
        border: 3px solid green; 
        }

        .vertical-center {
        margin: 0;
        position: absolute;
        top: 50%;
        -ms-transform: translateY(-50%);
        transform: translateY(-50%);
        }
    </style>
    """
    try:
        score = selection["selected_rows"][0]["Credit_Score"]
    except:
        score = ""
    st.markdown("""
    {1}
    <div class="container">
        <h1 class="vertical-center" style='text-align: center; color: grey;'>Credit Score: {0}</h1>
    </div>
    """.format(score, style), unsafe_allow_html=True)


col1, col2 = st.columns(2)

with col1:
    table = go.Figure(data=[go.Table(header=dict(values=['A Scores', 'B Scores']),
                    cells=dict(values=[[100, 90, 80, 90], [95, 85, 75, 95]]))
                        ])
    st.plotly_chart(table, theme="streamlit", use_container_width=True)


with col2:
   fig = px.histogram(df, x="x")
   st.plotly_chart(fig, theme="streamlit", use_container_width=True)