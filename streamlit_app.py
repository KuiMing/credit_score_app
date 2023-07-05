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
import pickle
import numpy as np
import shap

"""
# 信用風險

"""

st.info("This is a purely informational message", icon="ℹ️")


cola, colb = st.columns([2, 1])


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
    options.configure_pagination(
        enabled=True, paginationAutoPageSize=True, paginationPageSize=10
    )
    selection = AgGrid(
        df,
        enable_enterprise_modules=True,
        gridOptions=options.build(),
        # theme="dark",
        update_mode=GridUpdateMode.MODEL_CHANGED,
        allow_unsafe_jscode=True,
    )

    return selection


with open("model.pkl", "rb") as f:
    model = pickle.load(f)
f.close()
cols = [
    "Annual_Income",
    "Monthly_Inhand_Salary",
    "Num_Bank_Accounts",
    "Num_Credit_Card",
    "Interest_Rate",
    "Num_of_Loan",
    "Delay_from_due_date",
    "Num_of_Delayed_Payment",
    "Credit_Mix",
    "Outstanding_Debt",
    "Credit_History_Age",
    "Monthly_Balance",
]
cols_ch = [
    "年薪",
    "月薪",
    "銀行帳戶數",
    "信用卡張數",
    "利息",
    "信貸數量",
    "遲交天數",
    "遲交次數",
    "混合信用分數",
    "未償債務",
    "信用紀錄月數",
    "每月餘額",
]
test_data = pd.DataFrame(
    np.array(
        [
            -1.10399851e-02,
            -8.04193807e-04,
            -1.48516614e-02,
            -8.07955399e-03,
            -4.98199064e-02,
            -1.97553948e-02,
            -6.60816043e-02,
            5.58308875e-03,
            -8.72436039e-03,
            -6.40038640e-02,
            -2.16789976e-02,
            8.16568990e-05,
        ]
    ).reshape(1, -1),
    columns=cols_ch,
)

with cola:
    data = pd.read_csv("/Users/chenkuiming/Downloads/Credit_score/train.csv", nrows=10)
    data = data[cols]

    data["Credit_Mix"] = data["Credit_Mix"].map({"Standard": 1, "Good": 2, "Bad": 0})
    data.columns = cols_ch
    selection = aggrid_interactive_table(df=data)

    # if selection:
    #     st.write("You selected:")
    #     try:
    #         st.json(dict(Name=selection["selected_rows"][0]["Name"]))
    #     except:
    #         st.json([])

with colb:
    style = """
    <style>
        .container { 
        height: 400px;
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
        test_data = selection["selected_rows"][0]
        test_data.pop("_selectedRowNodeInfo")
        test_data = np.array(list(test_data.values()))
        score = model.predict(test_data.reshape(1, -1))[0]
        test_data = pd.DataFrame(test_data.reshape(1, -1), columns=data.columns)
    except:
        score = "Good"
    st.markdown(
        """
    {1}
    <div class="container">
        <h1 class="vertical-center" style='text-align: center; color: grey;'>Credit Score: {0}</h1>
    </div>
    """.format(
            score, style
        ),
        unsafe_allow_html=True,
    )

st.markdown("---")
col1, col2 = st.columns([2, 3])

with col1:
    text = "#### 該交易的各項相關數值\n"
    for i in test_data.columns:
        text += f"- {i}: {test_data[i].values[0]}\n"
    st.markdown(text)


shap.initjs()
explainer = shap.TreeExplainer(model)

with col2:
    try:
        # Compute SHAP values for the random person
        shap_values = explainer.shap_values(test_data)
        print(shap_values[1][0])
        fig = []
        fig.append(
            go.Bar(
                y=test_data.columns,
                x=shap_values[list(model.classes_).index(score)][0],
                name=score,
                orientation="h",
            )
        )

        figure = go.Figure(fig)
        figure.update_layout(title="影響該交易信用風險的因子")
        st.plotly_chart(figure, theme="streamlit", use_container_width=True)
    except:
        pass
