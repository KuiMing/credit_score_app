import altair as alt
import pandas as pd
import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder, ColumnsAutoSizeMode
import pickle
import shap

st.set_page_config(layout="wide")

data = pd.read_csv("train.csv")
data["Gender"] = data["Name"].apply(lambda n: "Male" if hash(n) % 2 == 0 else "Female")

with open("model.pkl", "rb") as f:
    model = pickle.load(f)
f.close()
col_translate = {
    "Customer_ID": "客戶編號",
    "Name": "姓名",
    "Gender": "性別",
    "Age": "年齡",
    "Annual_Income": "年薪",
    "Monthly_Inhand_Salary": "月薪",
    "Num_Bank_Accounts": "銀行帳戶數",
    "Num_Credit_Card": "信用卡張數",
    "Interest_Rate": "信貸最高利息",
    "Num_of_Loan": "貸款數量",
    "Delay_from_due_date": "延遲還款天數",
    "Num_of_Delayed_Payment": "延遲還款數量",
    "Credit_Mix": "混合信用分數",
    "Outstanding_Debt": "未償債務",
    "Credit_History_Age": "信用紀錄月數",
    "Monthly_Balance": "每月結餘",
    "Occupation": "職業",
    "Payment_Behaviour": "付款行為",
}

predict_col = [
    "Annual_Income",
    "Monthly_Inhand_Salary",
    "Num_Bank_Accounts",
    "Num_Credit_Card",
    "Num_of_Loan",
    "Interest_Rate",
    "Delay_from_due_date",
    "Num_of_Delayed_Payment",
    # "Credit_Mix",
    "Outstanding_Debt",
    "Credit_History_Age",
    "Monthly_Balance",
]
behavior = {
    "High_spent_Small_value_payments": "低單價高頻消費",
    "Low_spent_Large_value_payments": "高單價低頻消費",
    "Low_spent_Medium_value_payments": "中單價低頻消費",
    "Low_spent_Small_value_payments": "低單價低頻消費",
    "High_spent_Medium_value_payments": "中單價高頻消費",
    "High_spent_Large_value_payments": "高單價高頻消費",
}


def aggrid_interactive_table(df: pd.DataFrame):
    options = GridOptionsBuilder.from_dataframe(
        df, enableRowGroup=True, enableValue=True, enablePivot=True
    )
    options.configure_selection("single")
    selection = AgGrid(
        df.round(),
        gridOptions=options.build(),
        enable_enterprise_modules=True,
        columns_auto_size_mode=ColumnsAutoSizeMode.FIT_ALL_COLUMNS_TO_VIEW,
        suppressColumnVirtualisation=True,
        allow_unsafe_jscode=True,
        height=350,
    )
    return selection


table_view, credit_score_view = st.columns([2, 1])

with table_view:
    table_data = data[
        [
            "Customer_ID",
            "Name",
            "Gender",
            "Age",
            "Annual_Income",
            "Outstanding_Debt",
            "Monthly_Balance",
        ]
    ]
    table_data["Annual_Income"] *= 30
    table_data["Outstanding_Debt"] *= 30
    table_data["Monthly_Balance"] *= 30
    table_data = (
        table_data.groupby(["Customer_ID", "Name", "Gender"])
        .mean()
        .reset_index()
        .round()
    )
    table_data.columns = [col_translate[col] for col in table_data.columns]

    selection = aggrid_interactive_table(df=table_data)
    # if selection:
    #     st.write("You selected:")
    #     try:
    #         st.json(dict(Name=selection["selected_rows"][0]["Name"]))
    #     except:
    #         st.json([])


with credit_score_view:
    style = """
    <style>
        .container { 
        height: 400px;
        position: relative;
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
    credit = dict(Good="低", Standard="中", Poor="高", Not_Selected="<br>請選擇一名客戶")
    try:
        selected_user_id = selection["selected_rows"][0]["客戶編號"]
        selected_data = data.loc[
            data["Customer_ID"] == selected_user_id, predict_col
        ].head(1)
        predict_data = selected_data.copy()
        # predict_data["Credit_Mix"] = predict_data["Credit_Mix"].map(
        #     {"Standard": 1, "Good": 2, "Bad": 0}
        # )
        score = model.predict(predict_data.to_numpy())[0]
    # selected_data = pd.DataFrame(selected_data.reshape(1, -1), columns=data.columns
    except:
        score = "Not_Selected"
    st.markdown(
        """
    {1}
    <div class="container">
        <h1 class="vertical-center" style='text-align: center; color: grey;'>信用風險: {0}</h1>
    </div>
    """.format(
            credit[score], style
        ),
        unsafe_allow_html=True,
    )

# st.markdown("---")
user_detail, score_sorting = st.columns([1, 3])

with user_detail:
    try:
        text = "#### 該客戶的財務相關資訊：\n"
        occupation = data.loc[
            data["Customer_ID"] == selected_user_id, "Occupation"
        ].values[0]

        text += f"- 職業: {occupation}\n"
        Payment_Behaviour = data.loc[
            data["Customer_ID"] == selected_user_id, "Payment_Behaviour"
        ].values[0]
        text += f"- 消費習慣: {behavior[Payment_Behaviour]}\n"
        selected_data["Annual_Income"] *= 30
        selected_data["Outstanding_Debt"] *= 30
        selected_data["Monthly_Balance"] *= 30
        selected_data["Monthly_Inhand_Salary"] *= 30
        selected_data.columns = [col_translate[col] for col in selected_data.columns]
        for i in selected_data.columns:
            text += f"- {i}: {int(selected_data[i].values[0])}\n"
        st.markdown(text)
    except:
        pass


shap.initjs()
explainer = shap.TreeExplainer(model)

with score_sorting:
    try:
        # Compute SHAP values for the random person
        shap_values = explainer.shap_values(predict_data)
        selected_factor = pd.DataFrame(
            {
                "factor": selected_data.columns,
                "importance": shap_values[list(model.classes_).index(score)][0],
                "value": selected_data.values[0],
            }
        )

        c = (
            alt.Chart(selected_factor)
            .transform_calculate(
                abs_importance="abs(datum.importance)",  # 計算絕對值
                sign="datum.importance >= 0",  # 判斷正負
            )
            .mark_bar()
            .encode(
                y=alt.Y(
                    "factor",
                    sort=alt.EncodingSortField(field="importance", order="descending"),
                    title=None,
                ),
                x=alt.X("abs_importance:Q", title=None),
                color=alt.condition(
                    "datum.sign",  # 以 sign 作為條件判斷
                    alt.value("blue"),  # 如果是正值，以藍色表示
                    alt.value("red"),  # 如果是負值，以紅色表示
                ),
                tooltip=["factor", "importance", "value"],
            )
            .configure_axis(labelFontSize=15)  # 改變軸標籤字體大小
            .properties(height=450)  # 改變圖的高度
        )

        st.markdown("#### 影響信用風險的因子佔比：**:blue[正面影響]** **:red[負面影響]**")
        st.altair_chart(c, use_container_width=True)
    except:
        pass
