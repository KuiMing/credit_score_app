import altair as alt
import pandas as pd
import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder, ColumnsAutoSizeMode
import pickle
import shap

st.set_page_config(layout="wide")

data = pd.read_csv("train.csv")
name_gender = pd.read_csv("name_gender.csv")

with open("model.pkl", "rb") as f:
    model = pickle.load(f)
f.close()

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


def get_selected_data_and_score(selection, data, predict_col, model):
    selected_user_id = selection["selected_rows"][0]["客戶編號"]
    selected_data = data.loc[data["Customer_ID"] == selected_user_id, predict_col].head(
        1
    )
    predict_data = selected_data.copy()
    score = model.predict(predict_data.to_numpy())[0]
    return selected_data, score


def display_credit_score_view(score, credit, color):
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
    st.markdown(
        f"""
    {style}
    <div class="container">
        <h1 class="vertical-center" style='text-align: center; color: grey;'>
        信用風險:<font style='color: {color[score]};'> {credit[score]}</font></h1>
    </div>
    """,
        unsafe_allow_html=True,
    )


def display_user_detail(selection, data, col_translate):
    behavior = {
        "High_spent_Small_value_payments": "低單價高頻消費",
        "Low_spent_Large_value_payments": "高單價低頻消費",
        "Low_spent_Medium_value_payments": "中單價低頻消費",
        "Low_spent_Small_value_payments": "低單價低頻消費",
        "High_spent_Medium_value_payments": "中單價高頻消費",
        "High_spent_Large_value_payments": "高單價高頻消費",
    }

    career = {
        "Scientist": "科學家",
        "Teacher": "老師",
        "Engineer": "工程師",
        "Entrepreneur": "企業家",
        "Developer": "軟體工程師",
        "Lawyer": "律師",
        "Media_Manager": "媒體經理",
        "Doctor": "醫生",
        "Journalist": "記者",
        "Manager": "經理",
        "Accountant": "會計師",
        "Musician": "音樂家",
        "Mechanic": "技工",
        "Writer": "作家",
        "Architect": "建築師",
    }

    selected_user_id = selection["selected_rows"][0]["客戶編號"]
    text = "#### 該客戶的財務相關資訊：\n"
    occupation = data.loc[data["Customer_ID"] == selected_user_id, "Occupation"].values[
        0
    ]
    text += f"- 職業: {career[occupation]}\n"
    Payment_Behaviour = data.loc[
        data["Customer_ID"] == selected_user_id, "Payment_Behaviour"
    ].values[0]
    text += f"- 消費習慣: {behavior[Payment_Behaviour]}\n"
    selected_data = data.loc[data["Customer_ID"] == selected_user_id, predict_col].head(
        1
    )
    selected_data["Annual_Income"] *= 30
    selected_data["Outstanding_Debt"] *= 30
    selected_data["Monthly_Balance"] *= 30
    selected_data["Monthly_Inhand_Salary"] *= 30
    selected_data.columns = [col_translate[col] for col in selected_data.columns]
    for i in selected_data.columns:
        text += f"- {i}: {int(selected_data[i].values[0])}\n"
    st.markdown(text)


table_view, credit_score_view = st.columns([2, 1])

with table_view:
    table_data = data[
        [
            "Customer_ID",
            "Name",
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
        table_data.groupby(["Customer_ID", "Name"]).mean().reset_index().round()
    )
    table_data["Name"] = name_gender["Name"]
    table_data["Gender"] = name_gender["Gender"]
    table_data.columns = [col_translate[col] for col in table_data.columns]

    selection = aggrid_interactive_table(df=table_data)


with credit_score_view:
    credit = dict(Good="低", Standard="中", Poor="高", Not_Selected="<br>請選擇一名客戶")
    color = dict(
        Good="#6E94F3", Standard="#FD895F", Poor="#F1616D", Not_Selected="<br>grey"
    )
    if selection.selected_rows != []:
        selected_data, score = get_selected_data_and_score(
            selection, data, predict_col, model
        )
    else:
        score = "Not_Selected"

    display_credit_score_view(score, credit, color)

user_detail, score_sorting = st.columns([1, 3])

with user_detail:
    if selection.selected_rows != []:
        display_user_detail(selection, data, col_translate)


shap.initjs()
explainer = shap.TreeExplainer(model)

with score_sorting:
    try:
        # Compute SHAP values for the random person
        predict_data = selected_data.copy()
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
                    alt.value("#6E94F3"),  # 如果是正值，以藍色表示
                    alt.value("#F1616D"),  # 如果是負值，以紅色表示
                ),
                tooltip=["factor", "importance", "value"],
            )
            .configure_axis(labelFontSize=15)  # 改變軸標籤字體大小
            .properties(height=450)  # 改變圖的高度
        )

        # st.markdown("#### 影響信用風險的因子佔比：**:blue[正面影響]** **:red[負面影響]**")#E5765C
        st.markdown(
            """            
            <h4> 影響信用風險的因子佔比：<font style="color: #6E94F3;">正面影響</font> <font style="color: #F1616D;">負面影響</font></h4>
            """,
            unsafe_allow_html=True,
        )
        st.altair_chart(c, use_container_width=True)
    except:
        pass
