import warnings

warnings.filterwarnings("ignore")

import altair as alt
import pandas as pd
import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder, ColumnsAutoSizeMode, AgGridReturn
import pickle
import shap
import dalex as dx


def predict_good(model, newdata):
    return model.predict_proba(newdata)[:, 0]


class CreditPredictor:
    predict_col = [
        "Annual_Income",
        "Monthly_Inhand_Salary",
        "Num_Bank_Accounts",
        "Num_Credit_Card",
        "Num_of_Loan",
        "Interest_Rate",
        "Delay_from_due_date",
        "Num_of_Delayed_Payment",
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
        "Interest_Rate": "信貸最高利息%",
        "Num_of_Loan": "貸款數量",
        "Delay_from_due_date": "延遲還款天數",
        "Num_of_Delayed_Payment": "延遲還款次數",
        "Outstanding_Debt": "未償債務",
        "Credit_History_Age": "信用紀錄月數",
        "Monthly_Balance": "每月結餘",
        "Occupation": "職業",
        "Payment_Behaviour": "付款行為",
        "intercept": "起始點",
        "prediction": "整體情況",
    }
    credit = dict(Good="低", Standard="中", Poor="高", Not_Selected="<br>請選擇一名客戶")
    color = dict(
        Good="#6E94F3", Standard="#FD895F", Poor="#F1616D", Not_Selected="<br>grey"
    )

    def __init__(self, data_file: str, model_file: str):
        self.data = pd.read_json(data_file, orient="records")
        self.model = pickle.load(open(model_file, "rb"))
        # self.explainer = shap.TreeExplainer(self.model)
        self.explainer_good = dx.Explainer(
            self.model,
            self.data[self.predict_col],
            predict_function=predict_good,
            verbose=0,
        )

    def process_table_data(self) -> pd.DataFrame:
        table_cols = [
            "Customer_ID",
            "Name",
            "Age",
            "Gender",
            "Occupation",
            "Annual_Income",
            "Outstanding_Debt",
            "Monthly_Balance",
        ]
        table_data = self.data[table_cols]
        table_data.columns = [self.col_translate[col] for col in table_data.columns]
        return table_data

    def aggrid_interactive_table(self, df: pd.DataFrame) -> AgGridReturn:
        options = GridOptionsBuilder.from_dataframe(
            df, enableRowGroup=True, enableValue=True, enablePivot=True
        )
        options.configure_selection("single")
        selection = AgGrid(
            df,
            gridOptions=options.build(),
            enable_enterprise_modules=True,
            columns_auto_size_mode=ColumnsAutoSizeMode.FIT_ALL_COLUMNS_TO_VIEW,
            suppressColumnVirtualisation=True,
            allow_unsafe_jscode=True,
            height=350,
            custom_css={
                "#gridToolBar .ag-row-odd": {"background-color": "#F1F1F1"},
            },
        )
        return selection

    def get_selected_data_and_score(
        self, selection: AgGridReturn
    ) -> tuple([pd.DataFrame, str]):
        selected_user_id = selection["selected_rows"][0]["客戶編號"]
        selected_data = self.data.loc[
            self.data["Customer_ID"] == selected_user_id, self.predict_col
        ].head(1)
        predict_data = selected_data.copy()
        score = self.model.predict(predict_data.to_numpy())[0]
        return selected_data, score

    def display_credit_score_view(self, score: str) -> None:
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
            信用風險:<font style='color: {self.color[score]};'> {self.credit[score]}</font></h1>
        </div>
        """,
            unsafe_allow_html=True,
        )

    def display_user_detail(self, selection: AgGridReturn) -> None:
        text = "#### 該客戶的財務相關資訊：\n"
        st.markdown(text)

        selected_user_id = selection["selected_rows"][0]["客戶編號"]

        selected_data = self.data.loc[
            self.data["Customer_ID"] == selected_user_id, self.predict_col
        ].head(1)
        selected_data.columns = [
            self.col_translate[col] for col in selected_data.columns
        ]

        df_to_show = pd.DataFrame(
            {"欄位": selected_data.columns, "數值": selected_data.values[0].tolist()}
        )
        edited_df = st.data_editor(df_to_show, width=350, height=430, hide_index=True)

        # make breakdown dataframe

        df_breakdown = (
            pd.Series(edited_df["數值"].values, index=edited_df["欄位"]).to_frame().T
        )

        return df_breakdown

    def display_credit_risk_factor_chart(
        self, selected_data: pd.DataFrame, score: str
    ) -> None:
        predict_data = selected_data.copy()

        shap_values = self.explainer.shap_values(predict_data)
        selected_factor = pd.DataFrame(
            {
                "factor": [self.col_translate[col] for col in selected_data.columns],
                "importance": shap_values[list(self.model.classes_).index(score)][0],
                "value": selected_data.values[0],
            }
        )

        chart = (
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
            .configure_axis(labelFontSize=15)
            .properties(height=450)
        )
        st.markdown(
            """            
            <h4> 影響信用風險的因子佔比：<font style="color: #6E94F3;">正面影響</font> <font style="color: #F1616D;">負面影響</font></h4>
            """,
            unsafe_allow_html=True,
        )
        st.altair_chart(chart, use_container_width=True)

    def sort_contribution(self, df: pd.DataFrame) -> pd.DataFrame:
        df.loc[0, "contribution"] = 1e100
        df.loc[len(df) - 1, "contribution"] = -1e100
        df.sort_values(by="contribution", inplace=True, ascending=False)
        df.loc[0, "contribution"] = df.loc[0, "cumulative"]
        df.loc[len(df) - 1, "contribution"] = df.loc[len(df) - 1, "cumulative"]
        df.loc[0 : len(df) - 1, "cumulative"] = df.contribution[
            0 : len(df) - 1
        ].cumsum()
        df.reset_index(drop=True, inplace=True)
        df.loc[len(df) - 1, "cumulative"] = df.loc[len(df) - 2, "cumulative"]
        return df

    def display_breakdown(self, selected_data: pd.DataFrame) -> None:
        result = self.explainer_good.predict_parts(selected_data, type="break_down")
        result.result = self.sort_contribution(result.result)
        _label = result.result["variable_name"].tolist()
        _label[_label.index("")] = "prediction"
        result.result["variable"] = _label
        result.result["variable"] = [
            self.col_translate[col] for col in result.result.variable
        ]
        result.result["label"] = ""
        fig = result.plot(
            show=False, max_vars=12, vcolors=["RebeccaPurple", "#6E94F3", "#F1616D"]
        )
        fig.layout["annotations"][0]["text"] = ""
        fig.update_annotations(visible=False)
        fig.update_layout(
            title="",
            height=400,
            font_size=16,
            font_color="black",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=10, b=10, t=10, pad=10),
        )
        fig.update_xaxes(gridcolor="#E1E1E1")
        fig.update_yaxes(gridcolor="#E1E1E1")
        score = self.model.predict(selected_data.to_numpy())[0]
        st.markdown(
            f"""            
                <h4> 各因子對信用風險的影響分析：<font style="color: #6E94F3;">正面影響</font> <font style="color: #F1616D;">負面影響</font></h4>
                <h4> 調整後的信用風險:<font style='color: {self.color[score]};'> {self.credit[score]}</font></font></h4>
                """,
            unsafe_allow_html=True,
        )
        st.plotly_chart(fig, use_container_width=True)


def main() -> None:
    predictor = CreditPredictor("preprocessed_data.json", "model.pkl")

    st.set_page_config(layout="wide")

    table_view, credit_score_view = st.columns([2, 1])

    with table_view:
        table_data = predictor.process_table_data()

        selection = predictor.aggrid_interactive_table(df=table_data)

    with credit_score_view:
        if selection.selected_rows != []:
            selected_data, score = predictor.get_selected_data_and_score(selection)

        else:
            score = "Not_Selected"

        predictor.display_credit_score_view(score)

    user_detail, score_sorting = st.columns([1, 3])

    with user_detail:
        if selection.selected_rows != []:
            df_breakdown = predictor.display_user_detail(selection)

    with score_sorting:
        if selection.selected_rows != []:
            predictor.display_breakdown(df_breakdown)


if __name__ == "__main__":
    main()
