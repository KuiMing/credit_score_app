# 個人智慧信用風險評級分析系統：

- 背景設定：
    - 這是一個財務諮詢公司的專案，旨在為會員提供諮詢服務，而信用風險評級是其財務諮詢的核心服務項目之一。
    - 藉由會員的目前財務情況，透過財務公司利用 Tukey 平台開發的人工智慧，衡量會員信用風險評級、貸款的能力好壞，並提供改善信用風險的建議。
- 說明影片：https://youtu.be/CSWmDkJlgCw
- 資料：
    - 來源：https://statso.io/credit-score-classification-case-study/
    - 姓名、性別：https://github.com/cyy0523xc/chinese-name-gender-analyse
    - 資料預處理：
        1. 將美金欄位，全部以匯率 30 換算成台幣金額，並無條件捨去至小數位
        2. 幫使用者加入姓名、性別欄位
        3. 針對同一使用者，只取其中一筆資料
        4. 留下僅需要的資料
- 作法：
    - 演算法：隨機生成樹模型(Random Forest)
    - 使用欄位：年收入、每月收入、銀行數、信用卡數、信用卡最高利息、貸款數量、延遲還款日數、延遲還款次數、信用年紀、未償債務、每月結餘
    - 預測結果：預測會員的信用風險為高、中、低。
- 效益：
    - 此專案導入後，協助此財務公司能夠看快速有初步的判斷，並藉由模型的判讀結果後跟客人開始深度的討論與分析，討論未來財務規劃該如何進行。
        - 可以省下初步評估的人力約 70% 。
        - 可以省下加速客戶諮詢服戶的效率約 30%。


## 預處理後的資料：
- json file: https://drive.google.com/file/d/1_cXiKe2ZtYkv35lByuoZ_GaC8TPnDUGo/view?usp=sharing

### 資料說明：
| 資料欄位 | 欄位名稱 | 欄位解釋 |
| -------- | -------- | -------- |
|Customer_ID | 客戶編號 | 會員唯一的編號 |
|Name | 姓名 | 該會員的名字 |
|Age | 年齡 | 該會員的年齡 |
|Gender | 性別 | 該會員的性別 |
|Occupation | 職業 | 該該會員的職業別 |
|Payment_Behaviour | 付款行為 | 該該會員過去付款行為分析 |
|Annual_Income | 年薪 | 該會員的年收入 |
|Monthly_Inhand_Salary | 月薪 | 該會員的每月實際收入 |
|Num_Bank_Accounts | 銀行帳戶數 | 該會員擁有的銀行賬戶數量 |
|Num_Credit_Card | 信用卡張數 | 該會員擁有的信用卡張數 |
|Interest_Rate | 信貸最高利息 | 該會員各種信貸中最高的利息百分比 |
|Num_of_Loan | 貸款數量 | 該會員借貸的數量 |
|Outstanding_Debt | 未償債務 | 該會員尚未償還的債務金額總和 |
|Credit_History_Age | 信用紀錄月數 | 該會員使用信用的月份數量 |
|Delay_from_due_date | 延遲還款天數 | 平均該會員延遲幾天還款 |
|Num_of_Delayed_Payment | 延遲還款次數 | 該會員延遲還款的次數 |
|Monthly_Balance | 每月結餘 | 該會員每月的結餘金額 |

## Model
- https://drive.google.com/file/d/1ZINWXv8Hj5QdiSUkzVlb17lAHM9geA87/view?usp=sharing

## Docker 

```bash
docker build -t streamlit .
docker run -p 8501:8501 streamlit
```