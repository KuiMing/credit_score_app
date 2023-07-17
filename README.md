# 專案說明文件：
- 專案說明：
	- 背景設定：
		- 這專案是設想有一間財務諮詢公司，提供會員諮詢的服務，包含：財務狀況衡量、貸款額度、利率評估、信用風險評級...等，而這個應用就是他們提供會員信用風險評級用的工具。
		- 藉由會員的目前財務情況，透過財務公司利用 Tukey 平台開發的人工智慧，衡量會員信用風險評級、貸款的能力好壞，進一步提供若要改善信用風險評級，應該從何處開始改善。是本公司提供諮詢的工具。
	- 資料：
		- 來源：https://statso.io/credit-score-classification-case-study/
		- 姓名、性別：https://github.com/cyy0523xc/chinese-name-gender-analyse
	- 作法：
		- 模型：隨機生成樹模型(Random Forest)
		- 使用欄位：年收入、每月收入、銀行數、信用卡數、信用卡最高利息、貸款數量、延遲還款日數、延遲還款數量、信用年紀、未償債務、每月結餘
		- 推論：信用評級為高、中、低，其中一個。
	- 效益：
		- 此專案導入後，協助此財務公司能夠看快速有初步的判斷，並藉由模型的判讀結果後跟客人開始深度的討論與分析，討論未來財務規劃該如何進行。
			- 可以省下初步評估的人力約 70% 。
			- 可以省下加速客戶諮詢服戶的效率約 30%。


## 預處理後的資料：
- 

## Model
- https://drive.google.com/file/d/1ZINWXv8Hj5QdiSUkzVlb17lAHM9geA87/view?usp=sharing

## Docker 

```bash
docker build -t streamlit .
docker run -p 8501:8501 streamlit
```