### A/B testing

#### Description

A/B testing to evaluate which version of a landing page converts users the best.

use of:

- `pandas` for data manipulation and exploratory data analysis
- `plotly` for data visualization
- `scipy.stats` for hypothesis testing

#### Files

- `ab_testing.ipynb`: Jupyter notebook with the analysis
- `ab_data.csv`: data from [Kaggle](https://www.kaggle.com/datasets/zhangluyuan/ab-testing) used in the analysis
- `README.md`: this file
- `ab_script.py`: Python script with the analysis

#### Results

The analysis shows that the new landing page does not convert users better than the old landing page. The p-value is 0.1596, which is higher than the significance level of 0.05. Therefore, we fail to reject the null hypothesis.
