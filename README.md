# Marketing Mix Modelling for Product Promotion

## Project Description
This project applies **Marketing Mix Modelling (MMM)** to analyze how different advertising channels drive product sales. By comparing Linear Regression, Ridge, and Gradient Boosting models, we estimate channel contributions, capture diminishing returns, and provide **data-driven budget allocation strategies**.  

---

## Data Source
- **Advertising_Data.csv**: Contains 300 observations of marketing spend across six channels (TV, Billboards, Google Ads, Social Media, Influencer Marketing, Affiliate Marketing) and the target variable `Product_Sold`.

---

## Aim
To evaluate the effectiveness of marketing channels in driving sales and to optimize advertising spend to maximize **ROI**.  

---

## Objectives
1. **Data Exploration & Cleaning** – Check for missing values, outliers, and distributions.  
2. **Channel Contribution Analysis** – Quantify the impact of each channel on sales.  
3. **Model Development** – Build and compare regression-based models.  
4. **Optimisation** – Simulate budget reallocation to maximize sales.  
5. **Strategic Insights** – Provide actionable recommendations.  

---

## Analysis Approach
- **Exploratory Data Analysis (EDA)**: Pairplots, boxplots, and correlation heatmaps.  
- **Feature Engineering**:  
  - *Adstock Transformation*: Captures advertising carryover effect.  
  - *Log Transformation*: Models diminishing returns.  
  - *Moving Averages & Lag Features*: Smooth short-term noise and capture autocorrelation.  
- **Model Comparison**: Linear Regression, Ridge Regression, Gradient Boosting.  
- **Residual Analysis**: Validate model assumptions.  

---

## Tools & Libraries
- **Python** (pandas, numpy, matplotlib, seaborn)  
- **Scikit-learn** (Linear Regression, Ridge, Gradient Boosting, metrics)  
- **Statsmodels** (econometric checks)  
- **Jupyter Notebook** for interactive analysis  

---
## Model Comparison

| Model              | R²     | RMSE   | MAE  | Notes |
|--------------------|--------|--------|------|-------|
| Linear Regression  | >0.999 | ~8.7   | ~7.1 | Best performer – simple & interpretable |
| Ridge Regression   | >0.999 | ~8.7   | ~7.1 | Similar to Linear, stable coefficients |
| Gradient Boosting  | 0.93   | ~457   | High | Overcomplicated, poorer fit |

**Best Model: Linear Regression** – balances accuracy and interpretability, making it ideal for marketing mix analysis.  

---

## Key Insights
- **Affiliate Marketing** → strongest driver of sales (highest elasticity).  
- **Billboards & Social Media** → substantial contributions with strong incremental effects.  
- **TV** → moderate impact, useful for brand visibility.  
- **Google Ads & Influencer Marketing** → weakest contributors, show diminishing returns.  

---

## Final Recommendations
- **Increase investment** in Affiliate Marketing, Billboards, and Social Media.  
- **Maintain moderate spend** on TV for awareness.  
- **Reduce or re-evaluate spend** on Google Ads and Influencer Marketing unless justified by brand goals.  
- Use **elasticity measures** to prioritize channels with the highest marginal ROI.  

---

## Outcome
This project demonstrates how **data-driven marketing mix models** can:  
- Accurately forecast sales performance.  
- Reveal the true contribution of each channel.  
- Guide **optimal budget allocation** to maximize ROI.  
