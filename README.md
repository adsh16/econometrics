# Groundwater Quality Analysis Project

## Overview
This project investigates the relationship between economic growth and groundwater quality across Indian districts from 2000 to 2019. Using a combination of economic and environmental data, the study aims to uncover patterns and relationships that inform policy and decision-making.

## Data Sources
- **Groundwater Quality Data**: Multiple indicators measured in milligrams per liter for all Indian districts.
- **Economic Data**: Net State Domestic Product (SDP) at constant prices from the Reserve Bank of India’s Database for the Indian Economy (DBIE) portal.
- **Income Inequality Data**: District-level Gini index from Mohanty et al. (2016).

## Methodology
1. **Data Integration**:
   - Merged district-year level groundwater quality data with state-year wise SDP data.
   - Further merged the dataset with district-level Gini index data.

2. **Regression Analysis**:
   - Estimated the relationship between groundwater quality (GWQ) and economic output (SDP):
     \[
     \text{GWQ}_{i,t} = \beta_0 + \beta_1 \text{SDP}_{i,t} + u_{i,t}
     \]
   - Summarized and interpreted regression results.

3. **Residual Analysis**:
   - Visualized model residuals with plots:
     - Groundwater quality indicator on Y-axis and SDP on X-axis.
     - Residuals (\(\hat{u}_{i,t}\)) on Y-axis and SDP on X-axis.
   - Analyzed the expected patterns in the plots.
   - Plotted a histogram of residuals and verified that their sum is zero (\(\sum_{i,t} \hat{u}_{i,t} = 0\)).

4. **Environmental Kuznets Curve (EKC)**:
   - Enhanced the regression model to reflect the non-linear relationship between environmental quality and economic growth.
   - Summarized regression results in a table and prepared detailed summary statistics.
   - Identified and addressed outliers and influential observations.

5. **Temporal Analysis**:
   - Explored if the relationship between economic growth and groundwater quality varied by year.

6. **Regional Disparities**:
   - Enhanced the model to examine regional differences across Indian states as defined by the Reserve Bank of India (RBI).
   - Analyzed the varying estimates of the Kuznets curve across different regions.

## Key Findings
- Articulated the relationship between economic growth (SDP) and groundwater quality.
- Identified non-linear patterns consistent with the Environmental Kuznets Curve.
- Highlighted regional disparities in the economic-environmental relationship.

## Conclusion
This analysis provides valuable insights into how economic growth impacts groundwater quality in India, revealing important non-linear and regional dynamics. The findings can guide policymakers in balancing economic development with environmental sustainability.
