# Syria Conflict Dashboard

An interactive Streamlit dashboard analyzing conflict dynamics across Syrian regions using ACLED conflict data and food price data.

## Live Dashboard
🔗 https://syria-conflict-dashboard.streamlit.app/

**Note:** If the app has been inactive, it may need a few seconds to wake up on first load.

## Research Focus

This project examines whether protest activity is associated with:

- Lagged food price changes  
- Lagged repression (proxied by civilian targeting)  
- The interaction between economic shocks and repression  

## Regression Framework

- **Dependent Variable:** Protests  
- **Independent Variables:**  
  - Lagged food price change  
  - Lagged civilian targeting  
  - Interaction term

## Dashboard Features

- Interactive dashboard interface  
- Compare regions over time  
- Track protests, battles, riots, remote violence, and civilian targeting  
- Monitor food prices and price shocks  
- District-level interactive map  
- District-specific regression results

## Run Locally

```bash
streamlit run app.py