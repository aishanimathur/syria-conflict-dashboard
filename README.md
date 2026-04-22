# Syria Conflict Dashboard

An interactive Streamlit dashboard analyzing protests, violence, food prices, and repression across Syrian districts over time.

## Live App
[Open Dashboard](https://syria-conflict-dashboard.streamlit.app/)

*(App may take a few seconds to wake up.)*

## Project Goals

- Visualize protests, conflict events, and food prices across Syria
- Compare districts over time
- Explore whether food prices and repression relate to protest activity
- Run district-level exploratory regressions

## Data Used

- ACLED conflict and protest event data
- WFP food price data
- Syria administrative boundary shapefiles

## Key Findings

- Protest dynamics differ sharply across districts
- Aleppo and Idleb show consistently high protest activity
- As-Sweida drives the major 2023 protest surge
- Food prices alone do not uniformly predict protest
- Repression often correlates positively with later protest activity
- The strongest evidence supports an interaction effect: economic shocks matter differently depending on repression levels

## Methods

- Time-series visualization
- Monthly lags
- District-level correlations
- OLS regressions by district

## Limitations

- Observational data (not causal)
- Small samples in some districts
- Omitted variables likely matter
- Event counts do not capture protest size

## Next Steps

- Improve model specification
- Add more controls
- Expand case-study annotations