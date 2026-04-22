# Syria Conflict, Prices, and Protest Dashboard – Key Observations

## Project Aims

This project has two main goals. First, it aims to visually present available Syria data across districts, including protests, different forms of violence, food prices, and food price change over time. Second, it explores whether there are observable relationships between these variables, particularly whether changes in food prices and civilian targeting are associated with later protest activity. To examine this, I also run district-level regressions. These results should be interpreted cautiously and as exploratory rather than causal.

## Important Limitations

Several limitations should be acknowledged. First, this is observational data, so correlations and regressions cannot establish causation. Second, the models are intentionally simple and likely omit important drivers of protest such as local politics, aid access, military offensives, governance differences, migration, and reporting intensity. Third, protest counts are event data, meaning they capture frequency rather than protest size or political importance. Fourth, monthly lagging by one month may not perfectly match real behavioral timelines. Finally, relatively few districts show statistically significant effects, which may reflect both genuine heterogeneity and limited model power.

---

## What I Observe from the Data

## 1. Protest Trends Over Time

Overall protests show a clear spike in 2022 and another major increase in 2023. Looking at the maps, Idleb and Aleppo account for consistently high protest activity over the full period, with As-Sweida also becoming highly important. The monthly view makes clear that As-Sweida is a major driver of the 2023 surge. This corresponds to the documented protest wave that began in August 2023 following subsidy cuts, fuel price increases, and broader economic collapse. Prior to this, protest levels in As-Sweida were relatively low, suggesting that mobilization there was triggered by a specific political-economic turning point rather than a long-running trend.

For the 2022 increase, filtering the map to only 2022 shows Aleppo as especially prominent, indicating that the earlier spike appears more connected to unrest in northern districts than to the later Sweida movement.

---

## 2. Food Price Change and Protests

I next examined whether lagged food price change helps explain protests. Lagging here means comparing current protests to food price change in the previous month. At the national level, the scatterplot shows no obvious simple relationship. When I calculate district-level correlations, many are weak or negative rather than strongly positive.

This suggests that higher food prices do not automatically generate more protest. In some places, severe hardship may reduce people’s capacity to mobilize. In others, humanitarian aid, displacement, informal coping systems, or local political conditions may weaken the direct relationship between prices and protest.

Several districts on the map exhibit weak or negative correlations, reiterating the need for contextual understanding.

---

## 3. Repression and Protests

I then examined whether lagged civilian targeting (used here as a proxy for repression) is associated with later protests. At the overall Syria level, the scatterplot does not show a clean linear relationship. Even when lagged civilian targeting becomes very high, protest counts are often modest.

This could mean several things: repression may suppress protest in some contexts, high repression may occur in places where protest capacity is already low, or national pooling may hide important regional differences.

The district-level correlation map is more informative. In many districts, higher lagged civilian targeting is associated with somewhat higher later protest activity. This is especially visible in Aleppo, where the relationship appears stronger than in most other districts. This suggests that repression may sometimes generate backlash or intensify grievances rather than simply deter dissent.

---

## 4. Current Protests by Lagged Repression

The grouped bar chart shows that current protest totals are highest when lagged repression is high, followed by medium repression, and lowest when repression is low. This is counterintuitive if one expects repression to always suppress protest. Instead, it may indicate backlash dynamics, conflict spirals, or that authorities repress places already prone to unrest.

---

## 5. Current Protests by Lagged Price Change and Lagged Repression

When examining price change and repression together, protests are highest when food prices are rising and repression is high. This is important because it suggests that economic stress alone may not tell the full story. Protest activity appears strongest when material hardship is combined with coercive pressure.

---

## 6. Regression Results

The district-level regressions suggest that there is no single national model explaining protest dynamics in Syria. Instead, the relationship between economic hardship, repression, and mobilization varies sharply across regions.

Lagged food price changes matter in some districts, but not uniformly. In Ar-Raqqa and Deir Ez-Zor, higher food price increases are significantly associated with fewer later protests, which may indicate that severe hardship can suppress collective action rather than fuel it. In contrast, Idleb shows a large positive, though only marginally significant, price effect, suggesting that in some politically mobilized contexts economic shocks may generate unrest.

Lagged civilian targeting more often shows a positive relationship with later protests than a negative one, with the clearest evidence in Al Hasakah, Aleppo, and Dara. This implies that repression may create backlash or intensify grievances rather than simply deter dissent.

Most importantly, the interaction between repression and food prices is significant in Al Hasakah, Ar-Raqqa, and Deir Ez-Zor. This indicates that repression changes how economic shocks translate into protest behavior. This provides the strongest support for a conditional rather than universal theory of protest: economic grievances matter differently depending on local coercive environments and regional political context.

Overall, the results point to substantial subnational heterogeneity, where neither prices nor repression alone consistently explain protest, but their combination helps explain mobilization in key districts.

---

## What I Plan to Do Next

- Explore additional controls and more robust model specifications.
- Compare monthly and yearly relationships more systematically.
- Add clearer case-study annotations for major spikes such as Aleppo (2022) and As-Sweida (2023).