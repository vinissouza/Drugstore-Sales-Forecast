# Drugstore-Sales-Forecast
This repos contain all files from sales prediction project


## Business Problem

Rossmann is a drugstore chain in Europe with more than 4000 stores over Germany, Poland, Hungary
and others.

The CFO of company want to invest in a stores refurbishment and for that, want to know how much money put in each store.
Furthermore, he required a sale prediction for the next 6 weeks for each store that can be access by mobile.

For this job, company provide a base data with results of last 2 years with relevant information like, promotion, 
store type etc.

All the context about this project is completely fictitious, including CEO and business issues.

This is a project provided by _Comunidade DS_.


## Business Assumptions
Assumptions to fill na:
- competition_distance -> NA data mean that dont have competitors near, so is adopted a large distance to fill na based in the max distance.
- open_distance[month/year] -> NA data mean that don't have a competitors near, or don't know when is open, so is adopted the equal to date row to when calculate the distance be considerer 0.
- promo2_since_[month/week] -> Na data mean that store dont adopted the second promo round, so is adopted the same strategy from open_distance_[month/year]
- promo_interval -> NA data mean that store dont adopted the promo, so a column was created with boolean value to mean if there is a promo and if row date is in interval 


## Solution Strategy
