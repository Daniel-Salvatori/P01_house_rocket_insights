# <p align="center"> <b> INSIGHTS PROJECT - HOUSE ROCKET COMPANY </p> </b>


![1](houserocket.png)

## 1. About
This repository contains codes for the portfolio analysis of a real estate company. All business context involving this project is fictitious. The database was extracted from Kaggle.

The objectives of this project are:
- Determine which properties have the best conditions for buying and identify best season for reselling  
- Develop interactive [Dashboard](https://p01-house-rocket-insights.herokuapp.com/) on Heroku, which the company's CEO can analyze the properties from a mobile or computer
- Extract business insights from available catalog data.

---

## 2. Business Problem
House Rocket's business model consists of purchasing and reselling properties through a digital platform. The data scientist is responsible for developing an online dashboard to help the CEO company overview properties available on House Rocket's portfolio and identifying better business opportunities.

The dashboard must contain:
  * Which properties the company should buy and better season to reselling
  * Map view with selected properties
  * Table view with attributes filters
  * Expected profit 

<br>

## 3. Business questions

  * 1 Which properties should House Rocket buy?
  Business criteria to determine whether a property should be bought:
      - Property must have a ‘condition’ bigger than 3;
      - Property price must be below or equal the median price on the region (zip code)
 
  * 2 When is the best time to resell? At what price? 
  Business criteria to determine the best time to resell and at what price:
      - Identify the best season of the year to resell
      - Properties with lower prices than the region average: price + 30%
      - Properties with equal prices than the region average: price + 10%


## 4. Business Results

There are 21,435 available properties. Based on business criteria, 3844 should be bought by House Rocket resulting in a $277,47M profit. This result represents 18% of the gross value.
  * Maximum value invested: $1,52B
  * Maximum value return: $1,8B
  * Maximum Expected Profit: $ 277,47M  


## 5. Conclusion

We have concluded that there are 3844 properties that are worth purchasing and reselling them, with a total possible profit of $277,47M. An interactive dashboard was also made available that can be accessed by any digital platform (mobile, PC), through a browser.


## 6. Next steps
 
- Expand this methodology to other regions;
- We could make a sale price predictions using ML;


## 7. References

- Dataset from [Kaggle](https://www.kaggle.com/harlfoxem/housesalesprediction)
- Variables meaning on [Kaggle discussion](https://www.kaggle.com/harlfoxem/housesalesprediction/discussion/207885)
- Python from Zero to [DS Youtube](https://www.youtube.com/watch?v=1xXK_z9M6yk&list=PLZlkyCIi8bMprZgBsFopRQMG_Kj1IA1WG&ab_channel=SejaUmDataScientist)

If you have any other suggestion or question, feel free to contact me via [LinkedIn](https://linkedin.com/in/daniel-salvatori)

## 8. How to contribute
1. Fork the project.
2. Create a new branch with your changes: `git checkout -b my-feature`
3. Save your changes and create a commit message telling you what you did: `git commit -m" feature: My new feature "`
4. Submit your changes: `git push origin my-feature`
