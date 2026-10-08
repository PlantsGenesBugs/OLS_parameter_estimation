Manual OLS for parameter estimation
================
PGB
2026-10-07

In this file I will manually estimate the parameters (or coefficients)
of a multiple linear regression, where I will describe the current price
of houses (outcome variable) as a function of the number of bathrooms,
bedrooms and living rooms, and floor area in square meters. I will be
using a data set available on
[Kaggle](https://www.kaggle.com/datasets/jakewright/house-price-data/data).

``` r
# load required packages
library(tidyverse)
```

    ## Warning: package 'ggplot2' was built under R version 4.4.3

    ## ── Attaching core tidyverse packages ──────────────────────── tidyverse 2.0.0 ──
    ## ✔ dplyr     1.1.4     ✔ readr     2.1.5
    ## ✔ forcats   1.0.0     ✔ stringr   1.5.1
    ## ✔ ggplot2   4.0.2     ✔ tibble    3.3.0
    ## ✔ lubridate 1.9.4     ✔ tidyr     1.3.1
    ## ✔ purrr     1.1.0     
    ## ── Conflicts ────────────────────────────────────────── tidyverse_conflicts() ──
    ## ✖ dplyr::filter() masks stats::filter()
    ## ✖ dplyr::lag()    masks stats::lag()
    ## ℹ Use the conflicted package (<http://conflicted.r-lib.org/>) to force all conflicts to become errors

``` r
library(MASS)
```

    ## 
    ## Attaching package: 'MASS'
    ## 
    ## The following object is masked from 'package:dplyr':
    ## 
    ##     select

``` r
# download data from Kaggle: https://www.kaggle.com/datasets/jakewright/house-price-data/data
# or download using RKaggle
#install.packages("RKaggle")
#library(RKaggle)

#london_housing <- RKaggle::get_dataset("jakewright/house-price-data")

# save data in 'data' folder

# load data into environment
london_housing <- read.csv("data/kaggle_london_house_price_data.csv")

# tidy the data
# remove NA
london_housing <- na.omit(london_housing)
```

Firstly, this is a huge dataset and will put a substantial computational
burden on the computer, so we select a subset. Then, in order to preapre
the data for multiple linear regression, we have to put the output
variable into its own vector, while we put the covariables (or input
features) into a design matrix. Because we will estimate parameters for
a linear model of the form y = b0 + b1x1 + b2x2… we will add a column of
value = 1 at the front of our design matrix. That way we ensure that the
intercept is retained (i.e. we create a variable x0 for the intercept
parameter b0, and set it to 1).

``` r
# filter for date and remove unwanted variables to decrease computational burden
london_housing_short <- london_housing %>%
  mutate(history_date = as.Date(history_date)) %>%
  filter(history_date > "2024-06-01") %>%
  dplyr::select(outcode,
         bathrooms,
         bedrooms,
         floorAreaSqM,
         livingRooms,
         saleEstimate_currentPrice
  )

# clean up to decrease memory usage
rm(london_housing)

# specify output variable and convert to matrix for calculations
y <- as.matrix(london_housing_short$saleEstimate_currentPrice)

# create feature matrix
feature_matrix <- london_housing_short %>%
  dplyr::select(-c(outcode, saleEstimate_currentPrice)) %>%
  as.matrix()

# Create x0 variable for intercept
b0_vector <- rep(1, length(y))

# Add x0 vector to feature matrix, and define as input matrix X
X <- cbind(b0_vector, feature_matrix)
```

In the next step we will use the Normal Equation: $$
\hat{\beta} = (X^T X)^{-1} X^T y
$$ This provides a closed-form solution to linear regression, allowing
for the computation of optimal coefficients in one step. While it uses
the simplicity of matrix multiplication, the inversion of the input
matrix (X) can become computationally expensive for large datasets,
which is why we trimmed the one we’re using here. Once we’ve completed
the computation, we’ll have a vector of parameters (designated as
beta-hat in the equation above), relating to the coefficient by which
each variable has to be multiplied to return the output variable.

``` r
# define a function to calculate the parameters (beta-hat)
parameters <- function(response, features) {
  matrix_T_multiply = t(features) %*% features
  invert_matrix = solve(matrix_T_multiply)
  param_estim = invert_matrix %*% t(features) %*% response
  return(round(param_estim, 2))
}

params <- parameters(y, X)

intersect <- params[1]
bathrooms_param <- params[2]
bedrooms_param <- params[3]
area_param <- params[4]
livingrooms_param <- params[5]
```

The model estimates that price increases with increasing bathrooms,
living rooms and floor area, but decreases with increasing bedrooms. The
intercept is set at a negative value (when all other values are zero,
which is not a feasible scenario). Let’s see how well this model
predicts current prices in London.

Let’s look at a one-bedroom flat in [Golder’s
Green](https://www.rightmove.co.uk/properties/173789558#/). It has one
living room, one bathroom and a floor area of 54.72 square meters.

``` r
small_flat_price <- intersect + bathrooms_param*1 + bedrooms_param*1 + area_param*54.72 + livingrooms_param*1

print(small_flat_price)
```

    ## [1] 386261.2

The model predicts that the price of the flat will be £386,261.20. It is
currently for sale at £360,000 which shows that the model has a
reasonable amount of accuracy.

OK, let’s look at a more upmarket abode, in the heart of Kensington, on
[Palace Green](https://www.rightmove.co.uk/properties/93801582#/). This
luxury residence has 3 bedrooms, 3 bathrooms, 309 square meters, but
only a single livingroom…

``` r
luxury_apartment_price <- intersect + bathrooms_param*3 + bedrooms_param*3 + area_param*309 + livingrooms_param*1

print(luxury_apartment_price)
```

    ## [1] 2813227

The predicted price is £2,813,227. And here we see the effect of
geographic location within London, because the house is currently for
sale at £23,500,000 - nearly 10x higher than the predicted values. A
useful addition to the model would be the inclusion of the postal code.
So, let’s firstly check whether our manual model returned similar
parameters as would be calculated through the `lm()` function in R, and
then see if the built-in function will add this categorical variable,
and improve the predictive power.

``` r
house_model_lm <- lm(saleEstimate_currentPrice ~ bathrooms + bedrooms + floorAreaSqM + livingRooms,
                     data = london_housing_short)

print(summary(house_model_lm))
```

    ## 
    ## Call:
    ## lm(formula = saleEstimate_currentPrice ~ bathrooms + bedrooms + 
    ##     floorAreaSqM + livingRooms, data = london_housing_short)
    ## 
    ## Residuals:
    ##      Min       1Q   Median       3Q      Max 
    ## -3585535  -160409   -13668   112020  8080312 
    ## 
    ## Coefficients:
    ##               Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)  -300897.3    17284.0 -17.409  < 2e-16 ***
    ## bathrooms     185804.2    11106.6  16.729  < 2e-16 ***
    ## bedrooms      -43494.7     8355.1  -5.206 2.01e-07 ***
    ## floorAreaSqM    8425.1      213.1  39.543  < 2e-16 ***
    ## livingRooms    83824.7    13342.1   6.283 3.61e-10 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 393000 on 5028 degrees of freedom
    ## Multiple R-squared:  0.6127, Adjusted R-squared:  0.6124 
    ## F-statistic:  1989 on 4 and 5028 DF,  p-value: < 2.2e-16

The built-in linear model function returns the same parameters as the
manual calculation above.

``` r
house_model_incl_area <- lm(saleEstimate_currentPrice ~ outcode + bathrooms + bedrooms + floorAreaSqM + livingRooms, data = london_housing_short)
```

Let’s use this new model to predict the prices of the small flat and the
luxury apartment. The data for these properties is as follows:

| Property   | outcode | bathrooms | bedrooms | area (sq m) | livingrooms |
|:-----------|:--------|:----------|:---------|:------------|:------------|
| Small flat | NW11    | 1         | 1        | 54.72       | 1           |
| Luxury apt | W8      | 3         | 3        | 309         | 1           |

``` r
new_data <- data.frame(outcode=c("NW11", "W8"),
                       bathrooms=c(1, 3),
                       bedrooms=c(1,3),
                       floorAreaSqM=c(54.72, 309),
                       livingRooms=c(1,1)
                       )
rownames(new_data) <- c("small_flat", "lux_apt")

prediction_incl_area <- predict(house_model_incl_area, newdata = new_data)
print(prediction_incl_area)
```

    ## small_flat    lux_apt 
    ##     373399    3630641

Adding in the area code to the model has adjusted the price for the
small flat downwards (closer to the asking price). Without this factor,
the predicted price was £386,261.20. Including the `outcode` factor
drops that to £373,399, which is closer to the asking price of £360,000.
The luxury apartment’s predicted price has increased by about £800,000,
but still hasn’t boosted it to reflect the asking price. The previous
model predicted an asking price of £2,813,227, and adding the `outcode`
factor raised that to £3,630,641 but it is still far from the asking
price of £23,500,000, which shows the limitation of models like this. It
is possible that the dataset contains much more data about 1 bed, 1 bath
flats than large, luxurious apartments, and so the model might be
somewhat biased towards smaller/cheaper properties.
