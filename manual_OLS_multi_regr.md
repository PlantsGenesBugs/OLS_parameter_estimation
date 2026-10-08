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

print(summary(house_model_incl_area))
```

    ## 
    ## Call:
    ## lm(formula = saleEstimate_currentPrice ~ outcode + bathrooms + 
    ##     bedrooms + floorAreaSqM + livingRooms, data = london_housing_short)
    ## 
    ## Residuals:
    ##      Min       1Q   Median       3Q      Max 
    ## -3108413  -123793     3357   117191  7431891 
    ## 
    ## Coefficients:
    ##               Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)  -247369.4    64783.9  -3.818 0.000136 ***
    ## outcodeE10   -155066.3    77912.7  -1.990 0.046618 *  
    ## outcodeE11   -132303.3    74062.1  -1.786 0.074099 .  
    ## outcodeE12   -222795.3   146736.8  -1.518 0.128995    
    ## outcodeE13   -313769.8   113107.7  -2.774 0.005557 ** 
    ## outcodeE14    -94718.1    74114.2  -1.278 0.201309    
    ## outcodeE15   -212014.2    78343.2  -2.706 0.006829 ** 
    ## outcodeE16   -228597.3    76052.8  -3.006 0.002663 ** 
    ## outcodeE17    -95547.7    67932.7  -1.407 0.159638    
    ## outcodeE18   -259879.4    83491.5  -3.113 0.001865 ** 
    ## outcodeE1W     48236.9   237699.3   0.203 0.839196    
    ## outcodeE2      26828.7    84412.2   0.318 0.750627    
    ## outcodeE3     -79685.9    81199.1  -0.981 0.326462    
    ## outcodeE4    -313081.8    71371.5  -4.387 1.18e-05 ***
    ## outcodeE5      -7128.3    88267.5  -0.081 0.935638    
    ## outcodeE6    -293077.6    89924.3  -3.259 0.001125 ** 
    ## outcodeE7    -223458.0    91731.3  -2.436 0.014886 *  
    ## outcodeE8      48485.3    80141.9   0.605 0.545212    
    ## outcodeE9      -1893.7    82990.0  -0.023 0.981796    
    ## outcodeEC1A    58956.0   330095.1   0.179 0.858257    
    ## outcodeEC1M   122230.8   146751.8   0.833 0.404937    
    ## outcodeEC1N   153611.9   237689.6   0.646 0.518134    
    ## outcodeEC1R   156177.0   237672.6   0.657 0.511141    
    ## outcodeEC1V   126737.8   146740.4   0.864 0.387802    
    ## outcodeEC1Y    30286.9   237724.2   0.127 0.898626    
    ## outcodeEC2A    87192.8   237691.8   0.367 0.713761    
    ## outcodeEC2Y   280413.4   173996.1   1.612 0.107112    
    ## outcodeEC3R   405544.2   330088.2   1.229 0.219283    
    ## outcodeEC4A    98619.3   330076.2   0.299 0.765123    
    ## outcodeEC4V    -8455.8   330088.2  -0.026 0.979564    
    ## outcodeN1     197321.0    73361.4   2.690 0.007176 ** 
    ## outcodeN10    -36271.9    86248.8  -0.421 0.674103    
    ## outcodeN11   -181062.5   113094.6  -1.601 0.109446    
    ## outcodeN12   -259957.6    96582.1  -2.692 0.007136 ** 
    ## outcodeN13   -358446.8    99528.4  -3.601 0.000320 ***
    ## outcodeN14   -277170.2   120587.5  -2.298 0.021575 *  
    ## outcodeN15   -215543.7    81329.4  -2.650 0.008069 ** 
    ## outcodeN16    -16547.1    78914.1  -0.210 0.833922    
    ## outcodeN17   -279339.7    76159.2  -3.668 0.000247 ***
    ## outcodeN18   -308514.3   138011.7  -2.235 0.025435 *  
    ## outcodeN19   -101253.6   113089.3  -0.895 0.370648    
    ## outcodeN2      50102.2    96504.7   0.519 0.603667    
    ## outcodeN20   -160607.4   113173.3  -1.419 0.155925    
    ## outcodeN21   -307626.4    90798.8  -3.388 0.000710 ***
    ## outcodeN22   -190616.2    77912.4  -2.447 0.014458 *  
    ## outcodeN3    -230818.9   131053.4  -1.761 0.078257 .  
    ## outcodeN4      -9158.4    75677.2  -0.121 0.903681    
    ## outcodeN5      86779.3    90739.9   0.956 0.338941    
    ## outcodeN6       2518.9    92745.2   0.027 0.978334    
    ## outcodeN7     -17855.6    78580.9  -0.227 0.820258    
    ## outcodeN8      11716.0    79836.2   0.147 0.883335    
    ## outcodeN9    -332111.3   101248.7  -3.280 0.001045 ** 
    ## outcodeNW1    160915.9   102925.8   1.563 0.118020    
    ## outcodeNW10  -122235.7    89031.6  -1.373 0.169832    
    ## outcodeNW11   -39963.5    92980.5  -0.430 0.667356    
    ## outcodeNW2   -234226.0    95091.3  -2.463 0.013806 *  
    ## outcodeNW3    379116.2    87519.3   4.332 1.51e-05 ***
    ## outcodeNW4   -357585.2   102993.9  -3.472 0.000521 ***
    ## outcodeNW5    187541.1   110047.0   1.704 0.088409 .  
    ## outcodeNW6    137742.4    83490.8   1.650 0.099050 .  
    ## outcodeNW7   -372985.9    93937.6  -3.971 7.27e-05 ***
    ## outcodeNW8    607558.2    82079.0   7.402 1.57e-13 ***
    ## outcodeNW9   -270188.8    87518.8  -3.087 0.002032 ** 
    ## outcodeSE1     56622.4    72895.5   0.777 0.437337    
    ## outcodeSE10  -105899.5    71300.4  -1.485 0.137540    
    ## outcodeSE11    -5761.9    83893.2  -0.069 0.945246    
    ## outcodeSE12  -286338.2    79885.3  -3.584 0.000341 ***
    ## outcodeSE13  -155194.0    77877.4  -1.993 0.046339 *  
    ## outcodeSE14   -76602.9    82079.0  -0.933 0.350720    
    ## outcodeSE15   -56582.5    75182.3  -0.753 0.451724    
    ## outcodeSE16   -92308.2    80819.9  -1.142 0.253449    
    ## outcodeSE17    -8467.3    87482.4  -0.097 0.922899    
    ## outcodeSE18  -269850.2    71618.2  -3.768 0.000167 ***
    ## outcodeSE19  -148965.7    74901.7  -1.989 0.046777 *  
    ## outcodeSE2   -290548.9    92870.4  -3.129 0.001767 ** 
    ## outcodeSE20  -239983.5    78673.3  -3.050 0.002298 ** 
    ## outcodeSE21    19431.2    84740.0   0.229 0.818642    
    ## outcodeSE22   -34921.5    85530.1  -0.408 0.683075    
    ## outcodeSE23  -111817.1    77140.6  -1.450 0.147256    
    ## outcodeSE24   -32518.8    85018.9  -0.382 0.702115    
    ## outcodeSE25  -307679.8    75462.3  -4.077 4.63e-05 ***
    ## outcodeSE26  -216351.7    77699.5  -2.784 0.005382 ** 
    ## outcodeSE27  -219018.3    83006.4  -2.639 0.008352 ** 
    ## outcodeSE28  -249915.8    90759.2  -2.754 0.005916 ** 
    ## outcodeSE3   -144679.7    72758.5  -1.988 0.046813 *  
    ## outcodeSE4   -134550.3    78388.1  -1.716 0.086141 .  
    ## outcodeSE5    -72979.3    77085.4  -0.947 0.343822    
    ## outcodeSE6   -258365.0    74231.9  -3.481 0.000505 ***
    ## outcodeSE7   -291856.9    91831.6  -3.178 0.001491 ** 
    ## outcodeSE8   -117078.2    86783.1  -1.349 0.177371    
    ## outcodeSE9   -303630.8    70622.5  -4.299 1.75e-05 ***
    ## outcodeSW10   920547.2   125281.5   7.348 2.35e-13 ***
    ## outcodeSW11    85243.0    68814.7   1.239 0.215504    
    ## outcodeSW12    88825.1    72476.8   1.226 0.220421    
    ## outcodeSW13   275114.2    84597.5   3.252 0.001154 ** 
    ## outcodeSW14   107893.7    76286.2   1.414 0.157329    
    ## outcodeSW15   -14137.6    72798.8  -0.194 0.846026    
    ## outcodeSW16  -276472.1    69008.4  -4.006 6.26e-05 ***
    ## outcodeSW17   -30722.4    71289.7  -0.431 0.666522    
    ## outcodeSW18    82599.8    68886.0   1.199 0.230555    
    ## outcodeSW19    -9312.6    71821.3  -0.130 0.896838    
    ## outcodeSW1H  1840056.7   175013.9  10.514  < 2e-16 ***
    ## outcodeSW1P   323990.2   125320.7   2.585 0.009758 ** 
    ## outcodeSW1V    30829.8    96559.7   0.319 0.749526    
    ## outcodeSW1W   717057.0   131708.2   5.444 5.46e-08 ***
    ## outcodeSW1X   845346.2   138468.2   6.105 1.11e-09 ***
    ## outcodeSW2   -143746.1    76303.2  -1.884 0.059640 .  
    ## outcodeSW20   -93827.0    86836.0  -1.081 0.279970    
    ## outcodeSW3   1157277.0    97817.4  11.831  < 2e-16 ***
    ## outcodeSW4     33286.7    75373.1   0.442 0.658780    
    ## outcodeSW5   -206432.0   125640.8  -1.643 0.100440    
    ## outcodeSW6    276253.1    82091.3   3.365 0.000771 ***
    ## outcodeSW7    964000.6   105136.6   9.169  < 2e-16 ***
    ## outcodeSW8     83064.7    79792.6   1.041 0.297923    
    ## outcodeSW9    -73179.3    76272.1  -0.959 0.337379    
    ## outcodeW10    192609.6   107376.8   1.794 0.072911 .  
    ## outcodeW11    366607.0   113055.6   3.243 0.001192 ** 
    ## outcodeW12    -11354.3    96416.7  -0.118 0.906261    
    ## outcodeW13   -138997.8    91738.9  -1.515 0.129800    
    ## outcodeW14    392173.9   131036.7   2.993 0.002778 ** 
    ## outcodeW1B    355000.2   330077.1   1.076 0.282201    
    ## outcodeW1G    -57149.1   330145.2  -0.173 0.862578    
    ## outcodeW1H    384102.3   137949.4   2.784 0.005384 ** 
    ## outcodeW1J    769264.8   330101.2   2.330 0.019826 *  
    ## outcodeW1K    677161.2   237688.7   2.849 0.004405 ** 
    ## outcodeW1T    384996.8   173970.5   2.213 0.026944 *  
    ## outcodeW1U    687425.5   116527.0   5.899 3.90e-09 ***
    ## outcodeW1W   1642766.9   158376.3  10.373  < 2e-16 ***
    ## outcodeW2     498165.8    80604.2   6.180 6.91e-10 ***
    ## outcodeW3    -133923.3    84437.7  -1.586 0.112790    
    ## outcodeW4     113629.9    76321.6   1.489 0.136596    
    ## outcodeW5    -163435.0    89917.0  -1.818 0.069183 .  
    ## outcodeW6     160184.2   110071.1   1.455 0.145656    
    ## outcodeW7    -375590.8   110134.0  -3.410 0.000654 ***
    ## outcodeW8    1135051.9   107762.8  10.533  < 2e-16 ***
    ## outcodeW9     475004.7    83864.7   5.664 1.56e-08 ***
    ## outcodeWC1A   394146.5   237682.4   1.658 0.097323 .  
    ## outcodeWC1B   596428.6   174009.1   3.428 0.000614 ***
    ## outcodeWC1H    -6013.1   330060.2  -0.018 0.985466    
    ## outcodeWC1X    84278.1   197554.1   0.427 0.669684    
    ## bathrooms     117072.9     9672.1  12.104  < 2e-16 ***
    ## bedrooms       20972.8     7284.8   2.879 0.004007 ** 
    ## floorAreaSqM    7102.9      183.4  38.720  < 2e-16 ***
    ## livingRooms   134013.4    11322.6  11.836  < 2e-16 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 323900 on 4889 degrees of freedom
    ## Multiple R-squared:  0.7443, Adjusted R-squared:  0.7368 
    ## F-statistic: 99.53 on 143 and 4889 DF,  p-value: < 2.2e-16

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
