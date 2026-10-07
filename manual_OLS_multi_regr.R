# loading MASS library
library(MASS)

# download data from Kaggle: https://www.kaggle.com/datasets/jakewright/house-price-data/data
# or download using RKaggle
install.packages("RKaggle")
library(RKaggle)

london_housing <- RKaggle::get_dataset("jakewright/house-price-data")

# save data in 'data' folder

# load data into environment
london_housing <- read.csv("data/kaggle_london_house_price_data.csv")

# tidy the data
# remove NA
london_housing <- na.omit(london_housing)

# remove unwanted variables
london_housing <- london_housing %>%
  select(outcode,
         bathrooms,
         bedrooms,
         floorAreaSqM,
         livingRooms,
         saleEstimate_currentPrice
         )
