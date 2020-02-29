library(dplyr)
library(readr)
library(ggplot2)
house_df <-  read_csv('train.csv')
View(house_df)
subset <-  house_df[1:11]

subset <- subset %>% 
  select(-Utilities) %>% 
  mutate(SalePrice  = house_df$SalePrice)

summary(subset)
View(subset)
#MSSubClass
subset %>% 
  group_by(MSSubClass) %>% 
  count()

ggplot(subset,aes(x=MSSubClass)) + geom_bar()

#MSZoning
subset %>% 
  group_by(MSZoning) %>% 
  count()
#Lot config
subset %>% 
  group_by(LotConfig) %>% 
  count()


#Lot frontage
plot_frontage <- ggplot(subset, aes(x = LotFrontage)) + 
  geom_histogram()  
plot_frontage
subset %>% 
  mutate(n= n(LotFrontage))

#Lotshape
subset %>% 
  group_by (LotShape) %>% 
  count()

ggplot(subset, aes(x = LotShape)) + geom_bar()
  
  
plot_lotshape <- ggplot(subset, aes(x = LotShape, y = (SalePrice))) + geom_boxplot()
plot_lotshape


#Street
subset %>% 
  group_by(Street) %>% 
  count()
 6/(6+1454)
 #Alley
 subset %>% 
   group_by(Alley) %>% 
   count()
 (50 +41)/( 1460)
 
ggplot(subset, aes(x = Alley, y = SalePrice)) + geom_boxplot()

ggplot(subset, aes(x = LotConfig, y = SalePrice)) + geom_boxplot()
 
ggplot(subset,aes(x = LotConfig, y = log(LotArea))) + geom_boxplot()

ggplot(subset, aes(x = log(LotArea), y = log(SalePrice))) + geom_point() 
       