library(forecast)
library(fpp2)
library(ggplot2)
library(dplyr)
library(MASS)
library(zoo)
library(ggcorrplot)
library(ggplot2)
library(plyr)
library(scales)
library(zoo)

df <- read.csv ("forex_csv.csv")
colnames(df) <- c("DATE","USDEUR","USDGBP","USDCAD","USDJPY","NASDAQ","S&P","VIX","DOW","EURO","WTI","TIPS","FEDRATE","PRIMERATE","MIX","SPOT","SWAP")
head(df)

na_count <- sum(is.na(df))
na_count
df<- na.locf(df)

##Correlation matrix
data<- df[,c(2,3,4,5,6,7,8,9,10,11,12,13)]
data1<- data.matrix(data, rownames.force = NA)
corr <- cor(data1)
ggcorrplot(corr, hc.order = TRUE, 
           type = "lower", 
           lab = TRUE, 
           lab_size = 3, 
           method="circle", 
           colors = c("tomato2", "white", "springgreen3"), 
           title="Correlogram of USDEUR", 
           ggtheme=theme_bw)



train <- 1:350
test <- 351:445
ts.df<- ts(df$USDEUR)
plot(decompose(ts.df))
boxplot(ts.df ~ cycle(ts.df))
Y_train <- df[train,"USDEUR"]
X_train <- df[train,c("USDCAD","USDJPY","Nasdaq_std","Dow_std")]
X_test <- as.matrix(df[test,c("USDCAD","USDJPY","Nasdaq_std","Dow_std")])
str(Y_train)
str(X_train)
str(X_test)

model <- auto.arima(y = Y_train,
                        xreg = as.data.frame(X_train))
class(model$xreg)
summary(model)
pred = forecast(model$fitted,88)
plot(pred, xlim=c(10,445),main = "ARIMA plot",ylab="USDEUR", xlab="Days")
lines(df$USDEUR)

summary(pred)
summary(model)

Nasdaq_mean = mean(df$NASDAQ)
Nasdaq_sd = sd(df$NASDAQ)
df$Nasdaq_std <- (df$NASDAQ- Nasdaq_mean)/Nasdaq_sd

Dow_mean = mean(df$DOW)
Dow_sd = sd(df$DOW)
df$Dow_std <- (df$DOW- Dow_mean)/Dow_sd
