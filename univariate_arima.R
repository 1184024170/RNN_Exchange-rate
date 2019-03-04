library(forecast)
library(ggplot2)

data = read.csv('daily_forex_3177.csv', header = TRUE)
ggplot(data,aes(x=df$index,y=df$value))+geom_point()

detach(data)
attach(data)
ts = ts(data)
plot(ts)
y = value
t = index

#checking autocorrelation
acf(data$value)
pacf(data$value)
## checking stationarity with ADF test
adf.test(data$value)

# splitting data into train and valid sets
train = data[1:2540,]
valid = data[2541:nrow(data),]


# removing "Month" column
train$index = NULL
valid$index = NULL

# training model
model = auto.arima(train)

# model summary
checkresiduals(model) 
summary(model)

# forecasting
pred = forecast(model,638)
plot(pred, xlim=c(1,3177))
lines(y)

predv2 <- ts(pred2$fitted)
y2 <- ts(y)
accuracy(predv2,y2)



