# Core Tidyverse
library(tidyverse)
library(lubridate)
library(glue)
library(forcats)
library(magrittr)
library(date)
library(ModelMetrics)

# Time Series
library(timetk)
library(tidyquant)
library(tibbletime)

# Visualization
library(ggplot2)
library(cowplot)
library(scales)

# Preprocessing
library(recipes)
library(scales)


# Sampling / Accuracy
library(rsample)
library(yardstick) 

# Modeling of LSTM in keras of python
library(keras)

##loading data
df <- read.csv("daily_forex_3177.csv",as.is = T)
class(df)
class(df$index)
class(df$value)
str(df$index)
## change the format of index, to date format
df$index <- as_date(df$index, tz=NULL, format = NULL)


#### df$index <- mdy_hm( as.character(df$index))
#### df$index <- anytime(as.factor(df$index,format = "%Y-%m-%d" ))
## can convert to tibble using tk_tbl()


## plotting the forex rate (only variable)
plot <- df %>%   ggplot(aes(index, value)) +  geom_point(color = palette_dark()[[1]], alpha = 0.5) +  theme_tq() +  labs(
    title = "USDEUR daily")

plot_title <- ggdraw() + draw_label("USDEUR", size = 18, fontface = "bold", colour = palette_light()[[1]])

plot_grid(plot_title, plot, ncol = 1, rel_heights = c(0.1, 1, 1))
##  ACF and pacf plots 

acf(df$value)
pacf(df$value)
adf.test(df$value)

## cross validation : Back testing method 
## initial for training set
samples_train <- 1200  
## for testing and validation set
samples_test  <- 600
## skip to make an even distribution of 6 slices
skip_span     <- 263   

backtesting_resamples <- rolling_origin(df,initial = samples_train,assess= samples_test,cumulative = FALSE,skip=skip_span)
## cumulative false to avoid higher weightage to recent points
# check the samples in slices, for now trying with 6
backtesting_resamples

# Plotting backtesting function for a single split using keras function in ggplot (function from python)
slice_graph <- function(split, expand_y_axis = TRUE, alpha = 1, size = 1, base_size = 14) {
  
  # conversting data to tbl data
  train_tbl <- training(split) %>%
    add_column(key = "training") 
  
  test_tbl  <- testing(split) %>%
    add_column(key = "testing") 
  
  data_manipulated <- bind_rows(train_tbl, test_tbl) %>%
    as_tbl_time(index = index) %>%
    mutate(key = fct_relevel(key, "training", "testing"))
  
  # Collect attributes
  train_time_summary <- train_tbl %>%
    tk_index() %>%
    tk_get_timeseries_summary()
  
  test_time_summary <- test_tbl %>%
    tk_index() %>%
    tk_get_timeseries_summary()
  
  # Visualize
  g <- data_manipulated %>%
    ggplot(aes(x = index, y = value, color = key)) +
    geom_line(size = size, alpha = alpha) +
    theme_tq(base_size = base_size) +
    scale_color_tq() +
    labs(
      title    = glue("Split: {split$id}"),
      subtitle = glue("{train_time_summary$start} to {test_time_summary$end}"),
      y = "", x = ""
    ) +
    theme(legend.position = "none") 
  
  if (expand_y_axis) {
    
    df_time_summary <- df %>% 
      tk_index() %>% 
      tk_get_timeseries_summary()
    
    g <- g +
      scale_x_date(limits = c(df_time_summary$start, 
                              df_time_summary$end))
  }
  
  return(g)
}


#plot sample back testing - slice 06 - on whole scale
backtesting_resamples$splits[[6]] %>%  slice_graph(expand_y_axis = TRUE) +   theme(legend.position = "bottom")

## plot further back testing:-
# Plotting function that scales to all splits uses ggplot + cowplot+purr and sclaes slice_graph()
slice_all_graph <- function(sampling_tbl, expand_y_axis = TRUE,ncol = 3, alpha = 1, size = 1, base_size = 14, 
                               title = "Sampling Plan") {sampling_tbl_with_plots <- sampling_tbl %>%
    mutate(gg_plots = map(splits, slice_graph,expand_y_axis = expand_y_axis,alpha = alpha, base_size = base_size))
  plot_list <- sampling_tbl_with_plots$gg_plots 
  p_temp <- plot_list[[1]] + theme(legend.position = "bottom")
  legend <- get_legend(p_temp)
  p_body  <- plot_grid(plotlist = plot_list, ncol = ncol)
  p_title <- ggdraw() + draw_label(title, size = 14, fontface = "bold", colour = palette_light()[[1]])
  g <- plot_grid(p_title, p_body, legend, ncol = 1, rel_heights = c(0.05, 1, 0.05))
  return(g)
}

##samples of all slices together in cowplot:-
backtesting_resamples %>%  slice_all_graph(expand_y_axis = F,ncol = 3,alpha = 1,size = 1,base_size = 10,title = "Backtesting Strategy:all slices")

## plot sample as a whole to check back testing - taking a random slice - training the same below
example_split    <- backtesting_resamples$splits[[1]]
example_split_id <- backtesting_resamples$id[[1]]
slice_graph(example_split, expand_y_axis = FALSE, size = 0.5) +
  theme(legend.position = "bottom") +
  ggtitle(glue("Split: {example_split_id}"))

## keeping 66% as train and rest for validation 
df_trn <- analysis(example_split)[1:800, , drop = FALSE]
df_val <- analysis(example_split)[801:1200, , drop = FALSE]
df_tst <- assessment(example_split)

# creating a df matrix for the same
df1 <- bind_rows(df_trn %>% add_column(key = "training"),df_val %>% add_column(key = "validation"),df_tst %>% add_column(key = "testing")
) %>%as_tbl_time(index = index)

df1
str(df1)
summary(df1)

## normalising/scaling the data - centered and scaled - recipes package used
scale_df <- recipe(value ~ ., df1) %>%step_sqrt(value) %>%step_center(value) %>%step_scale(value) %>%prep()
??bake
df1_processed_tbl <- bake(scale_df, df1)
df1_processed_tbl
## inverting for the model:-
mean_scale <- scale_df$steps[[2]]$means["value"]
sd_scale  <- scale_df$steps[[3]]$sds["value"]

c("center" = mean_scale, "scale" = sd_scale)

## reshaping the whole data for ML
## creasting a 3d array:- samples(no of obser).timesteps(hidden state length).features(no of predictors, 1 here)
#  defined by the the order in (first the data, then the model)
#superseded by FLAGS$hidden_len, FLAGS$batch_size and n_predictions later
hidden_len <- 12
n_predictions <- hidden_len
batch_size <- 10

# matrix in tensor format
build_matrix <- function(tseries, overall_timesteps) {t(sapply(1:(length(tseries) - overall_timesteps + 1), function(x) 
    tseries[x:(x + overall_timesteps - 1)]))}

reshape_X_3d <- function(X) {dim(X) <- c(dim(X)[1], dim(X)[2], 1) 
 X
}

# extract values from data frame
train_vals <- df1_processed_tbl %>%filter(key == "training") %>%select(value) %>%pull()
valid_vals <- df1_processed_tbl %>%filter(key == "validation") %>%select(value)%>%pull()
test_vals <- df1_processed_tbl %>%filter(key == "testing") %>%select(value) %>%pull()


# the three windowed matrices 
??build_matrix
train_matrix <- build_matrix(train_vals, hidden_len + n_predictions)
valid_matrix <-  build_matrix(valid_vals, hidden_len + n_predictions)
test_matrix <- build_matrix(test_vals, hidden_len + n_predictions)

dim(train_matrix)
dim(valid_matrix)
dim(test_matrix)
# creating two new separate matrices of training and testing 
# conditioning on discarding the last batch - to avoid fewer than batch_size 
X_train <- train_matrix[, 1:hidden_len]
y_train <- train_matrix[, (hidden_len + 1):(hidden_len * 2)]
X_train <- X_train[1:(nrow(X_train) %/% batch_size * batch_size), ]
y_train <- y_train[1:(nrow(y_train) %/% batch_size * batch_size), ]
dim(X_train)
dim(y_train)

X_valid <- valid_matrix[, 1:hidden_len]
y_valid <- valid_matrix[, (hidden_len + 1):(hidden_len * 2)]
X_valid <- X_valid[1:(nrow(X_valid) %/% batch_size * batch_size), ]
y_valid <- y_valid[1:(nrow(y_valid) %/% batch_size * batch_size), ]
dim(X_valid)
dim(y_valid)
X_test <- test_matrix[, 1:hidden_len]
y_test <- test_matrix[, (hidden_len + 1):(hidden_len * 2)]
X_test <- X_test[1:(nrow(X_test) %/% batch_size * batch_size), ]
y_test <- y_test[1:(nrow(y_test) %/% batch_size * batch_size), ]
dim(X_test)
dim(y_test)
# adding the third axis
X_train <- reshape_X_3d(X_train)
X_valid <- reshape_X_3d(X_valid)
X_test <- reshape_X_3d(X_test)
y_train <- reshape_X_3d(y_train)
y_valid <- reshape_X_3d(y_valid)
y_test <- reshape_X_3d(y_test)
dim(X_train)
dim(y_train)
dim(X_valid)
dim(y_valid)
dim(X_test)
dim(y_test)

## Defining all required LSTM PARAMETERS as per keras 
# using RMSE, stochastic gradient, less learning rate for now
FLAGS <- flags(flag_boolean("stateful", FALSE),flag_boolean("stack_layers", FALSE),flag_integer("batch_size", 10),
   flag_integer("hidden_len", 12),flag_integer("n_epochs", 50),flag_numeric("dropout", 0.2),flag_numeric("recurrent_dropout", 0.2),
   flag_string("loss", "logcosh"),flag_string("optimizer_type", "sgd"),flag_integer("n_units", 128),flag_numeric("lr", 0.003),
  flag_numeric("momentum", 0.9),flag_integer("patience", 10))

n_predictions <- FLAGS$hidden_len
# only one variable so one feature
n_features <- 1
callbacks <- list(callback_early_stopping(patience = FLAGS$patience))
##model running starts here
model <- keras_model_sequential()

# adding two layers - one for shape and other for result
model %>%layer_lstm(units = FLAGS$n_units,batch_input_shape  = c(FLAGS$batch_size, FLAGS$hidden_len, n_features),
    dropout = FLAGS$dropout,recurrent_dropout = FLAGS$recurrent_dropout,return_sequences = TRUE) %>% time_distributed(layer_dense(units = 1))

## defining a simple keras optimizer before we estimate the model
##compiling the model with simple adam optimize
model %>%compile(loss = FLAGS$loss,optimizer = 'adam',metrics = list("mean_squared_error"))

## Fittling LSTM 
fitting_sequence <- model %>% fit(x= X_train,y= y_train,validation_data = list(X_valid, y_valid),batch_size = FLAGS$batch_size,
  epochs= FLAGS$n_epochs,callbacks = callbacks)
## seems epochs = 29 is fine. plotting the loss as a whole
plot(fitting_sequence, metrics = "loss")
## seems overfitting for slice 6 not for slice 1
## checking the charac of training - predict and return tidy data
pred_train <- model %>%predict(X_train, batch_size = FLAGS$batch_size) %>%.[, , 1]

# Retransform values to original scale
pred_train <- (pred_train * sd_scale + mean_scale) ^2
compare_train <- df1 %>% filter(key == "training")
head(pred_train)

# dataframe with both actual and predicted values
for (i in 1:nrow(pred_train)) {varname <- paste0("pred_train", i)
compare_train <-mutate(compare_train,!!varname := c(
      rep(NA, FLAGS$hidden_len + i - 1),pred_train[i,],rep(NA, nrow(compare_train) - FLAGS$hidden_len * 2 - i + 1)
))
}
summary(compare_train)
head(compare_train)
## check for rmse for efficiency  of train
rmse()
??quo
coln <- colnames(compare_train)[4:ncol(compare_train)]
cols <- map(coln, quo(sym(.)))
rsme_train <-
  map_dbl(cols, function(col)
    rmse(
      compare_train,
      truth = value,
      estimate = !!col,
      na.rm = TRUE
    )) %>% mean()

rsme_train


## yhat prediction for training 
ggplot(compare_train, aes(x = index, y = value)) + geom_line() +
  geom_line(aes(y = pred_train1), color = "cyan") +
  geom_line(aes(y = pred_train50), color = "red") +
  geom_line(aes(y = pred_train100), color = "green") +
  geom_line(aes(y = pred_train150), color = "violet") +
  geom_line(aes(y = pred_train200), color = "cyan") +
  geom_line(aes(y = pred_train250), color = "red") +
  geom_line(aes(y = pred_train300), color = "red") +
  geom_line(aes(y = pred_train350), color = "green") +
  geom_line(aes(y = pred_train400), color = "cyan") +
  geom_line(aes(y = pred_train450), color = "red") +
  geom_line(aes(y = pred_train500), color = "green") +
  geom_line(aes(y = pred_train550), color = "violet") +
  geom_line(aes(y = pred_train600), color = "cyan") +
  geom_line(aes(y = pred_train650), color = "red") +
  geom_line(aes(y = pred_train700), color = "red") +
  geom_line(aes(y = pred_train750), color = "green") +
  ggtitle("Predictions on the training set")+ theme(text=element_text(size=16,  family="Nirmala UI Semilight"))

## yhat for testing set
pred_test <- model %>%predict(X_test, batch_size = FLAGS$batch_size) %>%.[, , 1]

#transform to original scale
pred_test <- (pred_test * sd_scale + mean_scale) ^2
pred_test[1:10, 1:5] %>% print()
compare_test <- df1 %>% filter(key == "testing")

# both actual and predicted values together
for (i in 1:nrow(pred_test)) {
  varname <- paste0("pred_test", i)
  compare_test <-
    mutate(compare_test,!!varname := c(
      rep(NA, FLAGS$hidden_len + i - 1),
      pred_test[i,],
      rep(NA, nrow(compare_test) - FLAGS$hidden_len * 2 - i + 1)
    ))
}

compare_test[FLAGS$hidden_len:(FLAGS$hidden_len + 10), c(2, 4:8)] %>% print()

coln <- colnames(compare_test)[4:ncol(compare_test)]
cols <- map(coln, quo(sym(.)))
rsme_test <-
  map_dbl(cols, function(col)
    rmse(
      compare_test,
      truth = value,
      estimate = !!col,
      na.rm = TRUE
    )) %>% mean()

rsme_test



## plot - test hat
ggplot(compare_test, aes(x = index, y = value)) + geom_line() +
  geom_line(aes(y = pred_test1), color = "red") +
  geom_line(aes(y = pred_test50), color = "red") +
  geom_line(aes(y = pred_test100), color = "green") +
  geom_line(aes(y = pred_test150), color = "red") +
  geom_line(aes(y = pred_test200), color = "red") +
  geom_line(aes(y = pred_test250), color = "red") +
  geom_line(aes(y = pred_test300), color = "green") +
  geom_line(aes(y = pred_test350), color = "red") +
  geom_line(aes(y = pred_test400), color = "red") +
  geom_line(aes(y = pred_test450), color = "green") +  
  geom_line(aes(y = pred_test500), color = "red") +
  geom_line(aes(y = pred_test550), color = "violet") +
  ggtitle("Predictions on test set")+ theme(text=element_text(size=16,  family="Nirmala UI Semilight"))

## not as good as training set

## can do the same for all different slices  - repeat the same for all slices again
## did this in another R file - WIP

