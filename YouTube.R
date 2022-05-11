
# CSUEB STAT632 Final Project
# Multiple Linear Regression for YouTube Videos View Prediction
# Xinyi Lu, Daiyan Zhang
# 5/10/2022


# loading all the libraries
library(pacman)
p_load(tidyverse, dplyr, stringr, randomForest, vip, rpart, rpart.plot, caret, tidytext, tidyr, MASS, car)

# read original data set
df_raw <- read.csv("data.csv")

dim(df_raw)
# Functions for standardize variable unit 
# Unify units and convert string to number, like: 10K views -> 10, 10M views -> 10000
cleanViews <- function(str) {
  str <- str_remove(str, " views")
  last <- str_sub(str, -1)
  views <- str %>% str_remove(last) %>% as.numeric()
  if (last == "M") return(1000*views)
  else return(views)
}

# Unify units and convert string to number, like: 10K subscribers -> 10, 10M subscribers -> 10000
cleanSubscribers <- function(str) {
  str <- str_remove(str, " subscribers")
  last <- str_sub(str, -1)
  views <- str %>% str_remove(last) %>% as.numeric()
  if (last == "M") return(1000*views)
  else return(views)
}

# Convert time in string format to number of minutes, like: 12:00 -> 12, 1:12:00 -> 72
cleanLength <- function(str) {
  list <- str_split(str, ":")
  len <- length(list[[1]])
  if (len == 3) {
    h <- as.numeric(list[[1]][1])
    m <- as.numeric(list[[1]][2])
    return((m + 1) + 60*h)
  } else {
    m <- as.numeric(list[[1]][1])
    return(m+1)
  }
}

# Convert time to number of months ago, like: 1 years ago -> 12, 10 months ago to 10
cleanReleased <- function(str) {
  str <- str_remove(str, "Streamed ")
  list <- str_split(str, " ")
  if (list[[1]][2] == "years") return(as.numeric(list[[1]][1])*12)
  else return(as.numeric(list[[1]][1]))
}

# Remove NAs
df <- df_raw %>%  
  na.omit() %>%
  filter(
    Released != "",
    Title != "",
    Transcript != ""
  ) 

# Clean the data
df <- df %>% mutate(
  Views = map_dbl(Views, cleanViews),
  Subscribers = map_dbl(Subscribers, cleanSubscribers),
  Length = map_dbl(Length, cleanLength),
  Released = map_dbl(Released, cleanReleased)
)

df %>% dplyr::select(URL, Channel, Views, Subscribers, Released, Length) %>% head(10)

# function to compute RMSE
RMSE <- function(y, y_hat) {
  sqrt(mean((y - y_hat)^2))
}


# Save for future use
write_csv(df, "cleaned_data.csv")

df <- read_csv("cleaned_data.csv")
df$CC <- as.factor(df$CC)
df$Category <- as.factor(df$Category)
df$Subscribers <- as.numeric(df$Subscribers)

# Sentiment Analysis / Text mining
df_script <- df %>% 
  dplyr::select(Id, Title, Transcript)

data("stop_words")
custom_stop_words <- rbind(stop_words, c("_", "custom"))

df_word <- df %>% 
  group_by(Id) %>% 
  unnest_tokens(word, Transcript) %>% 
  anti_join(custom_stop_words) %>% 
  count(word, sort = TRUE) %>% 
  mutate(total = sum(n)) %>% 
  ungroup()


df_title_word <- df %>% 
  group_by(Id) %>% 
  unnest_tokens(word, Title) %>% 
  anti_join(custom_stop_words) %>% 
  count(word, sort = TRUE) %>% 
  mutate(total = sum(n)) %>% 
  ungroup()

# Below codes:
# use tf-idf to find the importance of a word in the transcript,
# then times it to a word's afinn score, and sum up all the words' socre to a video afinn_score.

# Transcript sentiment
df_afinn <- df_word %>% 
  left_join(get_sentiments("afinn")) %>% 
  group_by(Id) %>% 
  bind_tf_idf(word, Id, n) %>% 
  mutate(value = ifelse(is.na(value), 0, value)) %>%
  mutate(afinn_score = sum(value*tf_idf)) %>%
  ungroup() 

df_afinn <- df_afinn %>% 
  dplyr::select(Id, afinn_score) %>% 
  unique() %>% 
  ungroup()

# Title sentiment
df_title_afinn <- df_title_word %>% 
  left_join(get_sentiments("afinn")) %>% 
  group_by(Id) %>% 
  bind_tf_idf(word, Id, n) %>% 
  mutate(value = ifelse(is.na(value), 0, value)) %>%
  mutate(afinn_title_score = sum(value*tf_idf)) %>%
  ungroup() 

df_title_afinn <- df_title_afinn %>% dplyr::select(Id, afinn_title_score) %>% 
  unique() %>% 
  ungroup()


df1 <- df %>% 
  left_join(df_afinn) %>% 
  left_join(df_title_afinn) %>% 
  mutate(
    afinn_title_score = ifelse(is.na(afinn_title_score), 0, afinn_title_score)
  ) %>% 
  unique() 



# use df1 for all the analysis 
write_csv(df1, "cleaned_data_with_sentiment.csv")
df1 <- read_csv("cleaned_data_with_sentiment.csv")

df1$CC <- as.factor(df1$CC)
df1$Category <- as.factor(df1$Category)
df1$Subscribers <- as.numeric(df1$Subscribers)

# Randomly split the data set in a 70% training and 30% test set. 
# Make sure to use set.seed() so that your results are reproducible
set.seed(652)
n <- nrow(df1)
train_index <- sample(1:n, round(0.7*n))
df_train <- df1[train_index,]
df_test <- df1[-train_index,]

# visualize variables
hist(df1$Views)

table(df1$CC)
boxplot(log(Views) ~ CC, data = df1)

hist(df1$Length, xlab = "Minutes", breaks = 50)
summary(df1$Length) 

summary(df1$Released)
hist(df1$Released, xlab = "Month Ago")

summary(df1$Subscribers)
hist(df1$Subscribers, xlab = "K Subscribers")

# scatter plot matrix
pairs(Views ~ CC + Released + Length + Subscribers + Category, data=df1)

# log response and other predictors that are right skewed
pairs(log(Views) ~ CC + log(Released) + log(Length) + log(Subscribers) + Category + afinn_score + afinn_title_score, data=df1)

# correlation table for predictors
round(cor(df1[, c(3,7,11,12,13)]), 2)





# Linear Regression

# full model without log transf' on training set
lm_full_train <- lm(Views ~ CC + Released + Length + Subscribers + Category + afinn_score+ afinn_title_score, data=df_train)
summary(lm_full_train)

# full model without log transf' on 100% original data set
lm_full <- lm(Views ~ CC + Released + Length + Subscribers + Category + afinn_score+ afinn_title_score, data=df1)
summary(lm_full)

# make prediction for full model without transformer
pred1 <- predict(lm_full_train, newdata = df_test)
pred_full <- pred1; length(pred_full)

# Compute the RMSE
lm_full_RMSE <- RMSE(df_test$Views, pred_full); lm_full_RMSE 

# MLR assumption check for lm_full_train model
par(mfrow=c(1,3))
plot(predict(lm_full_train), resid(lm_full_train), xlab = "Fitted values", ylab = "Residuals")
abline(h=0)
hist(resid(lm_full_train))
qqnorm(resid(lm_full_train))
qqline(resid(lm_full_train))

# include all the predictors and with log transf's on training set
lm1_train <- lm(log(Views) ~ CC + log(Released) + log(Length) + log(Subscribers) + Category + afinn_score+ afinn_title_score, data=df_train)
summary(lm1_train)

# MLR assumption check for lm1_train model
par(mfrow=c(1,3))
plot(predict(lm1_train), resid(lm1_train), xlab = "Fitted values", ylab = "Residuals")
abline(h=0)
hist(resid(lm1_train))
qqnorm(resid(lm1_train))
qqline(resid(lm1_train))

# make prediction for full model with log transformer
pred2 <- predict(lm1_train, newdata = df_test)
pred2 <- exp(pred2); length(pred2)

# Compute the RMSE
lm_log_RMSE <- RMSE(df_test$Views, pred2); lm_log_RMSE

# include all the predictors and with log transf's on 100% set
lm1 <- lm(log(Views) ~ CC + log(Released) + log(Length) + log(Subscribers) + Category + afinn_score+ afinn_title_score, data=df1)
summary(lm1)

# MLR assumption check for lm1 model
plot(predict(lm1), resid(lm1), xlab = "Fitted values", ylab = "Residuals")
abline(h=0)
par(mfrow=c(1,2))
hist(resid(lm1))
qqnorm(resid(lm1))
qqline(resid(lm1))

# variable selection
lm_step_train <- step(lm1_train)
summary(lm_step_train)

# MLR assumption check for lm_step_train model
par(mfrow=c(1,3))
plot(predict(lm_step_train), resid(lm_step_train), xlab = "Fitted values", ylab = "Residuals")
hist(resid(lm_step_train))
qqnorm(resid(lm_step_train))
qqline(resid(lm_step_train))


# Make prediction
pred_lm <- predict(lm_step_train, newdata = df_test);

# Compute the RMSE
lm_RMSE <- RMSE(df_test$Views, pred_lm); lm_RMSE


# make this prediction and to calculate a 95% prediction interval.

# randomly pick one youtube video from data set
df1[300, ]

# use the above to predict View and compare with the actual View
new_x <- data.frame(CC = "0", Length = 12,  Subscribers = 1590, Category = "Tech,Comedy", afinn_score = 0.1143453)
exp(predict(lm_step_train, newdata = new_x, interval="prediction"))

# do it agian with anothe video
df1[2000, ]
new_x <- data.frame(CC = "1", Length = 7,  Subscribers = 18700, Category = "Food", afinn_score = 0.4008991)
exp(predict(lm_step_train, newdata = new_x, interval="prediction"))





# Regression Tree

# Fit a regression tree on the training set.
# Fit tree model
t1 <- rpart(Views ~ CC + Released + Category + Length + Subscribers + afinn_score + afinn_title_score,
            data = df_train,
            method = "anova")
summary(t1)

# Plot the desicion tree
rpart.plot(t1)

# Plot R-square vs Splits and the Relative Error vs Splits.
rsq.rpart(t1)

# Make prediction
pred_tree <- predict(t1, newdata = df_test)

# Compute the RMSE
t1_RMSE <- RMSE(df_test$Views, pred_tree); t1_RMSE

# Compute R^2
t1_R2 <- cor(df_test$Views, pred_tree)^2; t1_R2


# Random Forest

# Fit a Random Forest on the training set usinng the defaults for mtry and ntree = 500.
# default mtry: p/3 = 28/3 = 9 (mtry: Number of predictors randomly sampled as candidates at each split.)

set.seed(652)
# Fit random forest mode using the training set
rf1 <- randomForest(Views ~ CC + Released + Category + 
                      Length + Subscribers + afinn_score + 
                      afinn_title_score, importance = TRUE, 
                    data = df_train)
rf1

# Make a variable importance plot
vip(rf1, num_features = 14,  include_type = TRUE)

plot(c(1: 500), rf1$mse, xlab="ntree", ylab="MSE", type="l")

# Make prediction
pred_rf <- predict(rf1, newdata = df_test);

# Compute the RMSE
rf1_RMSE <- RMSE(df_test$Views, pred_rf); rf1_RMSE

# Compute R^2
rf1_R2 <- cor(df_test$Views, pred_rf)^2; rf1_R2

# test R^2
rf1_R2 <- cor(df_test$Views, pred_rf)^2; rf1_R2


# Conclusion for comparing models

conclusion1 <- data.frame(Model=c("lm_full_train","full model without transformation",  "lm1_train","lm1", "lm_step_train"),
           R_squared = c(summary(lm_full_train)$r.squared,
                         summary(lm_full)$r.squared,
                         summary(lm1_train)$r.squared,
                         summary(lm1)$r.squared,
                         summary(lm_step_train)$r.squared),
           adj_R_squared = c( summary(lm_full_train)$adj.r.squared,
                              summary(lm_full)$adj.r.squared,
                              summary(lm1_train)$adj.r.squared,
                              summary(lm1)$adj.r.squared,
                             summary(lm_step_train)$adj.r.squared),
           formula = c("lm(Views ~ CC + Released + Length + Subscribers + Category + afinn_score+ afinn_title_score, data=df_train)",
                       "lm(Views ~ CC + Released + Length + Subscribers + Category + afinn_score+ afinn_title_score, data=df1)",
                       "lm(log(Views) ~ CC + log(Released) + log(Length) + log(Subscribers) + Category + afinn_score+ afinn_title_score, data=df_train)",
                       "lm(log(Views) ~ CC + log(Released) + log(Length) + log(Subscribers) + Category + afinn_score+ afinn_title_score, data=df1)",
                       "lm(log(Views) ~ CC + log(Released) + log(Length) + log(Subscribers) + Category + afinn_title_score, data = df_train)"))



conclusion2 <- data.frame(Model=c("linear regression", "regression tree", "random forest"),
           R_squared = c(summary(lm_step_train)$r.squared,
                         cor(df_test$Views, pred_tree)^2,
                         cor(df_test$Views, pred_rf)^2),
           RMSE = c(lm_RMSE,
                    t1_RMSE,
                    rf1_RMSE))






