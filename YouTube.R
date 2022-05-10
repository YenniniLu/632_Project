
# CSUEB STAT632 Final Project
# Multiple Linear Regression for YouTube Videos View Prediction
# Xinyi Lu, Daiyan Zhang
# 5/10/2022


# loading all the libraries
library(pacman)
p_load(tidyverse, dplyr, stringr, randomForest, vip, rpart, rpart.plot, caret, tidytext, tidyr, MASS, car)

df_raw <- read.csv("data.csv")

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

# Save for future use
write_csv(df, "cleaned_data.csv")


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
# use tf-idf to find the importance of a word in the transcript, then times it to a word's afinn score, and sum up all the words' socre to a video afinn_score.

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





# Data Discovery / Diagnostics for Linear Regression

df <- read_csv("cleaned_data.csv")
df$CC <- as.factor(df$CC)
df$Category <- as.factor(df$Category)
df$Subscribers <- as.numeric(df$Subscribers)













