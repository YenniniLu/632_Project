# YouTube Videos View Prediction Modeling Project

## Reserach Goal

Find the best model to predict the number of Views using sub-set of predictors: Subscribers, CC, Released, Category, Sentiment and Length.

## Data Sets
data from: "YouTubers saying things" by PRANESH MUKHOPADHYAY 

Kaggle: https://www.kaggle.com/datasets/praneshmukhopadhyay/youtubers-saying-things 

Github: https://github.com/Mukhopadhyay/YouTubers-saying-things


## Description
This dataset contains subtitles from over `91` different YouTubers, ranging from all different kinds of categories. The data were collected and cleaned (as much as necessary) by PRANESH MUKHOPADHYAY. The `Transcript` column in the dataset contains the subtitles for the respective videos. However, the reliability of the subtitles may vary. Even though the auto-generated subtitles work great (most of the time). Sometimes under heavy pressure of thick accents, it lets go of the ball. Please consult the `CC` attribute to check whether the subtitle is auto-generated or not. `1381` of these video subtitles are auto-generated, the rest of the `1134` are manual ones. Currently, the dataset contains `2515` unique videos and their subtitles. There are `11` columns in the dataset. The purpose of each column is as follows:

|**No**|**Column**|**Dtype**|**Description**|
|:-----|:---------|:--------|:--------------|
|1|`Id`|`str`|Unique ID for the video. (e.g., `dQw4w9WgXcQ`)|
|2|`Channel`|`str`|Name of the YouTube channel.|
|3|`Subscribers`|`str`|How many subscribers did the channel have while collecting the dataset|
|4|`Title`|`str`|Title of the video|
|5|`CC`|`int`|Did the video have manual subtitles? (Possible values `0` or `1`, where `0` means that the Transcript is auto-generated, and may be less reliable)|
|6|`URL`|`str`|URL of the video (e.g., `https://www.youtube.com/watch?v=dQw4w9WgXcQ`)|
|7|`Released`|`str`|When the video was released|
|8|`Views`|`str`|How many views did the video have during the collection of this dataset|
|9|`Category`|`str`|Category of the channel (e.g., **Science**, **Comedy**, etc)|
|10|`Transcript`|`str`|Subtitle for the video|
|11|`Length`|`str`|Duration of the video|

## Genres and channels

|**Genre**|**Channels**| **#** |
|:--------|:-----------|:------|
|`Comedy`|**Brooklyn Nine-Nine**, **DRIVETRIBE**, **Incognito Mode**, **Internet Historian**, **Joma Tech**, **Key & Peele**, **OverSimplified**, **Parks and Recreation**, **penguinz0**, **Screen Junkies**, **Team Coco**, **The Graham Norton Show**, **The Grand Tour**, **The Office**, **Top Gear**|15|
|`Science`|**3Blue1Brown**, **Dr. Becky**, **ElectroBOOM**, **Kurzgesagt – In a Nutshell**, **Lex Clips**, **Mark Rober**, **NileRed**, **PBS Space Time**, **SEA**, **SmarterEveryDay**, **Veritasium**, **Vsauce**|12|
|`Automobile`|**Car Throttle**, **carwow**, **ChrisFix**, **Donut Media**, **DRIVETRIBE**, **The Grand Tour**, **The Stig**, **TheStraightPipes**, **Throttle House**, **Top Gear**|10|
|`VideoGames`|**EpicNameBro**, **gameranx**, **jacksepticeye**, **Let's Game It Out**, **LevelCapGaming**, **LGR**, **Markiplier**, **Shroud**, **TheWarOwl**, **videogamedunkey**|10|
|`Food`|**About To Eat**, **Bon Appétit**, **Eater**, **Epicurious**, **First We Feast**, **FoodTribe**, **Gordon Ramsay**, **Hell's Kitchen**, **Munchies**, **Mythical Kitchen**, **The F Word**|11|
|`Entertainment`|**Brooklyn Nine-Nine**, **BuzzFeedVideo**, **Doctor Who**, **First We Feast**, **MrBeast**, **Parks and Recreation**, **Screen Junkies**, **Team Coco**, **The Graham Norton Show**, **The Office**, **The Try Guys**|11|
|`Informative`|**Barely Sociable**, **Captain Disillusion**, **CNET**, **Coder Coder**, **Internet Historian**, **JCS - Criminal Psychology**, **LEMMiNO**, **OverSimplified**, **Sam O'Nella Academy**, **The Infographics Show**, **Vox**|11|
|`Blog`|**Abroad in Japan**, **CDawgVA**, **DramaAlert**, **Drew Gooden**, **Incognito Mode**, **Joma in NYC**, **JRE Clips**, **Lex Clips**, **MrBeast**, **penguinz0**, **The Try Guys**|11|
|`News`|**A&E**, **BBC News**, **Insider News**, **NBC News**, **NowThis News**, **Sky News**, **SomeGoodNews**, **TechLinked**, **The Daily Show with Trevor Noah**, **VICE**|10|
|`Tech`|**Austin Evans**, **Coder Coder**, **Fireship**, **Hardware Canucks**, **Joma Tech**, **Linus Tech Tips**, **Marques Brownlee**, **TechLinked**, **Techquickie**, **Web Dev Simplified**|10|



## Tasks

1. Data Cleaning and manipulation. -> Daiyan
2. Sentiment Analysis -> Xinyi
3. Regression Model -> Xinyi
4. Random Forest -> Daiyan
5. Compare models and make conclusion -> Daiyan & Xinyi






