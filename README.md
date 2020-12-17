# Hotel Review Analysis Using NLP and Machine Learning

#### Can hotels easily get insights out of thousand of reviews?

##### Author:  Ismael Araujo

![](https://github.com/Ismaeltrevi/hotel-reviews-analysis-using-nlp/blob/main/images/hotel-image.jpg?raw=true)

##### This project is expected to be concluded on January 6, 2021.

## Overview



## Business Problem
One of the biggest problems that many companies have been battleling is how to take advantage of all the data that is collected by them. The travel industry has been challenged by the amount of data. One of the types of data is reviews left by guests in websites such as Booking.com, TripAdvisor, and Yelp.

Hotels have been trying to find ways to analyze the reviews and get insights out of it. However, some hotels can receive thousands of guests every week and hundreds of reviews. It becomes nearly impossible and expensive for hotels to keep track of the reviews. Thus, multiple hotels might negligete these valuable data due to the cost and energy that need to be allocated.

For these reasons, here are a few questions that I will try to answer with this project:

1. Can machine learning be used to correctly identify positive or negative reviews?
2. Can NLP be used to create a useful word cloud with the the positive and negative reviews?
3. How is a specific hotel compared to the other hotels in the same city? Can we find more information about the guests based on their review?

## Data and Methods
The dataset for this project was originally used in the study Text Mining in Hotel Reviews: Impact of Words Restriction in Text Classification by Diego Campos, Rodrigo Rocha Silva, and Jorge Bernadino and a team at University of Coimbra. You can find the paper [here](https://www.researchgate.net/publication/336224346_Text_Mining_in_Hotel_Reviews_Impact_of_Words_Restriction_in_Text_Classification "here"). The raw dataset is from [Kaggle](https://www.kaggle.com/jiashenliu/515k-hotel-reviews-data-in-europe/data "Kaggle"). Since some datasets were large, you can download the cvs files [here](https://drive.google.com/drive/folders/1mjoGF17DR8bcqLhHQ78IqXgY81rSonVM?usp=sharing "here"). The original dataset had 515,738 observations and 17 columns.

#### Dataset Features
| Column Name  | Description  |
| ------------ | ------------ |
|Hotel_Address   | The full address of the hotel  |
|Review_Date|The date when the review was given   |
|Average_Score|Average Hotel Score   |
|Hotel_Name   |Name of Hotel   |
|Reviewer_Nationality|Nationality of the reviewer   |
|Negative_Review   |Text of the negative review   |
|Review_Total_Negative_Word_Counts   |Number of words in the negative review   |
|Positive_Review   |Text of the positive review   |
|Review_Total_Positive_Word_Counts   |Number of words in the positive review   |
|Reviewer_Score   |Score given by user (this information is arguably)   |
|Total_Number_of_Reviews_Reviewer_Has_Given   |Number of reviews written by user   |
|Total_Number_of_Reviews   |Number of reviews given to the hotel   |
|Tags   |Tags related to the hotel   |
|Days_Since_Review   |How many days the review was written   |
|Additional_Number_of_Scoring   |This is how many users gave a scoring without giving a written review   |
|lat   |The latitude of the hotel   |
|lng   |The longitude of the hotel   |

#### Libraries Used

|Used for |Libraries   |
| ------------ | ------------ |
|Cleaning and EDA   | pandas, nltk, matplotlib, seaborn, string, wordcloud, numpy, pickle, geopy  |
|Modeling   | pandas, matplotlib, sklearn, pickle  |
|Custom Functions   |matplotlib, numpy, pandas, sklearn, xgboost   |



## Challenges
The data set was quite organized. However, it had a few challenges. For example, the review was divided between positive and negative reviews. Although this is useful for specific cases, most reviews will not be separated by positive and negative reviews. Thus, creating a model that is able to idenfity positive and negative reviews could be useless if we add reviews that are not separated. Other uses for the model, such as using it in social media would not work. A solution was merging the together.

Other big challenge with dataset is that the address is a long string, which means that is difficult to collect information such as city and country due to the different formats which is written. This problem was solved grabbing the city and country of the hotel using Geopy.

## Data Cleaning
As in any NLP project, the data cleaning is the most important step. We need to make sense of the information we have and create features for the models only with the information that matters. A few cleaning steps that I took were: 
- Remove any puntuations, stop words, and numbers
- Fix the spelling using TextBlob
- Lemmatization
- Remove words from the word clouds that didn't help with insights, such as complimentatory adjectives (great, good, nice, excelent, etc.)

## Data Understanding
Data understanding was a very interesting step in this project. I was able to understand the dataset. First thing I noticed was that the dataset was very large. This could be a problem when modeling. Thus, for the MVP, I decided to use only 20% of the dataset.

Since I had a classification problem, I first created a target feature with the score. In this step I noticed that the lowest score was 2.5 and the highest was 10. Since there were users who wrote that there was anything good about the hotel, I assume that 2.5 was actually given by Booking.com, not the user. Then I converted the score into a classification problem. Since 2.5 was the lowest and 10 the highest, I decided to dived the target in two: positive and negative, where anything below 6 was negative and 6 and above was positive. 6.25 would be the middle point, so I decided to go with 6.

### Class Distribution

![Class Distribution - Before](https://github.com/Ismaeltrevi/hotel-reviews-analysis-using-nlp/blob/main/images/class_distribution-before.png?raw=true "Class Distribution - Before")

We can see that there is a big class imbalance. Since our dataset is large, we can fix this using the pandas sample function. Thus, I will only use 12% of the positive reviews, which will get closer to the number of negative reviews, so it will get closer to the number of negative reviews.

![Class Distribution - After](https://github.com/Ismaeltrevi/hotel-reviews-analysis-using-nlp/blob/main/images/class_distribution-after.png?raw=true "Class Distribution - After")

We can see above that the class distribution problem was solved.

### Exploratory Data Analysis
I wanted a few answer from the dataset and check if I could find any patterns. Thus, I had questions, which I will answer right next:

**What nationalities give the highest number of reviews?**

![](https://github.com/Ismaeltrevi/hotel-reviews-analysis-using-nlp/blob/main/images/nationality.png?raw=true)

Reviewers from the United Kingdom have the highest number of reviews left. To understand the main reason behind this, I checked where the hotels with the highest reviews are located. In the next image we can understand two things:
- The hotels are located in only 6 countries only, not the whole Europe, as I previously thought.
- The reviewers are mostly from the UK (over 50%), which explain the higher number of people from the UK giving reviews.

![](https://github.com/Ismaeltrevi/hotel-reviews-analysis-using-nlp/blob/main/images/countries-hotels.png?raw=true)

**How does the hotel which I will focus on perform compared to other hotels in London?**

![](https://github.com/Ismaeltrevi/hotel-reviews-analysis-using-nlp/blob/main/images/score-density-hotel-vs-london.png?raw=true)

As we can see above, the Britannia International Hotel Canary Wharf underperform compared to other hotels in London. While the hotels in London have the score density skewed to the right where the positive scores are, Britannia International Hotel Canary Wharf is closer to a evenly distribution, where the amount of negative reviews are as high as the negative reviews. This tells as the the hotel has a lot to improve.

For the results section, I had two question in mind:

- What words appear the most in positive and negative reviews?
- Can we get any insights from it?

To get these answers, I had to clean the data through different steps. You can see more details of each step  in the [Data Cleaning notebook](https://github.com/Ismaeltrevi/hotel-reviews-analysis-using-nlp/blob/main/preprossessing/data-cleaning.ipynb "Data Cleaning notebook"). Now let's check the word clouds and see if we can get insights from it.

Below we can see the positive reviews word cloud. Note that you can relate every word to things that you could image coming from a hotel review. For example, we can see the words location, clean, comfortable, staff, service, price, and room. We can certainly assure that these are high points to the hotel and they can use these words to promote the hotel.

![](https://github.com/Ismaeltrevi/hotel-reviews-analysis-using-nlp/blob/main/images/positive_wordcloud.png?raw=true)

On the other hand, looking at the negative reviews word cloud, we can see many words that could be points for improvement to the hotel. Staff seems it’s mentioned multiple times as well as the word rude and reception. Maybe the staff was rude to these guests? We can also see the words old, dated, dirty, and uncomfortable, which could be points of attention to the management.

![](https://github.com/Ismaeltrevi/hotel-reviews-analysis-using-nlp/blob/main/images/negative_wordcloud.png?raw=true)


## Results

## Final Recommendations

## Conclusion

## Repository Content

## For More Information or Suggestions
If you have any questions or suggestions, please reach me out on:
Email: alves.trevi@gmail.com
LinkedIn: https://www.linkedin.com/in/ismael-araujo/
Twitter: https://twitter.com/ish_araujo

## References
Campos, D., Rocha Silva, R., and Bernadino, J., 2019. Text Mining in Hotel Reviews: Impact of Words Restriction in Text Classification. University of Coimbra, Coimbra, Portugal.



 



