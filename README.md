# Hotel Review Analysis Using NLP and Machine Learning

#### Can hotels quickly get insights out of thousands of reviews?

##### Author:  Ismael Araujo

![](https://github.com/Ismaeltrevi/hotel-reviews-analysis-using-nlp/blob/main/images/hotel-image.jpg?raw=true)

##### This project is expected to be concluded on January 6, 2021.

## Overview
In this project, I will create a model that can predict if a hotel review is negative or positive so that hotels can use it to classify their reviews correctly. I will analyze a specific hotel in London and compare it to other hotels in London as well. We will walk through multiple Natural Language Processing to understand how we can use machines to read reviews and get insights out of it. Baseline models include Logistic Regression, Random Forest, Naive Bayes, and Support Vector Machine (SVM). Ensemble models include Voting, Bagging, GridSearch, AdaBoost, and Gradient Boosting. The final model was a GridSearch SVM with an accuracy of 0.8268 and F1-Score 0.8247.

This project walks through exploratory data analysis, data cleaning, sentiment analysis, data preprossessing, vanilla model, and ensemble model iterations. You can find a summary of the project in the final notebook.

## Business Problem
One of the biggest problems that many companies have been trying to overcome is how to take advantage of all the data collected from guests. The amount of data has challenged the travel industry. One type of data is reviews left by guests on websites such as Booking.com, TripAdvisor, and Yelp.

Hotels have been trying to find ways to analyze the reviews and get insights out of them. However, some hotels can receive thousands of guests every week and hundreds of reviews. It becomes nearly impossible and expensive for hotels to keep track of the reviews. Thus, multiple hotels might ignore these valuable data due to the cost and energy that need to be allocated. The other problem is that hotels such as Booking.com doesn't allow users to choose their score. The score is determined by questions asked to the user, and then the review is calculated. This is problematic because guests could have had a bad experience and the hotel would still get a 7 or 8 score, which gives a false illusion that the guest didn't have any problems.

This project will build a model that can correctly predict if a hotel review is negative or positive so that hotels can input their reviews and get a non-biased score.

### Setting the hypothetical scenario

Our actual client is a hotel in London called Britannia International Hotel Canary Wharf. They have thousands of reviews and a 6.7 overall score on Booking.com. They think this is a low score compared to other hotels in London, and they want to understand what is causing this low score. Due to COVID-19, they don't have the resources to read all the reviews and make sense of them. Thus, they want to find a way to get quick insights without having to read every review. They have a few business questions:

- Can we create a model that can correctly identify the most important features when predicting if a review is positive or negative for all the reviews we have available? What are these features?
- What are the most mentioned words in negative and positive reviews? What insights could they get from them? How would a word cloud for negative and positive reviews look like for their hotel and in comparison to other hotels?
- How does the client score performs compared to other hotels in the city?

#### Why Britannia International Hotel Canary Wharf?

While doing the Exploratory Data Analysis, I noticed that Britannia International Hotel Canary Wharf was the hotel with the highest number of reviews. The average score is 6.7, which means that there is probably room for improvement. It is more likely to find different word clouds for negative and positive reviews.


## Data and Methods
The dataset for this project was originally used in the study Text Mining in Hotel Reviews: Impact of Words Restriction in Text Classification by Diego Campos, Rodrigo Rocha Silva, and Jorge Bernadino and a team at the University of Coimbra. You can find the paper [here](https://www.researchgate.net/publication/336224346_Text_Mining_in_Hotel_Reviews_Impact_of_Words_Restriction_in_Text_Classification "here"). The raw dataset is from [Kaggle](https://www.kaggle.com/jiashenliu/515k-hotel-reviews-data-in-europe/data "Kaggle"). Since some datasets were large, you can download the CVS files [here](https://drive.google.com/drive/folders/1mjoGF17DR8bcqLhHQ78IqXgY81rSonVM?usp=sharing "here"). The original dataset had 515,738 observations and 17 columns. It's important to mention that this dataset was collected from Booking.com, a website where you can book hotels. It's a great dataset because it contains real reviews that we can use to train our model so that our model will be able to read real-life reviews and predict if they are good or bad.

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
|Custom Functions   |matplotlib, NumPy, pandas, sklearn, xgboost   |



## Challenges
The data set was quite organized. However, it had a few challenges. For example, the review was divided between positive and negative reviews. Although this is useful for specific cases, most reviews will not be separated by positive and negative reviews. Thus, creating a model that is able to identify positive and negative reviews could be useless if we add reviews that are not separated. Other uses for the model, such as using it in social media, would not work. A solution was merging them together.

Another big challenge with the dataset is that the address is a long string, which means that it is difficult to collect information such as city and country due to the different formats which are written. This problem was solved by grabbing the city and country of the hotel using Geopy.

## Data Cleaning
As in any NLP project, the data cleaning is the most important step. We need to make sense of the information we have and create features for the models only with the information that matters. A few cleaning steps that I took were: 
- Remove any punctuation, stop words, and numbers
- Fix the spelling using TextBlob
- Lemmatization
- Remove words from the word clouds that didn't help with insights, such as complimentary adjectives (great, good, nice, excellent, etc.)

## Data Understanding
Data understanding was a very interesting step in this project. I was able to understand the dataset. The first thing I noticed was that the dataset was very large. This could be a problem when modeling. Thus, for the MVP, I decided to use only 20% of the dataset.

Since I had a classification problem, I first created a target feature with the score. In this step, I noticed that the lowest score was 2.5 and the highest was 10. Since there were users who wrote that there was anything good about the hotel, I assume that 2.5 was actually given by Booking.com, not the user. Then I converted the score into a classification problem. Since 2.5 was the lowest and 10 the highest, I decided to dived the target in two: positive and negative, where anything below 6 was negative and 6 and above was positive. 6.25 would be the middle point, so I decided to go with 6.

### Class Distribution

![Class Distribution - Before](https://github.com/Ismaeltrevi/hotel-reviews-analysis-using-nlp/blob/main/images/class_distribution-before.png?raw=true "Class Distribution - Before")

We can see that there is a big class imbalance. Since our dataset is large, we can fix this using the pandas sample function. Thus, I will only use 12% of the positive reviews, which will get closer to the number of negative reviews, so it will get closer to the number of negative reviews.

![Class Distribution - After](https://github.com/Ismaeltrevi/hotel-reviews-analysis-using-nlp/blob/main/images/class_distribution-after.png?raw=true "Class Distribution - After")

We can see above that the class distribution problem was solved.

### Exploratory Data Analysis
I wanted a few answers from the dataset and check if I could find any patterns. Thus, I had questions, which I will answer right next:

**What nationalities give the highest number of reviews?**

![](https://github.com/Ismaeltrevi/hotel-reviews-analysis-using-nlp/blob/main/images/nationality.png?raw=true)

Reviewers from the United Kingdom have the highest number of reviews left. To understand the main reason behind this, I checked where the hotels with the highest reviews are located. In the next image we can understand two things:
- The hotels are located in only 6 countries only, not the whole of Europe, as I previously thought.
- The reviewers are mostly from the UK (over 50%), which explain the higher number of people from the UK giving reviews.

![](https://github.com/Ismaeltrevi/hotel-reviews-analysis-using-nlp/blob/main/images/countries-hotels.png?raw=true)

**How does the hotel which I will focus on perform compared to other hotels in London?**

![](https://github.com/Ismaeltrevi/hotel-reviews-analysis-using-nlp/blob/main/images/score-density-hotel-vs-london.png?raw=true)

As we can see above, the Britannia International Hotel Canary Wharf underperform compared to other hotels in London. While the hotels in London have the score density skewed to the right where the positive scores are, Britannia International Hotel Canary Wharf is closer to a evenly distribution, where the amount of negative reviews are as high as the negative reviews. This tells us the the hotel has a lot to improve.

## Word Cloud

#### Why is the word cloud important?
While choosing the appropriate dataset, I noticed that the reviews score were not matching to the guest sentiment about the hotel. The mismatching becomes clear in the scores between 6 and 7. Please see an example below. Keep in mind that the punctuations were removed in the data cleaning and misspellings are common in the reviews.

|Hotel   |Negative Review   |Posittive Review   |Score   |
| ------------ | ------------ | ------------ | ------------ |
|Hotel Arena   | Even though the pictures show very clean rooms the actual room was quite dirty and outlived Also check-in is at 15 o clock but our room was not ready at that time   |No Positive   |6.5   |

As we can see above, the review doesn't match the overall score. If there is nothing positive about the hotel, how can they still get a 6.5 score? This is misleading to the hotel who are looking for areas to improve and to the users who are looking for a trustworthy score.

For the results section, I had two question in mind:

- What words appear the most in positive and negative reviews?
- Can we get any insights from it?

To get these answers, I had to clean the data through different steps. You can see more details of each step  in the [Data Cleaning notebook](https://github.com/Ismaeltrevi/hotel-reviews-analysis-using-nlp/blob/main/preprossessing/data-cleaning.ipynb "Data Cleaning notebook"). Now let's check the word clouds and see if we can get insights from it.

Below we can see the positive reviews word cloud. Note that you can relate every word to things that you could imagine coming from a hotel review. For example, we can see the words location, clean, comfortable, staff, service, price, and room. We can certainly assure that these are high points to the hotel and they can use these words to promote the hotel.

![](https://github.com/Ismaeltrevi/hotel-reviews-analysis-using-nlp/blob/main/images/positive_wordcloud.png?raw=true)

On the other hand, looking at the negative reviews word cloud, we can see many words that could be points for improvement to the hotel. Staff seems it's mentioned multiple times as well as the word rude and reception. Maybe the staff was rude to these guests? We can also see the words old, dated, dirty, and uncomfortable, which could be points of attention to the management.

![](https://github.com/Ismaeltrevi/hotel-reviews-analysis-using-nlp/blob/main/images/negative_wordcloud.png?raw=true)

## Sentiment Analysis

In this step we will create a sentiment analysis and compare the performance the actual score that users give. I want to visualize if the sentiment analysis can do a better job analysing when a review is negative or positive.

![](https://github.com/Ismaeltrevi/hotel-reviews-analysis-using-nlp/blob/main/images/sentiment_analysis.png?raw=true)

The graph above shows that the sentiment analysis does a good job identifying positive reviews with a positive polarity. However, it does not perform as well with the negative reviews. We can see that there is a big number of neutral reviews where sentiment analysis couldn't understand if the review was positive or negative.

## Modeling Process

### Vanilla Models

For the modeling process, I chose multiple models, testing them with different vectorizers in different stages of data cleaning. For the baseline models, I ran Logistic Regression, Random Forest, Naive Bayes, and Support Vector Machine.

I ran the models with the CountVectorizer and TF-IDF vectorizers to compared which one would have the best performance. I also tried these models with and without lemmatization. I did not include other features such as the name of the hotel or location because the main objective is to train a model using the reviews only.

The Vanilla Models performed fairly well since the beggining with an accuracy of 0.7981. The time I spent cleaning the text reviews paid off. The best performing model was a SVM model with the accuracy score of 0.8233 and F1 Score of 0.8205 using the RBF kernel. However, SVM models using RBF kernel don't allow feature importance retrieve. I tried running a SVM model using linear kernel, but the performance was poor compared to the RBF. Thus, I Random Forest with lemmatized words was the winner between the vanilla models. You can see all the models I ran in the [vanilla models notebook](https://github.com/Ismaeltrevi/hotel-reviews-analysis-using-nlp/blob/main/models/baseline-models.ipynb "vanilla models notebook") below.

|Model   |Accuracy   |Precision   |Recall   |F1 Score   |
| ------------ | ------------ | ------------ | ------------ | ------------ |
|Vanilla SVC SMOTE|0.823312	|0.865979	|0.779722	|0.820590|
|Vanilla Random Forest Lem|0.823127	|0.849331	|0.790264	|0.818733|
|Vanilla SVC TF-IDF|0.822202	|0.862205	|0.781864	|0.820071|
|Vanilla Log Reg Lem|0.817761	|0.831499	|0.801977	|0.816471|
|Vanila Log Reg TF-IDF|0.817021	|0.848194	|0.787933	|0.816954|
|Vanilla Random Forest CV|0.805365	|0.843615	|0.766512	|0.803217|
|Vanilla Random Forest TF-IDF|0.803885	|0.840971	|0.766512	|0.802017|
|Vanilla Random Forest SMOTE|0.800925	|0.831200	|0.760615	|0.794343|
|Vanilla Logisitic Regression CV|0.798150	|0.810458	|0.796858	|0.803600|
|Vanilla SVC CV	|0.783904|0.794234	|0.786862	|0.790531|
|Vanilla Naive Bayes CV|0.781129	|0.793541	|0.780793	|0.787115|




## Results



The best model was a GridSearch SVC. We can see the metrics below:

|Best Model  |Accuracy  |Precision   |Recall   |F1-Score   |
| ------------ | ------------ | ------------ | ------------ | ------------ |
|GridSearch SVC   |0.826827   |0.867271   |0.786148   |0.824719   |

Although I was looking for a high accuracy, the F1-Score is also very important, because I want to correctly classify reviews that need special attention. In this case, the negative reviews.

![](https://github.com/Ismaeltrevi/hotel-reviews-analysis-using-nlp/blob/main/images/confusion-matrix-ensemble-model.png?raw=true)

As we can see,  the GridSearch SVC was able to predict correctly 85.13% of the positive reviews and 72.79%. It definitely have room for improvement, but it's a good result for this stage of the project.

## Final Recommendations
- Use machine learning models to quickly identify negative and positive reviews without having to read all of them.
- Use word clouds to get quick insights from negative and positive reviews. The negative reviews can be used to improve the business and positive reviews can be used for advertisement, for example.

## Conclusion

- Machine learning can be used to correctly identify positive and negative reviews. However, identify with 100% confidence would be extremely difficult.
- Word clouds can be used to understand what words appear the most in negative and positive reviews. The management can quickly take a look and get insights out of it.
- The Britannia International Hotel Canary Wharf performes poorly in the reviews compared to other hotels in London. There is a lot of room for improvement.

## Next Steps:

- Test the model in the whole data set
- Use the features of the model with the highest coefficient to understand which words can predict better if a review is good or not and test
- Use the features with the highest coefficient to predict if posts in social media are positive or negative
- Create a recommendation system to the user based on reviews
- Create a dashboard for guests and hotels easily get information about hotels.

## Repository Content


```
├── .ipynb_checkpoints             # file created by GitHub
├── CSV                                     # contains CSV files used in the project
├── functions                             # contains functions applied in the notebook
├── images                                # contains all the images used in this README.md and in the final notebook
├── models                                # contains model iterations
│   ├── baseline-models.ipynb     # contains baseline models
│   ├── ensemble-models.ipynb   # contains ensemble models and final model
├── pickle                                  # contains pickle files
├── preprocessing                       # contains preprocessing notebooks
│   ├── data-cleaning.ipynb         # contains data cleaning process
│   ├── eda.ipynb                       # contains eda process
│   ├── geocoding.ipynb              # contains feature engineering using geopy
│   ├── sentiment-analysis.ipynb  # contains sentiment analysis notebook
├── README.md                          # public-facing preview
├── final_notebook.ipynb               # final version of data cleaning, EDA, feature engineering, and modeling process
└── presentation.pdf    # deck
```
## For More Information or Suggestions
If you have any questions or suggestions, please reach me out on:

- Email: alves.trevi@gmail.com
- LinkedIn: https://www.linkedin.com/in/ismael-araujo/
- Twitter: https://twitter.com/ish_araujo


## References
Campos, D., Rocha Silva, R., and Bernadino, J., 2019. Text Mining in Hotel Reviews: Impact of Words Restriction in Text Classification. University of Coimbra, Coimbra, Portugal.


