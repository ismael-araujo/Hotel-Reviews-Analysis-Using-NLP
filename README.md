# Hotel Reviews Analysis Using NLP and Machine Learning

#### Can hotels quickly get insights out of thousands of reviews?

![](https://github.com/Ismaeltrevi/hotel-reviews-analysis-using-nlp/blob/main/images/hotel-image.jpg?raw=true)

## Overview
In this project, I will create a model that can predict if a hotel review is negative or positive so that hotels can use it to classify their reviews correctly. I will analyze a specific hotel in London and compare it to other hotels in the same city. We will walk through multiple Natural Language Processing to understand how we can use machines to read reviews and get insights from them. Baseline models include Logistic Regression, Random Forest, Naive Bayes, and Support Vector Machine (SVM). Ensemble models include Voting, Bagging, GridSearch, AdaBoost, and Gradient Boosting. The final model was a GridSearch SVM with an accuracy of 0.8268 and F1-Score 0.8247.

This project walks through exploratory data analysis, data cleaning, sentiment analysis, data preprocessing, vanilla model, an ensemble model iterations. You can find a summary of the project in the final notebook.

## Objectives
1. Create a model that can classify reviews as positive and negative for every website using the same algorithm
2. Discover the most important features for positive and bad reviews
3. Find insights and areas for improvement for hotels from the reviews
4. Create a web app where hotels can easily upload their reviews and get instantaneous insights (IN DEVELOPMENT)
5. Create a recommendation system based on hotel reviews (IN DEVELOPMENT)

## Business Problem
One of the biggest problems that many companies have been trying to overcome is how to take advantage of all the data collected from guests. The amount of data has challenged the travel industry. One type of data is reviews left by guests on websites such as Booking.com, TripAdvisor, and Yelp.

Hotels have been trying to find ways to analyze the reviews and get insights out of them. However, some hotels can receive thousands of guests every week and hundreds of reviews. It becomes nearly impossible and expensive for hotels to keep track of the reviews. Thus, multiple hotels might ignore these valuable data due to the cost and energy that need to be allocated. The other problem is that hotels such as Booking.com don't allow users to choose their score. The score is determined by questions asked to the user, and then the review is calculated. This is problematic because guests could have had a bad experience, and the hotel would still get a 7 or 8 score, which gives a false illusion that the guest didn't have any problems.

This project will build a model that can correctly predict if a hotel review is negative or positive so that hotels can input their reviews and get a non-biased score from any website, which will turn the review analysis uniform. Although the model will be trained with reviews from over 1,400 hotels, it can be used for any hotel to predict if a review is positive or negative correctly.

### Setting the hypothetical scenario

Our actual client is a hotel in London called Britannia International Hotel Canary Wharf. They have thousands of reviews and a 6.7 overall score on Booking.com. They think this is a low score compared to other London hotels, and they want to understand what is causing this low score. Due to COVID-19, they don't have the resources to read all the reviews and make sense of them. Thus, they want to find a way to get quick insights without having to read every review. They have a few business questions:

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
As in any NLP project, data cleaning is the most important step. We need to make sense of the information we have and create features for the models only with the information that matters. A few cleaning steps that I took were: 
- Remove any punctuation, stop words, and numbers
- Fix the spelling using TextBlob
- Lemmatization
- Remove words from the word clouds that didn't help with insights, such as complimentary adjectives (great, good, nice, excellent, etc.)

## Data Understanding
Data understanding was a very interesting step in this project. I was able to understand the dataset. The first thing I noticed was that the dataset was very large. This could be a problem when modeling. Thus, for the MVP, I decided to use only 20% of the dataset.

Since I had a classification problem, I first created a target feature with the score. In this step, I noticed that the lowest score was 2.5 and the highest was 10. Since there were users who wrote that there was anything good about the hotel, I assume that 2.5 was actually given by Booking.com, not the user. Then I converted the score into a classification problem. Since 2.5 was the lowest and 10 the highest, I decided to dived the target in two: positive and negative, where anything below 6 was negative and 6 and above was positive. 6.25 would be the middle point, so I decided to go with 6.

### Class Distribution

<img align='center' alt='Class Distribution - Before' src="https://github.com/Ismaeltrevi/hotel-reviews-analysis-using-nlp/blob/main/images/class_distribution-before.png?raw=true" width="65%" height="65%">

We can see that there is a significant class imbalance. Since our dataset is large, we can fix this using the pandas sample function. Thus, I will only use 12% of the positive reviews, which will get closer to the number of negative reviews, so it will get closer to the number of negative reviews.

<img align='center' alt='Class Distribution - After' src="https://github.com/Ismaeltrevi/hotel-reviews-analysis-using-nlp/blob/main/images/class_distribution-after.png?raw=true" width="65%" height="65%">

We can see above that the class distribution problem was solved.

### Exploratory Data Analysis
I wanted a few answers from the dataset and checked if I could find any patterns. Thus, I had questions which I will answer right next:

**What nationalities give the highest number of reviews?**

<img align='center' alt='Hotel Nationalities' src="https://github.com/Ismaeltrevi/hotel-reviews-analysis-using-nlp/blob/main/images/nationality.png?raw=true" width="75%" height="75%">

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
|Hotel Arena   | Even though the pictures show very clean rooms the actual room was quite dirty and outlived also check-in is at 15 o clock but our room was not ready at that time   |No Positive   |6.5   |

As we can see above, the review doesn't match the overall score. If there is nothing positive about the hotel, how can they still get a 6.5 score? This is misleading to the hotel looking for areas to improve and to the users looking for a trustworthy score.

For the results section, I had two question in mind:

- What words appear the most in positive and negative reviews?
- Can we get any insights from it?

To get these answers, I had to clean the data through different steps. You can see more details of each step  in the [Data Cleaning notebook](https://github.com/Ismaeltrevi/hotel-reviews-analysis-using-nlp/blob/main/preprossessing/data-cleaning.ipynb "Data Cleaning notebook"). Now let's check the word clouds and see if we can get insights from it.

Below we can see the positive reviews word cloud. Note that you can relate every word to things that you could imagine coming from a hotel review. For example, we can see the words location, clean, comfortable, staff, service, price, and room. We can certainly assure that these are high points to the hotel and they can use these words to promote the hotel.

![](https://github.com/Ismaeltrevi/hotel-reviews-analysis-using-nlp/blob/main/images/positive_wordcloud.png?raw=true)

On the other hand, looking at the negative reviews word cloud, we can see many words that could be points for improvement to the hotel. Staff seems it's mentioned multiple times as well as the word rude and reception. Maybe the staff was rude to these guests? We can also see the words old, dated, dirty, and uncomfortable, which could be points of attention to the management.

![](https://github.com/Ismaeltrevi/hotel-reviews-analysis-using-nlp/blob/main/images/negative_wordcloud.png?raw=true)

## Sentiment Analysis

In this step, we will create a sentiment analysis and compare the performance the actual score that users give. I want to visualize if the sentiment analysis can do a better job analyzing when a review is negative or positive.

![](https://github.com/Ismaeltrevi/hotel-reviews-analysis-using-nlp/blob/main/images/sentiment_analysis.png?raw=true)

The graph above shows that sentiment analysis does a good job identifying positive reviews with a positive polarity. However, it does not perform as well with the negative reviews. We can see that there is a big number of neutral reviews where sentiment analysis couldn't understand if the review was positive or negative.

## Modeling Process

### Vanilla Models

For the modeling process, I chose multiple models, testing them with different vectorizers in different stages of data cleaning. For the baseline models, I ran Logistic Regression, Random Forest, Naive Bayes, and Support Vector Machine.

I ran the models with the CountVectorizer and TF-IDF vectorizers to compare which one would have the best performance. I also tried these models with and without lemmatization. I did not include other features such as the name of the hotel or location because the main objective is to train a model using the reviews only.

The Vanilla Models performed fairly well since the beginning with an accuracy of 0.7981. The time I spent cleaning the text reviews paid off. The best performing model was an SVM model with an accuracy score of 0.8233 and F1 Score of 0.8205 using the RBF kernel. However, SVM models using RBF kernel don't allow feature importance retrieval. I tried running an SVM model using a linear kernel, but the performance was poor compared to the RBF. Thus, I Random Forest with lemmatized words was the winner between the vanilla models. You can see all the models I ran in the [vanilla models notebook](https://github.com/Ismaeltrevi/hotel-reviews-analysis-using-nlp/blob/main/models/score-density-hotel-vs-london.png.ipynb "vanilla models notebook") below.

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

### Ensemble Models

I tried multple ensemble models. My main focus was to find the most important features, so ensemble models were useful to find the best hyperparameters to tune the best vanilla models and understand how the models were behaving. Although GridSearch SVC was the model with the best accuracy, Logistic Regression is able to inform the best features for both negative and positive classification.

|Models	|	Accuracy	|	Precision	|	Recall	|	F1 Score	|
| ------------ | ------------ | ------------ | ------------ | ------------ |
|	GridSearch SVC	|	0.826827	|	0.867271	|	0.786148	|	0.824719	|
|	Voting	|	0.823497	|	0.86032	|	0.787219	|	0.822148	|
|	GridSearch Logistic Regression	|	0.818316	|	0.851566	|	0.786505	|	0.817743	|
|	GridSearch Random Forest - First Model	|	0.808511	|	0.846275	|	0.770439	|	0.806578	|
|	GridSearch Random Forest - Second Model	|	0.809436	|	0.847937	|	0.770439	|	0.807333	|
|	Gradient Boosting	|	0.780204	|	0.834786	|	0.717958	|	0.771977	|
|	Bagging	|	0.769658	|	0.823897	|	0.706533	|	0.760715	|
|	GridSearch ADABoost	|	0.731175	|	0.747067	|	0.727597	|	0.737204	|
|	ADABoost and Random Forest	|	0.731175	|	0.747067	|	0.727597	|	0.737204	|

## Final Model

Logistic Regression using GridSearch was the final model for its high accuracy and showed the feature importance for each class. The accuracy was 0.8183, which means that the model can correctly classify the target variable 81.83% of the time. Looking at cross-validation, we can see that the model performed similarly in the train set. I used 5 folds, and the range difference between the highest accuracy and lowest accuracy was a very small difference.

|   |Accuracy   |Precision   |Recall   |F1 Score   |
| ------------ | ------------ | ------------ | ------------ | ------------ |
|Logistic Regression|0.818316 |0.851565|0.7865048|0.8177431|

Looking at the confusion matrix, we can see that the model needs improvements classifying Positive reviews, since it has a higher number of False Negatives compared to False Positives.

<img align='center' src="https://github.com/Ismaeltrevi/hotel-reviews-analysis-using-nlp/blob/main/images/confusion-matrix-final-model.png?raw=true" width="50%" height="50%">

Now, let's take a look at the 50 most important features for each class using ELI5.

Note: Weight is how important a feature is for each class. For the positive class. The 

|Weight - Positive|	 Positive Feature|Weight - Negative |	Negative Feature|
| ------------ | ------------ | ------------ | ------------ |
|	6.092	|	Excellent	|	5.308	|	Dirty	|
|	5.27	|	Great	|	4.277	|	Rude	|
|	5.082	|	Amazing	|	3.543	|	Terrible	|
|	5.01	|	Comfortable	|	3.451	|	Poor	|
|	4.544	|	Lovely	|	3.248	|	Worst	|
|	4.222	|	Bit	|	3.23	|	Tired	|
|	4.15	|	Perfect	|	3.208	|	Old	|
|	3.823	|	Spacious	|	3.105	|	Bad	|
|	3.818	|	Loved	|	2.865	|	Basic	|
|	3.596	|	Friendly	|	2.843	|	Uncomfortable	|
|	3.407	|	Wonderful	|	2.83	|	Star	|
|	3.221	|	Fantastic	|	2.801	|	Horrible	|
|	2.963	|	Quiet	|	2.776	|	Money	|
|	2.935	|	Brilliant	|	2.699	|	Paid	|
|	2.896	|	Superb	|	2.631	|	Overpriced	|
|	2.685	|	Nice	|	2.556	|	Awful	|
|	2.674	|	Negative	|	2.428	|	Unfriendly	|
|	2.61	|	Come	|	2.252	|	Charged	|
|	2.56	|	Beautiful	|	2.234	|	Broken	|
|	2.482	|	Modern	|	2.211	|	Management	|
|	2.424	|	Helpful	|	2.133	|	Dated	|
|	2.266	|	Little	|	2.118	|	Tiny	|
|	2.235	|	Large	|	2.077	|	Work	|
|	2.234	|	Fabulous	|	2.066	|	Run	|
|	1.944	|	Cooked	|	2.064	|	Attitude	|

### Positive Features

The top features for predicting positive reviews weren't a surprise. Most of the features are adjectives that we can relate to positive reviews, such as excellent, great, and amazing. Although these words might not give much insight, some others are very related to hotels and can give us insights. Hotels should make sure that there are delivering this aspect to their guests. Let's check a few that I believe can carry insights:

- Comfortable: The most important aspect of a hotel is comfortable, so the guest can rest
- Bit: It doesn't carry much meaning
- Spacious: It's good when a hotel has a spacious room
- Friendly: It could be talking about how friendly the staff is, a very important aspect
- Quiet: It seems like quiet places are something that guests are looking for
- Negative: Although it is a negative word, I assume that the guests are saying that there isn't anything negative about the hotel
- Modern: Modern hotels seem to be noticed in the reviews
- Helpful: It's probably walking about the staff

### Negative Features

When looking at the top features for the negative class predictor, we can find words that everyone could expect from negative reviews such as awful, horrible, and bad. However, with the negative class feature predictor, we can have more insights and areas that every hotel should consider as critical. It's interesting to see that dirty has a higher weight than overpriced and dated.
Let's take a look at a few features:

- Dirty: It's the most crucial feature when predicting negative reviews. This means that dirty is a giant red flag.
- Rude: Probably talking about the staff, which means that the hotel needs improvement in training
- Old: It's probably related to the hotel being outdated
- Overpriced: This is obvious to me. If people pay more money than they think it's worth, they will complain.


## Final Recommendations

All models were trained with reviews of over 1,400 hotels. Thus, the model can be used for any hotel because it used over 515k reviews. Britannia International Hotel Canary Wharf can use our model to classify reviews at any point correctly. However, the most important takeaway here is the feature importance. Since guests tend to expect the same things in every hotel, we learned that words such as staff, location, comfortable, and dirty will make have a higher value in their reviews. The words also match the word cloud created for Britannia International Hotel Canary Wharf, which proves that the hotel, similarly to others, needs to focus on those words.

I recommend the hotel start using word clouds to get quick insights from negative and positive reviews. The negative reviews can be used to improve the business and should be used as soon as possible if the hotel wants to increase its overall score. The hotel should also use positive reviews that can be used for advertisement, for example.

## Conclusion

- Machine learning can be used to identify positive and negative reviews correctly. However, identity with 100% confidence is difficult. My final model can be used for any hotel to find feature importance
- Word clouds can be used to understand what words appear the most in negative and positive reviews. The management can quickly take a look and get insights out of it.
- The Britannia International Hotel Canary Wharf performs poorly in the reviews compared to other hotels in London. There is a lot of room for improvement.

## Next Steps:

- Test the model in the whole data set as well as social media posts.
- Create a recommendation system to the user based on reviews
- Create a dashboard for guests and hotels to easily get information about hotels.

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
