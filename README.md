# Hotel Review Analysis Using NLP and Machine Learning

![Photo by Martin Péchy from Pexels](https://github.com/Ismaeltrevi/capstone-project/blob/main/visualizations/hotel-image.jpg?raw=true "Photo by Martin Péchy from Pexels")

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
The dataset for this project was originally used in the study Text Mining in Hotel Reviews: Impact of Words Restriction in Text Classification by Diego Campos, Rodrigo Rocha Silva, and Jorge Bernadino and a team at University of Coimbra. You can find the paper [here](https://www.researchgate.net/publication/336224346_Text_Mining_in_Hotel_Reviews_Impact_of_Words_Restriction_in_Text_Classification "here"). The raw dataset is from [Kaggle](https://www.kaggle.com/jiashenliu/515k-hotel-reviews-data-in-europe/data "Kaggle"). 

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



## Results
### Exploratory Data Analysis


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



 

