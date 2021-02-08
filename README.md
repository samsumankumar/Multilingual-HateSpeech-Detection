# Multilingual-HateSpeech-Detection

Deep Learning Models for Multilingual Hate Speech Detection

I. Introduction: 
Many online platforms like social media, online shopping or food delivery, online service-based platforms have multiple ways which support users to share their thoughts, views, feedback and hopes. These platforms like Facebook and Twitter try to provide a pleasant user experience. But social bullying and hate speech will spoil user experience und undermines their right to free speech. Targeting a group of individuals (based on their race, ethnicity, national origin, religion, sex, gender, sexual orientation, disability, or disease) is known as Hate speech. To tackle the problem of hate speech many social media platforms have heavily invested in the research, thereby making it one of the emerging research interests for many.
II. Baseline Approach:
To overcome the need of multilingual training to classify different languages hate speech using a single a model is the source for the paper “Deep Learning Models for Multilingual Hate Speech Detection”. In this paper the authors increase the dataset size by using nine different languages as their data for multilingual hate speech detection. The source of these 9 different languages data come from 16 different datasets. 

Datasets Used:
The paper uses tweets from 9 different languages, to classify whether a tweet is hateful or not. We have tried to gather all the datasets of all the languages used in the paper.  As the data contains hateful tweets; most of the hateful tweets have been taken down by Twitter. We could only find data in some languages by contacting the original owners of the dataset mentioned in the paper.
The Languages we have used are:
•	 Arabic    
•	 English 
•	 French
Data Preparation:
For the experimentation, the data is prepared based on how the authors prepared it. They performed a stratified split on the datasets of each language in the ratio of 70, 10, 20 for the train, validation, test sets. To maintain the classes in all the three sets train, validation and test set they used stratified split and then randomly split using 5 different seed values. They increased the training speed with a sample batch size of 16 at a time. They stratified the samples in sizes of 16, 32, 64, 128, 256 and whole data, and passed them to the models to observe model’s performance respective to low-resource and high-resource settings. They measured the performance of the model using the F1 score for each of different training sample sizes.

The baseline consists of two settings:

Monolingual Setting 
In which model is trained in samples of 16, 32, 64, 128, 256 and full data for only one language and tested on same language.

Multilingual Setting 
In this the model is trained on N-1 languages and tested on Nth language for Zero-shot case. Next the model is trained on N-1 languages and samples of 16, 32, 64, 128, 256 and full-data of the Nth language and then tested with the Nth language’s test set.
Limitations of Baseline:
•	Lack of data cleaning.
•	Model’s inability to understand the hidden context.
•	Models are dependent or focused on the cuss words in classification.

III. Need to Overcome the Limitations of Baseline:
Why data cleaning is needed?
Data cleaning and preprocessing is one of the important steps in most machine learning model constructions. The data which we considered as our input dataset to train the models is twitter data and it contains many unnecessary parts which should be removed to make more interpretable. Often these unnecessary and extra parts of the data confuse the model and leads to erroneous predictions. The extra parts of the twitter data are retweets/mentions, special characters in the data and etc.
Why the models need to understand the context?
The model needs to understand the hidden context because many examples of hate speech does not seem to be hateful. For the model to efficiently classify the different examples it is important for the model to understand the context. 
Why the model should not focus on the hate words in the sentence?
The model should also not completely depend on the hateful words in the sentence to classify an example, it will deviate the model from the actual context and the way these words are used. So, in this case also the model should less focus on hateful words and try to figure out the context of words to limit the number of prediction errors.
IV. Methods Used to Overcome Baseline Limitations and to Improve Results:
Data Cleaning: We implemented data cleaning with the help of Natural Language Tool Kit (NLTK) one of the most useful preprocessing tool available for natural language processing. With the help of NLTK we removed mentions or usernames most probably present in the twitter dataset, special characters, punctuation marks, extra spaces, URLs, numbers, emojis and stop words. Thus, making the data containing just normal words.
Language Specific BERT: BERT stands for Bidirectional Encoder Representations from Transformers is an open sourced pre-trained model from Google AI. BERT is popular due to its application of training bidirectional Transformers. Transformers an attention mechanism are famous for their applications in language modelling by training them bidirectionally, now they are more applicable to complex language modelling tasks. Transformer includes two separate mechanisms an encoder that reads the text input and a decoder that produces a prediction for the task. Since BERT’s goal is to generate a language model, only the encoder mechanism is necessary. BERT is trained in several regional languages and we used BERT models trained on specific testing dataset language (French, English, or Arabic). [3]
XLMR: XLM-R is one of the most recent multilingual models in the NLP world. Here the R means RoBERTa. So, it is a combination of XLM and RoBERTa models. It is based on Facebook’s RoBERTa model released in 2019. It is a large multilingual model, trained on 2.5TB of filtered CommonCrawl data. XLM-R is a multilingual model trained on 100 different languages. Unlike some XLM multilingual models, it does not require ‘lang’ tensors to understand which language is used and is able to determine the correct language from the input ids. This implementation is the same as RoBERTa. So, after using LASER and mBERT multilingual models in our project. We used XLM-R to see which model performs better in monolingual and multilingual setting. As we already discussed about Zero Shot, we use this model to test that as well and code switch (switch back and forth between their common languages). [1]
Adapters: Adapters are the small bottleneck layers that are inserted inside the pre-trained models. Adapters reduce the work of updating all the parameters of the model during training, thereby reducing the time and memory to load, store, and update the model. While using adapters, we keep the parameters(θ) of the model fixed/frozen only to update the parameters(Φ) of the adapter. This enables task-sharing by adding the adapter layers throughout the model. For monolingual we trained the adapters and tested with the left-out test set for the same language. And for the multilingual setting we combined the N-1 dataset and then tested on the Nth language to fulfill the Zero shot case and then used N languages to train the adapters and then tested with left out dataset of nth language. [2]
Weighted F1 Score: For binary or multiclass classification with class imbalances the accurate metric to measure the performance of the model is using F1 score. We have three types in F1 scores Macro, Weighted and Micro. Previously we used Macro F1 score as our performance metric. Macro calculates F1 scores separately for each class but does not multiply with each class weight of F1 scores. Thus, penalizing the scores with slight decrease in performance for the minority class. Weighted F1 score calculates F1 score individually for each class but while adding the scores it uses a weight that depends on the number of true labels of each class by which it favors the majority class. We wanted to observe how different models favor to our imbalanced data, like we want to understand how much the low performing models with low Macro F1 scores are penalized. So, we also consider weighted F1 as a metric, applied it on different models and compare its changes. 
V. Observations:
Implementing Language Specific BERT- We previously used mBERT for the monolingual training and testing, mBERT is many different languages. To be more precise with the monolingual modelling we trained each language data on the BERT pretrained model, which is trained, particularly in that language. This makes the training model specific to the single language that is to be tested instead using a model which is trained on multiple languages. We then trained the BERT with different samples, then recorded the macro F1 score to observe exactly how model is performing on each class and weighted F1 score to observe how much the model is favoring the majority class.
 
                                Table1: Describes the newly added language specific BERT model metrics

As in the table, BERT-Arabic, BERT-English and BERT-French represent the BERT models trained in that language. For the Arabic language, in the low resource setting there is not much effect of BERT-Arabic till the sample size of 64. For samples 16 and 32 mBERT and BERT-Arabic report exact sample macro F1 scores. After sample 64 we can observe that there is a constant increase in the performance by using BERT-Arabic model. If we observe the weighted F1 scores of BERT-Arabic model there is a clear 20-30% difference in the F1 scores of macro and weighted case, representing the much penalization due to imbalance data in the low resource setting from the macro case. But for the high resource setting the difference becomes as low as 10% again representing the model’s performance in the high resource setting because of more examples for each class and less penalization due to imbalance data.

Being the most used and popular language English’s performance is very close in both the mBERT and BERT-English models. This could be due vast availability of English data and major training examples in mBERT being English. Still we can observe a slight increase in the macro F1 score using BERT-English model. Comparing the macro and weighted F1 scores, there is a similar pattern of BERT-English metrics to the BERT-Arabic model i.e., huge gap between macro and weighted score in low resource setting and less gap in metrics for the high resource case.

For the French language, BERT-French is not able to improve the performance. It is stagnant with same value recorded in low resource and high resource setting. Same with weighted F1 setting the model produces same output for all the cases.

Implementing and comparing the performance metrics of new models in monolingual setting-
We observed that most of the outputs in the low resource setting, i.e., samples of 16, 32, 64, 128, 256 result in very similar scores and we can observe very little change in the F1 scores. So, we mainly focus on the scores resulting when trained on full dataset for the new models. We implemented XLMR model and mBERT model with the adapter layers as an architectural addition to the mBERT model. We recorded the macro and weighted F1 scores, then compared them with previously recorded model metrics.
 
Table2: Compares performance metrics of different models in the monolingual setting
In the monolingual setting we have different models like LASER+LR, mBERT, BERT, XLMR and mBERT+Adapter. For all the three languages we compare all the languages’ metrics using the macro F1 score and their weighted F1 score in detail. For Arabic language, BERT-Arabic performs better than all the other languages in terms of macro F1 score, but mBERT and XLMR are not too far away. Like BERT-Arabic has the best weighted F1 score. By using adapters there was a huge computational time reduction but the F1 scores almost drop by half. 

For the English language, we can observe from the table that XLMR performs better than all the other languages in terms of macro and weighted F1 scores. The macro and weighted F1 scores are also near, from which we can say that there is not much penalization in terms of class imbalances and these models are able to handle the imbalances effectively. As English has data near to 105K sentences it used to a lot of time around 2 hours to run the model but with the use of adapters we observed 5 times less time was needed for computation.

For the French language, still the LASER+LR model performs better than all the other models. The new models are still stagnant with their performance. But still XLMR performed better than BERT-French model.

Implementing and comparing the performance metrics of new models in multilingual setting-
We applied new models like XLMR and made architectural changes to mBERT by adding adapter layers as the bottom layer. We considered two cases zero shot and full data. In zero short we trained the model on two languages other than the language we are testing. In the case of full data, we trained on all the three languages and tested on different language each time. We recorded the macro and weighted F1 scores for new models and compared them with the previous model’s metrics.
  
Table3: Compares performance metrics of different models in the multilingual setting
In the multilingual setting we observe performance metrics of the models LASER+LR, mBERT, XLMR and mBERT+Adapter. We compare the macro and weighted F1 scores for both zero shot setting and full data. For the English language in Zero shot setting, we can observe that XLMR outperforms mBERT model still compared to the mBERT scores recorded in the paper the XLMR score is less. But for the full data setting XLMR beats all the other models. 

For the English language we can observe that English language performs better in both the zero-shot case and full data case with the macro F1 score. Even the weighted F1 scores are high for XLMR model compared to mBERT. For the French language, XLMR performs better than all the other models in both Zero shot setting and full data setting. Similarly, the weighted F1 score is also higher for XLMR model.

Overall, from the observations XLMR performs best for all the languages in multilingual case for both zero-shot and full data setting. Even for the monolingual setting XLMR performs best in some cases and for some case the performance metrics of XLMR are close to best performing model. mBERT has also gave challenging performance metrics, so in order to not just consider F1 scores to conclude which model is the best, we ran a thorough error analysis of each model and to check their interpretability in terms of classifying sentences with hidden context. 

VI. Error Analysis
We considered three tricky sentences to observe how each model classifies them. The first sentence is an advertisement for a book download. Even though it has some hateful words like destruction it is still considered as the wrongly annotated category as it is not offensive, by this even if the model classifies it as non-hateful due to the annotation error it will not be correct. The second sentence is wrongly classified by the model as it misses the hidden context in the sentence. Rather the model checks for any cuss words and this example have it, so it is classified as hate. Similarly, whenever there are no abusive or negative words the model classifies the sentence as non-hate speech while it is hateful.

We consider the models LASER+LR, mBERT and XLMR trained on multiple languages. We test the following example on each of the model with data cleaning as one case and model without data cleaning as another case. We try to observe the effects of data cleaning on the model in prediction and try to find a best model which that can solve the previous classification limitations by understanding the context. 

Tweet	Label
@user!!!!Click below for a FREE download of a colorfully illustrated 132page e-book on the “Zionist-engineered INTENTIONAL destruction of Western civilization” #new #trending	1
In my opinion using cuss words like “bitch” is demeaning to women.	0
I hope they all starve!!!!!!	1
                  Table4: Example test set for error analysis
We observed that LASER+LR is producing an output as [0, 1, 0] even with and without data cleaning. LASER fails to understand the hidden context in both the cases and just depends on the cuss words to classify the sentences. Now if we see in mBERT, with data cleaning it produces the output of [0, 0, 0]. It performs better in understanding the context in the case of second example but still confuses with the plain third sentence and classifies it wrongly. XLMR which performed better with higher F1 score when used without any data cleaning and tested with the above data it still classifies them as [0, 0, 0]. But with data cleaning and removing the stop words the XLMR produces the correct output i.e., [0, 0, 1]. Thus, completely able to understand the context of the above examples.

Overall, XLMR with competitive performance metrics and the ability to understand the context is best for monolingual and multilingual classification.

VII. Conclusion
To conclude, we took the problem of classifying multilingual hate speech recognition. We constructed different machine learning models to classify the hate speech in three different languages. We considered various parameters in terms of classification like training in sample, saving the model and re-training it. Also used newer better models which can possibly increase the performance. We analyzed performance metrics of different models and overcame the limitation of the state-of-the-art baseline by implementing data cleaning and using models which can understand the context better. We implemented adapters one of the recent concepts to decrease the computational time and observed 5 times decrease in computational time of the previous models used but with an expense of certain performance decrease. We observed that XLMR with multilingual training and data cleaning outperforms most of the existing models and best for multilingual hate speech classification task.


