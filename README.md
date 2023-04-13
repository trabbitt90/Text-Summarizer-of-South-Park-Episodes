![image](https://user-images.githubusercontent.com/100429663/231789102-9ce1b9a2-cddc-42dc-9be3-96755ce9f169.png)

# Extractive and Abstractive Text Summarization of South Park Episodes
By: Tim Rabbitt

## Overview

Text summarization is an exciting yet challenging task in Natural Language Processing (NLP). Developing a model that can digest a large amount of textual data and produce an output that encapsulates the most important information can be a very powerful tool that saves a lot of time and resources. Today, text summarization models are being deployed on legal documents, CNN news articles, medical records, and even whole books. For my capstone project, I wanted to train a text summarization model on South Park dialogue to provide an overview of what happens in the episode. 

## Business Understanding
A new start-up would like to create a website that is the source of all things TV, both past and present. Providing information such as air date, character names, episode titles and much more the hope is to become a hub where fanatics can gather all of the information they need about their favorite TV shows. To save some time they would like to investigate using NLP to produce summaries from dialogue that explain the events that take place in a given episode, starting with the popular show South Park.

Creating a text summarization model trained on South Park dialogue can save the developers of this new website time to focus their efforts on other content of the website, while providing a powerful summarization tool for future TV show applications.

## Data
For this project, we utilized the South Park Scripts Dataset from Kaggle. This data set has two files. One contains episode names, air dates, seasons, episode numbers, and descriptions of the episode. While the other includes the episode name, character, and each character's lines. This is a large collection of text with over 300 episodes and 95000 lines. More detailed information on each dataset and its features are listed below:

#### Episodes 
* `Title`- Title of the episode
* `Air Date`- Air date of the episode
* `Code`- Code for the episode, it represents the season and episode
* `#`- Episode number (all time)
* `Description`- General description for episode
* `Season`- Episode's season
* `Episode`- Episode number in the season

#### Lines
* `Title`- Title of the episode
* `Character`- Character
* `Line`- What the character said

For more information on the datasets and the sources behind them please visit the [South Park Scripts Dataset](https://www.kaggle.com/datasets/mustafacicek/south-park-scripts-dataset?select=SouthPark_Lines.csv) on the Kaggle website.

## Eploratory Data Analysis

![image](https://user-images.githubusercontent.com/100429663/231789683-275d68d3-9cc5-4f26-81b9-9567755442fe.png)

If you are familiar with the show we can see that the usual suspects such as Cartman, Stan, and Kyle have the lion's share of the dialogue in the show. The Scene Description character has the second most lines, this makes sense as that character is used to describe the setting throughout each episode.

![image](https://user-images.githubusercontent.com/100429663/231789793-2ce422f5-a31e-47cd-8b42-b5090a6487ee.png)

Characters such as **Cartman**, **Stan**, and **Kyle** dominate the text. Considering the text mainly consists of a dialogue between those characters this makes a lot of sense. Frequent words such as **going**, **boy**, **walk** and **kid** could be attributed to the scene description "character" in the text.

![image](https://user-images.githubusercontent.com/100429663/231789850-cd14b5d8-032b-428b-8c89-223fa51b2ce9.png)

While there are certainly similarities between the text and the summary corpus and their respective word frequencies (characters are again very prevalent), we see more narrative vocabulary in the summary text. Words such as **south**, **park**, **meanwhile**, and **elementary** are examples of this narrative vocabulary.


## Text Summarization Methods and ROUGE Score
Text summarization can generally be categorized into two classes: **Extractive Summarization** and **Abstractive Summarization**.

* Extractive summarization compiles the most important words, sentences, or phrases from a corpus into a summary. This approach does not necessarily aim to understand the meaning of the text therefore the extracted summary is just a subset of the original text.

* Abstractive summarization models use more advanced NLP techniques to understand semantics and structure and can therefore have new phrases and sentences that are not contained in the original text. Abstractive summarization is most similar to how humans summarize, as humans often summarize by paraphrasing.

For this analysis, we used the **ROUGE Score** to measure how well our models produced a viable summary. In particular, we will be using ROUGE-1, ROUGE-2, and ROUGE-L.

* ROUGE-1 refers to the overlap of unigrams (each word) between the predicted and reference summaries.
* ROUGE-2 refers to the overlap of bigrams (two words) between the predicted and reference summaries.
* ROUGE-L takes into account sentence level structure similarity naturally and identifies longest co-occurring in sequence n-grams.

A ROUGE score close to zero indicates poor similarity between the summary and references. A ROUGE score close to one indicates a strong similarity between the summary and reference. If the summary is identical to one of the reference documents, then the score is 1.

## Extractive Summarization using Gensim TextRank

Gensim TextRank summarizer is an unsupervised algorithm that summarizes text by extracting the most important sentences from it.

### Example:

-----------------*Original Summary*-----------------

When Cartman's environmental ***essay*** wins a national contest, America's sweetheart, ***Kathie Lee Gifford***, comes to ***South Park*** to ***present the award***.

-----------------*TextRank Summary*-----------------

Children, as you all know, Miss ***Kathie Lee Gifford*** will be in ***South Park*** to ***present the award*** to some kid for an ***essay***.
Mr. Garrison's Bed. He hears ***Kathie Lee*** singing in his head: "If they could see me now, that little crowd of mine, and eating fancy chow..." He wakes up startled.

### Results:

![image](https://user-images.githubusercontent.com/100429663/231789913-4c9f4d3d-0556-4a85-834e-b36854c55928.png)

* Average ROUGE-1 score for TextRank is: 0.13

* Average ROUGE-2 score for TextRank  is: 0.01

* Average ROUGE-L score for TextRank  is: 0.11

Averaging all of the above ROUGE scores we get a score of 0.09. This isn't great as ROUGE scores range from 0-1 and scores close to 0 indicate poor similarity between texts. We do see some instances where we achieved an average ROUGE score of > 0.20, which for a purely extractive unsupervised approach is fairly adequate!

## Abstractive Text Summarization using Seq-2-Seq Modeling

A Sequence to Sequence (Seq2Seq) model involves any problem that deals with sequential information. In regards to natural language processing, this could be useful in translating a text to a different language, providing sentiment analysis, or in our case producing a summary of a text.

### Example:

-----------------*Original Summary*-----------------

town call child ***boy*** caught


-----------------*Model Summary*-----------------

***boy*** boy boy 

### Results:

![image](https://user-images.githubusercontent.com/100429663/231790000-ab2f2138-a079-403f-8288-779261fa2bbb.png)

* Average ROUGE-1 score for Model is: 0.07

* Average ROUGE-2 score for Model is: 0.00

* Average ROUGE-L score for Model is: 0.07

Our models scored very poorly, with an average ROUGE score of 0.05. Because the most common word in our summary vocabulary is **boy** and we have reduced the number of possible words that our model can use as output in its target sequence, it is likely that by repeatedly producing the word **boy** our model yields inflated ROUGE scores. This suggests our ROUGE scores are lower than they seem. The Gensim text rank algorithm has performed better.

## Abstractive Text Summarization using Facebook's BART
Bidirectional Auto-Regressive Transformers (BART) is a  seq-2-seq model that combines an autoregressive decoder with a bidirectional encoder. BART performs well for comprehension tasks and is especially successful when applied to text generation, such as summary and translation. 

### Example:

-----------------*Original Summary*-----------------

The boys go to **China*** to compete in a ***dodgeball*** championship. The town holds an appreciation week for the school nurse.


-----------------*BART Summary*-----------------

The South Park Cows Elementary School ***Dodgeball*** Team is going to the national finals. The Cows are on their way to the world championship in ***China***.

### Results:

![image](https://user-images.githubusercontent.com/100429663/231790046-dba0fbc8-35ee-4a59-aaa2-35e7a85674f7.png)

* The average Rouge-1 score for BART is: 0.16

* The average Rouge-2 score for BART  is: 0.02

* The average Rouge-L score for BART  is: 0.14

Averaging all of the above ROUGE scores we get a score of 0.11. This isn't great but it does confirm BART as our highest-performing model. We do see some instances where we achieved an average ROUGE score of close 0.30, which is great! When you consider the abstractive, grammatically correct, and concise summaries produced by BART this is easily the best model.
 

## Conclusion
In this project, we explored different avenues of text summarization. We began with Gensim's TextRank algorithm, trained our sequence-2-sequence model from scratch and then went on to use a state-of-the-art model like Facebook's BART. The goal of this project was to attempt to create a Sequence-to-Seqence model that can provide text summarizations of South Park episodes. Attempting to create such a model is always a good practice when it comes to understanding certain concepts of NLP and neural networks. Having said that it is hard to compete with the pre-trained models available on [Hugging Face Transformers](https://huggingface.co/docs/transformers/index). 

Recommendations:

* Attempting to build your Sequence-2-Sequence model from scratch for text summarization that is specific to South Park is very limiting. Incorporating additional dialogue from other texts and TV shows would help to produce a more useful model for our business case. 


* Transformers provides APIs and tools to easily download and train state-of-the-art pre-trained models. Using these pre-trained models can reduce your computing costs, and carbon footprint, and save you the time and resources required to train a model from scratch. 


* Fine-tune a pre-trained model on a large corpus of dialogue and save it for future use. This will allow you to produce summaries quickly as new episodes are written.


# Limitations and Future Analysis
* Future analysis should include additional dialogue from more popular TV shows and texts.


* Explore attention layers and different neural network architectures. This could help improve model performance.


* Compare and contrast different pre-trained models and their performance in summarizing dialogue.


* Compare different evaluation metrics for model performance, particularly BLEU.

# For More Information

Please check out my [Notebook](https://github.com/trabbitt90/Text-Summarizer-of-South-Park-Episodes/blob/main/Text-Summarization-of-South-Park-Episodes.ipynb) and [Presentation](https://github.com/trabbitt90/Text-Summarizer-of-South-Park-Episodes/blob/main/presentation.pdf)

# Navigating the Repository

README.md ----> Document for reviewers of the project

Text-Summarization-of-South-Park-Episodes.ipynb ---> Modeling and methods explanation in notebook


presentation.pdf ---> PDF of presentation slides

# References
[Abstractive Text Summarization](https://www.kaggle.com/code/akashsdas/abstractive-text-summarization/comments)

[Understanding Abstractive Text Summarization from Scratch](https://pub.towardsai.net/understanding-abstractive-text-summarization-from-scratch-baaf83d446b3)

[Text Summarization with NLP: TextRank vs Seq2Seq vs BART](https://towardsdatascience.com/text-summarization-with-nlp-textrank-vs-seq2seq-vs-bart-474943efeb09) 

[Hugging Face](https://huggingface.co/docs/transformers/index)

[Comprehensive Guide to Text Summarization using Deep Learning in Python](https://www.analyticsvidhya.com/blog/2019/06/comprehensive-guide-text-summarization-using-deep-learning-python/)


```python

```
