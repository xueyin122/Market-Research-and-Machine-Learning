# Market-Research-and-Machine-Learning

I used “activision blizzard” as my keyword and web-scraped 1,000 English tweets on 1/13/2023. After getting the raw tweets, I did some basic cleaning, including removing stop words, URLs, and at-mentions. I then ran a counter and found the most common 15 words in the scraped tweets. The top three words are nvidia (mentioned 716 times), google (mentioned 573 times), and microsoft (mentioned 444 times).

I proceeded with an LDA topic model with pre-defined 3 topics, and in the images topic 1, topic 2, topic 3 in this repo you can see the topic distributions on a distance map, along with the word frequencies of each topic. From the word distributions, we can roughly say that topic 1 is mainly about FTC and/or Sony opposing Microsoft's offer to Blizzard, topic 2 is about Nvidia having concerns about Microsoft's acquisition plan, and topic 3 is about Google reportedly joined raising concerns about the deal between Microsoft and Blizzard. 

The current model is being asked to identify 3 topics from the web-scraped tweets, but what will be the optimal number of topics that the model should be looking for? To analyze how the performance of topic models will change with the pre-defined number of topics, I plotted the relationship between the number of topics (x-axis) and coherence score (y-axis). The higher the coherence score is, the better the model performance is. The plot (Topic model performance.png) shows a U shape, which indicates that the model will perform better if I ask my model to find 1-2 topics, or 30+ topics from the scraped tweets; if I tell the model to divide the tweets into 20 topics, the model won’t yield good results.
