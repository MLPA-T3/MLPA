# Introduction to machine learning

> 🎥 Click the image above for a short video working through this lesson.

Welcome to this course on classical machine learning for beginners! Whether you're completely new to this topic, or an experienced ML practitioner looking to brush up on an area, we're happy to have you join us!

[![Introduction to ML](https://img.youtube.com/vi/h0e2HAPTGF4/0.jpg)](https://youtu.be/h0e2HAPTGF4 "Introduction to ML")

> 🎥 Click the image above for a video: MIT's John Guttag introduces machine learning

---
## Getting started with machine learning

Before starting with this curriculum, you need to have your computer set up and ready to run notebooks locally.

- **Configure your machine with these videos**. Use the following links to learn [how to install Python](https://youtu.be/CXZYvNRIAKM) in your system and [setup a text editor](https://youtu.be/EU8eayHWoZg) for development.
- **Learn Python**. It's also recommended to have a basic understanding of [Python](https://docs.microsoft.com/learn/paths/python-language/?WT.mc_id=academic-77952-leestott), a programming language useful for data scientists that we use in this course.
- **Learn Node.js and JavaScript**. We also use JavaScript a few times in this course when building web apps, so you will need to have [node](https://nodejs.org) and [npm](https://www.npmjs.com/) installed, as well as [Visual Studio Code](https://code.visualstudio.com/) available for both Python and JavaScript development.
- **Create a GitHub account**. Since you found us here on [GitHub](https://github.com), you might already have an account, but if not, create one and then fork this curriculum to use on your own. (Feel free to give us a star, too 😊)
- **Explore Scikit-learn**. Familiarize yourself with [Scikit-learn](https://scikit-learn.org/stable/user_guide.html), a set of ML libraries that we reference in these lessons.

---
## What is Machine Learning?

Machine learning is a subfield of artificial intelligence that gives computers the ability to learn without explicitly being programmed.

It is a branch of artificial intelligence (AI) and computer science that focuses on using data and algorithms to enable AI to imitate the way that humans learn, gradually improving its accuracy. It was defined in the 1950s by AI pioneer Arthur Samuel as “the field of study that gives computers the ability to learn without explicitly being programmed.”

Machine learning is behind chatbots and predictive text, language translation apps, the shows Netflix suggests, and how our social media feeds are presented. It powers autonomous vehicles and machines that can diagnose medical conditions based on images.

Machine learning starts with data—the data is gathered and prepared to be used as training data, or the information the machine learning model will be trained on. The more data, the better the program. From there, programmers choose a machine learning model to use, supply the data, and let the computer model train itself to find patterns or make predictions.

---
## Types of Machine Learning Algorithms

There are four subcategories of machine learning:

### Supervised Machine Learning

In supervised machine learning, the data is labeled, meaning each example comes with a correct answer, and the machine is trained on this data to generate a model that gives accurate predictions for similar kinds of data.

For example, an algorithm would be trained with pictures of dogs and other things, all labeled by humans, and the machine would learn ways to identify pictures of dogs on its own.

The operator provides the machine learning algorithm with a known dataset that includes desired inputs and outputs, and the algorithm must find a method to determine how to arrive at those inputs and outputs. While the operator knows the correct answers to the problem, the algorithm identifies patterns in data, learns from observations, and makes predictions. The algorithm makes predictions and is corrected by the operator — and this process continues until the algorithm achieves a high level of accuracy/performance.

#### Types of Supervised Learning Problems:

- **Regression**: In regression problems, the goal is to predict a continuous numeric value. This could be predicting house prices based on features like location, size, and number of rooms, forecasting stock prices, or estimating the temperature based on weather variables.
- **Classification**: In classification problems, the goal is to predict a category or class label for a given input. For example, classifying emails as spam or non-spam, identifying handwritten digits, or predicting whether a customer will churn or not.

#### Applications of Supervised Learning:

- **Healthcare**: Diagnosing diseases based on patient symptoms.
- **Finance**: Predicting stock prices or credit risk assessment.
- **Marketing**: Targeted advertising based on customer behavior.
- **Natural Language Processing (NLP)**: Sentiment analysis, text classification.
- **Autonomous Vehicles**: Recognizing objects and making driving decisions.

### Unsupervised Machine Learning

Imagine you’re given a basket filled with assorted fruits but without any labels. Your task is to group similar fruits together based on their features like color, shape, or texture. This process of discovering patterns or structures in data without explicit supervision is akin to unsupervised learning in machine learning.

In unsupervised machine learning, a program looks for patterns in unlabeled data. In an unsupervised learning process, the machine learning algorithm is left to interpret large datasets and address that data accordingly. The algorithm tries to organize that data in some way to describe its structure. This might mean grouping the data into clusters or arranging it in a way that looks more organized.

#### Types of Unsupervised Learning Problems:

- **Clustering**: Clustering involves grouping similar data points together into clusters or segments based on their features. Examples include grouping customers based on purchase behavior, segmenting news articles into topics, or identifying similar genes in biological data.
- **Dimensionality Reduction**: Dimensionality reduction techniques aim to reduce the number of features in a dataset while preserving its essential information. This helps in visualizing high-dimensional data, compressing data for efficient storage, or improving the performance of machine learning models.
- **Anomaly Detection**: Anomaly detection focuses on identifying rare events or outliers in the data that deviate from the norm. This is particularly useful in fraud detection, network security, or monitoring industrial equipment for faults.

#### Applications of Unsupervised Learning:

- **Market Segmentation**: Grouping customers based on purchasing behavior.
- **Image and Document Clustering**: Organizing similar images or documents into groups.
- **Anomaly Detection**: Identifying fraudulent transactions or unusual patterns in data.
- **Recommendation Systems**: Suggesting similar products or content based on user preferences.
- **Data Visualization**: Visualizing high-dimensional data in lower dimensions for exploration.

### Reinforcement Machine Learning

Reinforcement machine learning trains machines through trial and error to take the best action by establishing a reward system. Reinforcement learning can train models to play games or train autonomous vehicles to drive by telling the machine when it made the right decisions, which helps it learn over time what actions it should take.

By defining the rules, the machine learning algorithm then tries to explore different options and possibilities, monitoring and evaluating each result to determine which one is optimal.

#### Applications of Reinforcement Learning:

- **Game Playing**: Training agents to play board games (e.g., chess, Go) or video games (e.g., Atari games) at superhuman levels.
- **Robotics**: Teaching robots to perform complex tasks such as manipulation, navigation, or assembly in dynamic environments.
- **Autonomous Vehicles**: Developing self-driving cars that learn to navigate roads safely and efficiently.
- **Recommendation Systems**: Personalizing content recommendations (e.g., movies, music, products) based on user interactions and feedback.
- **Resource Management**: Optimizing resource allocation in dynamic systems, such as energy management or supply chain optimization.

### Semi-Supervised Machine Learning

Semi-supervised learning is similar to supervised learning, but instead uses both labeled and unlabeled data. In semi-supervised learning, the algorithm learns from a dataset that contains a small amount of labeled data and a much larger amount of unlabeled data. This approach is particularly useful in scenarios where obtaining labeled data is expensive or time-consuming, but unlabeled data is abundant.

---
## The hype curve

![ml hype curve](images/hype.png)

> Google Trends shows the recent 'hype curve' of the term 'machine learning'

---
## A mysterious universe

We live in a universe full of fascinating mysteries. Great scientists such as Stephen Hawking, Albert Einstein, and many more have devoted their lives to searching for meaningful information that uncovers the mysteries of the world around us. This is the human condition of learning: a human child learns new things and uncovers the structure of their world year by year as they grow to adulthood.

---
## The child's brain

A child's brain and senses perceive the facts of their surroundings and gradually learn the hidden patterns of life which help the child to craft logical rules to identify learned patterns. The learning process of the human brain makes humans the most sophisticated living creature of this world. Learning continuously by discovering hidden patterns and then innovating on those patterns enables us to make ourselves better and better throughout our lifetime. This learning capacity and evolving capability is related to a concept called [brain plasticity](https://www.simplypsychology.org/brain-plasticity.html). Superficially, we can draw some motivational similarities between the learning process of the human brain and the concepts of machine learning.

---
## The human brain

The [human brain](https://www.livescience.com/29365-human-brain.html) perceives things from the real world, processes the perceived information, makes rational decisions, and performs certain actions based on circumstances. This is what we called behaving intelligently. When we program a facsimile of the intelligent behavioral process to a machine, it is called artificial intelligence (AI).

---
## Some terminology

Although the terms can be confused, machine learning (ML) is an important subset of artificial intelligence. **ML is concerned with using specialized algorithms to uncover meaningful information and find hidden patterns from perceived data to corroborate the rational decision-making process**.

---
## AI, ML, Deep Learning

![AI, ML, deep learning, data science](images/ai-ml-ds.png)

> A diagram showing the relationships between AI, ML, deep learning, and data science. Infographic by [Jen Looper](https://twitter.com/jenlooper) inspired by [this graphic](https://softwareengineering.stackexchange.com/questions/366996/distinction-between-ai-ml-neural-networks-deep-learning-and-data-mining)

---
## Concepts to cover

In this curriculum, we are going to cover only the core concepts of machine learning that a beginner must know. We cover what we call 'classical machine learning' primarily using Scikit-learn, an excellent library many students use to learn the basics.  To understand broader concepts of artificial intelligence or deep learning, a strong fundamental knowledge of machine learning is indispensable, and so we would like to offer it here.

---
## In this course you will learn:

- core concepts of machine learning
- the history of ML
- ML and fairness
- regression ML techniques
- classification ML techniques
- neural networks
- clustering ML techniques
- natural language processing ML techniques
- time series forecasting ML techniques
- reinforcement learning
- AI
- real-world applications for ML

---

## Why study machine learning?

Machine learning, from a systems perspective, is defined as the creation of automated systems that can learn hidden patterns from data to aid in making intelligent decisions.

This motivation is loosely inspired by how the human brain learns certain things based on the data it perceives from the outside world.

✅ Think for a minute why a business would want to try to use machine learning strategies vs. creating a hard-coded rules-based engine.

---
## Applications of machine learning

Applications of machine learning are now almost everywhere, and are as ubiquitous as the data that is flowing around our societies, generated by our smart phones, connected devices, and other systems. Considering the immense potential of state-of-the-art machine learning algorithms, researchers have been exploring their capability to solve multi-dimensional and multi-disciplinary real-life problems with great positive outcomes.

---
## Examples of applied ML

**You can use machine learning in many ways**:

- To predict the likelihood of disease from a patient's medical history or reports.
- To leverage weather data to predict weather events.
- To understand the sentiment of a text.
- To detect fake news to stop the spread of propaganda.

Finance, economics, earth science, space exploration, biomedical engineering, cognitive science, and even fields in the humanities have adapted machine learning to solve the arduous, data-processing heavy problems of their domain.

---
## Conclusion

Machine learning automates the process of pattern-discovery by finding meaningful insights from real-world or generated data. It has proven itself to be highly valuable in business, health, and financial applications, among others.

In the near future, understanding the basics of machine learning is going to be a must for people from any domain due to its widespread adoption.



---
# Review & Self Study

To learn more about how you can work with ML algorithms in the cloud, follow this [Learning Path](https://docs.microsoft.com/learn/paths/create-no-code-predictive-models-azure-machine-learning/?WT.mc_id=academic-77952-leestott).

Take a [Learning Path](https://docs.microsoft.com/learn/modules/introduction-to-machine-learning/?WT.mc_id=academic-77952-leestott) about the basics of ML.


