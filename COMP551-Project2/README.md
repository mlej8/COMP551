<div align="center">  
    
# MiniProject 2: Reddit Classification

## COMP 551, Fall 2019, McGill University

## Professor: William L. Hamilton

## September 30, 2019

</div>

Please read this entire document before beginning the assignment.

## Preamble

- This mini-project isdue on Oct. 18th at 11:59pm. Late work will be automatically subject to a
    20% penalty and can be submitted up to 5 days after the deadline. No submissions will accepted after
    this 5 day period.
- This mini-project is to completed in groups of three. All members of a group will recieve the same
    grade. It is not expected that all team members will contribute equally to all components. However
    every team member should make integral contributions to the project.
- You will submit your assignment on MyCourses as a group, and you will also submit to a Kaggle
    competition. You must register in the Kaggle competition using the email that you are
    associated with on MyCourses (i.e., @mail.mcgill.ca for McGill students). You can reg-
    ister for the competition at:https://www.kaggle.com/t/cbd6c7bc66394bd682983a6daeefe
    As with MiniProject 1, you must register your group on MyCourses and any group member can submit.
    You must also form teams on Kaggle and you must use your MyCourses group name as
    your team name on Kaggle. All Kaggle submissions must be associated with a valid team
    registered on MyCourses.
- Except where explicitly noted in this specification, you are free to use any Python library or utility for
    this project.

## Background

In this mini-project you will develop models to analyze text from the website Reddit (https://www.reddit.
com/), a popular social media forum where users post and comment on content in different themed commu-
nities, orsubreddits.The goal of this project is to develop a supervised classification model that can predict
what community a comment came from. You will be competing with other groups to achieve the best ac-
curacy in a competition for this prediction task.However, your performance on the competition is
only one aspect of your grade. We also ask that you implement a minimum set of models and
report on their performance in a write-up.

The Kaggle website has a link to the data, which is a 20-class classification problem with a (nearly) balanced
dataset (i.e., there are equal numbers of comments from 20 different subreddits). The data is provided in
CSVs, where the text content of the comment is enclosed in quotes. Each entry in the training CSV contains


a comment ID, the text of the comment, and the name of the target subreddit for that comment. For the
test CSV, each line contains a comment ID and the text for that comment. You can view and download the
data via this link:https://www.kaggle.com/c/reddit-comment-classification-comp-551/data

You need to submit a prediction for each comment in the test CSV; i.e., you should make a prediction CSV
where each line contains a comment ID and the predicted subreddit for that comment. Since the data is
balanced and involves multiple classes, you will be evaluated according to the accuracy score your the model.
An example of the proper formatting for the submission file can be viewed at:https://www.kaggle.com/
c/reddit-comment-classification-comp-551/overview/evaluation.

## Tasks

You are welcome to try any model you like on this task, and you are free to use any libraries you like to
extract features. However,you must meet the following requirements:

- You must implement a Bernoulli Naive Bayes model (i.e., the Naive Bayes model from Lecture 5) from
    scratch (i.e., without using any external libraries such as SciKit learn). You are free to use any text
    preprocessing that you like with this model.Hint 1:you many want to use Laplace smoothing with
    your Bernoulli Naive Bayes model.Hint 2:you can choose the vocabulary for your model (i.e, which
    words you include vs. ignore), but you should provide justification for the vocabulary you use.
- You must run experiments using at least two different classifiers from the SciKit learn package (which
    are not Bernoulli Naive Bayes). Possible options are:
       - Logistic regression
          (https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
       - Decision trees
          (https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)
       - Support vector machines [to be introduced in Lecture 10 on Oct. 7th]
          (https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html)
- You must develop a model validation pipeline (e.g., using k-fold cross validation or a held-out validation
    set) and report on the performance of the above mentioned model variants.
- You should evaluate all the model variants above (i.e., Naive Bayes and the SciKit learn
    models) using your validation pipeline (i.e., without submitting to Kaggle) and report on
    these comparisons in your write-up. Ideally, you should only run your “best” model on
    the Kaggle competition, since you are limited to two submissions to Kaggle per day.

## Final remarks

You are expected to display initiative, creativity, scientific rigour, critical thinking, and good communication
skills. You don’t need to restrict yourself to the requirements listed above - feel free to go beyond, and
explore further.

You can discuss methods and technical issues with members of other teams, but you cannot share any code
or data with other teams. Any team found to cheat (e.g. use external information, use resources without
proper references) on either the code, predictions or written report will receive a score of 0 for all components
of the project.


Rules specific to the Kaggle competition:

- Don’t cheat! You must submit code that can reproduce the numbers of your leaderboard solution.
- The classification challenge is based on a public dataset. You must not attempt to cheat by searching
    for information about the test set. Submissions with suspicious accuracies and/or predictions will be
    flagged and your group will receive a 0 if you used external information about the test set at any point.
- Do not make more than one team for your group (e.g., to make more submissions). You will receive a
    grade of 0 for intentionally creating new groups with the purpose of making more Kaggle submissions.


