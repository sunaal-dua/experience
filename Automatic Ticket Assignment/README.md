# AutomaticTicketAssignment
### <a target="_blank" href="https://github.com/sunaal-dua/experience/blob/master/Automatic%20Ticket%20Assignment/PROJECT%20REPORT/Automatic%20Ticket%20Assignment.pdf">(Find project report here. You can download or read the report online)</a>

### Greetings! Please read the following to understand the directory structure:
You will find a folder named <b>capstone</b>.<br>'Capstone' is the root folder of the code that contains all the jupyter notebooks. We have 6 parts for the project. All the noteooks are well documented and easy to follow. We recommend you to start the project in the order mentioned below. Lets get started:<br><br>
<b>1) M1(part1) - Preprocessing and Visualization.ipynb</b><br>
Start with this notebook. This is milestone-1 part-1. It contains all the Visualisations, Data Pre-processing and Exploration.<br><br>

<b>2) M1(part2a) - Base Model Approach 1.ipynb</b><br>
This is the following notebook. This is milestone-1 part-2a. As the name suggests, it contains our first set of base models which are traditional machine learning models, implemented with BagOfWords and Tf-Idf transformation.<br><br>

<b>3) M1(part2b) - Base Model Approach 2.ipynb</b><br>
This is milestone-1 part-2b. In 2nd set of base models we went for deep learning architecture; LSTMs.<br><br>

<b>4) M2(part1) - Model Tuning.ipynb</b><br>
This is milestone-2 part-1. After all the exploration and model analysis we finalised LSTMs to be the model of our choice. Continuing with LSTMs, we first tuned the LSTM for the best architecture in this part. We tried various architectures which are explained in details in this notebook.<br><br>

<b>5) M2(part2) - Model Tuning.ipynb</b><br>
This is milestone-2 part-2. After deciding upon the apt architecture in previous part, we explored the learning rate, tried different optimizers and did sanity checks of our model in this notebook. The aim was to hit the right learning rate and go for longer training sessions with various callbacks in order to avoid overfitting and best generalize the model. Detailed analysis can be found in this notebook.<br><br>

<b>6) M2(part3) - Model Tuning.ipynb</b><br>
This is milestone-2 part-3. Shortcomings from previous part are attended in this notebook. We closed the project with 92% accuracy and 93% macro fi-score on the test set.

<hr>

Coming to other folders in the root directory, we have made certain custom modules for making the code more readable and for the ease of programming. These modules contain the .py files that are used in our code. We have explained each and every module in the notebook. This is just a pre-cap of what folder contains what:<br><br>
1) <b>DataFiles:</b> Contains all the csv files for the project.<br>
2) <b>DataTransformation:</b> Module that contains functions for transforming dataset to Tf-Idf and BOW matrices<br>
3) <b>Model:</b> Contains functions for model tuning and evaluation for both traditional machine learning as well as deep learning approaches. It also contains the model checkpoints andtraining history<br>
4) <b>ProjectModules:</b> This model contains functions related to text processing, cleaning, lemmatization etc. It also contains language detection function as well as general functions for plotting various plots that are used in a number of places in the project
<hr>
