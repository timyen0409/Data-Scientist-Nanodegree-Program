
### Installation
The project is supported by Python 3.x 
-- Numpy 1.15.2
-- Pandas 0.23.4
-- IPython 6.5.0
-- Matplotlib 3.0.0
-- scikit-learn 0.20.0
-- sqlalchemy 1.2.12
-- nltk 3.3.0

### Project Overview
We analyze disaster data with data enginering method, and build a model for API, in order to classify disaster messages.

First, we build the ETL pipeline and store the data into the database. Second, we use the cleaned data to build a Machine Learning pipeline, in order to classify the messages.

Last, we deploy the model at the web application,so that we can classify the messages from the web.

### File Description

app/run.py: Launch the web app.
data/process_data.py: Execute the ETL Pipeline and generate a database with clean data.
data/disaster_categories.csv: csv file containing the categories asociated to the messages.
data/disaster_messages.csv: csv file containing the messages.
models/train_classifier.py: Load data from SQLite database. And stores the final model after training.

### Web application

1. Run the following commands to set up database and model.

  - To run ETL pipeline that cleans data and stores in database:<pre><code>python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db</code></pre>
  
  - To run ML pipeline that trains classifier and saves the model:<pre><code>python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl</code></pre>
  - To start your web app.<pre><code>python run.py</code></pre>

2. Go to [http://0.0.0.0:3001/](http://0.0.0.0:3001/)

### Acknowledge
Most credits to Udacity to the data and web application templates. The project focus on full pipelines about the ETL and the machine learning.
