
### Table of Contents

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [Project Components](#components)
4. [File Descriptions](#files)
5. [Instruction](#results)
6. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation <a name="installation"></a>

To use all content of the repository the following python libraries are used:
* pandas
* numpy
* sqlalchemy
* nltk
* re
* sklearn
* pickle
* sys
* flask
* plotly
* json

## Project Motivation<a name="motivation"></a>

This repository analyze disaster data from Figure Eight to build a model for an API that classifies disaster messages.

In the Project Workspace, you'll find a data set containing real messages that were sent during disaster events. Creating a machine learning pipeline to categorize these events to send the messages to an appropriate disaster relief agency is part of it.

The project includes a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data.

## Project Components<a name="components"></a>
There are three components you'll need to complete for this project.

### 1. ETL Pipeline
The Python script, `process_data.py`, is builds the ETL Pipeline:

* Loads the `messages` and `categories` datasets
* Merges the two datasets
* Cleans the data
* Stores it in a SQLite database

### 2. ML Pipeline
The Python script, `train_classifier.py`, includes a machine learning pipeline:

* Loads data from the SQLite database
* Splits the dataset into training and test sets
* Builds a text processing and machine learning pipeline
* Trains and tunes a model using GridSearchCV
* Outputs results on the test set
* Exports the final model as a pickle file

### 3. Flask Web App
The flask web app builds visulaizations powered by flask, html, css and javascript in combination with Plotly


## File Descriptions <a name="files"></a>

The files structure is arranged as below:

* README.md
* ETL Pipeline Preparation.ipynb
* ML Pipeline Preparation.ipynb
* Workspace
  * app
    * templates
    * run.py
  * data
    * DisasterResponse.db
    * disaster_categories.csv
    * disaster_messages.csv
    * process_data.py
  * models
    * classifier.pkl
    * train_classifier.py  


## Instruction<a name="results"></a>

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

## Licensing, Authors, Acknowledgements<a name="licensing"></a>

Data and content of the project are part of the Udacity [Data Scientist](https://www.udacity.com/course/data-scientist-nanodegree--nd025) nanodegree . 
