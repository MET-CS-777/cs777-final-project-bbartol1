from __future__ import print_function

import re
import sys
import numpy as np
from operator import add
import os
os.environ['PYSPARK_PYTHON'] = "C:/Program Files/Python310/python.exe"
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
from pyspark import SparkContext
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.classification import LinearSVC
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from pyspark.sql.types import *
from pyspark.sql import functions as func
from pyspark.sql.functions import *
from pyspark.sql import functions as F
from pyspark.mllib.evaluation import MulticlassMetrics
import time
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType
from pyspark.sql.functions import udf
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import sum as spark_sum

#sc = SparkContext.getOrCreate()
#spark = SparkSession.builder.getOrCreate()


# Get performance metrics for logictic regression model.
def m_metrics_log(ml_model,test_data):
    predictions = ml_model.transform(test_data).cache()
    predictionAndLabels = predictions.select("SHOT_OUTCOME","prediction").rdd.map(lambda x: (float(x[0]), float(x[1]))).cache()
    
    # Print some predictions vs labels
    # print(predictionAndLabels.take(10))
    metrics = MulticlassMetrics(predictionAndLabels)
    
    # Overall statistics
    accuracy = metrics.accuracy
    precision = metrics.precision(1.0)
    recall = metrics.recall(1.0)
    f1Score = metrics.fMeasure(1.0)
    print(f"Accuracy: {accuracy} Precision = {precision:.4f} Recall = {recall:.4f} F1 Score = {f1Score:.4f}")
    print("Confusion matrix \n", metrics.confusionMatrix().toArray().astype(int))


# Initialize the Spark Session
spark = SparkSession.builder.master("local[*]").getOrCreate()
sc = SparkContext.getOrCreate()
sqlContext = SQLContext(sc)

####################################################################
# 1. DATA INPUT
####################################################################
# Location containing data files contianing shot data
path_1="file:///C:/Users/brand/OneDrive/Documents/CS777/nba_shot_data/"
nba_2020_season = path_1 + "NBA_2020_Shots.csv"
nba_2021_season = path_1 + "NBA_2021_Shots.csv"
nba_2022_season = path_1 + "NBA_2022_Shots.csv"
nba_2023_season = path_1 + "NBA_2023_Shots.csv"
nba_2024_season = path_1 + "NBA_2024_Shots.csv"
# Location of the data file containing NBA All Stars
path_2="file:///C:/Users/brand/OneDrive/Documents/CS777/"
nba_all_stars = path_2 + "NBA_ALL_STAR_BY_YEAR.csv"

# Read the csv Files and all stars list
corpus_1 = spark.read.format('csv').options(header='true', inferSchema='true',  sep =",").load(nba_2020_season)
corpus_2 = spark.read.format('csv').options(header='true', inferSchema='true',  sep =",").load(nba_2021_season)
corpus_3 = spark.read.format('csv').options(header='true', inferSchema='true',  sep =",").load(nba_2022_season)
corpus_4 = spark.read.format('csv').options(header='true', inferSchema='true',  sep =",").load(nba_2023_season)
corpus_5 = spark.read.format('csv').options(header='true', inferSchema='true',  sep =",").load(nba_2024_season)
nba_all_stars = spark.read.format('csv').options(header='true', inferSchema='true',  sep =",").load(nba_all_stars)

# Select Distinct list of all star Players
all_stars = nba_all_stars.select('Player').distinct()

# Full dataframe of every shot
unified_corpus = corpus_1.union(corpus_2).union(corpus_3).union(corpus_4).union(corpus_5)
# Total number of shots
print("Total Number of Shots:")
print(unified_corpus.count())
# Turn the outcome to a binary integer 1 or 0, to be used as the prediction class
made_shot_udf = udf(lambda EVENT_TYPE: 1 if EVENT_TYPE == "Made Shot" else 0, IntegerType())
# Add in column showing the value of the shot (3 points, 2 points, or 0 if missed)
shot_value_udf = udf(lambda EVENT_TYPE, SHOT_TYPE: 3 if EVENT_TYPE == "Made Shot" and SHOT_TYPE == "3PT Field Goal" else (2 if EVENT_TYPE == "Made Shot" else 0), IntegerType())
# Add column for prediction
unified_corpus = unified_corpus.withColumn("SHOT_OUTCOME", made_shot_udf(unified_corpus.EVENT_TYPE))
unified_corpus = unified_corpus.withColumn("SHOT_VALUE", shot_value_udf(unified_corpus.EVENT_TYPE, unified_corpus.SHOT_TYPE))


# Add flag to all-star dataframe
all_stars_flagged = all_stars.withColumn('is_all_star', lit(1))
# Join with unified_corpus
unified_corpus = unified_corpus.join(all_stars_flagged, unified_corpus.PLAYER_NAME == all_stars_flagged.Player, how='left')
# Replace null values in is_all_star column with 0
unified_corpus = unified_corpus.na.fill({'is_all_star': 0})

# Number of shots made and missed
shots_made = unified_corpus.filter(col('SHOT_OUTCOME') == 1).count()
shots_missed = unified_corpus.filter(col('SHOT_OUTCOME') == 0).count()
print(f'Shots Made: {shots_made}')
print(f'Shots Missed: {shots_missed}')


# View the dataset (Optional)
result = unified_corpus.select(['PLAYER_NAME','GAME_ID', 'SHOT_VALUE','SHOT_OUTCOME','ACTION_TYPE', 'BASIC_ZONE',
                                'ZONE_NAME','ZONE_RANGE','LOC_X', 'LOC_Y', 'SHOT_DISTANCE']).show()
#print(result)


##############################################################################
# 2. Preprocess and define model features
##############################################################################
# Define columns that will be used as features
categorical_features = ['ACTION_TYPE', 'ZONE_NAME', 'ZONE_RANGE']

# Index and encode categorical columns
# Index tags each distinct value with a number so it can be mapped back
indexers = [StringIndexer(inputCol=column, outputCol=column+"_index") for column in categorical_features]
# Encoders turn my index values to the column header, it allows me to use each categorical value as a binary 1 or 0
encoders = [OneHotEncoder(inputCol=column+"_index", outputCol=column+"_encoded") for column in categorical_features]

# Assemble the feature columns
encoded_features = [column+"_encoded" for column in categorical_features]
assembler = VectorAssembler(inputCols=encoded_features, outputCol="features")
# Create a pipeline for preprocessing steps
pipeline_p = Pipeline(stages=indexers + encoders + [assembler])
# Transform the unified_corpus
full_data = pipeline_p.fit(unified_corpus).transform(unified_corpus) #.cache() not enough memory on my system



############################################################### 
# Exploritory - Used to view the preprocessing steps
###############################################################
'''
# Fit the StringIndexers separately
index_models = [indexer.fit(unified_corpus) for indexer in indexers]
indexed_data = unified_corpus
for indexer, model in zip(indexers, index_models):
    indexed_data = model.transform(indexed_data)

# Create a DataFrame for each indexed column
index_value_dataframes = []
for model in index_models:
    input_col = model.getInputCol()
    index_col = model.getOutputCol()
    labels = model.labels
    data = [(i, label) for i, label in enumerate(labels)]
    df = spark.createDataFrame(data, ["index", "value"])
    df = df.withColumn("column", lit(input_col))
    index_value_dataframes.append(df)

# Union all DataFrames to get a combined view
index_value_table = index_value_dataframes[0]
for df in index_value_dataframes[1:]:
    index_value_table = index_value_table.union(df)

# Show the combined table
index_value_table.show(truncate=False)

# Inspect the indexed values
for model in index_models:
    print(f"Indexer for {model.getInputCol()}: {model.labels}")

# Fit the OneHotEncoders separately
encoder_models = [encoder.fit(indexed_data) for encoder in encoders]
encoded_data = indexed_data
for encoder, model in zip(encoders, encoder_models):
    encoded_data = model.transform(encoded_data)

# Inspect the encoded values
encoded_data.select(*encoded_features).show(truncate=False)

# Assemble the feature columns
assembler = VectorAssembler(inputCols=encoded_features, outputCol="features")

# Fit and transform the final dataset
final_data = assembler.transform(encoded_data)

# Verify the final assembled features
final_data.select("features").show(truncate=False)

'''
#######################################################################
# 3. Create Logistic Regression Model
#######################################################################
# Logistic Regression model
logr = LogisticRegression(maxIter=20, regParam=0.1, labelCol="SHOT_OUTCOME", featuresCol="features")

# Split data
train_data, test_data = full_data.randomSplit([0.8, 0.2], seed=42)

# Fit model
log_model = logr.fit(train_data)


####################################################################
# 4. Create Linear Regression Model
####################################################################
# Linear Regression Model

# Filter to 2 dataframes
all_stars_df = full_data.filter(col('is_all_star') == 1)
non_all_stars_df = full_data.filter(col('is_all_star') == 0)

# Linear Regression model
linr = LinearRegression(maxIter=20, regParam=0.1, labelCol="SHOT_VALUE", featuresCol="features")

# Split into train and test datasets 1 set for all stars, one set for non-all-stars
as_train_data, as_test_data = all_stars_df.randomSplit([0.8, 0.2], seed=42)
non_as_train_data, non_as_test_data = non_all_stars_df.randomSplit([0.8, 0.2], seed=42)

# Create the linear regression model for the shots of all stars
as_linr_model = linr.fit(as_train_data)
# make predictions on the test data
as_predictions = as_linr_model.transform(as_test_data)

# Create the linear regression model for the shots of non-all stars
non_as_linr_model = linr.fit(non_as_train_data)
# make predictions on the test data
non_as_predictions = non_as_linr_model.transform(non_as_test_data)

# Hyper parameters were altered to attempt to improve performance. Found a 0.1 regerlarization parameter and 0 elasticNetParam ahd optimal results
'''
# Create ParamGrid for Cross Validation
paramGrid = (ParamGridBuilder()
             .addGrid(linr.regParam, [0.1, 0.01, 0.001])
             .addGrid(linr.elasticNetParam, [0.0, 0.5, 1.0])
             .build())

# Evaluator for Cross Validation
evaluator = RegressionEvaluator(labelCol="SHOT_VALUE", predictionCol="prediction", metricName="rmse")

# CrossValidator setup
crossval = CrossValidator(estimator=linr,
                          estimatorParamMaps=paramGrid,
                          evaluator=evaluator,
                          numFolds=5)

# Run cross validations
cvModel = crossval.fit(train_data)

# Make predictions
predictions = cvModel.transform(test_data)

# Evaluate best model
rmse = evaluator.evaluate(predictions)
print(f"Root Mean Squared Error (RMSE) = {rmse}")
'''



# Count to ensure number of shots is sufficient from both populations
#all_stars_size = all_stars_df.count()
#non_all_stars_size = non_all_stars_df.count()
#print(all_stars_size) # 244160 shots
#print(non_all_stars_size) # 787582 shots

#####################################################################################
# Model Performance
#####################################################################################
# TEST LOGISTIC REGRESSION MODEL PERFORMANCE
#############################################
# Test Performance
m_metrics_log(log_model,test_data)

#############################################
# TEST LINEAR REGRESSION MODEL PERFORMANCE
#############################################
#Performance evaluation by calculating the root mean squared error (RMSE):
evaluator1 = RegressionEvaluator(labelCol='SHOT_VALUE', predictionCol='prediction', metricName='rmse')
evaluator2 = RegressionEvaluator(labelCol='SHOT_VALUE', predictionCol='prediction', metricName='r2')
# All Star Model
as_rmse = evaluator1.evaluate(as_predictions)
as_r2 = evaluator2.evaluate(as_predictions)
# Non All-Star Model
non_as_rmse = evaluator1.evaluate(non_as_predictions)
non_as_r2 = evaluator2.evaluate(non_as_predictions)
print(f"All Star Root Mean Squared Error (RMSE) = {as_rmse}, R2 = {as_r2}")
print(f"non-All Star Root Mean Squared Error (RMSE) = {non_as_rmse}, R2 = {non_as_r2}")


# View Model Coefficient Values
# print the coefficients and intercept of the All Star linear regression model
print("All-Star Coefficients: " + str(as_linr_model.coefficients))
print("All-Star Intercept: " + str(as_linr_model.intercept))
#as_predictions.toPandas().head(5)
#as_predictions.show(5) # View Predicted values
# print the coefficients and intercept of the linear regression model
print("non All-Star Coefficients: " + str(non_as_linr_model.coefficients))
print("non All-Star Intercept: " + str(non_as_linr_model.intercept))
#non_as_predictions.toPandas().head(5)
#non_as_predictions.show(5) view predicted values

# Get the coefficients and intercept - All Stars
as_coefficients = as_linr_model.coefficients
as_intercept = as_linr_model.intercept
# Get the coefficients and intercept - non All-Stars
non_as_coefficients = non_as_linr_model.coefficients
non_as_intercept = non_as_linr_model.intercept

# Create dictionary of index values to show alongside coefficients
# Manually collect the categories for each indexed feature
categories = {}
for indexer in indexers:
    idx_col = indexer.getOutputCol()
    model = indexer.fit(unified_corpus)
    categories[idx_col] = model.labels
# Create encoded feature names
encoded_feature_names = []
for encoder in encoders:
    input_col = encoder.getInputCol()
    index_col = input_col.replace("_index", "")
    for i, category in enumerate(categories[input_col]):
        encoded_feature_names.append(f"{index_col}_{category}")

# All-Star Model Coefficient Analysis
# Create a list of features and coefficients
as_data = [(feature, float(coef)) for feature, coef in zip(encoded_feature_names, as_coefficients)]
# Define the schema for the DataFrame
schema = StructType([
    StructField("feature", StringType(), True),
    StructField("as_coefficient", FloatType(), True)
])
# Create a DataFrame from the list with the defined schema
as_df = spark.createDataFrame(as_data, schema=schema)
# Show the DataFrame and sort by coefficient
as_df.orderBy(col("as_coefficient").desc()).show(truncate=False)

# Non All Star Coefficient Anaylsis
# Create a list of features and coefficients
non_as_data = [(feature, float(coef)) for feature, coef in zip(encoded_feature_names, non_as_coefficients)]
# Define the schema for the DataFrame
schema = StructType([
    StructField("feature", StringType(), True),
    StructField("non_as_coefficient", FloatType(), True)
])
# Create a DataFrame from the list with the defined schema
non_as_df = spark.createDataFrame(non_as_data, schema=schema)
# Show the DataFrame and sort by coefficient
non_as_df.orderBy(col("non_as_coefficient").desc()).show(truncate=False)


# Create a dictionary mapping from encoded feature name to coefficient
coef_dict = {}
for feature, coef in zip(encoded_feature_names, as_coefficients):
    coef_dict[feature] = coef

# Display the coefficients for each encoded feature
print("All-Star:")
for feature, coef in coef_dict.items():
    print(f"{feature}: {coef}")
# Create a dictionary mapping from encoded feature name to coefficient
coef_dict = {}
for feature, coef in zip(encoded_feature_names, non_as_coefficients):
    coef_dict[feature] = coef
print("Non-All-Star:")
# Display the coefficients for each encoded feature
for feature, coef in coef_dict.items():
    print(f"{feature}: {coef}")

# Compare values
# Join the coefficient tables
feature_coefficent_table =  as_df.join(non_as_df, as_df.feature == non_as_df.feature, how='left').select(as_df.feature, as_df.as_coefficient.alias('as_coefficient'), non_as_df.non_as_coefficient.alias('non_as_coefficient'))
# Calculate the difference between coefficients
feature_coefficent_table = feature_coefficent_table.withColumn('difference', col('as_coefficient') - col('non_as_coefficient'))
# Count how many coefficients are higher for all-stars
higher_count = feature_coefficent_table.filter(col('difference') > 0).count()
# Find the top 10 highest differences for AS
top_differences_as = feature_coefficent_table.orderBy((col('difference')).desc()).limit(10)
# Find the top 10 highest differences for non AS
top_differences_non_as = feature_coefficent_table.orderBy((col('difference')).asc()).limit(10)
# Find the top 10 highest differences total
top_differences_abs = feature_coefficent_table.orderBy(abs(col('difference')).desc()).limit(10)
# Show the results
print(f"Number of coefficients higher for all-stars: {higher_count}")
top_differences_as.show(truncate=False)
top_differences_non_as.show(truncate=False)
top_differences_abs.show(truncate=False)
feature_coefficent_table.show(truncate=False)


# Sum predicted and actual shot values for all-star predictions
as_total_predicted = as_predictions.agg(spark_sum("prediction")).collect()[0][0]
as_total_actual = as_predictions.agg(spark_sum("SHOT_VALUE")).collect()[0][0]

# Sum predicted and actual shot values for non-all-star predictions
non_as_total_predicted = non_as_predictions.agg(spark_sum("prediction")).collect()[0][0]
non_as_total_actual = non_as_predictions.agg(spark_sum("SHOT_VALUE")).collect()[0][0]

# Print the results
print(f"All-Star Total Predicted Shot Value: {as_total_predicted}")
print(f"All-Star Total Actual Shot Value: {as_total_actual}")
print(f"Non All-Star Total Predicted Shot Value: {non_as_total_predicted}")
print(f"Non All-Star Total Actual Shot Value: {non_as_total_actual}")



sc.stop()


