import numpy as np
from pyspark.ml import Pipeline
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, VectorIndexer,VectorAssembler
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from pyspark.sql.functions import mean, col

from pyspark.ml.linalg import Vectors, DenseMatrix
from pyspark.mllib.stat import Statistics


# 创建 SparkSession
spark = SparkSession.builder.getOrCreate()

# 读取数据
data = spark.read.csv("student_info_final_score.csv", header=True, inferSchema=True)

# 使用平均值填充
column_mean = data.select(mean(col("OptionScore"))).first()[0]
data = data.na.fill(column_mean, subset=["OptionScore"])

# 获取对应的特征列表
feature_names = [
    "OptionScore",
    "ProgramScore",
    "FirstStartTime",
    "UsedTime",
    "Submit",
    "ChaChong",
    "CorrectRate",
    "ResponseRate",
]

# Index labels, adding metadata to the label column.
# Fit on the whole dataset to include all labels in the index.
labelIndexer = StringIndexer(inputCol="level", outputCol="indexedLabel").fit(data)

# 构建对应的特征向量
assembler = VectorAssembler(inputCols=feature_names, outputCol="features")
data = assembler.transform(data)

# Automatically identify categorical features and index them.
# Set maxCategories so features with > 4 distinct values are treated as continuous.
featureIndexer = VectorIndexer(inputCol="features", outputCol="indexedFeatures").fit(data)

# Split the data into training and test sets (20% held out for testing)
(trainingData, testData) = data.randomSplit([0.8, 0.2])

# Train a RandomForest model.
rf = RandomForestClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures", numTrees=10)

# Chain indexers and forest in a Pipeline
pipeline = Pipeline(stages=[labelIndexer, featureIndexer, rf])

# Train the model
model = pipeline.fit(trainingData)

# Make predictions.
predictions = model.transform(testData)

# Select example rows to display.
# predictions.select("predictedLabel", "label", "features").show(5)
# predictions.select('label', 'prediction', 'probability').show()

# test_preds = xgb_model.transform(test_dmatrix)

# 将预测结果转换回Spark DataFrame，便于后续评估
# test_preds_df = test_preds.toPandas()

# 计算AUC（此处假设您已经实现了计算AUC的函数）
# auc = roc_auc_score(test_data["indexed_label"], test_preds_df["prediction"])

# # 预测测试集
# predictions = model.transform(test_data)

# 评估模型性能（这里使用多类分类评估器计算准确率）
evaluator = MulticlassClassificationEvaluator(
    labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy"
)
accuracy = evaluator.evaluate(predictions)
print("Accuracy: %.3f" % accuracy)

evaluator.setMetricName("f1")
f1 = evaluator.evaluate(predictions)
print("f1: %.3f" % f1)

evaluator.setMetricName("precisionByLabel")
precision = evaluator.evaluate(predictions)
print("precision: %.3f" % precision)

evaluator.setMetricName("recallByLabel")
recall = evaluator.evaluate(predictions)
print("recall: %.3f" % recall)

# 如果模型支持特征重要性（如决策树、随机森林），获取特征重要性
rfModel = model.stages[2]

if hasattr(rfModel, "featureImportances"):
    feature_importances = rfModel.featureImportances.toArray()
else:
    # 如果模型不直接提供特征重要性，可能需要通过其他方法（如PermutationImportance）来计算
    raise NotImplementedError("特征重要性计算未实现，取决于具体模型类型")

# create a feature importance dataframe
df_importance = pd.DataFrame({'feature': feature_names, 'importance': feature_importances})

# reshape it suitable for sns.heatmap
df_importance = df_importance.set_index('feature').T

# 画对应的热力图
plt.figure(figsize=(12, 8))
sns.heatmap(df_importance, cmap='viridis', annot=True)

plt.title('Feature Importances Heatmap')
# plt.show()
plt.savefig('feature_importance_heatmap.png')

# 如果要展示特征相关性热力图（适用于数值型特征）
# if all([isinstance(col, str) for col in data.columns if col.startswith("feature_")]):
#     # 计算特征相关系数矩阵
#     corr_matrix = Statistics.corr(data.select(feature_names), method="pearson")

#     # 转换为Pandas DataFrame以方便绘制热力图
#     corr_df = pd.DataFrame(
#         corr_matrix.toArray(),
#         columns=corr_matrix.columnNames(),
#         index=corr_matrix.rowNames(),
#     )

#     # 绘制热力图
#     plt.figure(figsize=(10, 8))
#     sns.heatmap(corr_df, annot=True, cmap="coolwarm", center=0)
#     plt.title("特征相关性热力图")
#     plt.show()

# # 绘制特征重要性热力图
# plt.figure(figsize=(10, 8))
# sns.heatmap(
#     DenseMatrix(feature_importances.reshape(1, -1)),
#     annot=True,
#     cmap="Blues",
#     yticklabels=False,
# )
# plt.title("特征重要性热力图")
# plt.show()

# 或者直接在新数据上进行预测
# new_data = spark.read.csv("path_to_new_data.csv", header=True, inferSchema=True)
# prepared_new_data = assembler.transform(new_data)
# predictions_new = lr_model.transform(prepared_new_data)
# predictions_new.show()
