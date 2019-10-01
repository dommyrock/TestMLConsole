using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using static Microsoft.ML.DataOperationsCatalog;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms.Text;

namespace TestMLApp
{
    class Program
    {
        static readonly string _dataPath = Path.Combine(Environment.CurrentDirectory, "Data", "yelp_labelled.txt");

        static void Main(string[] args)
        {
            MLContext mlContext = new MLContext();
            //Load data
            TrainTestData splitDataView = LoadData(mlContext);
            //Train model
            ITransformer model = BuildAndTrainModel(mlContext, splitDataView.TrainSet);
            //Assesses the model
            Evaluate(mlContext, model, splitDataView.TestSet);

            UseModelWithSingleItem(mlContext, model);

            UseModelWithBatchItems(mlContext, model);

            Console.ReadKey();
        }

        private static TrainTestData LoadData(MLContext mlContext)
        {
            //Load from file

            IDataView dataView = mlContext.Data.LoadFromTextFile<SentimentData>(_dataPath, hasHeader: false);

            //Split the dataset fomr model training
            TrainTestData splitDataView = mlContext.Data.TrainTestSplit(dataView, testFraction: 0.2);
            return splitDataView;
        }

        private static ITransformer BuildAndTrainModel(MLContext mlCtx, IDataView splitTrainSet)
        {
            //FeaturizeText() method in the previous code converts the text column (SentimentText)
            //into a numeric key type Features column used by the machine learning algorithm and adds it as a new dataset column:
            var estimator = mlCtx.Transforms.Text.FeaturizeText(outputColumnName: "Features", inputColumnName: nameof(SentimentData.SentimentText))
                .Append(mlCtx.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName: "Label", featureColumnName: "Features"));

            Console.WriteLine("=============== Create and Train the Model ===============");
            var model = estimator.Fit(splitTrainSet); // Fit() method trains your model by transforming the dataset and applying the training.
            Console.WriteLine("=============== End of training ===============");
            Console.WriteLine();

            return model;
        }

        public static void Evaluate(MLContext mlContext, ITransformer model, IDataView splitTestSet)
        {
            Console.WriteLine("=============== Evaluating Model accuracy with Test data===============");
            IDataView predictions = model.Transform(splitTestSet); // Transform() method to make predictions for multiple provided input rows of a test dataset.
            /*
             *  Evaluate() method assesses the model, which compares the predicted values with the actual Labels in the test dataset
             *  and returns a CalibratedBinaryClassificationMetrics object on how the model is performing.
             */
            CalibratedBinaryClassificationMetrics metrics = mlContext.BinaryClassification.Evaluate(predictions, "Label");

            Console.WriteLine();
            Console.WriteLine("Model quality metrics evaluation");
            Console.WriteLine("--------------------------------");
            Console.WriteLine($"Accuracy: {metrics.Accuracy:P2}");//accuracy of a model, which is the proportion of correct predictions in the test set.
            Console.WriteLine($"Auc: {metrics.AreaUnderRocCurve:P2}");//how confident the model is correctly classifying the positive and negative classes. You want the AreaUnderRocCurve to be as close to one as possible.
            Console.WriteLine($"F1Score: {metrics.F1Score:P2}");//measure of balance between precision and recall. You want the F1Score to be as close to one as possible.
            Console.WriteLine("=============== End of model evaluation ===============");
        }

        private static void UseModelWithSingleItem(MLContext mlContext, ITransformer model)
        {
            //"PredictionEngine" API ---allows you to perform a prediction on a single instance of data
            PredictionEngine<SentimentData, SentimentPrediction> predictionFunction = mlContext.Model.CreatePredictionEngine<SentimentData, SentimentPrediction>(model);

            SentimentData sampleStatement = new SentimentData
            {
                SentimentText = "This was a very bad steak"
            };

            var resultPrediction = predictionFunction.Predict(sampleStatement);

            Console.WriteLine();
            Console.WriteLine("=============== Prediction Test of model with a single sample and test dataset ===============");

            Console.WriteLine();
            Console.WriteLine($"Sentiment: {resultPrediction.SentimentText} | Prediction: {(Convert.ToBoolean(resultPrediction.Prediction) ? "Positive" : "Negative")} | Probability: {resultPrediction.Probability} ");

            Console.WriteLine("=============== End of Predictions ===============");
            Console.WriteLine();
        }

        public static void UseModelWithBatchItems(MLContext mlContext, ITransformer model)
        {
            IEnumerable<SentimentData> sentiments = new[]
            {
                new SentimentData
                {
                    SentimentText = "This was a horrible meal"
                },
                new SentimentData
                {
                    SentimentText = "I love this spaghetti."
                },
                new SentimentData
                {
                    SentimentText = "Was a  terrible lasagne."
                },
                new SentimentData
                {
                    SentimentText = "Meal was shit."
                },
                new SentimentData
                {
                    SentimentText = "Pizza was moldy and old."
                },
                new SentimentData
                {
                    SentimentText = "Pizza was so terribly good!"
                },
                new SentimentData
                {
                    SentimentText = "Cheese was crusty and it smelled nice."
                },
                new SentimentData
                {
                    SentimentText = "Matej is young but not that good looking."
                },
                new SentimentData
                {
                    SentimentText = "Good, bad."
                },
                new SentimentData
                {
                    SentimentText = "Bad, good."
                },
            };
            IDataView batchComments = mlContext.Data.LoadFromEnumerable(sentiments);

            IDataView predictions = model.Transform(batchComments);

            // Use model to predict whether comment data is Positive (1) or Negative (0).
            IEnumerable<SentimentPrediction> predictedResults = mlContext.Data.CreateEnumerable<SentimentPrediction>(predictions, reuseRowObject: false);

            Console.WriteLine();
            Console.WriteLine("=============== Prediction Test of loaded model with multiple samples ===============");

            foreach (SentimentPrediction prediction in predictedResults)
            {
                Console.WriteLine($"Sentiment: {prediction.SentimentText} | Prediction: {(Convert.ToBoolean(prediction.Prediction) ? "Positive" : "Negative")} | Probability: {prediction.Probability} ");
            }
            Console.WriteLine("=============== End of predictions ===============");
        }
    }
}