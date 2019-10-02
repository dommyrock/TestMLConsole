using System;
using System.IO;
using System.Linq;
using Microsoft.ML;
using TestMLApp.Models;

namespace TestMLApp
{
    class Program2
    {
        private static string _trainDataPath => Path.Combine(Environment.CurrentDirectory, "Data", "issues_train.tsv");
        private static string _testDataPath => Path.Combine(Environment.CurrentDirectory, "Data", "issues_test.tsv");
        private static string _modelPath => Path.Combine(Environment.CurrentDirectory, "Models", "model.zip");

        private static MLContext _mlContext;
        private static PredictionEngine<GitHubIssueData, IssuePrediction> _predEngine;
        private static ITransformer _trainedModel;
        static IDataView _trainingDataView;

        //static void Main(string[] args)
        //{
        //    //andom seed (seed: 0) for repeatable/deterministic results across multiple trainings
        //    _mlContext = new MLContext(seed: 0);
        //    _trainingDataView = _mlContext.Data.LoadFromTextFile<GitHubIssueData>(_trainDataPath, hasHeader: true);

        //    var pipeline = ProcessData();
        //    var trainingPipeline = BuildAndTrainModel(_trainingDataView, pipeline);

        //    Evaluate(_trainingDataView.Schema);
        //    PredictIssue();
        //}

        public static IEstimator<ITransformer> ProcessData()
        {
            var pipeline = _mlContext.Transforms.Conversion.MapValueToKey(inputColumnName: "Area", outputColumnName: "Label")//transform the Area column into a numeric key type Label column
                .Append(_mlContext.Transforms.Text.FeaturizeText(inputColumnName: "Title", outputColumnName: "TitleFeaturized"))//transforms text into a numeric vector for each called TitleFeaturized and DescriptionFeaturized
                .Append(_mlContext.Transforms.Text.FeaturizeText(inputColumnName: "Description", outputColumnName: "DescriptionFeaturized"))//transforms text into a numeric vector for each called TitleFeaturized and DescriptionFeaturized
                .Append(_mlContext.Transforms.Concatenate("Features", "TitleFeaturized", "DescriptionFeaturized"))
                .AppendCacheCheckpoint(_mlContext);//to cache the DataView so when you iterate over the data multiple times using the cache might get better performance
            return pipeline;
        }

        /// <summary>
        /// Returns the model.
        /// </summary>
        /// <param name="trainingDataView"></param>
        /// <param name="pipeline"></param>
        /// <returns></returns>
        public static IEstimator<ITransformer> BuildAndTrainModel(IDataView trainingDataView, IEstimator<ITransformer> pipeline)
        {
            var trainingPipeline = pipeline.Append(_mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy("Label", "Features"))
                    .Append(_mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            Console.WriteLine("------transforming the dataset and applying the training.--------");
            _trainedModel = trainingPipeline.Fit(trainingDataView);//Fit()method trains your model by transforming the dataset and applying the training.

            _predEngine = _mlContext.Model.CreatePredictionEngine<GitHubIssueData, IssuePrediction>(_trainedModel);

            GitHubIssueData issue = new GitHubIssueData()
            {
                Title = "WebSockets communication is slow in my machine",
                Description = "The WebSockets communication used under the covers by SignalR looks like is going slow in my development machine.."
            };

            var prediction = _predEngine.Predict(issue);// Predict() function makes a prediction on a single row of data:
            Console.WriteLine($"=============== Single Prediction just-trained-model - Result: {prediction.Area} ===============");
            return trainingPipeline;
        }

        /// <summary>
        /// The model created in BuildAndTrainModel is passed in to be evaluated
        /// </summary>
        /// <param name="trainingDataViewSchema"></param>
        public static void Evaluate(DataViewSchema trainingDataViewSchema)
        {
            var testDataView = _mlContext.Data.LoadFromTextFile<GitHubIssueData>(_testDataPath, hasHeader: true);
            var testMetrics = _mlContext.MulticlassClassification.Evaluate(_trainedModel.Transform(testDataView));
            Console.WriteLine($"*************************************************************************************************************");
            Console.WriteLine($"*       Metrics for Multi-class Classification model - Test Data     ");
            Console.WriteLine($"*------------------------------------------------------------------------------------------------------------");
            Console.WriteLine($"*       MicroAccuracy:    {testMetrics.MicroAccuracy:0.###}");//Every sample-class pair contributes equally to the accuracy metric
            Console.WriteLine($"*       MacroAccuracy:    {testMetrics.MacroAccuracy:0.###}");//Every class contributes equally to the accuracy metric. Minority classes are given equal weight as the larger classes
            Console.WriteLine($"*       LogLoss:          {testMetrics.LogLoss:#.###}");//You want Log-loss to be as close to zero as possible
            Console.WriteLine($"*       LogLossReduction: {testMetrics.LogLossReduction:#.###}");//Ranges from [-inf, 100], where 100 is perfect predictions and 0 indicates mean predictions. You want Log-loss reduction to be as close to zero as possible.
            Console.WriteLine($"*************************************************************************************************************");

            SaveModelAsFile(_mlContext, trainingDataViewSchema, _trainedModel);
        }

        /// <summary>
        ///  Save method to serialize and store the trained model as a zip file.
        /// </summary>
        /// <param name="mlContext"></param>
        /// <param name="trainingDataViewSchema"></param>
        /// <param name="model"></param>
        private static void SaveModelAsFile(MLContext mlContext, DataViewSchema trainingDataViewSchema, ITransformer model)
        {
            mlContext.Model.Save(model, trainingDataViewSchema, _modelPath);
        }

        private static void PredictIssue()
        {
            //Load the saved model into your application
            ITransformer loadedModel = _mlContext.Model.Load(_modelPath, out var modelInputSchema);
            //Add a GitHub issue to test the trained model's prediction
            GitHubIssueData singleIssue = new GitHubIssueData() { Title = "Entity Framework crashes", Description = "When connecting to the database, EF is crashing" };
            _predEngine = _mlContext.Model.CreatePredictionEngine<GitHubIssueData, IssuePrediction>(loadedModel);
            var prediction = _predEngine.Predict(singleIssue);
            Console.WriteLine($"=============== Single Prediction - Result: {prediction.Area} ===============");
        }
    }
}