using System;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace IrisFlowers
{
    class Program
    {
        static readonly string _dataPath = Path.Combine(Environment.CurrentDirectory, "Data", "iris.data");
        static readonly string _modelPath = Path.Combine(Environment.CurrentDirectory, "Data", "IrisClusteringModel.zip");

        static void Main(string[] args)
        {
            var mlContext = new MLContext(seed: 0);

            IDataView dataView = mlContext.Data.LoadFromTextFile<IrisData>(_dataPath, hasHeader: false, separatorChar: ',');

            string featuresColumnName = "Features";
            var pipeline = mlContext.Transforms
                .Concatenate(featuresColumnName, "SepalLength", "SepalWidth", "PetalLength", "PetalWidth")
                .Append(mlContext.Clustering.Trainers.KMeans(featuresColumnName, numberOfClusters: 3));// KMeansTrainer trainer to train the model using the k-means++ clustering algorithm.

            var model = pipeline.Fit(dataView);//TODO ...error path not found
            //At this point, you have a model that can be integrated into any of your existing or new .NET applications
            using (var fileStream = new FileStream(_modelPath, FileMode.Create, FileAccess.Write, FileShare.Write))
            {
                //Save model to Zip
                mlContext.Model.Save(model, dataView.Schema, fileStream);
            }
            // PredictionEngine<TSrc,TDst> class takes instances of the input type through the transformer pipeline and produces instances of the output type
            var predictor = mlContext.Model.CreatePredictionEngine<IrisData, ClusterPrediction>(model);

            var prediction = predictor.Predict(TestIrisData.Setosa);
            Console.WriteLine($"Cluster: {prediction.PredictedClusterId}");
            Console.WriteLine($"Distances: {string.Join(" ", prediction.Distances)}");
        }
    }
}