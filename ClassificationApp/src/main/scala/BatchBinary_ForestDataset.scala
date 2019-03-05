import org.apache.log4j.Logger
import org.apache.log4j.Level

import scala.collection.mutable.ArrayBuffer

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.mllib.classification.LogisticRegressionWithSGD
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql.Row
import org.apache.spark.sql.types.{DoubleType, IntegerType}


object ForestBinaryBatchClassification {

  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)
    val conf = new SparkConf().setAppName("ForestBinaryClassification")
    val sc = new SparkContext(conf)
    val spark = org.apache.spark.sql.SparkSession.builder
      .appName("ForestBinaryClassification")
      .getOrCreate;

    var df = spark.read
      .format("csv")
      .option("header", "true")
      .load("/Users/jigyayadav/Desktop/UCSDAcads/Quarter5/CSE291F/" +
        "spark.nosync/Forest_Dataset/forest_dataset_sepfeatures_csv.csv")

    df = df.withColumn("labeli", df("labeli").cast(DoubleType))

    var feat_name_array = ArrayBuffer[String]()
    for (i <- 1 to 54) {
      var str = s"Feat$i"
      df = df.withColumn(str, df(str).cast(DoubleType))
      feat_name_array += str
    }
    val assembler = new VectorAssembler()
      .setInputCols(feat_name_array.toArray)
      .setOutputCol("features")

    val featDF = assembler.transform(df)

    var labeledPointsRDD = featDF.rdd.map(row => LabeledPoint(
      row.getAs[Double]("labeli"),
      org.apache.spark.mllib.linalg.Vectors.fromML(row.getAs[org.apache.spark.ml.linalg.Vector]("features").toDense)
    ))

    var dataFraction = args(0).toFloat
    println(s"Fraction of data being used: $dataFraction")
    val labeledPointsRDDSample = labeledPointsRDD.sample(false, dataFraction, seed = 4)

    val Array(training, test) = labeledPointsRDDSample.randomSplit(Array(0.8, 0.2), seed = 11L)
    training.cache()

    val t0 = System.nanoTime()

    // Set different values of the regularization parameters here
    // Create algorithm instance
    val lr = new LogisticRegressionWithSGD()
//    lr.optimizer.setRegParams(0.003, 0.001, 0.03, 0.01, 0.3, 0.1, 1, 3, 10, 0.005, 0.05, 0.5, 0.25, 0.26, 0.27, 1.2, 0.015, 0.048)
    lr.optimizer.setRegParams(0.003, 0.001, 0.01, 0.03, 0.1, 0.3, 0.7, 1, 3)

    // Train model
    val model = lr.run(training)

    val t2 = System.nanoTime()

    println(s"Multinomial coefficients: ${model.weightMatrix}")
    println(s"Multinomial intercepts: ${model.interceptVector}")
    println(s"Time taken for training: ${t2 - t0}")

    // Clear the prediction threshold so the model will return probabilities
    model.clearThreshold

    // Compute raw scores on the test set
    val batchPredictionAndLabels = test.map { case LabeledPoint(label, features) =>
      val batchPrediction = model.batchPredict(features)
      (batchPrediction, label)
    }

    (0 until lr.optimizer.getRegParams.size).foreach { i =>
      val predictionAndLabels = batchPredictionAndLabels.map { case (batchPrediction, label) =>
        (batchPrediction(i), label)
      }
      // Instantiate metrics object
      val metrics = new BinaryClassificationMetrics(predictionAndLabels)

      // Precision by threshold
      val precision = metrics.precisionByThreshold
      precision.foreach { case (t, p) =>
        println(s"Threshold: $t, Precision: $p")
      }

      // Recall by threshold
      val recall = metrics.recallByThreshold
      recall.foreach { case (t, r) =>
        println(s"Threshold: $t, Recall: $r")
      }

      // Precision-Recall Curve
      val PRC = metrics.pr

      // F-measure
      val f1Score = metrics.fMeasureByThreshold
      f1Score.foreach { case (t, f) =>
        println(s"Threshold: $t, F-score: $f, Beta = 1")
      }

      val beta = 0.5
      val fScore = metrics.fMeasureByThreshold(beta)
      f1Score.foreach { case (t, f) =>
        println(s"Threshold: $t, F-score: $f, Beta = 0.5")
      }

      // AUPRC
      val auPRC = metrics.areaUnderPR
      println(s"Area under precision-recall curve = $auPRC")

      // Compute thresholds used in ROC and PR curves
      val thresholds = precision.map(_._1)

      // ROC Curve
      val roc = metrics.roc

      // AUROC
      val auROC = metrics.areaUnderROC
      println(s"Area under ROC = $auROC")
      // $example off$
    }

    val t1 = System.nanoTime()
    println("Elapsed time: " + (t1 - t0) + "ns")

    println("Size of training set: ", training.count())
    sc.stop()
  }
}