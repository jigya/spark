package org.apache.batchexperiments

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
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Row
import org.apache.spark.sql.types.{DoubleType, IntegerType}

import org.apache.batchexperiments.CriteoClassificationUtils


object CriteoBatchBinaryClassification {

  def main(args: Array[String]): Unit = {

    // This needs to be changed based on the dataset
    val maxNumFeatures: Int = 1000000

    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)
    val conf = new SparkConf().setAppName("BatchedBinaryClassification")
    val sc = new SparkContext(conf)
    // $example on$
    // Load training data in LIBSVM format

    val (numRegParams, dataFraction, batchSizeFraction, featuresFraction) = CriteoClassificationUtils.parseConfig(args)

    // TODO: Replace path of file here
    var data: RDD[LabeledPoint] = null
    if (featuresFraction < 1.0) {
      data = MLUtils.loadLibSVMFileLimitFeatures(sc,
        "/Users/jigyayadav/Desktop/UCSDAcads/Quarter5/CSE291F/spark.nosync/spark/data/mllib/sample_libsvm_data.txt",
        limitFeatures = true, maxFeaturesVal = (maxNumFeatures * featuresFraction).toInt)
    } else {
      data = MLUtils.loadLibSVMFile(sc,
        "/Users/jigyayadav/Desktop/UCSDAcads/Quarter5/CSE291F/spark.nosync/spark/data/mllib/sample_libsvm_data.txt")
    }

    // Modify the data according to the passed parameters
    val regParams = CriteoClassificationUtils.getRegParamValues(numRegParams)
    val labeledPointsRDDSample = data.sample(false, dataFraction, seed = 4)

    val Array(training, test) = labeledPointsRDDSample.randomSplit(Array(0.8, 0.2), seed = 11L)
    training.cache()

    val t0 = System.nanoTime()

    // Set different values of the regularization parameters here
    val lr = new LogisticRegressionWithSGD()
    lr.optimizer.setRegParams(regParams:_*)
    lr.optimizer.setMiniBatchFraction(batchSizeFraction)

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