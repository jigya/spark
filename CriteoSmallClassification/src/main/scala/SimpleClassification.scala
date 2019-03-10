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


object CriteoBinaryClassification {

  def main(args: Array[String]): Unit = {
    // This needs to be changed based on the dataset

    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)
    val conf = new SparkConf().setAppName("SequentialBinaryClassification")
    val sc = new SparkContext(conf)
    // $example on$
    // Load training data in LIBSVM format

    val (filename, numRegParams, dataFraction, batchSizeFraction, featuresFraction, maxNumFeatures) =
      CriteoClassificationUtils.parseConfig(args)

    // TODO: Replace path of file here
    var data: RDD[LabeledPoint] = null
    if (featuresFraction < 1.0) {
      data = MLUtils.loadLibSVMFileLimitFeatures(sc,
        filename,
        limitFeatures = true, maxFeaturesVal = (maxNumFeatures * featuresFraction).toInt)
    } else {
      data = MLUtils.loadLibSVMFile(sc, filename)
    }

    // Modify the data according to the passed parameters
    val regParams = CriteoClassificationUtils.getRegParamValues(numRegParams)
//    val regParams = Array(0.01)
    val labeledPointsRDDSample = data.sample(false, dataFraction, seed = 4)

    val Array(training, test) = labeledPointsRDDSample.randomSplit(Array(0.8, 0.2), seed = 11L)
    training.cache()

    var totalTime: Long = 0
    var totalTrainingTime: Long = 0

    for (currRegParam <- regParams) {
      val t0 = System.nanoTime()

      // Set different values of the regularization parameters here
      // Create algorithm instance
      val lr = new LogisticRegressionWithSGD()
      lr.optimizer.setRegParam(currRegParam)
      lr.optimizer.setMiniBatchFraction(batchSizeFraction)

      // Train model
      val model = lr.run(training)

      val t2 = System.nanoTime()

      println(s"Multinomial coefficients: ${model.weights}")
      println(s"Multinomial intercepts: ${model.intercept}")
      println(s"Time taken for training: ${t2 - t0}")
      totalTrainingTime += (t2 - t0)

      // Clear the prediction threshold so the model will return probabilities
      model.clearThreshold

      // Compute raw scores on the test set
      val predictionAndLabels = test.map { case LabeledPoint(label, features) =>
        val prediction = model.predict(features)
        (prediction, label)
      }

      // Instantiate metrics object
      val metrics = new BinaryClassificationMetrics(predictionAndLabels)

      // AUROC
      val auROC = metrics.areaUnderROC
      println(s"Area under ROC = $auROC" + s" with regularization = $currRegParam")

      val t1 = System.nanoTime()

      println("Elapsed time: " + (t1 - t0) + "ns")

      totalTime += (t1 - t0)
      // $example off$
    }
    println("Total time taken for all regularization values: " + totalTime + " " + totalTrainingTime)
    println("Size of training set: ", training.count())
    sc.stop()
  }
}