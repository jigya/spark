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
import org.apache.spark.sql.Row
import org.apache.spark.sql.types.{DoubleType, IntegerType}

// Different regularization parameters
// Different data sizes
// Different batch sizes
// Different number of features
// Different number of workers

object CriteoClassificationUtils {


  // The expected order of the passed parameters is
  // 1. Number of regularization parameter
  // 2. Fraction of the data to be used
  // 3. Fraction of the mini batch sizes
  // 4. Fraction of the features to be used
  def parseConfig(args: Array[String]) : (String, Int, Double, Double, Double, Int) = {
    var filePath: String =
      "/Users/jigyayadav/Desktop/UCSDAcads/Quarter5/CSE291F/spark.nosync/spark/data/mllib/sample_libsvm_data.txt"
    var numRegParams: Int = 5
    var dataFraction = 1.0
    var batchSizeFraction = 1.0
    var featuresFraction = 1.0
    var maxNumFeatures: Int = 1000000
    if (args.size > 0) {
      filePath = args(0).toString
    }
    if (args.size > 1) {
      numRegParams = args(1).toInt
    }
    if (args.size > 2) {
      dataFraction = args(2).toDouble
    }
    if (args.size > 3) {
      batchSizeFraction = args(3).toDouble
    }
    if (args.size > 4) {
      featuresFraction = args(4).toDouble
    }
    if (args.size > 5) {
      maxNumFeatures = args(5).toInt
    }
    println(s"Path of SVM file: $filePath")
    println(s"Number of regularization parameters: $numRegParams")
    println(s"Fraction of data being used: $dataFraction")
    println(s"Fraction for batch size being used: $batchSizeFraction")
    println(s"Fraction of features being used: $featuresFraction")
    println(s"Max number of features in the dataset: $maxNumFeatures")
    (filePath, numRegParams, dataFraction, batchSizeFraction, featuresFraction, maxNumFeatures)
  }

  // 50 values
  val sampleRegParams: Array[Double] = Array(0.01, 0.00623933, 0.00570339, 0.00574213, 0.00991926, 0.00575676,
  0.00648555, 0.00733108, 0.00012156, 0.00677153, 0.00554652, 0.04468551, 0.08963008, 0.02420132, 0.01055294, 0.02559068,
  0.00666728, 0.06396256, 0.04156763, 0.00247037, 0.039142, 0.04058291, 0.68601099, 0.40030564, 0.27374844, 0.03490986,
  0.31599488, 0.79967284, 0.097162  , 0.53898993, 0.17559417,
  0.22785145, 0.90312079, 0.57249303, 0.73986731, 0.37275708,
  0.45483588, 0.61873045, 0.47016149, 0.20316651, 0.25278749, 1.77981949, 8.3491931 , 9.744466  , 4.37778479, 6.34651022,
  5.31860376, 3.70928676, 7.15512508, 2.33975458)

  def getRegParamValues(num: Int): Array[Double] = {
    val chosenRegParams: ArrayBuffer[Double] = ArrayBuffer[Double]()

    val jumpValue: Int = (sampleRegParams.size / num).toInt
    for (i <- 0 until sampleRegParams.size by jumpValue) {
      chosenRegParams += sampleRegParams(i)
    }

    return chosenRegParams.toArray.slice(0, num)
  }
}



