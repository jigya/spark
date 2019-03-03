import org.apache.log4j.Logger
import org.apache.log4j.Level

import org.apache.spark.{SparkConf, SparkContext}
// $example on$
import org.apache.spark.mllib.classification.LogisticRegressionWithSGD
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
// $example off$

object SimpleBinaryClassification {

  def main(args: Array[String]): Unit = {

    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)
    val conf = new SparkConf().setAppName("SimpleBinaryClassification")
    val sc = new SparkContext(conf)
    // $example on$
    // Load training data in LIBSVM format
    val data = MLUtils.loadLibSVMFile(sc,
      "/Users/jigyayadav/Desktop/UCSDAcads/Quarter5/CSE291F/spark.nosync/spark/data/mllib/sample_libsvm_data.txt")

    // Split data into training (60%) and test (40%)
    val Array(training, test) = data.randomSplit(Array(0.6, 0.4), seed = 11L)
    training.cache()

    val t0 = System.nanoTime()

    // Create algorithm instance
    val lr = new LogisticRegressionWithSGD()
    lr.optimizer.setRegParam(0.01)

    // Train model
    val model = lr.run(training)

    // Clear the prediction threshold so the model will return probabilities
    model.clearThreshold

    // Compute raw scores on the test set
    val predictionAndLabels = test.map { case LabeledPoint(label, features) =>
      val prediction = model.predict(features)
      (prediction, label)
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

    val t1 = System.nanoTime()

    println("Elapsed time: " + (t1 - t0) + "ns")
    // $example off$
    sc.stop()
  }
}
// scalastyle:on println
