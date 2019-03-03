// import org.apache.log4j.Logger
// import org.apache.log4j.Level

// import org.apache.spark.{SparkConf, SparkContext}
// // $example on$
// import org.apache.spark.mllib.classification.LogisticRegressionWithSGD
// import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
// import org.apache.spark.mllib.regression.LabeledPoint
// import org.apache.spark.mllib.util.MLUtils
// // $example off$

// object BatchBinaryClassification {

//   def main(args: Array[String]): Unit = {

//     Logger.getLogger("org").setLevel(Level.OFF)
//     Logger.getLogger("akka").setLevel(Level.OFF)
//     val conf = new SparkConf().setAppName("BatchBinaryClassification")
//     val sc = new SparkContext(conf)
//     // $example on$
//     // Load training data in LIBSVM format
//     val data = MLUtils.loadLibSVMFile(sc,
//       "/Users/jigyayadav/Desktop/UCSDAcads/Quarter5/CSE291F/spark.nosync/spark/data/mllib/sample_libsvm_data.txt")

//     // Split data into training (60%) and test (40%)
//     val Array(training, test) = data.randomSplit(Array(0.6, 0.4), seed = 11L)
//     training.cache()

//     // Create algorithm instance
//     val lr = new LogisticRegressionWithSGD()
//     lr.optimizer.setRegParams(0.01, 0.1)

//     // Train model
//     val model = lr.run(training)

//     // Clear the prediction threshold so the model will return probabilities
//     model.clearThreshold

//     // Compute raw scores on the test set
//     val predictionsAndLabels = test.map { case LabeledPoint(label, features) =>
//       val predictions = model.predict(features)
//       (predictions, label)
//     }

//     // predictions.foreach{pred => 
//     //   val predictionAndLabel = (pred, label)
//     //   // Instantiate metrics object
//     //   val metrics = new BinaryClassificationMetrics(predictionAndLabels)

//     //   // Precision by threshold
//     //   val precision = metrics.precisionByThreshold
//     //   precision.foreach { case (t, p) =>
//     //     println(s"Threshold: $t, Precision: $p")
//     //   }

//     //   // Recall by threshold
//     //   val recall = metrics.recallByThreshold
//     //   recall.foreach { case (t, r) =>
//     //     println(s"Threshold: $t, Recall: $r")
//     //   }

//     //   // Precision-Recall Curve
//     //   val PRC = metrics.pr

//     //   // F-measure
//     //   val f1Score = metrics.fMeasureByThreshold
//     //   f1Score.foreach { case (t, f) =>
//     //     println(s"Threshold: $t, F-score: $f, Beta = 1")
//     //   }

//     //   val beta = 0.5
//     //   val fScore = metrics.fMeasureByThreshold(beta)
//     //   f1Score.foreach { case (t, f) =>
//     //     println(s"Threshold: $t, F-score: $f, Beta = 0.5")
//     //   }

//     //   // AUPRC
//     //   val auPRC = metrics.areaUnderPR
//     //   println(s"Area under precision-recall curve = $auPRC")

//     //   // Compute thresholds used in ROC and PR curves
//     //   val thresholds = precision.map(_._1)

//     //   // ROC Curve
//     //   val roc = metrics.roc

//     //   // AUROC
//     //   val auROC = metrics.areaUnderROC
//     //   println(s"Area under ROC = $auROC")
//     //   // $example off$
//     // }
//     sc.stop()
//   }
// }
// // scalastyle:on println

import org.apache.log4j.Logger
import org.apache.log4j.Level

import org.apache.spark.{SparkConf, SparkContext}
// $example on$
import org.apache.spark.mllib.classification.LogisticRegressionWithSGD
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
// $example off$

object BatchBinaryClassification {

  def main(args: Array[String]): Unit = {

    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)
    val conf = new SparkConf().setAppName("BatchBinaryClassification")
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
    lr.optimizer.setRegParams(0.01, 0.1, 0.03)

    // Train model
    val model = lr.run(training)

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

    sc.stop()
  }
}
// scalastyle:on println

