package sparkjobs


import org.apache.spark.SparkConf
import org.apache.spark.internal.Logging
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.sql.functions
import org.apache.spark.sql.functions.expr
import utils.RunMode.RunMode
// An assembler converts the input values to a vector
// A vector is what the ML algorithm reads to train a model
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.recommendation.ALS
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.regression.LinearRegressionModel
import org.apache.spark.mllib.regression.LinearRegressionWithSGD
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.streaming._
import org.apache.spark.sql.functions.{col, split, length}
import org.apache.spark.sql.SparkSession
import scala.io.Source
import utils.SparkFactory._
import utils.TestUtils.runModeFromOS


object PgaMLJob extends Logging {

  val runMode : RunMode = runModeFromOS()

  def main(args: Array[String]): Unit = {

    log.info("Creating spark session")
    val spark: SparkSession = createSparkSession(runMode, "PgaMlJob")
    import spark.implicits._
    // Prepare training and test data.
    val plainText = Source.fromResource("final_tour_data.csv").mkString
    val csvList = plainText.lines.toList
    val data = spark.read.option("header","true").option("inferSchema","true").csv(csvList.toDS()).drop("_c0")

    // Check out the Data
    data.printSchema()
    data.show(false)

    // A few things we need to do before Spark can accept the data!
    // It needs to be in the form of two columns
    // ("label","features")

    // This will allow us to join multiple feature columns
    // into a single column of an array of feature values

    // Rename to label column for naming convention.
    // Grab only numerical columns from the data
    val df = data.select(data("US_OPEN_TOP_10").as("label"),$"ROUNDS", $"SCORING1", $"SCORING2", $"SCORING3",
      $"SCORING4", $"SCORING5", $"DRIVE_DISTANCE", $"FWY_%1", $"FWY_%2", $"FWY_%3", $"FWY_%4", $"FWY_%5", $"GIR_%1",
      $"GIR_%2", $"GIR_%3", $"GIR_%4", $"GIR_%5", $"SG_P1", $"SG_P2", $"SG_P3", $"SG_P4", $"SG_P5", $"SG_TTG1",
      $"SG_TTG2", $"SG_TTG3", $"SG_TTG4", $"SG_TTG5", $"SG_T1", $"SG_T2", $"SG_T3", $"SG_T4", $"SG_T5",
      $"SCRAMBLING_P1", $"SCRAMBLING_P2", $"SCRAMBLING_P3", $"SCRAMBLING_P4", $"SCRAMBLING_P5", $"PAR OR BETTER",
      $"MISSED GIR", $"MISSED GIR", $"TOP 10", $"1ST")

    df.show(false)

    // Set the input columns from which we are supposed to read the values
    // Set the name of the column where the vector will be stored
    val featureColumns = Array("ROUNDS", "SCORING1", "SCORING2", "SCORING3",
      "SCORING4", "SCORING5", "DRIVE_DISTANCE", "FWY_%1", "FWY_%2", "FWY_%3", "FWY_%4", "FWY_%5", "GIR_%1",
      "GIR_%2", "GIR_%3", "GIR_%4", "GIR_%5", "SG_P1", "SG_P2", "SG_P3", "SG_P4", "SG_P5", "SG_TTG1",
      "SG_TTG2", "SG_TTG3", "SG_TTG4", "SG_TTG5", "SG_T1", "SG_T2", "SG_T3", "SG_T4", "SG_T5",
      "SCRAMBLING_P1", "SCRAMBLING_P2", "SCRAMBLING_P3", "SCRAMBLING_P4", "SCRAMBLING_P5", "PAR OR BETTER",
      "MISSED GIR", "MISSED GIR", "TOP 10", "1ST")
    val assembler = new VectorAssembler().setInputCols(featureColumns).setOutputCol("features").setHandleInvalid("skip")

    // Use the assembler to transform our DataFrame to the two columns
    val output = assembler.transform(df).select($"label",$"features")

    // Before modeling, we should split our data into train and test datasets where our train data has 80% of the whole data
    val Array(train, test) = output.randomSplit(Array(.8, .2), 42)  // TODO: look up seed for sampling

    // Linear Regression with Default Params
    // Create a Linear Regression Model object
    val lr = new LinearRegression()

    // Fit the model to the data
    val lrModel = lr.fit(train)

    // Print the coefficients and intercept for linear regression
    println(s"Coefficients: ${lrModel.coefficients} Intercept: ${lrModel.intercept}")

    // Summarize the model over the training set and print out some metrics!
    // Explore this in the spark-shell for more methods to call
    val trainingSummary = lrModel.summary
    println(s"numIterations (Default): ${trainingSummary.totalIterations}")
    println(s"objectiveHistory (Default): ${trainingSummary.objectiveHistory.toList}")
    trainingSummary.residuals.show()
    println(s"RMSE (Default): ${trainingSummary.rootMeanSquaredError}")
    println(s"MSE (Default): ${trainingSummary.meanSquaredError}")
    println(s"r2 (Default): ${trainingSummary.r2}")

    // Test model out
    // Resource: https://www.programcreek.com/scala/org.apache.spark.ml.evaluation.RegressionEvaluator
    val lrPredictions = lrModel.transform(test)
    lrPredictions.sort(col("prediction").desc_nulls_last).show()
    val evaluator = new RegressionEvaluator().setMetricName("rmse").setLabelCol("label").setPredictionCol("prediction")
    val rmse = evaluator.evaluate(lrPredictions)
    println(s"Root-mean-square error (Default) = $rmse")
    lrModel.write.overwrite().save("/Users/diegosierra/Box Sync/SparkPGA/pga-tour-ml/src/main/resources/lrModel")

    // Linear Regression with Elastic Net
    val lrWithElasticNet = new LinearRegression().setMaxIter(10).setRegParam(0.3).setElasticNetParam(0.5)
    val lrWithElasticModel = lrWithElasticNet.fit(train)
    val lrWithElasticPredictions = lrWithElasticModel.transform(test)
    lrWithElasticPredictions.sort(col("prediction").desc_nulls_last).show()
    val rmseWithElastic = evaluator.evaluate(lrWithElasticPredictions)
    println(s"Root-mean-square error (With Elastic) = $rmseWithElastic")
    lrWithElasticModel.write.overwrite().save("/Users/diegosierra/Box Sync/SparkPGA/pga-tour-ml/src/main/resources/lrModelWithElastic")

    val finalPredictions = lrWithElasticModel.transform(output)
    finalPredictions.sort(col("prediction").desc_nulls_last).show()
    val meta: org.apache.spark.sql.types.Metadata = lrWithElasticPredictions
      .schema(lrWithElasticPredictions.schema.fieldIndex("features"))
      .metadata
    print(meta.getMetadata("ml_attr").getMetadata("attrs"))
    // 2016 - 2021 Data
    // Most likely to be in Top 10 for any given tournament
    // 1. Dustin Johnson
    // 2. Justin Thomas
    // 3. Rory McIlroy
    // 4. Tony Finau
    // 5. Webb Simpson
    // 6. Chez Reavie
    // 7. Louis Oosthuizen
    // 8. Kevin Streelman
    // 9. Russell Knox
    // 10. Zach Johnson
  }

}