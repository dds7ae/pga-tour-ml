package sparkjobs


import org.apache.spark.internal.Logging
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.sql.functions.{col, split}
import org.apache.spark.sql.SparkSession
import java.io.InputStream
import scala.io.Source
import utils.RunMode.RunMode
import utils.SparkFactory._
import utils.TestUtils.runModeFromOS


// An assembler converts the input values to a vector
// A vector is what the ML algorithm reads to train a model
import org.apache.spark.ml.feature.VectorAssembler

object SampleMLJob extends Logging {

  val runMode : RunMode = runModeFromOS()

  def main(args: Array[String]): Unit = {

    log.info("Creating spark session")
    val spark: SparkSession = createSparkSession(runMode, "SampleMLJob")
    import spark.implicits._
    // Prepare training and test data.
    val plainText = Source.fromResource("pga_tour_data.csv").mkString
    val csvList = plainText.lines.toList
    val data = spark.read.option("header","true").option("inferSchema","true").csv(csvList.toDS()).drop("_c0")

    // Check out the Data
    data.printSchema()
    data.show(false)
    // See an example of what the data looks like
    // by printing out a Row
    val colnames = data.columns
    val firstrow = data.head(1)(0)
    println("\n")
    println("Example Data Row")
    for(ind <- Range(1,colnames.length)){
      println(colnames(ind))
      println(firstrow(ind))
      println("\n")
    }


    // A few things we need to do before Spark can accept the data!
    // It needs to be in the form of two columns
    // ("label","features")

    // This will allow us to join multiple feature columns
    // into a single column of an array of feature values

    // Rename to label column for naming convention.
    // Grab only numerical columns from the data
    val df = data.select(data("US_OPEN_TOP_10").as("label"),$"ROUNDS",$"SCORING",$"DRIVE_DISTANCE",$"FWY_%",
      $"GIR_%",$"SG_P",$"SG_TTG",$"SG_T",$"SCRAMBLING_P",$"PAR OR BETTER",$"MISSED GIR",$"MISSED GIR",$"POINTS",
      $"TOP 10",$"1ST")

    df.show(false)
    // Set the input columns from which we are supposed to read the values
    // Set the name of the column where the vector will be stored
    val featureColumns = Array("ROUNDS","SCORING","DRIVE_DISTANCE","FWY_%","GIR_%","SG_P","SG_TTG","SG_T",
      "SCRAMBLING_P","PAR OR BETTER","MISSED GIR","MISSED GIR","POINTS","TOP 10","1ST")
    val assembler = new VectorAssembler().setInputCols(featureColumns).setOutputCol("features").setHandleInvalid("skip")

    // Use the assembler to transform our DataFrame to the two columns
    val output = assembler.transform(df).select($"label",$"features")

    // Before modeling, we should split our data into train and test datasets where our train data has 80% of the whole data
    val Array(train, test) = df.randomSplit(Array(.8, .2), 42)   // TODO: look up seed for sampling

    // Create a Linear Regression Model object
    val lr = new LinearRegression()

    // Fit the model to the data
    val lrModel = lr.fit(train)

    // Print the coefficients and intercept for linear regression
    println(s"Coefficients: ${lrModel.coefficients} Intercept: ${lrModel.intercept}")

    // Summarize the model over the training set and print out some metrics!
    // Explore this in the spark-shell for more methods to call
    val trainingSummary = lrModel.summary

    println(s"numIterations: ${trainingSummary.totalIterations}")
    println(s"objectiveHistory: ${trainingSummary.objectiveHistory.toList}")

    trainingSummary.residuals.show()

    println(s"RMSE: ${trainingSummary.rootMeanSquaredError}")
    println(s"MSE: ${trainingSummary.meanSquaredError}")
    println(s"r2: ${trainingSummary.r2}")

    // Test model out
    val lrPredictions = lrModel.transform(test)

    // TODO: Evaluate lrPredictions
  }

}