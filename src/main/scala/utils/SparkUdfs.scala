package utils

import org.apache.spark.internal.Logging
import org.apache.spark.sql.SparkSession


object SparkUdfs extends Logging {

  // Write all your UDFs at one place

  // Sample udf to remove space in a string column, For demonstration only
  def removeSpaceUdf(input: String): String = {
    input.replaceAll(" ","")
  }

  // Registering udfs so that we can use in spark sql
  def registerUDFs()(implicit spark: SparkSession) : Unit = {
    try {
      spark.sqlContext.udf.register("removeSpace", removeSpaceUdf _)
      log.info("UDFs registered successfully")
    }
    catch {
      case e: Exception => log.error("Unable to register UDFs")
    }
  }

}
