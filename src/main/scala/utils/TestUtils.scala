package utils

import org.apache.spark.internal.Logging

object TestUtils extends Logging {

  def runModeFromOS(): RunMode.Value = {

    val TESTSYSTEMS_RE = "(Mac OS X|Windows).*".r

    System.getProperty("os.name") match {
      case TESTSYSTEMS_RE(_) => {
        log.info("Mac OS Found - Running in UNIT_TEST mode")
        RunMode.UNIT_TEST
      }
      case default => {
        log.info(s"$default Found - Running in PRODUCTION mode")
        RunMode.PRODUCTION
      }
    }
  }
}
