package com.developer27.lifind.videoprocessing

import Trilateration
import android.content.Context
import android.graphics.Bitmap
import android.os.Environment
import android.util.Log
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.opencv.android.Utils
import org.opencv.core.Mat
import org.opencv.core.Point
import org.opencv.core.Scalar
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.image.TensorImage
import java.io.File
import java.io.FileWriter
import java.io.IOException
import kotlin.math.pow
import kotlin.math.sqrt

// Holds the last set of LED distances: (classId, distance)
private var lastLedDistances: List<Pair<Int, Double>> = emptyList()

// Holds the last centers of each detected LED: (classId, centerPoint)
private var lastLedCenters: List<Pair<Int, Point>> = emptyList()

// Holds the last trilaterated user position
private var lastUserPosition: Pair<Double, Double> = Pair(0.0, 0.0)

data class DetectionResult(
    val xCenter: Float,
    val yCenter: Float,
    val width: Float,
    val height: Float,
    val confidence: Float,
    val classId: Int
)
data class BoundingBox(
    val x1: Float,
    val y1: Float,
    val x2: Float,
    val y2: Float,
    val confidence: Float,
    val classId: Int
)

var yoloInterpreter: Interpreter? = null
var distanceInterpreter: Interpreter? = null

object Settings {
    object DetectionMode {
        enum class Mode { YOLO }
        var current: Mode = Mode.YOLO
        var enableYOLOinference = true
    }
    object Inference {
        var confidenceThreshold: Float = 0.3f
    }
    object BoundingBox {
        var enableBoundingBox = true
        var boxColor = Scalar(255.0, 255.0, 255.0)
        var boxThickness = 2
    }
}

class VideoProcessor(private val context: Context) {
    // Accessors
    fun getLastLedDistances(): List<Pair<Int, Double>> = lastLedDistances
    fun getLastLedCenters(): List<Pair<Int, Point>> = lastLedCenters
    fun getLastUserPosition(): Pair<Double, Double> = lastUserPosition


    init {
        initOpenCV()
    }

    private fun initOpenCV() {
        try {
            System.loadLibrary("opencv_java4")
        } catch (e: UnsatisfiedLinkError) {
            Log.d("VideoProcessor", "OpenCV failed to load: ${e.message}", e)
        }
    }

    fun setYoloInterpreter(model: Interpreter) = synchronized(this) {
        yoloInterpreter = model
        Log.d("VideoProcessor", "YOLO Interpreter set")
    }

    fun setDistanceInterpreter(model: Interpreter) = synchronized(this) {
        distanceInterpreter = model
        Log.d("VideoProcessor", "Distance Interpreter set")
    }

    fun processFrame(bitmap: Bitmap, callback: (Pair<Bitmap, Bitmap>?) -> Unit) {
        CoroutineScope(Dispatchers.Default).launch {
            val result: Pair<Bitmap, Bitmap>? = try {
                when (Settings.DetectionMode.current) {
                    Settings.DetectionMode.Mode.YOLO -> processFrameInternalYOLO(bitmap)
                }
            } catch (e: Exception) {
                Log.d("VideoProcessor", "Error processing frame: ${e.message}", e)
                null
            }
            withContext(Dispatchers.Main) { callback(result) }
        }
    }

    private suspend fun processFrameInternalYOLO(
        bitmap: Bitmap
    ): Pair<Bitmap, Bitmap> = withContext(Dispatchers.IO) {

        val tag = javaClass.simpleName

        // Prepare model input
        val (inputW, inputH, outputShape) = getModelDimensions()
        val (letterboxed, offsets) =
            YOLOHelper.createLetterboxedBitmap(bitmap, inputW, inputH)
        val tensorImage = TensorImage(DataType.FLOAT32).apply { load(letterboxed) }
        val m = Mat().also { Utils.bitmapToMat(bitmap, it) }

        // Temporary lists
        val ledDistancesList = mutableListOf<Pair<Int, Double>>()
        val ledCentersList   = mutableListOf<Pair<Int, Point>>()

        // Run YOLO inference
        if (Settings.DetectionMode.enableYOLOinference && yoloInterpreter != null) {
            val yoloOut = Array(outputShape[0]) {
                Array(outputShape[1]) { FloatArray(outputShape[2]) }
            }
            tensorImage.buffer.also { yoloInterpreter!!.run(it, yoloOut) }

            val yoloDetections = YOLOHelper.parseTFLite(yoloOut)
                ?.filter { it.confidence > Settings.Inference.confidenceThreshold }
                ?.distinctBy { it.classId }
                ?.take(3)
                ?.sortedBy { it.classId }

            yoloDetections?.forEach { det ->
                val (box, center) = YOLOHelper.rescaleInferencedCoordinates(
                    det, bitmap.width, bitmap.height, offsets, inputW, inputH
                )
                ledCentersList.add(Pair(det.classId, center))

                if (Settings.BoundingBox.enableBoundingBox) {
                    val label = "${YOLOHelper.classNameForId(det.classId)} " +
                            "(${String.format("%.2f", det.confidence * 100)}%)"
                    YOLOHelper.drawDetectionCircleWithLabel(m, center, box, label)
                }

                val pixelDist = sqrt(
                    (center.x - bitmap.width / 2).pow(2) +
                            (center.y - bitmap.height / 2).pow(2)
                )
                ledDistancesList.add(Pair(det.classId, pixelDist))
            }
        }

        // Save latest lists
        lastLedDistances = ledDistancesList.sortedBy { it.first }.take(3)
        lastLedCenters   = ledCentersList  .sortedBy { it.first }.take(3)

        // Trilaterate
        lastUserPosition = if (lastLedDistances.size == 3) {
            val (dA, dB, dC) = lastLedDistances.map { it.second }
            Trilateration.solve(dA, dB, dC)
        } else {
            Pair(0.0, 0.0)
        }

        Log.d(tag, "User position: x=${lastUserPosition.first}, y=${lastUserPosition.second}")

        // Write to public Documents directory, clearing any old log first
        try {
            val docsDir = Environment
                .getExternalStoragePublicDirectory(Environment.DIRECTORY_DOCUMENTS)
            if (!docsDir.exists() && !docsDir.mkdirs()) {
                Log.e(tag, "Failed to create Documents directory")
            }
            val logFile = File(docsDir, "LiFind_Log.txt")

            // Clear any existing file
            if (logFile.exists()) {
                logFile.delete()
            }

            FileWriter(logFile, /*append=*/false).use { writer ->
                // User position
                writer.append("UserPosition: x=${lastUserPosition.first}, y=${lastUserPosition.second}\n")

                // LED centers
                lastLedCenters.forEach { (id, pt) ->
                    writer.append("LED$id Center: x=${pt.x}, y=${pt.y}\n")
                }

                // LED distances
                lastLedDistances.forEach { (id, dist) ->
                    writer.append("LED$id Distance: ${"%.2f".format(dist)}\n")
                }

                writer.append("\n")
            }
            Log.d(tag, "Wrote new log to ${logFile.absolutePath}")
        } catch (e: IOException) {
            Log.e(tag, "Failed to write log to Documents", e)
        }

        val outBmp = Bitmap.createBitmap(
            bitmap.width, bitmap.height, Bitmap.Config.ARGB_8888
        ).also { Utils.matToBitmap(m, it); m.release() }

        outBmp to letterboxed
    }

    fun getModelDimensions(): Triple<Int, Int, List<Int>> {
        val inTensor = yoloInterpreter?.getInputTensor(0)
        val shapeIn  = inTensor?.shape()
        val h = shapeIn?.getOrNull(1) ?: 416
        val w = shapeIn?.getOrNull(2) ?: 416
        val outTensor = yoloInterpreter?.getOutputTensor(0)
        val shapeOut = outTensor?.shape()?.toList() ?: listOf(1, 1, 9)
        return Triple(w, h, shapeOut)
    }
}
