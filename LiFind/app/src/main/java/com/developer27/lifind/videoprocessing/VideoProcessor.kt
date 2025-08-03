package com.developer27.lifind.videoprocessing

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
import org.opencv.imgproc.Imgproc
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.image.TensorImage
import java.io.File
import java.io.FileWriter
import kotlin.math.min
import kotlin.math.sqrt
import kotlin.math.pow
import kotlin.random.Random

import com.developer27.lifind.videoprocessing.YOLOHelper
import com.developer27.lifind.trilateration.Trilateration

// Holds the last set of LED distances: (classId, distance)
private var lastLedDistances: List<Pair<Int, Double>> = emptyList()

// Holds last trilaterated user position (x, y)
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

// Two different TFLite interpreters.
var yoloInterpreter: Interpreter? = null
var distanceInterpreter: Interpreter? = null

object Settings {
    object DetectionMode {
        enum class Mode { YOLO }
        var current: Mode = Mode.YOLO
        var enableYOLOinference = true
    }

    object Inference {
        var confidenceThreshold: Float = 0.3f  // adjust as needed
    }

    object BoundingBox {
        var enableBoundingBox = true
        var boxColor = Scalar(255.0, 255.0, 255.0)
        var boxThickness = 2
    }
}

class VideoProcessor(private val context: Context) {

    init {
        initOpenCV()
    }

    private fun initOpenCV() {
        try {
            System.loadLibrary("opencv_java4")
        } catch (e: UnsatisfiedLinkError) {
            Log.d("VideoProcessor","OpenCV failed to load: ${e.message}", e)
        }
    }

    fun setYoloInterpreter(model: Interpreter) = synchronized(this) {
        yoloInterpreter = model
        Log.d("VideoProcessor","YOLO Interpreter set")
    }

    fun setDistanceInterpreter(model: Interpreter) = synchronized(this) {
        distanceInterpreter = model
        Log.d("VideoProcessor","Distance Interpreter set")
    }

    // Return the last set of LED distances: (classId, distance). Use this in MainActivity
    fun getLastLedDistances(): List<Pair<Int, Double>> = lastLedDistances

    // Return the last trilaterated user position (x,y)
    fun getLastUserPosition(): Pair<Double, Double> = lastUserPosition

    fun processFrame(bitmap: Bitmap, callback: (Pair<Bitmap, Bitmap>?) -> Unit) {
        CoroutineScope(Dispatchers.Default).launch {
            val result: Pair<Bitmap, Bitmap>? = try {
                when (Settings.DetectionMode.current) {
                    Settings.DetectionMode.Mode.YOLO -> processFrameInternalYOLO(bitmap)
                }
            } catch (e: Exception) {
                Log.d("VideoProcessor","Error processing frame: ${e.message}", e)
                null
            }
            withContext(Dispatchers.Main) { callback(result) }
        }
    }

    /**
     * Runs YOLO-based detection and distance estimation on the given bitmap.
     * Also fills lastLedDistances after detection and computes trilateration!
     */
    private suspend fun processFrameInternalYOLO(
        bitmap: Bitmap
    ): Pair<Bitmap, Bitmap> = withContext(Dispatchers.IO) {

        val tag = javaClass.simpleName

        // 1) Get model input size/shape
        val (inputW, inputH, outputShape) = getModelDimensions()
        val (letterboxed, offsets) = YOLOHelper.createLetterboxedBitmap(bitmap, inputW, inputH)

        // Ensure letterboxed is Bitmap for TensorImage.load:
        val tensorImage = TensorImage(DataType.FLOAT32).apply { load(letterboxed) }

        val distTensorShape = distanceInterpreter
            ?.getOutputTensor(0)
            ?.shape()
            ?: intArrayOf(1, 1, YOLOHelper.numDistanceClasses)
        val distOut = Array(distTensorShape[1]) { FloatArray(distTensorShape[2]) }
        val m = Mat().also { Utils.bitmapToMat(bitmap, it) }
        var ledDistancesList = mutableListOf<Pair<Int, Double>>()   // For DA, DB, DC

        // --- Main detection ---
        if (Settings.DetectionMode.enableYOLOinference && yoloInterpreter != null) {
            val yoloOut = Array(outputShape[0]) {
                Array(outputShape[1]) { FloatArray(outputShape[2]) }
            }
            tensorImage.buffer.also { yoloInterpreter!!.run(it, yoloOut) }

            // Only keep above threshold
            val yoloDetections = YOLOHelper.parseTFLite(yoloOut)
                ?.filter { det -> det.confidence > Settings.Inference.confidenceThreshold }
                ?.distinctBy { det -> det.classId }
                ?.take(3)
                ?.sortedBy { det -> det.classId } // Ensure order

            yoloDetections?.forEach { det ->
                val (box, center) = YOLOHelper.rescaleInferencedCoordinates(
                    det, bitmap.width, bitmap.height, offsets, inputW, inputH
                )
                if (Settings.BoundingBox.enableBoundingBox) {
                    val yoloLabel = YOLOHelper.classNameForId(det.classId)
                    val labelText =
                        "$yoloLabel (${String.format("%.2f", det.confidence * 100)}%)"
                    YOLOHelper.drawDetectionCircleWithLabel(m, center, box, labelText)
                }
                // Calculate distance from image center (in pixels, adjust to your needs)
                val pixelDist = sqrt(
                    (center.x - bitmap.width / 2).pow(2) + (center.y - bitmap.height / 2).pow(2)
                )
                ledDistancesList.add(Pair(det.classId, pixelDist))
            }
        }

        // Save these for MainActivity to use after a process:
        lastLedDistances = ledDistancesList.sortedBy { it.first }.take(3)

        // Perform trilateration if we have 3 distances
        lastUserPosition = if (lastLedDistances.size == 3) {
            val dA = lastLedDistances[0].second
            val dB = lastLedDistances[1].second
            val dC = lastLedDistances[2].second
            Trilateration.solve(dA, dB, dC)
        } else {
            Pair(0.0, 0.0)
        }

        Log.d(tag, "Trilaterated user position: x=${lastUserPosition.first}, y=${lastUserPosition.second}")

        val outBmp = Bitmap
            .createBitmap(bitmap.width, bitmap.height, Bitmap.Config.ARGB_8888)
            .also { Utils.matToBitmap(m, it); m.release() }

        outBmp to letterboxed
    }

    fun getModelDimensions(): Triple<Int, Int, List<Int>> {
        val inTensor = yoloInterpreter?.getInputTensor(0)
        val inShape = inTensor?.shape()
        val h = inShape?.getOrNull(1) ?: 416
        val w = inShape?.getOrNull(2) ?: 416
        val outTensor = yoloInterpreter?.getOutputTensor(0)
        val outShape = outTensor?.shape()?.toList() ?: listOf(1, 1, 9)
        return Triple(w, h, outShape)
    }
}
