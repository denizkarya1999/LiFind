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
import org.opencv.imgproc.Imgproc
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.image.TensorImage
import java.io.File
import java.io.FileWriter
import java.io.IOException
import kotlin.math.max

data class DrawInfo(val x1: Int, val y1: Int, val x2: Int, val y2: Int, val label: String)

private fun Float.format(digits: Int) = "%.${digits}f".format(this)

// Holds the last set of LED distances: (classId, distance)
private var lastLedDistances: List<Pair<Int, Double>> = emptyList()

// Holds the last centers of each detected LED: (classId, centerPoint)
private var lastLedCenters: List<Pair<Int, Point>> = emptyList()

// Holds the last trilaterated user position
private var lastUserPosition: Pair<Double, Double> = 0.0 to 0.0

data class DetectionResult(
    val xCenter: Float,
    val yCenter: Float,
    val width: Float,
    val height: Float,
    val confidence: Float,
    val classId: Int
)

var distanceInterpreter: Interpreter? = null

object Settings {
    object DetectionMode {
        enum class Mode { YOLO }
        var current: Mode = Mode.YOLO
    }
    object Inference {
        var confidenceThreshold: Float = 0.00f
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
        try {
            System.loadLibrary("opencv_java4")
        } catch (e: UnsatisfiedLinkError) {
            Log.d("VideoProcessor", "OpenCV failed to load: ${e.message}", e)
        }
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

        val (inputW, inputH, outputShape) = getModelDimensions()
        val (letterboxed, offsets) = YOLOHelper.createLetterboxedBitmap(bitmap, inputW, inputH)
        val tensorImage = TensorImage(DataType.FLOAT32).apply { load(letterboxed) }

        // We'll draw on the letterboxed Mat, since that's what we feed the model
        val m = Mat().also { Utils.bitmapToMat(letterboxed, it) }

        val ledDistancesList = mutableListOf<Pair<Int, Double>>()
        val ledCentersList   = mutableListOf<Pair<Int, Point>>()

        val interpreter = distanceInterpreter
        if (interpreter == null) {
            Log.w(tag, "Interpreter is null; returning original frame.")
            val outBmp = Bitmap.createBitmap(letterboxed.width, letterboxed.height, letterboxed.config).also {
                Utils.matToBitmap(m, it); m.release()
            }
            return@withContext outBmp to letterboxed
        }

        // Run model
        val distOut = Array(outputShape[0]) { Array(outputShape[1]) { FloatArray(outputShape[2]) } }
        interpreter.run(tensorImage.buffer, distOut)

        // Parse -> keep best per LED id -> cap to 3
        val bestPerLed = YOLOHelper.parseTFLite(distOut)
            ?.filter { it.confidence >= Settings.Inference.confidenceThreshold }
            ?.groupBy { det -> YOLOHelper.classNameForId(det.classId).substringBefore('_').toInt() }
            ?.map { (_, dets) -> dets.maxByOrNull { it.confidence }!! }
            ?: emptyList()

        Log.d(tag, "Detections after grouping: ${bestPerLed.size}")

        val selected = bestPerLed
            .sortedByDescending { it.confidence }
            .take(3)
            .sortedBy { det -> YOLOHelper.classNameForId(det.classId).substringBefore('_').toInt() }

        val boxesToDraw = mutableListOf<DrawInfo>()

        selected.forEach { det ->
            val parts    = YOLOHelper.classNameForId(det.classId).split('_')
            val ledId    = parts[0].toInt()
            val distance = parts[1].toDouble()

            // Letterboxed coords (for drawing later)
            val xLb = det.xCenter * inputW
            val yLb = det.yCenter * inputH
            val wLb = det.width  * inputW
            val hLb = det.height * inputH

            // Original coords (for logging/trilateration)
            val (centerRaw, _) = YOLOHelper.rescaleToCenterAndRadius(
                det, bitmap.width, bitmap.height, offsets, inputW, inputH
            )

            // Log in original coords
            val classLabel = YOLOHelper.classNameForId(det.classId)
            Log.d(tag, "$classLabel: x=${centerRaw.x.toFloat().format(2)}, y=${centerRaw.y.toFloat().format(2)}")

            // Store for trilateration
            ledCentersList.add(ledId to centerRaw)
            ledDistancesList.add(ledId to distance)

            // Prepare draw info (Int px on the letterboxed canvas)
            val x1 = (xLb - wLb * 0.5f).toInt().coerceIn(0, inputW - 1)
            val y1 = (yLb - hLb * 0.5f).toInt().coerceIn(0, inputH - 1)
            val x2 = (xLb + wLb * 0.5f).toInt().coerceIn(0, inputW - 1)
            val y2 = (yLb + hLb * 0.5f).toInt().coerceIn(0, inputH - 1)
            boxesToDraw += DrawInfo(
                x1, y1, x2, y2,
                "LED = $ledId | Dist. = $distance"
            )

            // Keep for trilateration (original coords)
            ledCentersList.add(ledId to centerRaw)
            ledDistancesList.add(ledId to distance)
        }

        // === After all detections are processed, draw once (up to 3) ===
        if (Settings.BoundingBox.enableBoundingBox) {
            boxesToDraw.forEach { b ->
                Imgproc.rectangle(
                    m,
                    Point(b.x1.toDouble(), b.y1.toDouble()),
                    Point(b.x2.toDouble(), b.y2.toDouble()),
                    Settings.BoundingBox.boxColor,
                    max(2, Settings.BoundingBox.boxThickness),
                    Imgproc.LINE_AA,
                    0
                )
                Imgproc.putText(
                    m,
                    b.label,
                    Point(b.x1.toDouble(), (b.y1 - 6).coerceAtLeast(10).toDouble()),
                    Imgproc.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    Settings.BoundingBox.boxColor,
                    2,
                    Imgproc.LINE_AA,
                    false
                )
            }
        }

        // Trilaterate user position if we have 3
        lastLedDistances = ledDistancesList.sortedBy { it.first }.take(3)
        lastLedCenters   = ledCentersList.sortedBy { it.first }.take(3)
        lastUserPosition = if (lastLedDistances.size == 3) {
            val (dA, dB, dC) = lastLedDistances.map { it.second }
            Trilateration.solve(dA, dB, dC)
        } else 0.0 to 0.0

        Log.d(tag, "User position: x=${lastUserPosition.first}, y=${lastUserPosition.second}")

        // (Optional) write user position to a file (kept from your original)
        try {
            val docsDir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOCUMENTS)
            if (!docsDir.exists() && !docsDir.mkdirs()) {
                Log.e(tag, "Failed to create Documents directory")
            }
            val logFile = File(docsDir, "LiFind_Log.txt")
            if (logFile.exists()) logFile.delete()
            FileWriter(logFile, false).use { writer ->
                writer.append("UserPosition: x=${lastUserPosition.first}, y=${lastUserPosition.second}\n")
            }
            Log.d(tag, "Wrote user position to ${logFile.absolutePath}")
        } catch (e: IOException) {
            Log.e(tag, "Failed to write user position", e)
        }

        // Convert annotated Mat -> Bitmap to display
        val outBmp = Bitmap.createBitmap(letterboxed.width, letterboxed.height, letterboxed.config).also {
            Utils.matToBitmap(m, it); m.release()
        }
        outBmp to letterboxed
    }

    fun getModelDimensions(): Triple<Int, Int, List<Int>> {
        val inTensor  = distanceInterpreter?.getInputTensor(0)
        val shapeIn   = inTensor?.shape() // expected [1, H, W, 3]
        val h         = shapeIn?.getOrNull(1) ?: 416
        val w         = shapeIn?.getOrNull(2) ?: 416
        val outTensor = distanceInterpreter?.getOutputTensor(0)
        val shapeOut  = outTensor?.shape()?.toList() ?: listOf(1, 1, 9)
        return Triple(w, h, shapeOut)
    }
}
