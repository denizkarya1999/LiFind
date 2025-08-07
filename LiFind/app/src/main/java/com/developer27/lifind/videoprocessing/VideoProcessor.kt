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
import kotlin.math.min

private fun Float.format(digits: Int) = "%.${digits}f".format(this)

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

var distanceInterpreter: Interpreter? = null

object Settings {
    object DetectionMode {
        enum class Mode { YOLO }
        var current: Mode = Mode.YOLO
    }
    object Inference {
        var confidenceThreshold: Float = 0.00000f
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

        // 1) Prepare model input: letterbox the camera frame
        val (inputW, inputH, outputShape) = getModelDimensions()
        val (letterboxed, offsets) =
            YOLOHelper.createLetterboxedBitmap(bitmap, inputW, inputH)
        val tensorImage = TensorImage(DataType.FLOAT32).apply { load(letterboxed) }

        // 2) Create an OpenCV Mat from the letterboxed image
        val m = Mat().also { Utils.bitmapToMat(letterboxed, it) }

        // 3) Temporary holders
        val ledDistancesList = mutableListOf<Pair<Int, Double>>()
        val ledCentersList   = mutableListOf<Pair<Int, Point>>()

// 4) Run the distance model, collect circles but don’t draw yet
        val circlesToDraw = mutableListOf<Triple<Point, Int, String>>()

        distanceInterpreter?.let { interpreter ->
            val distOut = Array(outputShape[0]) {
                Array(outputShape[1]) { FloatArray(outputShape[2]) }
            }
            interpreter.run(tensorImage.buffer, distOut)

            val bestPerLed = YOLOHelper.parseTFLite(distOut)
                ?.filter { it.confidence > Settings.Inference.confidenceThreshold }
                ?.groupBy { det ->
                    YOLOHelper.classNameForId(det.classId).substringBefore('_').toInt()
                }
                ?.map { (_, dets) -> dets.maxByOrNull { it.confidence }!! }
                ?: emptyList()

            bestPerLed
                .sortedByDescending { it.confidence }
                .take(3)
                .sortedBy { det ->
                    YOLOHelper.classNameForId(det.classId).substringBefore('_').toInt()
                }
                .forEach { det ->
                    val parts    = YOLOHelper.classNameForId(det.classId).split('_')
                    val ledId    = parts[0].toInt()
                    val distance = parts[1].toDouble()

                    val (center, radius) = YOLOHelper.rescaleToCenterAndRadius(
                        det,
                        letterboxed.width,
                        letterboxed.height,
                        offsets,
                        inputW,
                        inputH
                    )

                    Log.i(tag, "LED$ledId raw center = x=${"%.1f".format(center.x)}, y=${"%.1f".format(center.y)}, radius=$radius")

                    // 1) Rescale from model‐input → camera frame:
                    val (rawW, rawH) = bitmap.width.toDouble() to bitmap.height.toDouble()
                    val (inW, inH)   = inputW.toDouble()    to inputH.toDouble()
                    val scaleX       = rawW / inW
                    val scaleY       = rawH / inH

                    val rawX    = center.x * scaleX
                    val rawY    = center.y * scaleY
                    val rawRad  = (radius * min(scaleX, scaleY)).toInt()

                    // 2) Clamp to screen bounds
                    val clampedX = rawX.coerceAtLeast(0.0).coerceAtMost(rawW)
                    val clampedY = rawY.coerceAtLeast(0.0).coerceAtMost(rawH)

                    // 4) Buffer the fitted values
                    circlesToDraw += Triple(Point(clampedX, clampedY), 300, "LED$ledId")

                    // record for trilateration
                    ledCentersList   .add(ledId to center)
                    ledDistancesList .add(ledId to distance)
                }
        }

        // 5) Trilaterate user position
        lastLedDistances = ledDistancesList.sortedBy { it.first }.take(3)
        lastLedCenters   = ledCentersList  .sortedBy { it.first }.take(3)
        lastUserPosition = if (lastLedDistances.size == 3) {
            val (dA, dB, dC) = lastLedDistances.map { it.second }
            Trilateration.solve(dA, dB, dC)
        } else {
            0.0 to 0.0
        }
        Log.d(tag, "User position: x=${lastUserPosition.first}, y=${lastUserPosition.second}")

        // 6) Write only the user position to log
        try {
            val docsDir = Environment
                .getExternalStoragePublicDirectory(Environment.DIRECTORY_DOCUMENTS)
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

        // draw the detection circle & label
        if (Settings.BoundingBox.enableBoundingBox) {
            for ((center, radius, label) in circlesToDraw) {
                YOLOHelper.drawDetectionCircleWithLabel(
                    m, center, radius, label
                )
            }
        }

        // 7) Convert the letterboxed Mat (with circles) back into a Bitmap
        val outBmp = Bitmap.createBitmap(
            letterboxed.width,
            letterboxed.height,
            letterboxed.config
        ).also {
            Utils.matToBitmap(m, it)
            m.release()
        }

        outBmp to letterboxed
    }

    fun getModelDimensions(): Triple<Int, Int, List<Int>> {
        val inTensor  = distanceInterpreter?.getInputTensor(0)
        val shapeIn   = inTensor?.shape()
        val h         = shapeIn?.getOrNull(1) ?: 416
        val w         = shapeIn?.getOrNull(2) ?: 416
        val outTensor = distanceInterpreter?.getOutputTensor(0)
        val shapeOut  = outTensor?.shape()?.toList() ?: listOf(1, 1, 9)
        return Triple(w, h, shapeOut)
    }
}