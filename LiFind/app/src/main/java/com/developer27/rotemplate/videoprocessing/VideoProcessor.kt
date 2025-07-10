package com.developer27.lifind.videoprocessing

import android.content.Context
import android.graphics.Bitmap
import android.os.SystemClock
import android.util.Log
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.opencv.android.Utils
import org.opencv.core.Core
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint
import org.opencv.core.Point
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.image.TensorImage
import kotlin.math.min

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
        enum class Mode { CONTOUR, YOLO }
        var current: Mode = Mode.YOLO
        var enableYOLOinference = true
    }

    object Inference {
        var confidenceThreshold: Float = 0.9f
    }

    object BoundingBox {
        var enableBoundingBox = true
        var boxColor = Scalar(0.0, 39.0, 76.0)
        var boxThickness = 2
    }

    object Brightness {
        var factor = 2.0
        var threshold = 150.0
    }
}

// Main VideoProcessor class.
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

    fun processFrame(bitmap: Bitmap, callback: (Pair<Bitmap, Bitmap>?) -> Unit) {
        CoroutineScope(Dispatchers.Default).launch {
            val result: Pair<Bitmap, Bitmap>? = try {
                when (Settings.DetectionMode.current) {
                    Settings.DetectionMode.Mode.CONTOUR -> processFrameInternalCONTOUR(bitmap)
                    Settings.DetectionMode.Mode.YOLO -> processFrameInternalYOLO(bitmap)
                }
            } catch (e: Exception) {
                Log.d("VideoProcessor","Error processing frame: ${e.message}", e)
                null
            }
            withContext(Dispatchers.Main) { callback(result) }
        }
    }

    private fun processFrameInternalCONTOUR(bitmap: Bitmap): Pair<Bitmap, Bitmap>? {
        return try {
            val (pMat, pBmp) = Preprocessing.preprocessFrame(bitmap)
            val (_, cMat) = ContourDetection.processContourDetection(pMat)
            val outBmp = Bitmap.createBitmap(cMat.cols(), cMat.rows(), Bitmap.Config.ARGB_8888)
                .also { Utils.matToBitmap(cMat, it) }
            pMat.release()
            cMat.release()
            outBmp to pBmp
        } catch (e: Exception) {
            Log.d("VideoProcessor","Error processing frame: ${e.message}", e)
            null
        }
    }

    private suspend fun processFrameInternalYOLO(
        bitmap: Bitmap
    ): Pair<Bitmap, Bitmap> = withContext(Dispatchers.IO) {

        val tag = javaClass.simpleName

        val (inputW, inputH, outputShape) = getModelDimensions()
        val (letterboxed, offsets) =
            YOLOHelper.createLetterboxedBitmap(bitmap, inputW, inputH)
        val tensorImage = TensorImage(DataType.FLOAT32).apply { load(letterboxed) }

        val distTensorShape = distanceInterpreter
            ?.getOutputTensor(0)
            ?.shape()
            ?: intArrayOf(1, 1, YOLOHelper.numDistanceClasses)

        val distOut = Array(distTensorShape[1]) {
            FloatArray(distTensorShape[2])
        }

        val m = Mat().also { Utils.bitmapToMat(bitmap, it) }
        var bestDistanceLabel = "Unknown"
        distanceInterpreter?.let { interp ->
            val t0 = SystemClock.elapsedRealtimeNanos()
            interp.run(tensorImage.buffer, arrayOf(distOut))
            val ms = (SystemClock.elapsedRealtimeNanos() - t0) / 1_000_000
            Log.d(tag, "Distance inference time: ${ms} ms")

            val scores = distOut[0]

            // --- Sort indices by score descending ---
            val sortedScores = scores
                .mapIndexed { idx, score -> idx to score }
                .filter { it.first < YOLOHelper.numDistanceClasses }
                .sortedByDescending { it.second }

            // Log only the highest‐confidence class, if any
            sortedScores.firstOrNull()?.let { (idx, score) ->
                val name = YOLOHelper.labelForDistanceIdx(idx)
                Log.d(tag, "Top Distance[$name] = ${"%.2f".format(score * 100)}%")
            }

            // Safely grab the best index, or -1 if the list is empty
            val bestIdx = sortedScores.firstOrNull()?.first ?: -1
            if (bestIdx >= 0) {
                bestDistanceLabel = YOLOHelper.labelForDistanceIdx(bestIdx)
            }
        } ?: Log.w(tag, "Distance interpreter not initialised")

        if (Settings.DetectionMode.enableYOLOinference && yoloInterpreter != null) {
            val yoloOut = Array(outputShape[0]) {
                Array(outputShape[1]) { FloatArray(outputShape[2]) }
            }
            tensorImage.buffer.also { yoloInterpreter!!.run(it, yoloOut) }

            YOLOHelper.parseTFLite(yoloOut)
                ?.distinctBy { it.classId }
                ?.take(3)
                ?.forEach { det ->
                    val (box, _) = YOLOHelper.rescaleInferencedCoordinates(
                        det, bitmap.width, bitmap.height, offsets, inputW, inputH
                    )
                    if (Settings.BoundingBox.enableBoundingBox) {
                        val yoloLabel = YOLOHelper.classNameForId(det.classId)
                        val labelText = "$yoloLabel | $bestDistanceLabel (${("%.2f".format(det.confidence * 100))}%)"
                        YOLOHelper.drawBoundingBoxesWithCustomLabel(m, box, labelText)
                    }
                }
        }

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

object Preprocessing {
    fun preprocessFrame(src: Bitmap): Pair<Mat, Bitmap> {
        val sMat = Mat().also { Utils.bitmapToMat(src, it) }
        val gMat = Mat().also {
            Imgproc.cvtColor(sMat, it, Imgproc.COLOR_BGR2GRAY)
            sMat.release()
        }
        val eMat = Mat().also {
            Core.multiply(gMat, Scalar(Settings.Brightness.factor), it)
            gMat.release()
        }
        val tMat = Mat().also {
            Imgproc.threshold(
                eMat,
                it,
                Settings.Brightness.threshold,
                255.0,
                Imgproc.THRESH_TOZERO
            )
            eMat.release()
        }
        val bMat = Mat().also {
            Imgproc.GaussianBlur(tMat, it, Size(5.0, 5.0), 0.0)
            tMat.release()
        }
        val k = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(3.0, 3.0))
        val cMat = Mat().also {
            Imgproc.morphologyEx(bMat, it, Imgproc.MORPH_CLOSE, k)
            bMat.release()
        }
        val bmp = Bitmap.createBitmap(cMat.cols(), cMat.rows(), Bitmap.Config.ARGB_8888).also {
            Utils.matToBitmap(cMat, it)
        }
        return cMat to bmp
    }
}

object ContourDetection {

    fun processContourDetection(mat: Mat): Pair<Point?, Mat> {
        val biggestContour = findContours(mat).maxByOrNull { Imgproc.contourArea(it) }
        val center = biggestContour?.let {
            Imgproc.drawContours(mat, listOf(it), -1, Settings.BoundingBox.boxColor, Settings.BoundingBox.boxThickness)
            val m = Imgproc.moments(it)
            Point(m.m10 / m.m00, m.m01 / m.m00)
        }
        Imgproc.cvtColor(mat, mat, Imgproc.COLOR_GRAY2BGR)
        return center to mat
    }

    private fun findContours(mat: Mat): MutableList<MatOfPoint> {
        val contours = mutableListOf<MatOfPoint>()
        val hierarchy = Mat()
        Imgproc.findContours(mat, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE)
        hierarchy.release()
        return contours
    }
}

object YOLOHelper {
    private val classNames = arrayOf("1000", "1001", "1010")

    val distanceLabels = arrayOf(
        "6_10","6_11","6_12","6_2","6_3","6_4","6_5","6_6","6_7","6_8","6_9",
        "7_10","7_11","7_12","7_2","7_3","7_4","7_5","7_6","7_7","7_8","7_9",
        "8_10","8_11","8_12","8_2","8_3","8_4","8_5","8_6","8_7","8_8","8_9"
    )
    val numDistanceClasses: Int get() = distanceLabels.size

    fun labelForDistanceIdx(idx: Int): String =
        if (idx in distanceLabels.indices) distanceLabels[idx] else "Unknown"

    fun classNameForId(classId: Int): String =
        if (classId in classNames.indices) classNames[classId] else "Unknown"

    fun parseTFLite(rawOutput: Array<Array<FloatArray>>): List<DetectionResult>? {
        val predictions = rawOutput[0]
        val detections = mutableListOf<DetectionResult>()
        for (row in predictions) {
            if (row.size < 6) continue
            val xCenter    = row[0]
            val yCenter    = row[1]
            val width      = row[2]
            val height     = row[3]
            val objectConf = row[4]
            val classScores= row.copyOfRange(5, row.size)
            val bestClassIdx   = classScores.indices.maxByOrNull { classScores[it] } ?: -1
            val bestClassScore = classScores.getOrNull(bestClassIdx) ?: 0f
            val finalConf      = objectConf * bestClassScore
            if (finalConf >= Settings.Inference.confidenceThreshold) {
                detections += DetectionResult(
                    xCenter, yCenter, width, height,
                    finalConf, bestClassIdx
                )
            }
        }
        return if (detections.isEmpty()) null else detections
    }

    private fun detectionToBox(d: DetectionResult) = BoundingBox(
        x1 = d.xCenter - d.width / 2,
        y1 = d.yCenter - d.height / 2,
        x2 = d.xCenter + d.width / 2,
        y2 = d.yCenter + d.height / 2,
        confidence = d.confidence,
        classId = d.classId
    )

    fun rescaleInferencedCoordinates(
        detection: DetectionResult,
        originalWidth: Int,
        originalHeight: Int,
        padOffsets: Pair<Int, Int>,
        modelInputWidth: Int,
        modelInputHeight: Int
    ): Pair<BoundingBox, Point> {
        val scale = min(
            modelInputWidth / originalWidth.toDouble(),
            modelInputHeight / originalHeight.toDouble()
        )
        val padLeft = padOffsets.first.toDouble()
        val padTop = padOffsets.second.toDouble()

        val xCenterLetterboxed = detection.xCenter * modelInputWidth
        val yCenterLetterboxed = detection.yCenter * modelInputHeight
        val boxWidthLetterboxed = detection.width * modelInputWidth
        val boxHeightLetterboxed = detection.height * modelInputHeight

        val xCenterOriginal = (xCenterLetterboxed - padLeft) / scale
        val yCenterOriginal = (yCenterLetterboxed - padTop) / scale
        val boxWidthOriginal = boxWidthLetterboxed / scale
        val boxHeightOriginal = boxHeightLetterboxed / scale

        val x1Original = xCenterOriginal - (boxWidthOriginal / 2)
        val y1Original = yCenterOriginal - (boxHeightOriginal / 2)
        val x2Original = xCenterOriginal + (boxWidthOriginal / 2)
        val y2Original = yCenterOriginal + (boxHeightOriginal / 2)

        Log.d("YOLOTest",
            "Adjusted BOX: x1=${"%.2f".format(x1Original)}, y1=${"%.2f".format(y1Original)}, " +
                    "x2=${"%.2f".format(x2Original)}, y2=${"%.2f".format(y2Original)}"
        )

        val boundingBox = BoundingBox(
            x1 = x1Original.toFloat(),
            y1 = y1Original.toFloat(),
            x2 = x2Original.toFloat(),
            y2 = y2Original.toFloat(),
            confidence = detection.confidence,
            classId = detection.classId
        )
        val center = Point(xCenterOriginal, yCenterOriginal)
        return Pair(boundingBox, center)
    }

    fun drawBoundingBoxesWithCustomLabel(mat: Mat, box: BoundingBox, labelText: String) {
        val topLeft = Point(box.x1.toDouble(), box.y1.toDouble())
        val bottomRight = Point(box.x2.toDouble(), box.y2.toDouble())
        Imgproc.rectangle(
            mat,
            topLeft,
            bottomRight,
            Settings.BoundingBox.boxColor,
            Settings.BoundingBox.boxThickness
        )
        val fontScale = 0.6
        val thickness = 1
        val baseline = IntArray(1)
        val textSize = Imgproc.getTextSize(labelText, Imgproc.FONT_HERSHEY_SIMPLEX, fontScale, thickness, baseline)
        val textX = box.x1.toInt()
        val textY = (box.y1 - 5).toInt().coerceAtLeast(10)
        Imgproc.rectangle(
            mat,
            Point(textX.toDouble(), textY.toDouble() + baseline[0]),
            Point(textX + textSize.width, textY - textSize.height),
            Settings.BoundingBox.boxColor,
            Imgproc.FILLED
        )
        Imgproc.putText(
            mat,
            labelText,
            Point(textX.toDouble(), textY.toDouble()),
            Imgproc.FONT_HERSHEY_SIMPLEX,
            fontScale,
            Scalar(255.0, 255.0, 255.0),
            thickness
        )
    }

    fun createLetterboxedBitmap(
        srcBitmap: Bitmap,
        targetWidth: Int,
        targetHeight: Int,
        padColor: Scalar = Scalar(0.0, 0.0, 0.0)
    ): Pair<Bitmap, Pair<Int, Int>> {
        val srcMat = Mat().also { Utils.bitmapToMat(srcBitmap, it) }
        val (srcWidth, srcHeight) = (srcMat.cols().toDouble()) to (srcMat.rows().toDouble())
        val scale = min(targetWidth / srcWidth, targetHeight / srcHeight)
        val newWidth = (srcWidth * scale).toInt()
        val newHeight = (srcHeight * scale).toInt()
        val resized = Mat().also {
            Imgproc.resize(srcMat, it, Size(newWidth.toDouble(), newHeight.toDouble()))
            srcMat.release()
        }
        val padWidth = targetWidth - newWidth
        val padHeight = targetHeight - newHeight
        val computePadding = { total: Int -> total / 2 to (total - total / 2) }
        val (top, bottom) = computePadding(padHeight)
        val (left, right) = computePadding(padWidth)
        val letterboxed = Mat().also {
            Core.copyMakeBorder(
                resized, it,
                top, bottom,
                left, right,
                Core.BORDER_CONSTANT,
                padColor
            )
            resized.release()
        }
        val outputBitmap = Bitmap.createBitmap(letterboxed.cols(), letterboxed.rows(), srcBitmap.config).apply {
            Utils.matToBitmap(letterboxed, this)
            letterboxed.release()
        }
        return Pair(outputBitmap, Pair(left, top))
    }
}