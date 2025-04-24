package com.developer27.rotemplate.videoprocessing

import android.content.Context
import android.graphics.Bitmap
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
import kotlin.math.max
import kotlin.math.min

data class DetectionResult(
    val xCenter: Float, val yCenter: Float,
    val width: Float, val height: Float,
    val confidence: Float
)
data class BoundingBox(
    val x1: Float, val y1: Float,
    val x2: Float, val y2: Float,
    val confidence: Float, val classId: Int
)

private var tfliteInterpreter: Interpreter? = null
// Removed rawDataList and smoothDataList since line drawing is no longer needed.

// Object to hold various configuration settings.
object Settings {
    object DetectionMode {
        enum class Mode { CONTOUR, YOLO }
        var current: Mode = Mode.YOLO
        var enableYOLOinference = true
    }
    object Inference {
        var confidenceThreshold: Float = 0.5f
        var iouThreshold: Float = 0.5f
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
    fun setInterpreter(model: Interpreter) {
        synchronized(this) { tfliteInterpreter = model }
        Log.d("VideoProcessor","TFLite Model set in VideoProcessor successfully!")
    }
    // Removed reset() and export/gather trace functions.

    // Processes a frame asynchronously and returns a Pair (outputBitmap, videoBitmap).
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
    // Processes a frame using Contour Detection.
    private fun processFrameInternalCONTOUR(bitmap: Bitmap): Pair<Bitmap, Bitmap>? {
        return try {
            val (pMat, pBmp) = Preprocessing.preprocessFrame(bitmap)
            // Removed trace drawing call.
            val (_, cMat) = ContourDetection.processContourDetection(pMat)
            val outBmp = Bitmap.createBitmap(cMat.cols(), cMat.rows(), Bitmap.Config.ARGB_8888).also { Utils.matToBitmap(cMat, it) }
            pMat.release()
            cMat.release()
            outBmp to pBmp
        } catch (e: Exception) {
            Log.d("VideoProcessor","Error processing frame: ${e.message}", e)
            null
        }
    }
    // Processes a frame using YOLO.
    private suspend fun processFrameInternalYOLO(bitmap: Bitmap): Pair<Bitmap, Bitmap> = withContext(Dispatchers.IO) {
        val (inputW, inputH, outputShape) = getModelDimensions()
        val (letterboxed, offsets) = YOLOHelper.createLetterboxedBitmap(bitmap, inputW, inputH)
        val m = Mat().also { Utils.bitmapToMat(bitmap, it) }
        if (Settings.DetectionMode.enableYOLOinference && tfliteInterpreter != null) {
            val out = Array(outputShape[0]) { Array(outputShape[1]) { FloatArray(outputShape[2]) } }
            TensorImage(DataType.FLOAT32).apply { load(letterboxed) }.also { tfliteInterpreter?.run(it.buffer, out) }
            YOLOHelper.parseTFLite(out)?.let {
                val (box, _) = YOLOHelper.rescaleInferencedCoordinates(it, bitmap.width, bitmap.height, offsets, inputW, inputH)
                if (Settings.BoundingBox.enableBoundingBox) YOLOHelper.drawBoundingBoxes(m, box)
                // Removed trace drawing call.
            }
        }
        val yoloBmp = Bitmap.createBitmap(bitmap.width, bitmap.height, Bitmap.Config.ARGB_8888).also {
            Utils.matToBitmap(m, it)
            m.release()
        }
        yoloBmp to letterboxed
    }

    // Retrieves the model input size dynamically.
    fun getModelDimensions(): Triple<Int, Int, List<Int>> {
        val inTensor = tfliteInterpreter?.getInputTensor(0)
        val inShape = inTensor?.shape()
        val (h, w) = (inShape?.getOrNull(1) ?: 416) to (inShape?.getOrNull(2) ?: 416)
        val outTensor = tfliteInterpreter?.getOutputTensor(0)
        val outShape = outTensor?.shape()?.toList() ?: listOf(1, 5, 3549)
        return Triple(w, h, outShape)
    }
}

// The TraceRenderer and KalmanHelper objects, as well as functions for exporting trace images,
// have been completely removed since the line drawing feature is no longer needed.

// Helper object for preprocessing frames with OpenCV.
object Preprocessing {
    fun preprocessFrame(src: Bitmap): Pair<Mat, Bitmap> {
        val sMat = Mat().also { Utils.bitmapToMat(src, it) }
        val gMat = Mat().also { Imgproc.cvtColor(sMat, it, Imgproc.COLOR_BGR2GRAY); sMat.release() }
        val eMat = Mat().also { Core.multiply(gMat, Scalar(Settings.Brightness.factor), it); gMat.release() }
        val tMat = Mat().also { Imgproc.threshold(eMat, it, Settings.Brightness.threshold, 255.0, Imgproc.THRESH_TOZERO); eMat.release() }
        val bMat = Mat().also { Imgproc.GaussianBlur(tMat, it, Size(5.0, 5.0), 0.0); tMat.release() }
        val k = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(3.0, 3.0))
        val cMat = Mat().also { Imgproc.morphologyEx(bMat, it, Imgproc.MORPH_CLOSE, k); bMat.release() }
        val bmp = Bitmap.createBitmap(cMat.cols(), cMat.rows(), Bitmap.Config.ARGB_8888).also { Utils.matToBitmap(cMat, it) }
        return cMat to bmp
    }
}

// Helper object for contour detection.
object ContourDetection {
    fun processContourDetection(mat: Mat) = findContours(mat).maxByOrNull { Imgproc.contourArea(it) }.let { c ->
        val center = c?.let {
            Imgproc.drawContours(mat, listOf(it), -1, Settings.BoundingBox.boxColor, Settings.BoundingBox.boxThickness)
            val m = Imgproc.moments(it)
            Point(m.m10 / m.m00, m.m01 / m.m00)
        }
        Imgproc.cvtColor(mat, mat, Imgproc.COLOR_GRAY2BGR)
        center to mat
    }
    private fun findContours(mat: Mat) = mutableListOf<MatOfPoint>().also {
        Mat().also { h -> Imgproc.findContours(mat, it, h, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE); h.release() }
    }
}

// Helper object for YOLO detection using TensorFlow Lite.
object YOLOHelper {
    fun parseTFLite(rawOutput: Array<Array<FloatArray>>): DetectionResult? {
        val numDetections = rawOutput[0][0].size
        // Parse detections and filter by confidence.
        val detections = mutableListOf<DetectionResult>()
        for (i in 0 until numDetections) {
            val xCenter = rawOutput[0][0][i]
            val yCenter = rawOutput[0][1][i]
            val width = rawOutput[0][2][i]
            val height = rawOutput[0][3][i]
            val confidence = rawOutput[0][4][i]
            if (confidence >= Settings.Inference.confidenceThreshold) {
                detections.add(DetectionResult(xCenter, yCenter, width, height, confidence))
            }
        }
        if (detections.isEmpty()) {
            Log.d("YOLOTest", "No detections above confidence threshold: ${Settings.Inference.confidenceThreshold}")
            return null
        }
        // Convert detections to bounding boxes.
        val detectionBoxes = detections.map { it to detectionToBox(it) }.toMutableList()
        detectionBoxes.sortByDescending { it.first.confidence }
        // Apply Non-Maximum Suppression.
        val nmsDetections = mutableListOf<DetectionResult>()
        while (detectionBoxes.isNotEmpty()) {
            val current = detectionBoxes.removeAt(0)
            nmsDetections.add(current.first)
            detectionBoxes.removeAll { other ->
                computeIoU(current.second, other.second) > Settings.Inference.iouThreshold
            }
        }
        val bestDetection = nmsDetections.maxByOrNull { it.confidence }
        bestDetection?.let { d ->
            Log.d(
                "YOLOTest",
                "BEST DETECTION: confidence=${"%.8f".format(d.confidence)}, x_center=${d.xCenter}, y_center=${d.yCenter}, width=${d.width}, height=${d.height}"
            )
        }
        return bestDetection
    }
    private fun detectionToBox(d: DetectionResult) = BoundingBox(
        d.xCenter - d.width / 2,
        d.yCenter - d.height / 2,
        d.xCenter + d.width / 2,
        d.yCenter + d.height / 2,
        d.confidence,
        1
    )
    private fun computeIoU(boxA: BoundingBox, boxB: BoundingBox): Float {
        val x1 = max(boxA.x1, boxB.x1)
        val y1 = max(boxA.y1, boxB.y1)
        val x2 = min(boxA.x2, boxB.x2)
        val y2 = min(boxA.y2, boxB.y2)
        val intersectionWidth = max(0f, x2 - x1)
        val intersectionHeight = max(0f, y2 - y1)
        val intersectionArea = intersectionWidth * intersectionHeight
        val areaA = (boxA.x2 - boxA.x1) * (boxA.y2 - boxA.y1)
        val areaB = (boxB.x2 - boxB.x1) * (boxB.y2 - boxB.y1)
        val unionArea = areaA + areaB - intersectionArea
        return if (unionArea > 0f) intersectionArea / unionArea else 0f
    }
    fun rescaleInferencedCoordinates(detection: DetectionResult, originalWidth: Int, originalHeight: Int, padOffsets: Pair<Int, Int>, modelInputWidth: Int, modelInputHeight: Int): Pair<BoundingBox, Point> {
        val scale = min(modelInputWidth / originalWidth.toDouble(), modelInputHeight / originalHeight.toDouble())
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
        Log.d("YOLOTest", "Adjusted BOUNDING BOX: x1=${"%.8f".format(x1Original)}, y1=${"%.8f".format(y1Original)}, x2=${"%.8f".format(x2Original)}, y2=${"%.8f".format(y2Original)}")
        val boundingBox = BoundingBox(
            x1Original.toFloat(),
            y1Original.toFloat(),
            x2Original.toFloat(),
            y2Original.toFloat(),
            detection.confidence,
            1
        )
        val center = Point(xCenterOriginal, yCenterOriginal)
        return Pair(boundingBox, center)
    }
    fun drawBoundingBoxes(mat: Mat, box: BoundingBox) {
        val topLeft = Point(box.x1.toDouble(), box.y1.toDouble())
        val bottomRight = Point(box.x2.toDouble(), box.y2.toDouble())
        Imgproc.rectangle(mat, topLeft, bottomRight, Settings.BoundingBox.boxColor, Settings.BoundingBox.boxThickness)
        val label = "User_1 (${("%.2f".format(box.confidence * 100))}%)"
        val fontScale = 0.6
        val thickness = 1
        val baseline = IntArray(1)
        val textSize = Imgproc.getTextSize(label, Imgproc.FONT_HERSHEY_SIMPLEX, fontScale, thickness, baseline)
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
            label,
            Point(textX.toDouble(), textY.toDouble()),
            Imgproc.FONT_HERSHEY_SIMPLEX,
            fontScale,
            Scalar(255.0, 255.0, 255.0),
            thickness
        )
    }
    fun createLetterboxedBitmap(srcBitmap: Bitmap, targetWidth: Int, targetHeight: Int, padColor: Scalar = Scalar(0.0, 0.0, 0.0)): Pair<Bitmap, Pair<Int, Int>> {
        val srcMat = Mat().also { Utils.bitmapToMat(srcBitmap, it) }
        val (srcWidth, srcHeight) = (srcMat.cols().toDouble()) to (srcMat.rows().toDouble())
        val scale = min(targetWidth / srcWidth, targetHeight / srcHeight)
        val (newWidth, newHeight) = (srcWidth * scale).toInt() to (srcHeight * scale).toInt()
        val resized = Mat().also { Imgproc.resize(srcMat, it, Size(newWidth.toDouble(), newHeight.toDouble())) }
        srcMat.release()
        val (padWidth, padHeight) = (targetWidth - newWidth) to (targetHeight - newHeight)
        val computePadding = { total: Int -> total / 2 to (total - total / 2) }
        val (top, bottom) = computePadding(padHeight)
        val (left, right) = computePadding(padWidth)
        val letterboxed = Mat().also { Core.copyMakeBorder(resized, it, top, bottom, left, right, Core.BORDER_CONSTANT, padColor) }
        resized.release()
        val outputBitmap = Bitmap.createBitmap(letterboxed.cols(), letterboxed.rows(), srcBitmap.config).apply {
            Utils.matToBitmap(letterboxed, this)
            letterboxed.release()
        }
        return Pair(outputBitmap, Pair(left, top))
    }
}
