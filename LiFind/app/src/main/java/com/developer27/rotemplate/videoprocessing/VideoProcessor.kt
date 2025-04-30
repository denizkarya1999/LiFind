package com.developer27.lifind.videoprocessing

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

private var tfliteInterpreter: Interpreter? = null

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
        var boxColor = Scalar(0.0, 39.0, 76.0) // Dark-ish blue
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
        synchronized(this) {
            tfliteInterpreter = model
        }
        Log.d("VideoProcessor","TFLite Model set in VideoProcessor successfully!")
    }

    /**
     * Processes a frame asynchronously and returns a Pair:
     *   (outputBitmap, letterboxedOrPreprocessedBitmap)
     */
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
            // Preprocess
            val (pMat, pBmp) = Preprocessing.preprocessFrame(bitmap)

            // Detect largest contour
            val (_, cMat) = ContourDetection.processContourDetection(pMat)

            // Convert cMat to Bitmap
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

    // Processes a frame using YOLO.
    private suspend fun processFrameInternalYOLO(bitmap: Bitmap): Pair<Bitmap, Bitmap> = withContext(Dispatchers.IO) {
        val (inputW, inputH, outputShape) = getModelDimensions()

        // Create letterboxed bitmap for consistent YOLO input
        val (letterboxed, offsets) = YOLOHelper.createLetterboxedBitmap(bitmap, inputW, inputH)

        // Convert original image to Mat for drawing bounding boxes
        val m = Mat().also { Utils.bitmapToMat(bitmap, it) }

        // Run TFLite inference if loaded and enabled
        if (Settings.DetectionMode.enableYOLOinference && tfliteInterpreter != null) {
            // Prepare the output array
            val out = Array(outputShape[0]) {
                Array(outputShape[1]) {
                    FloatArray(outputShape[2])
                }
            }

            // Convert letterboxed bitmap to TFLite buffer
            TensorImage(DataType.FLOAT32).apply { load(letterboxed) }
                .also { tfliteInterpreter?.run(it.buffer, out) }

            // Parse result
            YOLOHelper.parseTFLite(out)?.let { detection ->
                // Convert from letterboxed coords back to original image coords
                val (box, _) = YOLOHelper.rescaleInferencedCoordinates(
                    detection,
                    bitmap.width,
                    bitmap.height,
                    offsets,
                    inputW,
                    inputH
                )
                // Draw bounding box if enabled
                if (Settings.BoundingBox.enableBoundingBox) {
                    YOLOHelper.drawBoundingBoxes(m, box)
                }
            }
        }

        // Convert annotated Mat back to a Bitmap
        val yoloBmp = Bitmap.createBitmap(bitmap.width, bitmap.height, Bitmap.Config.ARGB_8888).also {
            Utils.matToBitmap(m, it)
            m.release()
        }

        // Return the annotated original + the letterboxed image
        yoloBmp to letterboxed
    }

    // Retrieves the model input size and output shape from TFLite interpreter.
    fun getModelDimensions(): Triple<Int, Int, List<Int>> {
        val inTensor = tfliteInterpreter?.getInputTensor(0)
        val inShape = inTensor?.shape()
        // Typically [1, inputH, inputW, 3]
        val h = inShape?.getOrNull(1) ?: 416
        val w = inShape?.getOrNull(2) ?: 416

        val outTensor = tfliteInterpreter?.getOutputTensor(0)
        // YOLOv5 TFLite often yields [1, N, 9], etc.
        val outShape = outTensor?.shape()?.toList() ?: listOf(1, 1, 9)

        return Triple(w, h, outShape)
    }
}

// Helper object for preprocessing frames with OpenCV.
object Preprocessing {
    /**
     * Converts input bitmap to gray, applies brightness factor,
     * threshold, morphological blur, and returns (processed_Mat, processed_Bitmap).
     */
    fun preprocessFrame(src: Bitmap): Pair<Mat, Bitmap> {
        val sMat = Mat().also { Utils.bitmapToMat(src, it) }

        // Convert to grayscale
        val gMat = Mat().also {
            Imgproc.cvtColor(sMat, it, Imgproc.COLOR_BGR2GRAY)
            sMat.release()
        }

        // Adjust brightness
        val eMat = Mat().also {
            Core.multiply(gMat, Scalar(Settings.Brightness.factor), it)
            gMat.release()
        }

        // Threshold
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

        // Gaussian blur
        val bMat = Mat().also {
            Imgproc.GaussianBlur(tMat, it, Size(5.0, 5.0), 0.0)
            tMat.release()
        }

        // Morphological close
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

// Helper object for contour detection.
object ContourDetection {

    fun processContourDetection(mat: Mat): Pair<Point?, Mat> {
        val biggestContour = findContours(mat).maxByOrNull { Imgproc.contourArea(it) }
        val center = biggestContour?.let {
            // Draw largest contour
            Imgproc.drawContours(mat, listOf(it), -1, Settings.BoundingBox.boxColor, Settings.BoundingBox.boxThickness)

            // Compute center via moments
            val m = Imgproc.moments(it)
            Point(m.m10 / m.m00, m.m01 / m.m00)
        }
        // Convert single-channel to BGR
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

// Helper object for YOLO detection using TensorFlow Lite.
object YOLOHelper {

    // If you have multiple classes, list them here in order:
    private val classNames = arrayOf("LED1", "LED2", "LED3", "LED4")

    /**
     * For YOLO models that output shape [1, N, 9] or [1, N, (5+numClasses)],
     * each row has [x, y, w, h, object_conf, class1_conf, class2_conf, ...].
     *
     * Returns the single best detection after NMS, or null if none are above threshold.
     */
    fun parseTFLite(rawOutput: Array<Array<FloatArray>>): DetectionResult? {
        // Usually [0] is the batch dimension
        val predictions = rawOutput[0]
        val detections = mutableListOf<DetectionResult>()

        for (row in predictions) {
            // row = [x, y, w, h, obj_conf, class1, class2, ... ]
            if (row.size < 6) continue

            val xCenter = row[0]
            val yCenter = row[1]
            val width = row[2]
            val height = row[3]
            val objectConf = row[4]

            // class scores start from row[5..]
            val classScores = row.copyOfRange(5, row.size)

            // SAFE approach: find best class index manually or using maxBy
            val bestClassIdx = classScores.indices.maxByOrNull { classScores[it] } ?: -1
            val bestClassScore = if (bestClassIdx >= 0) classScores[bestClassIdx] else 0f

            // Combined confidence
            val finalConf = objectConf * bestClassScore

            if (finalConf >= Settings.Inference.confidenceThreshold) {
                detections.add(
                    DetectionResult(
                        xCenter,
                        yCenter,
                        width,
                        height,
                        finalConf,
                        bestClassIdx
                    )
                )
            }
        }

        // If no detections above threshold, return null
        if (detections.isEmpty()) {
            Log.d("YOLOTest", "No detections above confidence threshold: ${Settings.Inference.confidenceThreshold}")
            return null
        }

        // Sort by descending confidence
        detections.sortByDescending { it.confidence }

        // Convert them to bounding boxes for NMS
        val detectionBoxes = detections.map { it to detectionToBox(it) }.toMutableList()
        val nmsDetections = mutableListOf<DetectionResult>()

        // Simple NMS
        while (detectionBoxes.isNotEmpty()) {
            val current = detectionBoxes.removeAt(0)
            nmsDetections.add(current.first)

            // Remove all boxes that overlap above the iou threshold
            detectionBoxes.removeAll { other ->
                computeIoU(current.second, other.second) > Settings.Inference.iouThreshold
            }
        }

        // Return the best detection from final list
        val bestDetection = nmsDetections.maxByOrNull { it.confidence }
        bestDetection?.let { d ->
            Log.d(
                "YOLOTest",
                "BEST DETECTION: conf=${"%.3f".format(d.confidence)}, classId=${d.classId}, " +
                        "x=${d.xCenter}, y=${d.yCenter}, w=${d.width}, h=${d.height}"
            )
        }
        return bestDetection
    }

    private fun detectionToBox(d: DetectionResult) = BoundingBox(
        x1 = d.xCenter - d.width / 2,
        y1 = d.yCenter - d.height / 2,
        x2 = d.xCenter + d.width / 2,
        y2 = d.yCenter + d.height / 2,
        confidence = d.confidence,
        classId = d.classId
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

    /**
     * Rescale letterboxed detection coordinates back to the original image size.
     */
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

        // Multiply normalized x,y,w,h by model input size
        val xCenterLetterboxed = detection.xCenter * modelInputWidth
        val yCenterLetterboxed = detection.yCenter * modelInputHeight
        val boxWidthLetterboxed = detection.width * modelInputWidth
        val boxHeightLetterboxed = detection.height * modelInputHeight

        // Undo letterbox offset and scaling
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

    /**
     * Draw a bounding box on the given Mat using its classId to form the label.
     */
    fun drawBoundingBoxes(mat: Mat, box: BoundingBox) {
        val topLeft = Point(box.x1.toDouble(), box.y1.toDouble())
        val bottomRight = Point(box.x2.toDouble(), box.y2.toDouble())

        // Draw rectangle
        Imgproc.rectangle(
            mat,
            topLeft,
            bottomRight,
            Settings.BoundingBox.boxColor,
            Settings.BoundingBox.boxThickness
        )

        // Build the label text from class ID
        val labelText = if (box.classId in classNames.indices) {
            val className = classNames[box.classId]
            "$className (${("%.2f".format(box.confidence * 100))}%)"
        } else {
            "Unknown (${("%.2f".format(box.confidence * 100))}%)"
        }

        val fontScale = 0.6
        val thickness = 1
        val baseline = IntArray(1)

        // Calculate text size
        val textSize = Imgproc.getTextSize(labelText, Imgproc.FONT_HERSHEY_SIMPLEX, fontScale, thickness, baseline)
        val textX = box.x1.toInt()
        val textY = (box.y1 - 5).toInt().coerceAtLeast(10)

        // Draw filled rectangle behind text
        Imgproc.rectangle(
            mat,
            Point(textX.toDouble(), textY.toDouble() + baseline[0]),
            Point(textX + textSize.width, textY - textSize.height),
            Settings.BoundingBox.boxColor,
            Imgproc.FILLED
        )

        // Put text
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

    /**
     * Creates a letterboxed bitmap to preserve aspect ratio when resizing to (targetWidth x targetHeight).
     * Returns the new bitmap + the top/left offset used for letterboxing.
     */
    fun createLetterboxedBitmap(
        srcBitmap: Bitmap,
        targetWidth: Int,
        targetHeight: Int,
        padColor: Scalar = Scalar(0.0, 0.0, 0.0)
    ): Pair<Bitmap, Pair<Int, Int>> {

        // Convert to Mat
        val srcMat = Mat().also { Utils.bitmapToMat(srcBitmap, it) }
        val (srcWidth, srcHeight) = (srcMat.cols().toDouble()) to (srcMat.rows().toDouble())

        // Scale factor
        val scale = min(targetWidth / srcWidth, targetHeight / srcHeight)
        val newWidth = (srcWidth * scale).toInt()
        val newHeight = (srcHeight * scale).toInt()

        // Resize
        val resized = Mat().also {
            Imgproc.resize(srcMat, it, Size(newWidth.toDouble(), newHeight.toDouble()))
            srcMat.release()
        }

        // Determine padding
        val padWidth = targetWidth - newWidth
        val padHeight = targetHeight - newHeight

        val computePadding = { total: Int -> total / 2 to (total - total / 2) }
        val (top, bottom) = computePadding(padHeight)
        val (left, right) = computePadding(padWidth)

        // Letterbox
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

        // Convert back to Bitmap
        val outputBitmap = Bitmap.createBitmap(letterboxed.cols(), letterboxed.rows(), srcBitmap.config).apply {
            Utils.matToBitmap(letterboxed, this)
            letterboxed.release()
        }

        // Return letterboxed image + offsets
        return Pair(outputBitmap, Pair(left, top))
    }
}