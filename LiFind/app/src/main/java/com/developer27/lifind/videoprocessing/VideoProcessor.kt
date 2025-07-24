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
import org.opencv.core.Point
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.image.TensorImage
import kotlin.math.min
import kotlin.random.Random

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
        var confidenceThreshold: Float = 0f
    }

    object BoundingBox {
        var enableBoundingBox = true
        var boxColor = Scalar(255.0, 255.0, 255.0)
        var boxThickness = 2
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
     * Returns a Pair of:
     *  1) outBmp: the original image annotated with bounding boxes and labels
     *  2) letterboxed: the resized/letterboxed version fed into the model
     */
    private suspend fun processFrameInternalYOLO(
        bitmap: Bitmap
    ): Pair<Bitmap, Bitmap> = withContext(Dispatchers.IO) {

        // Tag for logging
        val tag = javaClass.simpleName

        // 1) Get expected model input size and output tensor shape
        val (inputW, inputH, outputShape) = getModelDimensions()

        // 2) Resize and pad the input bitmap to match model input, preserving aspect ratio
        val (letterboxed, offsets) =
            YOLOHelper.createLetterboxedBitmap(bitmap, inputW, inputH)

        // 3) Load the letterboxed bitmap into a TensorImage for inference
        val tensorImage = TensorImage(DataType.FLOAT32).apply { load(letterboxed) }

        // 4) Prepare output buffer for distance estimation model
        val distTensorShape = distanceInterpreter
            ?.getOutputTensor(0)
            ?.shape()
            ?: intArrayOf(1, 1, YOLOHelper.numDistanceClasses)
        // 2D array: [batch=1][numDistanceClasses]
        val distOut = Array(distTensorShape[1]) {
            FloatArray(distTensorShape[2])
        }

        // 5) Convert original bitmap to OpenCV Mat to draw on later
        val m = Mat().also { Utils.bitmapToMat(bitmap, it) }

        // Default label if distance model is unavailable or yields no valid result
        var bestDistanceLabel = "Unknown"

        // 6) Run distance estimation model (if set) to predict distance class
        distanceInterpreter?.let { interp ->
            val scores = distOut[0]

            // Sort class indices by descending confidence
            val sortedScores = scores
                .mapIndexed { idx, score -> idx to score }
                .filter { it.first < YOLOHelper.numDistanceClasses }
                .sortedByDescending { it.second }

            // Log the top distance class and its confidence
            sortedScores.firstOrNull()?.let { (idx, score) ->
                val name = YOLOHelper.labelForDistanceIdx(idx)
                Log.d(tag, "Top Distance[$name] = ${"%.2f".format(score * 100)}%")
            }

            // Choose the best index or -1 if none
            val bestIdx = sortedScores.firstOrNull()?.first ?: -1
            if (bestIdx >= 0) {
                bestDistanceLabel = YOLOHelper.labelForDistanceIdx(bestIdx)
            }
        } ?: Log.w(tag, "Distance interpreter not initialised")

        // 7) If YOLO inference is enabled and interpreter is available, run object detection
        if (Settings.DetectionMode.enableYOLOinference && yoloInterpreter != null) {
            // Prepare output buffer for YOLO model: 3D array [1][grid][attributes]
            val yoloOut = Array(outputShape[0]) {
                Array(outputShape[1]) { FloatArray(outputShape[2]) }
            }

            // Run inference
            tensorImage.buffer.also { yoloInterpreter!!.run(it, yoloOut) }

            // Parse raw output into detection results
            YOLOHelper.parseTFLite(yoloOut)

                // Remove duplicate class detections
                ?.distinctBy { it.classId }
                // Limit to top 3 detections
                ?.take(3)
                ?.forEach { det ->
                    // Convert normalized box coordinates back to original image scale
                    val (box, center) = YOLOHelper.rescaleInferencedCoordinates(
                        det, bitmap.width, bitmap.height, offsets, inputW, inputH
                    )
                    if (Settings.BoundingBox.enableBoundingBox) {
                        // Build label combining class name, distance, and confidence
                        val yoloLabel = YOLOHelper.classNameForId(det.classId)
                        val labelText =
                            "$yoloLabel | $bestDistanceLabel (${"%.2f".format(det.confidence * 100)}%)"
                        // Draw bounding box and label on the Mat
                        YOLOHelper.drawDetectionCircleWithLabel(m, center, box, labelText)
                    }
                }
        }

        // draw three random circles each frame (This is a temproary code)
        val centers = mutableListOf<Point>()
        repeat(3) {
            // pick a random center within the image
            val x = Random.nextDouble(0.0, bitmap.width.toDouble())
            val y = Random.nextDouble(0.0, bitmap.height.toDouble())
            val center = Point(x, y)
            centers += center

            // build a box around it (100×100 px)
            val box = BoundingBox(
                x1 = (x - 50).toFloat(),
                y1 = (y - 50).toFloat(),
                x2 = (x + 50).toFloat(),
                y2 = (y + 50).toFloat(),
                confidence = 1.0f,
                classId = 0
            )

            // draw it
            YOLOHelper.drawDetectionCircleWithLabel(m, center, box, "Hardcoded")
        }

        // 8) Convert annotated Mat back to Bitmap and release Mat
        val outBmp = Bitmap
            .createBitmap(bitmap.width, bitmap.height, Bitmap.Config.ARGB_8888)
            .also { Utils.matToBitmap(m, it); m.release() }

        // Return the annotated image and the model input image
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

    /**
     * Parses raw TensorFlow Lite output into a list of DetectionResult,
     * applying a confidence threshold and selecting the most likely class.
     * @param rawOutput 3D array from YOLO model: [batch][rows][attributes]
     * @return List of DetectionResult if any detections pass the threshold, otherwise null
     */
    fun parseTFLite(rawOutput: Array<Array<FloatArray>>): List<DetectionResult>? {
        // 1) Extract predictions for the first (and only) batch
        val predictions = rawOutput[0]
        // 2) Prepare a list to collect valid detections
        val detections = mutableListOf<DetectionResult>()

        // 3) Iterate over each prediction row
        for (row in predictions) {
            // Skip rows that don't contain enough data (x,y,width,height,objectConf + at least one class score)
            if (row.size < 6) continue

            // 4) Unpack raw values
            val xCenter    = row[0]                   // normalized center X
            val yCenter    = row[1]                   // normalized center Y
            val width      = row[2]                   // normalized width
            val height     = row[3]                   // normalized height
            val objectConf = row[4]                   // objectness confidence

            // 5) Remaining entries are class probabilities
            val classScores = row.copyOfRange(5, row.size)

            // 6) Determine the most likely class index and its score
            val bestClassIdx   = classScores.indices.maxByOrNull { classScores[it] } ?: -1
            val bestClassScore = classScores.getOrNull(bestClassIdx) ?: 0f

            // 7) Combine object confidence and class confidence
            val finalConf = objectConf * bestClassScore

            // 8) Only keep detections above the configured threshold
            if (finalConf >= Settings.Inference.confidenceThreshold) {
                // 9) Create DetectionResult and add to list
                detections += DetectionResult(
                    xCenter,
                    yCenter,
                    width,
                    height,
                    finalConf,
                    bestClassIdx
                )
            }
        }

        // 10) Return null if no detections passed threshold, otherwise the list
        return if (detections.isEmpty()) null else detections
    }


    /**
     * Converts normalized detection coordinates from the model input space back to the original image space,
     * accounting for padding (letterboxing) and scaling.
     * Returns a Pair of:
     *  1) boundingBox: the rectangle in original image coordinates
     *  2) center: the center point of the bounding box
     */
    fun rescaleInferencedCoordinates(
        detection: DetectionResult,
        originalWidth: Int,
        originalHeight: Int,
        padOffsets: Pair<Int, Int>,       // (left, top) padding applied during letterboxing
        modelInputWidth: Int,
        modelInputHeight: Int
    ): Pair<BoundingBox, Point> {
        // 1) Compute the scale factor between model input and original image
        val scale = min(
            modelInputWidth / originalWidth.toDouble(),
            modelInputHeight / originalHeight.toDouble()
        )

        // 2) Extract the padding offsets applied on the left and top
        val padLeft = padOffsets.first.toDouble()
        val padTop = padOffsets.second.toDouble()

        // 3) Convert normalized center and size back to letterboxed pixel values
        val xCenterLetterboxed = detection.xCenter * modelInputWidth
        val yCenterLetterboxed = detection.yCenter * modelInputHeight
        val boxWidthLetterboxed = detection.width * modelInputWidth
        val boxHeightLetterboxed = detection.height * modelInputHeight

        // 4) Remove padding and apply inverse scaling to map back to original image
        val xCenterOriginal = (xCenterLetterboxed - padLeft) / scale
        val yCenterOriginal = (yCenterLetterboxed - padTop) / scale
        val boxWidthOriginal = boxWidthLetterboxed / scale
        val boxHeightOriginal = boxHeightLetterboxed / scale

        // 5) Compute the top-left and bottom-right coordinates of the bounding box
        val x1Original = xCenterOriginal - (boxWidthOriginal / 2)
        val y1Original = yCenterOriginal - (boxHeightOriginal / 2)
        val x2Original = xCenterOriginal + (boxWidthOriginal / 2)
        val y2Original = yCenterOriginal + (boxHeightOriginal / 2)

        // 6) Log adjusted box coordinates for debugging
        Log.d("YOLOHelper",
            "Adjusted BOX: x1=${"%.2f".format(x1Original)}, " +
                    "y1=${"%.2f".format(y1Original)}, " +
                    "x2=${"%.2f".format(x2Original)}, " +
                    "y2=${"%.2f".format(y2Original)}"
        )

        // 7) Create a BoundingBox data object in original image space
        val boundingBox = BoundingBox(
            x1 = x1Original.toFloat(),
            y1 = y1Original.toFloat(),
            x2 = x2Original.toFloat(),
            y2 = y2Original.toFloat(),
            confidence = detection.confidence,
            classId = detection.classId
        )

        // 8) Build a Point for the box center
        val center = Point(xCenterOriginal, yCenterOriginal)

        // 9) Return the bounding box and its center point
        return Pair(boundingBox, center)
    }

    /**
     * Draws a detection circle with a filled label background and text on the given Mat.
     * @param mat        the OpenCV Mat to draw on
     * @param center     the Point at which to center the circle
     * @param box        the original bounding box (used to size the circle)
     * @param labelText  the text to render above the circle
     */
    fun drawDetectionCircleWithLabel(
        mat: Mat,
        center: Point,
        box: BoundingBox,
        labelText: String
    ) {
        // 1) compute radius as half the smaller of box width/height
        val boxWidth  = box.x2 - box.x1
        val boxHeight = box.y2 - box.y1
        val baseRadius = (min(boxWidth, boxHeight) / 2).toInt()
        val radius = (baseRadius * 3.5f).toInt()

        // 2) draw the circle
        Imgproc.circle(
            mat,
            center,
            radius,
            Settings.BoundingBox.boxColor,
            Settings.BoundingBox.boxThickness
        )

        // 3) prepare text properties
        val fontScale = 2.0
        val thickness = 2
        val baseline  = IntArray(1)

        // 4) measure text size
        val textSize = Imgproc.getTextSize(
            labelText,
            Imgproc.FONT_HERSHEY_SIMPLEX,
            fontScale,
            thickness,
            baseline
        )

        // 5) position text centered above the circle
        val textX = (center.x - textSize.width / 2).toInt().coerceAtLeast(0)
        val textY = (center.y - radius - 5).toInt()
            .coerceAtLeast((textSize.height + baseline[0]).toInt())

        // 6) draw filled background behind text
        Imgproc.rectangle(
            mat,
            Point(textX.toDouble(), (textY + baseline[0]).toDouble()),
            Point((textX + textSize.width).toDouble(), (textY - textSize.height).toDouble()),
            Settings.BoundingBox.boxColor,
            Imgproc.FILLED
        )

        // 7) render the label text in white
        Imgproc.putText(
            mat,
            labelText,
            Point(textX.toDouble(), textY.toDouble()),
            Imgproc.FONT_HERSHEY_SIMPLEX,
            fontScale,
            Scalar(0.0, 0.0, 0.0),
            thickness
        )
    }

    /**
     * Resizes and pads the source bitmap to fit the target dimensions while maintaining aspect ratio.
     * Returns a Pair of:
     *  1) outputBitmap: the letterboxed image matching targetWidth x targetHeight
     *  2) Pair(leftPadding, topPadding): the pixel offsets applied on the left and top
     */
    fun createLetterboxedBitmap(
        srcBitmap: Bitmap,
        targetWidth: Int,
        targetHeight: Int,
        padColor: Scalar = Scalar(0.0, 0.0, 0.0)   // Color used for padding areas
    ): Pair<Bitmap, Pair<Int, Int>> {
        // 1) Convert source bitmap to OpenCV Mat for processing
        val srcMat = Mat().also { Utils.bitmapToMat(srcBitmap, it) }

        // 2) Get original dimensions
        val (srcWidth, srcHeight) = srcMat.cols().toDouble() to srcMat.rows().toDouble()

        // 3) Compute uniform scale factor to fit the target size
        val scale = min(
            targetWidth / srcWidth,
            targetHeight / srcHeight
        )

        // 4) Determine new dimensions after scaling
        val newWidth = (srcWidth * scale).toInt()
        val newHeight = (srcHeight * scale).toInt()

        // 5) Resize image to new dimensions
        val resized = Mat().also {
            Imgproc.resize(srcMat, it, Size(newWidth.toDouble(), newHeight.toDouble()))
            srcMat.release()  // Free original Mat
        }

        // 6) Calculate padding needed to reach target dimensions
        val padWidth = targetWidth - newWidth
        val padHeight = targetHeight - newHeight
        // Split padding evenly on both sides
        val computePadding = { total: Int -> total / 2 to (total - total / 2) }
        val (top, bottom) = computePadding(padHeight)
        val (left, right) = computePadding(padWidth)

        // 7) Apply border padding (letterbox) around the resized image
        val letterboxed = Mat().also {
            Core.copyMakeBorder(
                resized, it,
                top, bottom,
                left, right,
                Core.BORDER_CONSTANT,
                padColor
            )
            resized.release()  // Free resized Mat
        }

        // 8) Convert the letterboxed Mat back to a Bitmap
        val outputBitmap = Bitmap.createBitmap(
            letterboxed.cols(),
            letterboxed.rows(),
            srcBitmap.config
        ).apply {
            Utils.matToBitmap(letterboxed, this)
            letterboxed.release()  // Free letterboxed Mat
        }

        // 9) Return the new bitmap and the padding offsets for later use
        return Pair(outputBitmap, Pair(left, top))
    }
}