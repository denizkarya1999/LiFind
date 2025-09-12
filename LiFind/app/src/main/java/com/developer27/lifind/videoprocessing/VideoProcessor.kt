package com.developer27.lifind.videoprocessing

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.opencv.android.Utils
import org.opencv.core.Scalar
import org.opencv.imgproc.Imgproc
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.image.TensorImage
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.max

// ----------------- Data / Settings -----------------

data class DetectionResult(
    val xCenter: Float,   // normalized 0..1
    val yCenter: Float,   // normalized 0..1
    val width: Float,     // normalized 0..1
    val height: Float,    // normalized 0..1
    val confidence: Float,
    val classId: Int
)

object Settings {
    object DetectionMode {
        enum class Mode { YOLO }
        var current: Mode = Mode.YOLO
    }
    object Inference {
        var confidenceThreshold: Float = 0.00f
        var iouThreshold: Float = 0.45f
        var classAgnosticNms: Boolean = false
        var multiLabelPerBox: Boolean = true
        var topPerClass: Int = 1
    }
    object BoundingBox {
        var enableBoundingBox = true

        // BGR palette for OpenCV
        val perClassColors: Array<Scalar> = arrayOf(
            Scalar(0.0, 0.0, 255.0), // class 0 -> Red
            Scalar(0.0, 255.0, 0.0), // class 1 -> Green
            Scalar(255.0, 0.0, 0.0)  // class 2 -> Blue
        )
        var boxThickness = 2
        val fallbackBoxColor = Scalar(255.0, 255.0, 255.0)
        var drawCenterDot = false
    }
}

/**
 * Frame processor optimized to:
 *  - Reuse CoroutineScope, TensorImage, output arrays, Mat, Bitmaps
 *  - Avoid allocations in hot loops
 *  - Guard OpenCV init
 */
class VideoProcessor(private val context: Context) {

    private val tag = "VideoProcessor"
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Default)

    // --- Reusable TFLite input/output holders ---
    private val tensorImage = TensorImage(DataType.FLOAT32)

    // interpreter output shape, default small placeholder until first run
    private var outShape: IntArray = intArrayOf(1, 1, 8)

    // Backing array reused per frame; resized when model output shape changes
    @Volatile private var distOut: Array<Array<FloatArray>> =
        arrayOf(arrayOf(FloatArray(8)))

    // Reuse output bitmap to avoid realloc each frame
    private var reusedOutBitmap: Bitmap? = null

    // --- Reusable OpenCV buffers ---
    // Create Mat only after the native lib is loaded
    private lateinit var workMat: org.opencv.core.Mat

    // Guard against OpenCV init crash loops
    private val opencvLoaded = AtomicBoolean(false)

    init {
        try {
            System.loadLibrary("opencv_java4")
            opencvLoaded.set(true)
            workMat = org.opencv.core.Mat()   // SAFE: library is loaded now
        } catch (e: UnsatisfiedLinkError) {
            Log.d(tag, "OpenCV failed to load: ${e.message}", e)
        }
    }

    fun close() {
        scope.cancel()
        workMat.release()
        reusedOutBitmap = null
        try { YOLOHelperV2.closeInterpreter() } catch (_: Throwable) {}
    }

    fun processFrame(bitmap: Bitmap, callback: (Pair<Bitmap, Bitmap>?) -> Unit) {
        scope.launch {
            val result = try {
                when (Settings.DetectionMode.current) {
                    Settings.DetectionMode.Mode.YOLO -> processFrameInternalYOLO(bitmap)
                }
            } catch (e: Exception) {
                Log.d(tag, "Error processing frame: ${e.message}", e)
                null
            }
            withContext(Dispatchers.Main) { callback(result) }
        }
    }

    private fun ensureOutputBuffers(interp: org.tensorflow.lite.Interpreter) {
        val shape = interp.getOutputTensor(0).shape() // e.g. [1, 84, 8400] or [1, 8400, 84]
        if (!shape.contentEquals(outShape)) {
            outShape = shape
            distOut = Array(shape[0]) { Array(shape[1]) { FloatArray(shape[2]) } }
        }
    }

    private fun ensureReusedBitmaps() {
        val w = YOLOHelperV2.INPUT_SIZE
        val h = YOLOHelperV2.INPUT_SIZE
        if (reusedOutBitmap == null ||
            reusedOutBitmap!!.width != w ||
            reusedOutBitmap!!.height != h ||
            reusedOutBitmap!!.config != Bitmap.Config.ARGB_8888
        ) {
            reusedOutBitmap = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        }
    }

    private suspend fun processFrameInternalYOLO(
        bitmap: Bitmap
    ): Pair<Bitmap, Bitmap>? = withContext(Dispatchers.Default) {
        if (!opencvLoaded.get()) return@withContext null

        // 1) Auto-orient + resize to 640×640 ARGB_8888
        val inputBitmap = YOLOHelperV2.autoOrientAndResize(bitmap)
        ensureReusedBitmaps()

        // 2) Prepare TFLite input (reused TensorImage)
        tensorImage.load(inputBitmap)

        // 3) Run model (ensure interpreter + reusable output buffer)
        val interp = YOLOHelperV2.ensureInterpreter(context)
        ensureOutputBuffers(interp)
        synchronized(interp) {
            interp.run(tensorImage.buffer, distOut)
        }

        // 4) Parse + NMS + optional per-class top-K
        val rawDetections = YOLOHelperV2.parseTFLite(
            raw = distOut,
            confidenceThreshold = Settings.Inference.confidenceThreshold,
            classAgnosticNms = Settings.Inference.classAgnosticNms,
            multiLabelPerBox = Settings.Inference.multiLabelPerBox,
            expectedClasses = 3
        )
        val dets =
            if (Settings.Inference.topPerClass > 0)
                YOLOHelperV2.selectTopPerClass(rawDetections, Settings.Inference.topPerClass)
            else rawDetections

        // 5) Draw
        Utils.bitmapToMat(inputBitmap, workMat)
        if (Settings.BoundingBox.enableBoundingBox && dets.isNotEmpty()) {
            val size = YOLOHelperV2.INPUT_SIZE.toFloat()
            val right = (size - 1).toInt()
            val bottom = (size - 1).toInt()

            var xLb: Float; var yLb: Float; var wLb: Float; var hLb: Float
            var x1: Int; var y1: Int; var x2: Int; var y2: Int
            var color: Scalar
            var confPct: Int
            val labelBuf = StringBuilder(32)

            for (i in dets.indices) {
                val det = dets[i]

                xLb = det.xCenter * size
                yLb = det.yCenter * size
                wLb = det.width  * size
                hLb = det.height * size

                x1 = (xLb - 0.5f * wLb).toInt().coerceIn(0, right)
                y1 = (yLb - 0.5f * hLb).toInt().coerceIn(0, bottom)
                x2 = (xLb + 0.5f * wLb).toInt().coerceIn(0, right)
                y2 = (yLb + 0.5f * hLb).toInt().coerceIn(0, bottom)

                confPct = (YOLOHelperV2.getConfidence(det) * 100f + 0.5f).toInt()
                labelBuf.clear()
                labelBuf.append("OOK: ")
                    .append(YOLOHelperV2.classNameForId(det.classId))
                    .append(" - Acc: ")
                    .append(confPct)
                    .append('%')

                color = Settings.BoundingBox.perClassColors.getOrNull(det.classId)
                    ?: Settings.BoundingBox.fallbackBoxColor

                Imgproc.rectangle(
                    workMat,
                    org.opencv.core.Point(x1.toDouble(), y1.toDouble()),
                    org.opencv.core.Point(x2.toDouble(), y2.toDouble()),
                    color,
                    max(2, Settings.BoundingBox.boxThickness),
                    Imgproc.LINE_AA,
                    0
                )

                Imgproc.putText(
                    workMat,
                    labelBuf.toString(),
                    org.opencv.core.Point(x1.toDouble(), (y1 - 6).coerceAtLeast(10).toDouble()),
                    Imgproc.FONT_HERSHEY_SIMPLEX,
                    0.7, color, 2, Imgproc.LINE_AA, false
                )

                if (Settings.BoundingBox.drawCenterDot) {
                    Imgproc.circle(
                        workMat,
                        org.opencv.core.Point(xLb.toDouble(), yLb.toDouble()),
                        3, color, Imgproc.FILLED
                    )
                }
            }
        }

        // 6) Convert Mat back to bitmap(s)
        val annotated = reusedOutBitmap!!
        Utils.matToBitmap(workMat, annotated)

        // Return (annotated, 640x640 input used for inference)
        annotated to inputBitmap
    }
}
