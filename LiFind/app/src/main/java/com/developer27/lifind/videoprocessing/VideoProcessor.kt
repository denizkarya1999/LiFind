package com.developer27.lifind.videoprocessing

import android.content.Context
import android.graphics.Bitmap
import android.os.Environment
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
import java.io.BufferedWriter
import java.io.File
import java.io.FileWriter
import java.io.PrintWriter
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

private data class LedDistanceSample(
    val tMillis: Long,
    val led1: Pair<Int, Int>?,  // (x,y) in 640×640 px
    val led2: Pair<Int, Int>?,
    val led3: Pair<Int, Int>?,
    val distLabel: String?      // e.g., "Near" or "" if not detected
)

private val ledDistSamples = java.util.Collections.synchronizedList(mutableListOf<LedDistanceSample>())
private fun resetLedDistLogBuffer() = ledDistSamples.clear()

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
        try { YOLOLEDHelper.closeInterpreter() } catch (_: Throwable) {}
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
        val w = YOLOLEDHelper.INPUT_SIZE
        val h = YOLOLEDHelper.INPUT_SIZE
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

        // 1) Auto-orient + resize to 640×640 ARGB_8888 (shared input for both models)
        val inputBitmap = YOLOLEDHelper.autoOrientAndResize(bitmap)
        ensureReusedBitmaps()

        // 2) Prepare TFLite input (reused TensorImage)
        tensorImage.load(inputBitmap)

        // -------------------- LED MODEL (multi-box) --------------------
        val ledInterp = YOLOLEDHelper.ensureInterpreter(context)
        ensureOutputBuffers(ledInterp) // updates class-level outShape + distOut for LED head
        synchronized(ledInterp) {
            ledInterp.run(tensorImage.buffer, distOut)
        }

        val rawLedDetections = YOLOLEDHelper.parseTFLite(
            raw = distOut,
            confidenceThreshold = Settings.Inference.confidenceThreshold,
            classAgnosticNms = Settings.Inference.classAgnosticNms,
            multiLabelPerBox = Settings.Inference.multiLabelPerBox,
            expectedClasses = 3
        )
        val ledDets =
            if (Settings.Inference.topPerClass > 0)
                YOLOLEDHelper.selectTopPerClass(rawLedDetections, Settings.Inference.topPerClass)
            else rawLedDetections

        // -------------------- DISTANCE MODEL (one-box) --------------------
        val distInterp = YOLODISTANCEHelper.ensureInterpreter(context)
        val distShape = distInterp.getOutputTensor(0).shape() // e.g., [1,84,8400] or [1,8400,84], etc.
        val distHead: Array<Array<FloatArray>> =
            Array(distShape[0]) { Array(distShape[1]) { FloatArray(distShape[2]) } }

        synchronized(distInterp) {
            distInterp.run(tensorImage.buffer, distHead)
        }

        // Robust parse with auto-K + class-agnostic NMS, then take top-1
        val bestDistance = YOLODISTANCEHelper.parseTFLiteOneBox(
            raw = distHead,
            confidenceThreshold = Settings.Inference.confidenceThreshold,
            multiLabelPerBox = Settings.Inference.multiLabelPerBox,
            expectedClasses = 0,       // <=0 => auto-detect K (fixes your 8400/84 issue)
            classAgnosticNms = true
        )

        // -------------------- DRAWING --------------------
        Utils.bitmapToMat(inputBitmap, workMat)

        if (Settings.BoundingBox.enableBoundingBox) {
            val size = YOLOLEDHelper.INPUT_SIZE.toFloat()
            val right = (size - 1).toInt()
            val bottom = (size - 1).toInt()

            var xLb: Float; var yLb: Float; var wLb: Float; var hLb: Float
            var x1: Int; var y1: Int; var x2: Int; var y2: Int
            var color: Scalar
            var confPct: Int
            val labelBuf = StringBuilder(48)

            // Draw LED detections (multi-box)
            if (ledDets.isNotEmpty()) {
                for (det in ledDets) {
                    xLb = det.xCenter * size
                    yLb = det.yCenter * size
                    wLb = det.width  * size
                    hLb = det.height * size

                    x1 = (xLb - 0.5f * wLb).toInt().coerceIn(0, right)
                    y1 = (yLb - 0.5f * hLb).toInt().coerceIn(0, bottom)
                    x2 = (xLb + 0.5f * wLb).toInt().coerceIn(0, right)
                    y2 = (yLb + 0.5f * hLb).toInt().coerceIn(0, bottom)

                    confPct = (YOLOLEDHelper.getConfidence(det) * 100f + 0.5f).toInt()
                    labelBuf.clear()
                    labelBuf.append("OOK: ")
                        .append(YOLOLEDHelper.classNameForId(det.classId))
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

            // Draw the single best Distance detection (one-box)
            if (bestDistance != null) {
                xLb = bestDistance.xCenter * size
                yLb = bestDistance.yCenter * size
                wLb = bestDistance.width  * size
                hLb = bestDistance.height * size

                x1 = (xLb - 0.5f * wLb).toInt().coerceIn(0, right)
                y1 = (yLb - 0.5f * hLb).toInt().coerceIn(0, bottom)
                x2 = (xLb + 0.5f * wLb).toInt().coerceIn(0, right)
                y2 = (yLb + 0.5f * hLb).toInt().coerceIn(0, bottom)

                confPct = (YOLODISTANCEHelper.getConfidence(bestDistance) * 100f + 0.5f).toInt()
                val distLabel = YOLODISTANCEHelper.classNameForId(bestDistance.classId)
                val distText = "DIST: $distLabel - Acc: $confPct%"

                val distColor = Settings.BoundingBox.fallbackBoxColor

                Imgproc.rectangle(
                    workMat,
                    org.opencv.core.Point(x1.toDouble(), y1.toDouble()),
                    org.opencv.core.Point(x2.toDouble(), y2.toDouble()),
                    distColor,
                    max(2, Settings.BoundingBox.boxThickness),
                    Imgproc.LINE_AA,
                    0
                )
                Imgproc.putText(
                    workMat,
                    distText,
                    org.opencv.core.Point(x1.toDouble(), (y1 - 6).coerceAtLeast(10).toDouble()),
                    Imgproc.FONT_HERSHEY_SIMPLEX,
                    0.7, distColor, 2, Imgproc.LINE_AA, false
                )

                if (Settings.BoundingBox.drawCenterDot) {
                    Imgproc.circle(
                        workMat,
                        org.opencv.core.Point(xLb.toDouble(), yLb.toDouble()),
                        3, distColor, Imgproc.FILLED
                    )
                }
            }
        }

        // After computing `ledDets` and optional `bestDistance`...
        val nowMs = System.currentTimeMillis()

        val ledCenters = arrayOfNulls<Pair<Int, Int>>(3)
        run {
            val size = YOLOLEDHelper.INPUT_SIZE.toFloat()
            val right = (size - 1).toInt()
            val bottom = (size - 1).toInt()

            for (det in ledDets) {
                val cls = det.classId
                if (cls in 0..2 && ledCenters[cls] == null) {
                    val cx = (det.xCenter * size).toInt().coerceIn(0, right)
                    val cy = (det.yCenter * size).toInt().coerceIn(0, bottom)
                    ledCenters[cls] = cx to cy
                }
            }
        }

        var distLabel: String? = null
        bestDistance?.let { bd ->
            distLabel = YOLODISTANCEHelper.classNameForId(bd.classId)
        }

        // Push one entry into buffer
        ledDistSamples += LedDistanceSample(
            tMillis = nowMs,
            led1 = ledCenters[0],
            led2 = ledCenters[1],
            led3 = ledCenters[2],
            distLabel = distLabel
        )

        // 6) Convert Mat back to bitmap(s)
        val annotated = reusedOutBitmap!!
        Utils.matToBitmap(workMat, annotated)

        // Return (annotated, 640x640 input used for inference)
        annotated to inputBitmap
    }

    fun writeLedDistLogToFile(): File? {
        fun fmt(pt: Pair<Int, Int>?): String =
            pt?.let { "(x=${it.first}, y=${it.second})" } ?: "(x=N/A, y=N/A)"

        val name = "LiFind_Log.txt"

        // Public Documents dir
        val docsDir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOCUMENTS)
        if (!docsDir.exists()) docsDir.mkdirs()

        val outFile = File(docsDir, name)
        try {
            PrintWriter(BufferedWriter(FileWriter(outFile))).use { out ->
                synchronized(ledDistSamples) {
                    // ✅ only take the last entry if it exists
                    val s = ledDistSamples.lastOrNull()
                    if (s != null) {
                        val line =
                            "LED_1 - ${fmt(s.led1)}, " +
                                    "LED_2 - ${fmt(s.led2)}, " +
                                    "LED_3 - ${fmt(s.led3)}, " +
                                    "DISTANCE: ${s.distLabel ?: "N/A"}"
                        out.println(line)
                    }
                }
            }
            return outFile
        } catch (t: Throwable) {
            Log.e("LedDistLogger", "Failed to write log", t)
            return null
        } finally {
            resetLedDistLogBuffer()
        }
    }
}
