package com.developer27.lifind.videoprocessing

import android.content.Context
import android.graphics.Bitmap
import android.os.Environment
import android.util.Log
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
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
    val led1: Pair<Int, Int>?,
    val led2: Pair<Int, Int>?,
    val led3: Pair<Int, Int>?,
    val dist1Cm: Double?,   // distance matched to LED_1 (cm), or null
    val dist2Cm: Double?,   // matched to LED_2
    val dist3Cm: Double?    // matched to LED_3
)

private val ledDistSamples = java.util.Collections.synchronizedList(mutableListOf<LedDistanceSample>())
private fun resetLedDistLogBuffer() = ledDistSamples.clear()

object Settings {
    object DetectionMode {
        enum class Mode { YOLO }
        var current: Mode = Mode.YOLO
    }
    object Inference {
        var confidenceThreshold: Float = 0.95f
        var iouThreshold: Float = 0.45f
        var classAgnosticNms: Boolean = false
        var multiLabelPerBox: Boolean = true
        var topPerClass: Int = 1
    }
    object BoundingBox {
        var enableBoundingBox = true

        // BGR palette for OpenCV
        val perClassColors: Array<Scalar> = arrayOf(
            Scalar(255.0, 0.0, 0.0), // class 0 -> Red
            Scalar(80.0, 200.0, 120.0), // class 1 -> Green
            Scalar(0.0, 150.0, 255.0)  // class 2 -> Blue
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

        // 1) Prepare 640×640 input
        val inputBitmap = YOLOLEDHelper.autoOrientAndResize(bitmap)
        if (inputBitmap.isRecycled) return@withContext null
        ensureReusedBitmaps()

        // Make sure the drawing Mat contains the image we will annotate
        Utils.bitmapToMat(inputBitmap, workMat)

        // 3) Load shared TFLite input
        tensorImage.load(inputBitmap)

        // -------------------- LED MODEL (multi-box) --------------------
        val ledInterp = YOLOLEDHelper.ensureInterpreter(context)
        ensureOutputBuffers(ledInterp) // keeps distOut sized for (LED head) shape
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
        val ledDets: List<DetectionResult> =
            if (Settings.Inference.topPerClass > 0)
                YOLOLEDHelper.selectTopPerClass(rawLedDetections, Settings.Inference.topPerClass)
            else rawLedDetections

        // -------------------- DISTANCE MODEL (top-3) --------------------
        val distInterp = YOLODISTANCEHelper.ensureInterpreter(context)
        val distShape = distInterp.getOutputTensor(0).shape() // e.g., [1,84,8400] or [1,8400,84]
        val distHead = Array(distShape[0]) { Array(distShape[1]) { FloatArray(distShape[2]) } }
        synchronized(distInterp) {
            distInterp.run(tensorImage.buffer, distHead)
        }
        val distDets: List<DetectionResult> =
            YOLODISTANCEHelper.parseTFLite(
                raw = distHead,
                confidenceThreshold = Settings.Inference.confidenceThreshold,
                multiLabelPerBox = Settings.Inference.multiLabelPerBox,
                expectedClasses = 3,
                classAgnosticNms = true
            )
                .sortedByDescending { YOLODISTANCEHelper.getConfidence(it) }
                .take(3)

        // -------------------- ASSOCIATE Distance → nearest LED --------------------
        data class BoxCtr(val cx: Float, val cy: Float, val x1: Int, val y1: Int, val x2: Int, val y2: Int)
        val sizeF = YOLOLEDHelper.INPUT_SIZE.toFloat()
        val right = (sizeF - 1).toInt()
        val bottom = (sizeF - 1).toInt()

        // Per-LED (classId 0..2) first/top box with coords
        val ledBoxes = arrayOfNulls<BoxCtr>(3)
        run {
            for (det in ledDets) {
                val cls = det.classId
                if (cls in 0..2 && ledBoxes[cls] == null) {
                    val cx = det.xCenter * sizeF
                    val cy = det.yCenter * sizeF
                    val w  = det.width  * sizeF
                    val h  = det.height * sizeF
                    val x1 = (cx - 0.5f * w).toInt().coerceIn(0, right)
                    val y1 = (cy - 0.5f * h).toInt().coerceIn(0, bottom)
                    val x2 = (cx + 0.5f * w).toInt().coerceIn(0, right)
                    val y2 = (cy + 0.5f * h).toInt().coerceIn(0, bottom)
                    ledBoxes[cls] = BoxCtr(cx, cy, x1, y1, x2, y2)
                }
            }
        }

        // Distance centers
        val distCenters = distDets.map { d ->
            val cx = d.xCenter * sizeF
            val cy = d.yCenter * sizeF
            Triple(cx, cy, d) // keep detection along with center
        }.toMutableList()

        // Greedy nearest-neighbor assignment (no reuse)
        val distPerLedCm = arrayOf<Double?>(null, null, null)
        val used = BooleanArray(distCenters.size)
        for (ledIdx in 0..2) {
            val lb = ledBoxes[ledIdx] ?: continue
            var bestIdx = -1
            var bestD2 = Float.MAX_VALUE
            for (i in distCenters.indices) {
                if (used[i]) continue
                val (cx, cy, _) = distCenters[i]
                val dx = cx - lb.cx
                val dy = cy - lb.cy
                val d2 = dx*dx + dy*dy
                if (d2 < bestD2) { bestD2 = d2; bestIdx = i }
            }
            if (bestIdx >= 0) {
                used[bestIdx] = true
                val det = distCenters[bestIdx].third
                val label = YOLODISTANCEHelper.classNameForId(det.classId) // e.g., "15"
                distPerLedCm[ledIdx] = label.toDoubleOrNull()
            }
        }

        // -------------------- DRAWING --------------------
        if (Settings.BoundingBox.enableBoundingBox) {
            var xLb: Float; var yLb: Float; var wLb: Float; var hLb: Float
            var x1: Int; var y1: Int; var x2: Int; var y2: Int
            var confPct: Int
            val labelBuf = StringBuilder(64)

            // LED boxes (with associated distance text if present)
            if (ledDets.isNotEmpty()) {
                ledDets.forEach { det ->
                    xLb = det.xCenter * sizeF
                    yLb = det.yCenter * sizeF
                    wLb = det.width  * sizeF
                    hLb = det.height * sizeF

                    x1 = (xLb - 0.5f * wLb).toInt().coerceIn(0, right)
                    y1 = (yLb - 0.5f * hLb).toInt().coerceIn(0, bottom)
                    x2 = (xLb + 0.5f * wLb).toInt().coerceIn(0, right)
                    y2 = (yLb + 0.5f * hLb).toInt().coerceIn(0, bottom)

                    confPct = (YOLOLEDHelper.getConfidence(det) * 100f + 0.5f).toInt()
                    val ledName = YOLOLEDHelper.classNameForId(det.classId)
                    val distForThisLed = distPerLedCm.getOrNull(det.classId)
                    labelBuf.clear()
                    labelBuf.append("LED: ").append(ledName)
                        .append(" - Acc: ").append(confPct).append('%')

                    val color = Settings.BoundingBox.perClassColors.getOrNull(det.classId)
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

            // Distance boxes (up to 3) — unchanged drawing
            if (distDets.isNotEmpty()) {
                distDets.forEach { det ->
                    xLb = det.xCenter * sizeF
                    yLb = det.yCenter * sizeF
                    wLb = det.width  * sizeF
                    hLb = det.height * sizeF

                    x1 = (xLb - 0.5f * wLb).toInt().coerceIn(0, right)
                    y1 = (yLb - 0.5f * hLb).toInt().coerceIn(0, bottom)
                    x2 = (xLb + 0.5f * wLb).toInt().coerceIn(0, right)
                    y2 = (yLb + 0.5f * hLb).toInt().coerceIn(0, bottom)

                    confPct = (YOLODISTANCEHelper.getConfidence(det) * 100f + 0.5f).toInt()
                    val distLabel = YOLODISTANCEHelper.classNameForId(det.classId)
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
        }

        // -------------------- LOGGING (per LED distance) --------------------
        val nowMs = System.currentTimeMillis()

        // LED centers in pixels (for log)
        val ledCenters = arrayOfNulls<Pair<Int, Int>>(3)
        for (cls in 0..2) {
            ledBoxes[cls]?.let { b ->
                ledCenters[cls] = b.cx.toInt().coerceIn(0, right) to b.cy.toInt().coerceIn(0, bottom)
            }
        }

        ledDistSamples += LedDistanceSample(
            tMillis = nowMs,
            led1 = ledCenters[0],
            led2 = ledCenters[1],
            led3 = ledCenters[2],
            dist1Cm = distPerLedCm[0],
            dist2Cm = distPerLedCm[1],
            dist3Cm = distPerLedCm[2]
        )

        // 6) Convert Mat → Bitmap safely
        val matW = workMat.cols()
        val matH = workMat.rows()
        val annotated = reusedOutBitmap!!.let {
            if (it.width != matW || it.height != matH) {
                reusedOutBitmap = Bitmap.createBitmap(matW, matH, Bitmap.Config.ARGB_8888)
                reusedOutBitmap!!
            } else it
        }
        Utils.matToBitmap(workMat, annotated)

        // Return (annotated overlay, and the 640×640 model input)
        annotated to inputBitmap
    }

    fun writeLedDistLogToFile(): File? {
        fun fmtCoordBraced(pt: Pair<Int, Int>?): String =
            if (pt == null) "{x=N/A, y=N/A}" else "{x=${pt.first}, y=${pt.second}}"

        fun fmtDistanceBraced(v: Double?): String =
            if (v == null) "{N/A}" else "{${v.toInt()} CM}"

        val name = "LiFind_Log.txt"
        val docsDir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOCUMENTS)
        if (!docsDir.exists()) docsDir.mkdirs()

        val outFile = File(docsDir, name)
        try {
            PrintWriter(BufferedWriter(FileWriter(outFile))).use { out ->
                synchronized(ledDistSamples) {
                    val s = ledDistSamples.lastOrNull() ?: return@use
                    out.println("LED_1 -> Coordinates: {x=0, y=2} - Distance: ${fmtDistanceBraced(s.dist1Cm)}")
                    out.println("LED_2 -> Coordinates: {x=-2, y=-2) - Distance: ${fmtDistanceBraced(s.dist2Cm)}")
                    out.println("LED_3 -> Coordinates: {x=2, y=-2} - Distance: ${fmtDistanceBraced(s.dist3Cm)}")
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
