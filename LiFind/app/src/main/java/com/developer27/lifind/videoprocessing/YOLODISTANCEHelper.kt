package com.developer27.lifind.videoprocessing

import android.content.Context
import android.graphics.Bitmap
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.max

/**
 * YOLO distance helper for a "single-box" use case.
 *
 * Responsibilities:
 *  - Loads/caches the Distance TFLite interpreter
 *  - Normalizes/Parses YOLO head output ([1,84,8400] or [1,8400,84]) with/without objectness
 *  - Applies NMS (class-agnostic or per-class)
 *  - Exposes "one box" post-processor (best single box overall)
 *
 * Model: lifind_distance_100_yolo_11s.tflite
 * Classes: "11","12","13","14","15","16", "17",
 *         "18", "19", "20", "21", "22", "23", "24", "25", "26","27"
 */
object YOLODISTANCEHelper {

    // ------------- Model / I/O -------------

    const val INPUT_SIZE = 640
    private const val MODEL_ASSET_NAME = "lifind_iso_distance_100_yolo_11m_ultralytics_hub.tflite"

    /** Public for convenience when wiring expected class count. */
    val NUM_CLASSES: Int get() = labels.size

    @Volatile
    private var interpreter: Interpreter? = null

    @Synchronized
    fun ensureInterpreter(context: Context): Interpreter {
        interpreter?.let { return it }

        val opts = Interpreter.Options().apply {
            // Reasonable default threads; tune if you profile differently.
            setNumThreads(Runtime.getRuntime().availableProcessors().coerceAtLeast(2))
        }
        val mapped = loadModelFromAssets(context, MODEL_ASSET_NAME)
        return Interpreter(mapped, opts).also { interpreter = it }
    }

    fun closeInterpreter() {
        try {
            interpreter?.close()
        } finally {
            interpreter = null
        }
    }

    private fun loadModelFromAssets(context: Context, assetName: String): MappedByteBuffer {
        val afd = context.assets.openFd(assetName)
        FileInputStream(afd.fileDescriptor).channel.use { ch ->
            return ch.map(FileChannel.MapMode.READ_ONLY, afd.startOffset, afd.length)
        }
    }

    /** Ensures 640×640 ARGB_8888; no EXIF handling needed for camera frames. */
    fun autoOrientAndResize(src: Bitmap): Bitmap {
        val base = if (src.config != Bitmap.Config.ARGB_8888) {
            src.copy(Bitmap.Config.ARGB_8888, false)
        } else src
        if (base.width == INPUT_SIZE && base.height == INPUT_SIZE) return base
        return Bitmap.createScaledBitmap(base, INPUT_SIZE, INPUT_SIZE, true)
    }

    // ------------- Labels / utilities -------------

    private val labels = arrayOf("11","12","13","14","15","16", "17",
        "18", "19", "20", "21", "22", "23", "24", "25", "26","27")

    fun classNameForId(id: Int): String =
        if (id in labels.indices) labels[id] else "N/A"

    fun getConfidence(det: DetectionResult): Float = det.confidence

    // ------------- Parsing + NMS -------------

    private fun iou(a: DetectionResult, b: DetectionResult): Float {
        val ax1 = a.xCenter - 0.5f * a.width
        val ay1 = a.yCenter - 0.5f * a.height
        val ax2 = a.xCenter + 0.5f * a.width
        val ay2 = a.yCenter + 0.5f * a.height

        val bx1 = b.xCenter - 0.5f * b.width
        val by1 = b.yCenter - 0.5f * b.height
        val bx2 = b.xCenter + 0.5f * b.width
        val by2 = b.yCenter + 0.5f * b.height

        val x1 = max(ax1, bx1)
        val y1 = max(ay1, by1)
        val x2 = kotlin.math.min(ax2, bx2)
        val y2 = kotlin.math.min(ay2, by2)

        val iw = (x2 - x1).coerceAtLeast(0f)
        val ih = (y2 - y1).coerceAtLeast(0f)
        val inter = iw * ih
        val areaA = (ax2 - ax1).coerceAtLeast(0f) * (ay2 - ay1).coerceAtLeast(0f)
        val areaB = (bx2 - bx1).coerceAtLeast(0f) * (by2 - by1).coerceAtLeast(0f)
        val denom = areaA + areaB - inter
        return if (denom > 0f) inter / denom else 0f
    }

// In YOLODISTANCEHelper

    /**
     * Robust YOLO TFLite parser that:
     *  - Detects CHW/HWC by picking the plausible channel dimension (6..512)
     *  - Auto-detects classes K and presence of obj column if expectedClasses <= 0
     *    or when channels != 4+K and channels != 5+K
     *
     * @param expectedClasses If <= 0, auto-detect K from the head width.
     */
    fun parseTFLite(
        raw: Array<Array<FloatArray>>,
        confidenceThreshold: Float,
        classAgnosticNms: Boolean,
        multiLabelPerBox: Boolean,
        expectedClasses: Int
    ): List<DetectionResult> {

        if (raw.isEmpty() || raw[0].isEmpty()) return emptyList()

        val b = raw[0]
        val d1 = b.size
        val d2 = b[0].size

        // ---- Layout detection: pick the plausible channel dimension ----
        // Typical channels are small-ish (e.g., 6..~256), while points are large (hundreds/thousands).
        val plausibleMaxChannels = 512
        val d1LooksLikeChannels = d1 in 6..plausibleMaxChannels && d2 >= d1
        val d2LooksLikeChannels = d2 in 6..plausibleMaxChannels && d1 >= d2

        val channels: Int
        val numPoints: Int
        val get: (c: Int, i: Int) -> Float

        when {
            d1LooksLikeChannels && !d2LooksLikeChannels -> {
                channels = d1; numPoints = d2; get = { c, i -> b[c][i] } // CHW
            }
            d2LooksLikeChannels && !d1LooksLikeChannels -> {
                channels = d2; numPoints = d1; get = { c, i -> b[i][c] } // HWC
            }
            else -> {
                // Ambiguous: choose the smaller as channels
                if (d1 <= d2) {
                    channels = d1; numPoints = d2; get = { c, i -> b[c][i] } // assume CHW
                } else {
                    channels = d2; numPoints = d1; get = { c, i -> b[i][c] } // assume HWC
                }
            }
        }

        // ---- Auto-detect K and objectness if needed ----
        var k = expectedClasses
        val hasObjectness: Boolean = run {
            if (k > 0 && (channels == 4 + k || channels == 5 + k)) {
                channels == 5 + k
            } else {
                val kNoObj = channels - 4
                val kWithObj = channels - 5
                when {
                    kNoObj in 1..plausibleMaxChannels -> { k = kNoObj; false }
                    kWithObj in 1..plausibleMaxChannels -> { k = kWithObj; true }
                    else -> {
                        android.util.Log.e(
                            "YOLODistanceParser",
                            "Unexpected head width=$channels (d1=$d1, d2=$d2). " +
                                    "Cannot infer K from 4+K or 5+K."
                        )
                        return emptyList()
                    }
                }
            }
        }

        val clsStart = if (hasObjectness) 5 else 4
        val candidates = ArrayList<DetectionResult>(numPoints * (if (multiLabelPerBox) 2 else 1))

        for (i in 0 until numPoints) {
            val x = get(0, i)
            val y = get(1, i)
            val w = get(2, i)
            val h = get(3, i)
            val obj = if (hasObjectness) get(4, i) else 1f

            // Normalize from pixels -> 0..1 if needed
            val cxN = if (x > 1.5f) x / INPUT_SIZE else x
            val cyN = if (y > 1.5f) y / INPUT_SIZE else y
            val wN  = if (w > 1.5f) w / INPUT_SIZE else w
            val hN  = if (h > 1.5f) h / INPUT_SIZE else h

            if (multiLabelPerBox) {
                for (c in 0 until k) {
                    val clsP = get(clsStart + c, i)
                    val conf = if (hasObjectness) obj * clsP else clsP
                    if (conf >= confidenceThreshold) {
                        candidates.add(
                            DetectionResult(
                                xCenter = cxN.coerceIn(0f, 1f),
                                yCenter = cyN.coerceIn(0f, 1f),
                                width   = wN.coerceIn(0f, 1f),
                                height  = hN.coerceIn(0f, 1f),
                                confidence = conf,
                                classId = c
                            )
                        )
                    }
                }
            } else {
                var bestC = 0
                var bestP = get(clsStart, i)
                for (c in 1 until k) {
                    val p = get(clsStart + c, i)
                    if (p > bestP) {
                        bestP = p; bestC = c
                    }
                }
                val conf = if (hasObjectness) obj * bestP else bestP
                if (conf >= confidenceThreshold) {
                    candidates.add(
                        DetectionResult(
                            xCenter = cxN.coerceIn(0f, 1f),
                            yCenter = cyN.coerceIn(0f, 1f),
                            width   = wN.coerceIn(0f, 1f),
                            height  = hN.coerceIn(0f, 1f),
                            confidence = conf,
                            classId = bestC
                        )
                    )
                }
            }
        }

        if (candidates.isEmpty()) return emptyList()

        // ---- NMS (class-agnostic or per-class) ----
        val iouThresh = Settings.Inference.iouThreshold
        candidates.sortByDescending { it.confidence }

        val kept = ArrayList<DetectionResult>(candidates.size)
        val removed = BooleanArray(candidates.size)

        for (i in candidates.indices) {
            if (removed[i]) continue
            val a = candidates[i]
            kept.add(a)

            for (j in i + 1 until candidates.size) {
                if (removed[j]) continue
                val bDet = candidates[j]
                if (!classAgnosticNms && a.classId != bDet.classId)
                    continue
                // local IoU (same as in YOLOLEDHelper)
                val ax1 = a.xCenter - 0.5f * a.width
                val ay1 = a.yCenter - 0.5f * a.height
                val ax2 = a.xCenter + 0.5f * a.width
                val ay2 = a.yCenter + 0.5f * a.height
                val bx1 = bDet.xCenter - 0.5f * bDet.width
                val by1 = bDet.yCenter - 0.5f * bDet.height
                val bx2 = bDet.xCenter + 0.5f * bDet.width
                val by2 = bDet.yCenter + 0.5f * bDet.height
                val x1 = maxOf(ax1, bx1)
                val y1 = maxOf(ay1, by1)
                val x2 = kotlin.math.min(ax2, bx2)
                val y2 = kotlin.math.min(ay2, by2)
                val iw = (x2 - x1).coerceAtLeast(0f)
                val ih = (y2 - y1).coerceAtLeast(0f)
                val inter = iw * ih
                val areaA = (ax2 - ax1).coerceAtLeast(0f) * (ay2 - ay1).coerceAtLeast(0f)
                val areaB = (bx2 - bx1).coerceAtLeast(0f) * (by2 - by1).coerceAtLeast(0f)
                val denom = areaA + areaB - inter
                val iou = if (denom > 0f) inter / denom else 0f
                if (iou > iouThresh)
                    removed[j] = true
            }
        }
        return kept
    }
}