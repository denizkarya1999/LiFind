package com.developer27.lifind.videoprocessing

import android.content.Context
import android.graphics.Bitmap
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

/**
 * Helper that:
 *  - Loads/caches the TFLite interpreter
 *  - Normalizes/Parses YOLO head output ([1,84,8400] or [1,8400,84])
 *  - Applies NMS (class-agnostic or per-class)
 *  - Offers utilities used by VideoProcessor
 */
object YOLOLEDHelper {

    const val INPUT_SIZE = 640
    private const val MODEL_ASSET_NAME = "lifind_iso_100_yolo_11s.tflite"

    @Volatile
    private var interpreter: Interpreter? = null

    @Synchronized
    fun ensureInterpreter(context: Context): Interpreter {
        interpreter?.let { return it }

        val opts = Interpreter.Options().apply {
            // Reasonable default; change if needed
            setNumThreads(Runtime.getRuntime().availableProcessors().coerceAtLeast(2))
        }
        val mapped = loadModelFromAssets(context, MODEL_ASSET_NAME)
        val interp = Interpreter(mapped, opts)
        interpreter = interp
        return interp
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

    fun autoOrientAndResize(src: Bitmap): Bitmap {
        val base = if (src.config != Bitmap.Config.ARGB_8888) {
            src.copy(Bitmap.Config.ARGB_8888, false)
        } else src
        if (base.width == INPUT_SIZE && base.height == INPUT_SIZE) return base
        return Bitmap.createScaledBitmap(base, INPUT_SIZE, INPUT_SIZE, true)
    }

    // ---------- Parsing & NMS ----------
    private fun iou(a: DetectionResult, b: DetectionResult): Float {
        val ax1 = a.xCenter - 0.5f * a.width
        val ay1 = a.yCenter - 0.5f * a.height
        val ax2 = a.xCenter + 0.5f * a.width
        val ay2 = a.yCenter + 0.5f * a.height

        val bx1 = b.xCenter - 0.5f * b.width
        val by1 = b.yCenter - 0.5f * b.height
        val bx2 = b.xCenter + 0.5f * b.width
        val by2 = b.yCenter + 0.5f * b.height

        val x1 = maxOf(ax1, bx1)
        val y1 = maxOf(ay1, by1)
        val x2 = minOf(ax2, bx2)
        val y2 = minOf(ay2, by2)

        val iw = (x2 - x1).coerceAtLeast(0f)
        val ih = (y2 - y1).coerceAtLeast(0f)
        val inter = iw * ih
        val areaA = (ax2 - ax1).coerceAtLeast(0f) * (ay2 - ay1).coerceAtLeast(0f)
        val areaB = (bx2 - bx1).coerceAtLeast(0f) * (by2 - by1).coerceAtLeast(0f)
        val denom = areaA + areaB - inter
        return if (denom > 0f) inter / denom else 0f
    }

    private val labels = arrayOf("1010", "1000", "1001")

    fun classNameForId(id: Int): String =
        if (id in labels.indices) labels[id] else "N/A"

    fun getConfidence(det: DetectionResult): Float = det.confidence

    fun selectTopPerClass(
        dets: List<DetectionResult>,
        keepPerClass: Int = 1
    ): List<DetectionResult> {
        if (keepPerClass <= 0 || dets.isEmpty()) return dets
        return dets.groupBy { it.classId }
            .values
            .flatMap { it.sortedByDescending { d -> d.confidence }.take(keepPerClass) }
    }

    /**
     * Robust YOLO TFLite parser that supports heads with or without an "obj" column.
     *
     * expectedClasses: how many classes your model has (e.g., 3)
     * hasObjectness:   true if layout is [x,y,w,h,obj,classes...], false if [x,y,w,h,classes...]
     */
    /**
     * Robust YOLO TFLite parser that supports heads with or without an "obj" column.
     *
     * expectedClasses: how many classes your model has (e.g., 3)
     */
    fun parseTFLite(
        raw: Array<Array<FloatArray>>,
        confidenceThreshold: Float,
        classAgnosticNms: Boolean,
        multiLabelPerBox: Boolean,
        expectedClasses: Int
    ): List<DetectionResult> {

        val b = raw[0]
        val d1 = b.size
        val d2 = b[0].size

        val channels: Int
        val numPoints: Int
        val get: (c: Int, i: Int) -> Float

        // Layout detect: CHW [1,C,N] vs HWC [1,N,C]
        if (d1 <= 10 && d2 >= d1) { // CHW
            channels = d1
            numPoints = d2
            get = { c, i -> b[c][i] }
        } else {                    // HWC
            channels = d2
            numPoints = d1
            get = { c, i -> b[i][c] }
        }

        // Auto-detect objectness
        val hasObjectness = when (channels) {
            4 + expectedClasses -> false
            5 + expectedClasses -> true
            else -> {
                android.util.Log.e(
                    "YOLOParser",
                    "Unexpected head width=$channels (expected 4+K or 5+K with K=$expectedClasses)"
                )
                return emptyList()
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

            // Normalize boxes if they look like pixels
            val cxN = if (x > 1.5f) x / INPUT_SIZE else x
            val cyN = if (y > 1.5f) y / INPUT_SIZE else y
            val wN = if (w > 1.5f) w / INPUT_SIZE else w
            val hN = if (h > 1.5f) h / INPUT_SIZE else h

            if (multiLabelPerBox) {
                for (c in 0 until expectedClasses) {
                    val clsP = get(clsStart + c, i)
                    val conf = if (hasObjectness) obj * clsP else clsP
                    if (conf >= confidenceThreshold) {
                        candidates.add(
                            DetectionResult(
                                xCenter = cxN.coerceIn(0f, 1f),
                                yCenter = cyN.coerceIn(0f, 1f),
                                width = wN.coerceIn(0f, 1f),
                                height = hN.coerceIn(0f, 1f),
                                confidence = conf,
                                classId = c
                            )
                        )
                    }
                }
            } else {
                var bestC = 0
                var bestP = get(clsStart, i)
                for (c in 1 until expectedClasses) {
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
                            width = wN.coerceIn(0f, 1f),
                            height = hN.coerceIn(0f, 1f),
                            confidence = conf,
                            classId = bestC
                        )
                    )
                }
            }
        }

        if (candidates.isEmpty()) return emptyList()

        // NMS (class-agnostic or per-class)
        val iouThresh = Settings.Inference.iouThreshold
        candidates.sortByDescending { it.confidence }
        val kept = ArrayList<DetectionResult>(candidates.size)
        val removed = BooleanArray(candidates.size)
        for (i in candidates.indices) {
            if (removed[i])
                continue
            val a = candidates[i]
            kept.add(a)
            for (j in i + 1 until candidates.size) {
                if (removed[j]) continue
                val bDet = candidates[j]
                if (!classAgnosticNms && a.classId != bDet.classId)
                    continue
                if (iou(a, bDet) > iouThresh)
                    removed[j] = true
            }
        }
        return kept
    }
}