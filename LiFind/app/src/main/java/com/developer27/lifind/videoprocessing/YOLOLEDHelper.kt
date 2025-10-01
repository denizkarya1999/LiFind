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
    private const val MODEL_ASSET_NAME = "lifind_iso_led_100_yolo_11l_ultralytics_hub.tflite"

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

    private val labels = arrayOf("1010", "1001", "1000")

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
    fun parseTFLite(
        raw: Array<Array<FloatArray>>,
        confidenceThreshold: Float,
        classAgnosticNms: Boolean,
        multiLabelPerBox: Boolean,
        expectedClasses: Int
    ): List<DetectionResult> {

        val b = raw[0]
        val d1 = b.size           // first 2D dim
        val d2 = b[0].size        // second 2D dim

        // Robust layout detection:
        // Prefer the dimension that matches 4+K (no obj) or 5+K (with obj).
        val cNoObj = 4 + expectedClasses
        val cWithObj = 5 + expectedClasses

        val isCHW = when {
            d1 == cNoObj || d1 == cWithObj -> true          // [1, C, N] -> channels in d1
            d2 == cNoObj || d2 == cWithObj -> false         // [1, N, C] -> channels in d2
            // Fallback to legacy heuristic only if neither side matches
            d1 <= 10 && d2 >= d1 -> true
            else -> false
        }

        val channels = if (isCHW) d1 else d2
        val numPoints = if (isCHW) d2 else d1

        // Determine presence of objectness based on channels
        val hasObjectness = when (channels) {
            cNoObj    -> false
            cWithObj  -> true
            else -> {
                android.util.Log.e(
                    "YOLOParser",
                    "Unexpected head width=$channels (expected 4+K or 5+K with K=$expectedClasses)"
                )
                return emptyList()
            }
        }

        val clsStart = if (hasObjectness) 5 else 4
        val invInput = 1f / INPUT_SIZE
        val confThresh = confidenceThreshold

        // Pre-size to minimize resizes on large heads (YOLO 11M+)
        val est = if (multiLabelPerBox) (numPoints * 2.0f).toInt() else numPoints
        val candidates = ArrayList<DetectionResult>(est.coerceIn(16, 131072))

        fun clamp01(v: Float): Float = if (v < 0f) 0f else if (v > 1f) 1f else v

        if (isCHW) {
            // CHW: [1, C, N] => b[c][i]
            for (i in 0 until numPoints) {
                val x = b[0][i]; val y = b[1][i]; val w = b[2][i]; val h = b[3][i]
                val obj = if (hasObjectness) b[4][i] else 1f

                // Safe early prune when objectness is present
                if (hasObjectness && obj < confThresh) continue

                val cxN = if (x > 1.5f) x * invInput else x
                val cyN = if (y > 1.5f) y * invInput else y
                val wN  = if (w > 1.5f) w * invInput else w
                val hN  = if (h > 1.5f) h * invInput else h

                val xc = clamp01(cxN)
                val yc = clamp01(cyN)
                val ww = clamp01(wN.coerceAtLeast(0f))
                val hh = clamp01(hN.coerceAtLeast(0f))

                if (multiLabelPerBox) {
                    val base = if (hasObjectness) obj else 1f
                    for (c in 0 until expectedClasses) {
                        val clsP = b[clsStart + c][i]
                        val conf = if (hasObjectness) base * clsP else clsP
                        if (conf >= confThresh) {
                            candidates.add(
                                DetectionResult(
                                    xCenter = xc, yCenter = yc, width = ww, height = hh,
                                    confidence = conf, classId = c
                                )
                            )
                        }
                    }
                } else {
                    var bestC = 0
                    var bestP = b[clsStart][i]
                    for (c in 1 until expectedClasses) {
                        val p = b[clsStart + c][i]
                        if (p > bestP) { bestP = p; bestC = c }
                    }
                    val conf = if (hasObjectness) obj * bestP else bestP
                    if (conf >= confThresh) {
                        candidates.add(
                            DetectionResult(
                                xCenter = xc, yCenter = yc, width = ww, height = hh,
                                confidence = conf, classId = bestC
                            )
                        )
                    }
                }
            }
        } else {
            // HWC: [1, N, C] => b[i][c]
            for (i in 0 until numPoints) {
                val x = b[i][0]; val y = b[i][1]; val w = b[i][2]; val h = b[i][3]
                val obj = if (hasObjectness) b[i][4] else 1f

                if (hasObjectness && obj < confThresh) continue

                val cxN = if (x > 1.5f) x * invInput else x
                val cyN = if (y > 1.5f) y * invInput else y
                val wN  = if (w > 1.5f) w * invInput else w
                val hN  = if (h > 1.5f) h * invInput else h

                val xc = clamp01(cxN)
                val yc = clamp01(cyN)
                val ww = clamp01(wN.coerceAtLeast(0f))
                val hh = clamp01(hN.coerceAtLeast(0f))

                if (multiLabelPerBox) {
                    val base = if (hasObjectness) obj else 1f
                    for (c in 0 until expectedClasses) {
                        val clsP = b[i][clsStart + c]
                        val conf = if (hasObjectness) base * clsP else clsP
                        if (conf >= confThresh) {
                            candidates.add(
                                DetectionResult(
                                    xCenter = xc, yCenter = yc, width = ww, height = hh,
                                    confidence = conf, classId = c
                                )
                            )
                        }
                    }
                } else {
                    var bestC = 0
                    var bestP = b[i][clsStart]
                    for (c in 1 until expectedClasses) {
                        val p = b[i][clsStart + c]
                        if (p > bestP) { bestP = p; bestC = c }
                    }
                    val conf = if (hasObjectness) obj * bestP else bestP
                    if (conf >= confThresh) {
                        candidates.add(
                            DetectionResult(
                                xCenter = xc, yCenter = yc, width = ww, height = hh,
                                confidence = conf, classId = bestC
                            )
                        )
                    }
                }
            }
        }

        if (candidates.isEmpty()) return emptyList()

        // Sort once by confidence (desc)
        candidates.sortByDescending { it.confidence }

        // Precompute xyxy + area once for fast IoU
        val n = candidates.size
        val x1 = FloatArray(n)
        val y1 = FloatArray(n)
        val x2 = FloatArray(n)
        val y2 = FloatArray(n)
        val area = FloatArray(n)

        for (i in 0 until n) {
            val d = candidates[i]
            val hw = 0.5f * d.width
            val hh = 0.5f * d.height
            val xx1 = d.xCenter - hw
            val yy1 = d.yCenter - hh
            val xx2 = d.xCenter + hw
            val yy2 = d.yCenter + hh
            x1[i] = if (xx1 < 0f) 0f else xx1
            y1[i] = if (yy1 < 0f) 0f else yy1
            x2[i] = if (xx2 > 1f) 1f else xx2
            y2[i] = if (yy2 > 1f) 1f else yy2
            val wA = (x2[i] - x1[i]).coerceAtLeast(0f)
            val hA = (y2[i] - y1[i]).coerceAtLeast(0f)
            area[i] = wA * hA
        }

        val iouThresh = Settings.Inference.iouThreshold
        val removed = BooleanArray(n)
        val kept = ArrayList<DetectionResult>(n)

        fun iouQuick(i: Int, j: Int): Float {
            val xx1 = if (x1[i] > x1[j]) x1[i] else x1[j]
            val yy1 = if (y1[i] > y1[j]) y1[i] else y1[j]
            val xx2 = if (x2[i] < x2[j]) x2[i] else x2[j]
            val yy2 = if (y2[i] < y2[j]) y2[i] else y2[j]
            val iw = xx2 - xx1
            if (iw <= 0f) return 0f
            val ih = yy2 - yy1
            if (ih <= 0f) return 0f
            val inter = iw * ih
            val u = area[i] + area[j] - inter
            return if (u > 0f) inter / u else 0f
        }

        if (classAgnosticNms) {
            // Greedy global NMS
            for (i in 0 until n) {
                if (removed[i]) continue
                kept.add(candidates[i])
                for (j in i + 1 until n) {
                    if (!removed[j] && iouQuick(i, j) > iouThresh) {
                        removed[j] = true
                    }
                }
            }
        } else {
            // Per-class NMS to reduce IoU checks
            val buckets = Array(expectedClasses) { ArrayList<Int>() }
            for (i in 0 until n) {
                val cls = candidates[i].classId
                if (cls in 0 until expectedClasses) buckets[cls].add(i) else buckets[0].add(i)
            }
            for (idxs in buckets) {
                val m = idxs.size
                for (aPos in 0 until m) {
                    val i = idxs[aPos]
                    if (removed[i]) continue
                    kept.add(candidates[i])
                    for (bPos in aPos + 1 until m) {
                        val j = idxs[bPos]
                        if (!removed[j] && iouQuick(i, j) > iouThresh) {
                            removed[j] = true
                        }
                    }
                }
            }
            if (kept.size > 1) kept.sortByDescending { it.confidence }
        }

        return kept
    }

}