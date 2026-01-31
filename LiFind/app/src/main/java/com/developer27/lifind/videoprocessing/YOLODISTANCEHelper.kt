package com.developer27.lifind.videoprocessing

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.max
import kotlin.math.min

/**
 * YOLO distance helper for a "single-box" use case.
 *
 * Responsibilities:
 *  - Loads/caches the Distance TFLite interpreter
 *  - Parses YOLO outputs:
 *      Raw heads:    [1, (4+K or 5+K), N] OR [1, N, (4+K or 5+K)]
 *      Decoded head: [1, 6, N] OR [1, N, 6]  (end2end / fused)
 *  - Applies NMS (class-agnostic or per-class)
 *
 * Model: lifind_new_distance_detection_custom_yolo26m.tflite
 * Classes: 33 distance labels (see labels array)
 */
object YOLODISTANCEHelper {

    // ------------- Model / I/O -------------

    const val INPUT_SIZE = 640
    private const val MODEL_ASSET_NAME = "lifind_new_distance_detection_custom_yolo26m.tflite"

    /** Public for convenience when wiring expected class count. */
    val NUM_CLASSES: Int get() = labels.size

    @Volatile
    private var interpreter: Interpreter? = null

    @Synchronized
    fun ensureInterpreter(context: Context): Interpreter {
        interpreter?.let { return it }

        val opts = Interpreter.Options().apply {
            setNumThreads(Runtime.getRuntime().availableProcessors().coerceAtLeast(2))
        }
        val mapped = loadModelFromAssets(context, MODEL_ASSET_NAME)
        val interp = Interpreter(mapped, opts)

        // Optional: log output tensor shape to confirm head layout
        try {
            val out0 = interp.getOutputTensor(0)
            Log.d("TFLite", "YOLO DIST out[0] shape = ${out0.shape().contentToString()}")
        } catch (_: Throwable) {}

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

    /** Ensures 640×640 ARGB_8888; no EXIF handling needed for camera frames. */
    fun autoOrientAndResize(src: Bitmap): Bitmap {
        val base = if (src.config != Bitmap.Config.ARGB_8888) {
            src.copy(Bitmap.Config.ARGB_8888, false)
        } else src
        if (base.width == INPUT_SIZE && base.height == INPUT_SIZE) return base
        return Bitmap.createScaledBitmap(base, INPUT_SIZE, INPUT_SIZE, true)
    }

    // ------------- Labels / utilities -------------

    private val labels = arrayOf(
        "236.22",  // class 0
        "241",     // class 1
        "241.3",   // class 2
        "243.84",  // class 3
        "245",     // class 4
        "246",     // class 5
        "246.38",  // class 6
        "247",     // class 7
        "248.92",  // class 8
        "249",     // class 9
        "251.46",  // class 10
        "256.50",  // class 11
        "256.54",  // class 12
        "259.05",  // class 13
        "259.08",  // class 14
        "260.65",  // class 15
        "261",     // class 16
        "261.62",  // class 17
        "264.12",  // class 18
        "264.16",  // class 19
        "266.71",  // class 20
        "266.80",  // class 21
        "276",     // class 22
        "276.86",  // class 23
        "276.88",  // class 24
        "281.94",  // class 25
        "284.48",  // class 26
        "287.02",  // class 27
        "289",     // class 28
        "289.56",  // class 29
        "291.1",   // class 30
        "297.18",  // class 31
        "304.8"    // class 32
    )

    fun classNameForId(id: Int): String =
        if (id in labels.indices) labels[id] else "N/A"

    fun getConfidence(det: DetectionResult): Float = det.confidence

    private fun clamp01(v: Float): Float = when {
        v < 0f -> 0f
        v > 1f -> 1f
        else -> v
    }

    // ------------- IoU -------------

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
        val x2 = min(ax2, bx2)
        val y2 = min(ay2, by2)

        val iw = (x2 - x1).coerceAtLeast(0f)
        val ih = (y2 - y1).coerceAtLeast(0f)
        val inter = iw * ih

        val areaA = (ax2 - ax1).coerceAtLeast(0f) * (ay2 - ay1).coerceAtLeast(0f)
        val areaB = (bx2 - bx1).coerceAtLeast(0f) * (by2 - by1).coerceAtLeast(0f)
        val denom = areaA + areaB - inter

        return if (denom > 0f) inter / denom else 0f
    }

    // ------------- NMS -------------

    private fun applyNms(
        candidates: MutableList<DetectionResult>,
        expectedClasses: Int,
        classAgnosticNms: Boolean
    ): List<DetectionResult> {
        if (candidates.isEmpty()) return emptyList()

        candidates.sortByDescending { it.confidence }

        val n = candidates.size
        val removed = BooleanArray(n)
        val kept = ArrayList<DetectionResult>(n)

        for (i in 0 until n) {
            if (removed[i]) continue
            val a = candidates[i]
            kept.add(a)

            for (j in i + 1 until n) {
                if (removed[j]) continue
                val b = candidates[j]

                if (!classAgnosticNms && a.classId != b.classId) continue

                if (iou(a, b) > Settings.Inference.iouThreshold) {
                    removed[j] = true
                }
            }
        }

        // Keep descending by confidence
        if (kept.size > 1) kept.sortByDescending { it.confidence }
        return kept
    }

    /**
     * Robust YOLO TFLite parser that supports:
     *  - Decoded head width=6 -> [a0,a1,a2,a3,conf,classId]
     *  - Raw heads 4+K or 5+K -> [x,y,w,h,(obj),classes...]
     *
     * expectedClasses:
     *  - pass NUM_CLASSES (33) for this distance model.
     */
    fun parseTFLite(
        raw: Array<Array<FloatArray>>,
        confidenceThreshold: Float,
        classAgnosticNms: Boolean,
        multiLabelPerBox: Boolean,
        expectedClasses: Int
    ): List<DetectionResult> {

        if (raw.isEmpty() || raw[0].isEmpty() || raw[0][0].isEmpty()) return emptyList()

        val b = raw[0]
        val d1 = b.size
        val d2 = b[0].size

        // Decide CHW ([C][N]) vs HWC ([N][C]) by plausible channel sizes
        val plausibleMaxChannels = 512
        val d1LooksLikeChannels = d1 in 6..plausibleMaxChannels && d2 >= d1
        val d2LooksLikeChannels = d2 in 6..plausibleMaxChannels && d1 >= d2

        val channels: Int
        val numPoints: Int
        val get: (c: Int, i: Int) -> Float

        when {
            d1LooksLikeChannels && !d2LooksLikeChannels -> {
                channels = d1
                numPoints = d2
                get = { c, i -> b[c][i] } // CHW
            }
            d2LooksLikeChannels && !d1LooksLikeChannels -> {
                channels = d2
                numPoints = d1
                get = { c, i -> b[i][c] } // HWC
            }
            else -> {
                // Ambiguous: pick the smaller as channels
                if (d1 <= d2) {
                    channels = d1
                    numPoints = d2
                    get = { c, i -> b[c][i] } // assume CHW
                } else {
                    channels = d2
                    numPoints = d1
                    get = { c, i -> b[i][c] } // assume HWC
                }
            }
        }

        val invInput = 1f / INPUT_SIZE

        // ------------------------------------------------------------
        // CASE A: Decoded head width=6 (end2end/fused)
        // Layout usually: [a0,a1,a2,a3,conf,classId]
        // coords may be xyxy or xywh; we infer using a simple heuristic.
        // ------------------------------------------------------------
        if (channels == 6) {
            val candidates = ArrayList<DetectionResult>(numPoints.coerceIn(16, 131072))

            fun norm(v: Float): Float = if (v > 1.5f) v * invInput else v

            for (i in 0 until numPoints) {
                val a0 = get(0, i)
                val a1 = get(1, i)
                val a2 = get(2, i)
                val a3 = get(3, i)
                val conf = get(4, i)
                val clsF = get(5, i)

                if (conf < confidenceThreshold) continue

                val cls = clsF.toInt()

                // If a2>a0 and a3>a1 -> likely xyxy; else xywh
                val (xc, yc, ww, hh) = if (a2 > a0 && a3 > a1) {
                    val x1 = clamp01(norm(a0))
                    val y1 = clamp01(norm(a1))
                    val x2 = clamp01(norm(a2))
                    val y2 = clamp01(norm(a3))
                    val w = (x2 - x1).coerceAtLeast(0f)
                    val h = (y2 - y1).coerceAtLeast(0f)
                    val cx = x1 + 0.5f * w
                    val cy = y1 + 0.5f * h
                    listOf(cx, cy, w, h)
                } else {
                    val cx = clamp01(norm(a0))
                    val cy = clamp01(norm(a1))
                    val w  = clamp01(norm(a2).coerceAtLeast(0f))
                    val h  = clamp01(norm(a3).coerceAtLeast(0f))
                    listOf(cx, cy, w, h)
                }

                // clamp box sizes just in case
                val wClamped = ww.coerceIn(0f, 1f)
                val hClamped = hh.coerceIn(0f, 1f)

                candidates.add(
                    DetectionResult(
                        xCenter = xc,
                        yCenter = yc,
                        width = wClamped,
                        height = hClamped,
                        confidence = conf,
                        classId = cls
                    )
                )
            }

            // Apply NMS (safe; even if already NMS-fused, it won't break)
            return applyNms(candidates, expectedClasses, classAgnosticNms)
        }

        // ------------------------------------------------------------
        // CASE B: Raw head (4+K or 5+K)
        // ------------------------------------------------------------
        val cNoObj = 4 + expectedClasses
        val cWithObj = 5 + expectedClasses

        val hasObjectness = when (channels) {
            cNoObj -> false
            cWithObj -> true
            else -> {
                Log.e(
                    "YOLODistanceParser",
                    "Unexpected head width=$channels (d1=$d1, d2=$d2). Expected 4+K or 5+K with K=$expectedClasses."
                )
                return emptyList()
            }
        }

        val clsStart = if (hasObjectness) 5 else 4
        val candidates = ArrayList<DetectionResult>(numPoints.coerceIn(16, 131072))

        for (i in 0 until numPoints) {
            val x = get(0, i)
            val y = get(1, i)
            val w = get(2, i)
            val h = get(3, i)
            val obj = if (hasObjectness) get(4, i) else 1f

            // early prune when obj exists
            if (hasObjectness && obj < confidenceThreshold) continue

            // Normalize from pixels -> 0..1 if needed
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
                    val clsP = get(clsStart + c, i)
                    val conf = if (hasObjectness) base * clsP else clsP
                    if (conf >= confidenceThreshold) {
                        candidates.add(
                            DetectionResult(
                                xCenter = xc,
                                yCenter = yc,
                                width = ww,
                                height = hh,
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
                        bestP = p
                        bestC = c
                    }
                }
                val conf = if (hasObjectness) obj * bestP else bestP
                if (conf >= confidenceThreshold) {
                    candidates.add(
                        DetectionResult(
                            xCenter = xc,
                            yCenter = yc,
                            width = ww,
                            height = hh,
                            confidence = conf,
                            classId = bestC
                        )
                    )
                }
            }
        }

        return applyNms(candidates, expectedClasses, classAgnosticNms)
    }

    /**
     * Convenience for your “single-box” use case:
     * returns the best detection after parse+NMS (or null).
     */
    fun bestOne(
        dets: List<DetectionResult>
    ): DetectionResult? = dets.maxByOrNull { it.confidence }
}