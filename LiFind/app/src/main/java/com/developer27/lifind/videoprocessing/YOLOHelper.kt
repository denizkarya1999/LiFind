package com.developer27.lifind.videoprocessing

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Rect
import org.opencv.core.Point
import kotlin.math.max
import kotlin.math.min

/**
 * Utilities for TFLite YOLO inference and drawing.
 * Works with your DetectionResult data class and Settings object.
 */
object YOLOHelper {

    // === Your classes (LED id and distance buckets) ===
    private val classNames = arrayOf(
        "1000_3", "1000_4", "1000_5", "1000_6", "1000_7", "1000_8", "1000_9", "1000_10", "1000_11", "1000_12",
        "1001_3", "1001_4", "1001_5", "1001_6", "1001_7", "1001_8", "1001_9", "1001_10", "1001_11", "1001_12",
        "1010_3", "1010_4", "1010_5", "1010_6", "1010_7", "1010_8", "1010_9", "1010_10", "1010_11", "1010_12",
    )

    fun classNameForId(id: Int): String =
        if (id in classNames.indices) classNames[id] else "unknown"

    // ---- Image prep: letterbox to model input (returns offsets left/top) ----
    fun createLetterboxedBitmap(src: Bitmap, dstW: Int, dstH: Int): Pair<Bitmap, Pair<Int, Int>> {
        val out = Bitmap.createBitmap(dstW, dstH, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(out)

        val scale = min(dstW.toFloat() / src.width, dstH.toFloat() / src.height)
        val newW = (src.width * scale).toInt()
        val newH = (src.height * scale).toInt()
        val left = ((dstW - newW) / 2f).toInt()
        val top = ((dstH - newH) / 2f).toInt()

        // Optional: clear background
        canvas.drawColor(Color.BLACK)

        val srcRect = Rect(0, 0, src.width, src.height)
        val dstRect = Rect(left, top, left + newW, top + newH)
        canvas.drawBitmap(src, srcRect, dstRect, null)

        return out to (left to top)
    }

    // YOLOHelper.kt
    // Parse a YOLO-style TFLite head WITHOUT applying sigmoid in code.
    // Assumes the model already outputs probabilities in [0,1].
    // Expected output shape: [1, N, 5+numClasses]
    //   per row: [x, y, w, h, obj, c0, c1, ...]
    //   where x,y,w,h are normalized to the model input (letterboxed canvas).
    fun parseTFLite(out: Array<Array<FloatArray>>): List<DetectionResult>? {
        // 0) Quick sanity checks: if batch or rows are missing, return empty.
        if (out.isEmpty() || out[0].isEmpty()) return emptyList()

        // We expect [1, N, attrs]; grab the N rows.
        val rows = out[0]
        val results = ArrayList<DetectionResult>(rows.size)

        val numClasses = classNames.size

        for (r in rows) {
            // 1) Guard against malformed rows
            if (r.size < 5 + numClasses) continue

            // 2) Read box center/size (already normalized). Clamp to keep within [0,1].
            val x = r[0].coerceIn(0f, 1f)
            val y = r[1].coerceIn(0f, 1f)
            // Ensure width/height are finite and non-zero to avoid degenerate boxes.
            val w = if (r[2].isFinite()) r[2].coerceAtLeast(1e-6f).coerceAtMost(1f) else continue
            val h = if (r[3].isFinite()) r[3].coerceAtLeast(1e-6f).coerceAtMost(1f) else continue

            // 3) Objectness score (already a probability). Clamp to [0,1].
            val obj = if (r[4].isFinite()) r[4].coerceIn(0f, 1f) else 0f

            // 4) Pick the best class robustly (avoid -1 by initializing with class 0)
            var bestClass = 0
            var bestProb = if (r[5].isFinite()) r[5].coerceIn(0f, 1f) else 0f
            for (c in 1 until numClasses) {
                val p = if (r[5 + c].isFinite()) r[5 + c].coerceIn(0f, 1f) else 0f
                if (p >= bestProb) {            // >= so we always end with a valid class
                    bestProb = p
                    bestClass = c
                }
            }

            // 5) Final confidence = objectness * best class prob (stay in [0,1])
            val conf = (obj * bestProb).coerceIn(0f, 1f)

            // 6) Threshold and append as a DetectionResult
            if (conf >= Settings.Inference.confidenceThreshold) {
                results.add(
                    DetectionResult(
                        xCenter = x,
                        yCenter = y,
                        width = w,
                        height = h,
                        confidence = conf,
                        classId = bestClass
                    )
                )
            }
        }

        // 7) Light, class-agnostic NMS to clean up duplicates
        return nms(results, iouThreshold = 0.45f, maxKeep = 100)
    }

    // ---- NMS (class-agnostic) on center-format boxes normalized to [0..1] ----
    private fun nms(dets: List<DetectionResult>, iouThreshold: Float, maxKeep: Int): List<DetectionResult> {
        if (dets.isEmpty()) return emptyList()
        val sorted = dets.sortedByDescending { it.confidence }.toMutableList()
        val kept = ArrayList<DetectionResult>(min(maxKeep, sorted.size))

        while (sorted.isNotEmpty() && kept.size < maxKeep) {
            val best = sorted.removeAt(0)
            kept.add(best)

            val it = sorted.iterator()
            while (it.hasNext()) {
                val other = it.next()
                if (iou(best, other) > iouThreshold) it.remove()
            }
        }
        return kept
    }

    private fun toCorners(d: DetectionResult): FloatArray {
        val x1 = d.xCenter - d.width / 2f
        val y1 = d.yCenter - d.height / 2f
        val x2 = d.xCenter + d.width / 2f
        val y2 = d.yCenter + d.height / 2f
        return floatArrayOf(x1, y1, x2, y2)
    }

    private fun iou(a: DetectionResult, b: DetectionResult): Float {
        val ra = toCorners(a)
        val rb = toCorners(b)
        val x1 = max(ra[0], rb[0])
        val y1 = max(ra[1], rb[1])
        val x2 = min(ra[2], rb[2])
        val y2 = min(ra[3], rb[3])
        val interW = max(0f, x2 - x1)
        val interH = max(0f, y2 - y1)
        val inter = interW * interH
        val areaA = (ra[2] - ra[0]).coerceAtLeast(0f) * (ra[3] - ra[1]).coerceAtLeast(0f)
        val areaB = (rb[2] - rb[0]).coerceAtLeast(0f) * (rb[3] - rb[1]).coerceAtLeast(0f)
        val union = areaA + areaB - inter
        return if (union <= 0f) 0f else inter / union
    }

    // ---- Rescale: letterboxed -> original image coordinates (center + radius) ----
    /**
     * @param det detection in normalized letterboxed space [0..1]
     * @param rawW/H original bitmap size (NOT letterboxed)
     * @param offsets (left, top) returned by createLetterboxedBitmap
     * @param inputW/H model input (letterboxed) size
     */
    fun rescaleToCenterAndRadius(
        det: DetectionResult,
        rawW: Int,
        rawH: Int,
        offsets: Pair<Int, Int>,
        inputW: Int,
        inputH: Int
    ): Pair<Point, Int> {
        val (left, top) = offsets

        // position/size on the letterboxed canvas (in px)
        val xLb = det.xCenter * inputW
        val yLb = det.yCenter * inputH
        val rLb = 0.5f * min(det.width * inputW, det.height * inputH)

        // how original image was scaled to fit input
        val scale = min(inputW.toFloat() / rawW, inputH.toFloat() / rawH)

        // invert letterbox transform
        val xRaw = (xLb - left) / scale
        val yRaw = (yLb - top) / scale
        val rRaw = (rLb / scale)

        return Point(xRaw.toDouble(), yRaw.toDouble()) to rRaw.toInt()
    }
}
