package com.developer27.lifind.videoprocessing

import android.graphics.Bitmap
import org.opencv.android.Utils
import org.opencv.core.Core
import org.opencv.core.Mat
import org.opencv.core.Point
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import kotlin.math.min

object YOLOHelper {
    // Your distance‐label class names
    private val classNames = arrayOf(
        "1000_3", "1000_4", "1000_5", "1000_6", "1000_7", "1000_8", "1000_9", "1000_10", "1000_11", "1000_12",
        "1001_3", "1001_4", "1001_5", "1001_6", "1001_7", "1001_8", "1001_9", "1001_10", "1001_11", "1001_12",
        "1010_3", "1010_4", "1010_5", "1010_6", "1010_7", "1010_8", "1010_9", "1010_10", "1010_11", "1010_12",
    )

    /** Map a class index into its “led_distance” string */
    fun classNameForId(classId: Int): String =
        classNames.getOrNull(classId) ?: "Unknown"

    /** Parse model output into DetectionResult list above confidence threshold */
    fun parseTFLite(rawOutput: Array<Array<FloatArray>>): List<DetectionResult>? {
        val preds = rawOutput[0]   // preds.size == numCells
        val dets  = mutableListOf<DetectionResult>()
        for (row in preds) {
            if (row.size < 5 + classNames.size) continue

            // box coords + obj
            val xC = row[0]
            val yC = row[1]
            val w  = row[2]
            val h  = row[3]
            val objConf = row[4]

            // **only** look at the next 33 scores
            val classScores = row.sliceArray(5 until 5 + classNames.size)
            val bestIdx     = classScores.indices.maxByOrNull { classScores[it] } ?: -1
            val bestScore   = classScores.getOrNull(bestIdx) ?: 0f
            val conf        = objConf * bestScore

            if (conf >= Settings.Inference.confidenceThreshold) {
                dets += DetectionResult(xC, yC, w, h, conf, bestIdx)
            }
        }
        return dets.ifEmpty { null }
    }

    /**
     * Convert a DetectionResult back into:
     *  - the detection center in original‐image coordinates
     *  - a radius (half the smaller box dimension) in pixels
     */
    fun rescaleToCenterAndRadius(
        det: DetectionResult,
        origW: Int,
        origH: Int,
        padOffsets: Pair<Int, Int>,
        inW: Int,
        inH: Int
    ): Pair<Point, Int> {
        // undo letterbox + normalization
        val scale = min(inW.toDouble()/origW, inH.toDouble()/origH)
        val padX  = padOffsets.first.toDouble()
        val padY  = padOffsets.second.toDouble()

        val xCl = det.xCenter * inW
        val yCl = det.yCenter * inH
        val wL  = det.width   * inW
        val hL  = det.height  * inH

        val xC = (xCl - padX) / scale
        val yC = (yCl - padY) / scale
        val wO = wL  / scale
        val hO = hL  / scale

        val center = Point(xC, yC)
        val radius = ((min(wO, hO) / 2)).toInt()
        return center to radius
    }

    /**
     * Letterbox an image for model input. Returns the padded Bitmap plus
     * the (left, top) padding offsets needed to map back.
     */
    fun createLetterboxedBitmap(
        src: Bitmap,
        targetW: Int,
        targetH: Int,
        padColor: Scalar = Scalar(0.0, 0.0, 0.0)
    ): Pair<Bitmap, Pair<Int, Int>> {
        val mat = Mat().also { Utils.bitmapToMat(src, it) }
        val sw  = mat.cols().toDouble()
        val sh  = mat.rows().toDouble()
        val scale = min(targetW / sw, targetH / sh)
        val nw = (sw * scale).toInt()
        val nh = (sh * scale).toInt()

        val resized = Mat().also {
            Imgproc.resize(mat, it, Size(nw.toDouble(), nh.toDouble()))
            mat.release()
        }

        val padW = targetW - nw
        val padH = targetH - nh
        val (top, bottom) = padH/2 to (padH - padH/2)
        val (left, right) = padW/2 to (padW - padW/2)

        val letterboxed = Mat().also {
            Core.copyMakeBorder(resized, it,
                top, bottom, left, right,
                Core.BORDER_CONSTANT, padColor
            )
            resized.release()
        }

        val outBmp = Bitmap.createBitmap(
            letterboxed.cols(),
            letterboxed.rows(),
            src.config
        ).apply {
            Utils.matToBitmap(letterboxed, this)
            letterboxed.release()
        }

        return outBmp to Pair(left, top)
    }

    /**
     * Draws only a circle around a detection center (no rectangle),
     * plus a text label just above it.
     */
    fun drawDetectionCircleWithLabel(
        mat: Mat,
        center: Point,
        radius: Int,
        label: String
    ) {
        if (!Settings.BoundingBox.enableBoundingBox) return

        Imgproc.circle(
            mat, center, radius,
            Settings.BoundingBox.boxColor,
            Settings.BoundingBox.boxThickness
        )

        Imgproc.putText(
            mat, label,
            Point(center.x - radius, center.y - radius - 10),
            Imgproc.FONT_HERSHEY_SIMPLEX,
            0.5,
            Settings.BoundingBox.boxColor,
            Settings.BoundingBox.boxThickness
        )
    }
}
