package com.developer27.lifind.videoprocessing

import android.graphics.Bitmap
import android.util.Log
import org.opencv.core.*
import org.opencv.android.Utils
import org.opencv.imgproc.Imgproc
import kotlin.math.min

object YOLOHelper {
    private val classNames = arrayOf("LED1", "LED2", "LED3")

    fun classNameForId(classId: Int): String =
        if (classId in classNames.indices) classNames[classId] else "Unknown"

    // Model expects numDistanceClasses; placeholder here
    val numDistanceClasses = 33 // Update with your real value if needed

    /** Parse TFLite output into DetectionResult list above confidence threshold */
    fun parseTFLite(rawOutput: Array<Array<FloatArray>>): List<DetectionResult>? {
        // Only first batch
        val predictions = rawOutput[0]
        val detections = mutableListOf<DetectionResult>()
        for (row in predictions) {
            if (row.size < 6) continue
            val xCenter = row[0]
            val yCenter = row[1]
            val width = row[2]
            val height = row[3]
            val objectConf = row[4]
            val classScores = row.copyOfRange(5, row.size)
            val bestClassIdx = classScores.indices.maxByOrNull { classScores[it] } ?: -1
            val bestClassScore = classScores.getOrNull(bestClassIdx) ?: 0f
            val finalConf = objectConf * bestClassScore
            if (finalConf >= Settings.Inference.confidenceThreshold) {
                detections.add(
                    DetectionResult(
                        xCenter, yCenter, width, height, finalConf, bestClassIdx
                    )
                )
            }
        }
        return if (detections.isEmpty()) null else detections
    }

    /** Rescale detected box center from net input size to image coordinates, with aspect correction */
    fun rescaleInferencedCoordinates(
        detection: DetectionResult,
        originalWidth: Int,
        originalHeight: Int,
        padOffsets: Pair<Int, Int>,
        modelInputWidth: Int,
        modelInputHeight: Int
    ): Pair<BoundingBox, Point> {
        val scale = min(
            modelInputWidth / originalWidth.toDouble(),
            modelInputHeight / originalHeight.toDouble()
        )
        val padLeft = padOffsets.first.toDouble()
        val padTop = padOffsets.second.toDouble()
        val xCenterLetterboxed = detection.xCenter * modelInputWidth
        val yCenterLetterboxed = detection.yCenter * modelInputHeight
        val boxWidthLetterboxed = detection.width * modelInputWidth
        val boxHeightLetterboxed = detection.height * modelInputHeight
        val xCenterOriginal = (xCenterLetterboxed - padLeft) / scale
        val yCenterOriginal = (yCenterLetterboxed - padTop) / scale
        val boxWidthOriginal = boxWidthLetterboxed / scale
        val boxHeightOriginal = boxHeightLetterboxed / scale
        val x1Original = xCenterOriginal - (boxWidthOriginal / 2)
        val y1Original = yCenterOriginal - (boxHeightOriginal / 2)
        val x2Original = xCenterOriginal + (boxWidthOriginal / 2)
        val y2Original = yCenterOriginal + (boxHeightOriginal / 2)
        val boundingBox = BoundingBox(
            x1 = x1Original.toFloat(),
            y1 = y1Original.toFloat(),
            x2 = x2Original.toFloat(),
            y2 = y2Original.toFloat(),
            confidence = detection.confidence,
            classId = detection.classId
        )
        val center = Point(xCenterOriginal, yCenterOriginal)
        return Pair(boundingBox, center)
    }

    /** Letterbox (pad and scale) the bitmap to model input size, keeping aspect ratio */
    fun createLetterboxedBitmap(
        srcBitmap: Bitmap,
        targetWidth: Int,
        targetHeight: Int,
        padColor: Scalar = Scalar(0.0, 0.0, 0.0)
    ): Pair<Bitmap, Pair<Int, Int>> {
        val srcMat = Mat().also { Utils.bitmapToMat(srcBitmap, it) }
        val srcWidth = srcMat.cols().toDouble()
        val srcHeight = srcMat.rows().toDouble()
        val scale = min(
            targetWidth / srcWidth,
            targetHeight / srcHeight
        )
        val newWidth = (srcWidth * scale).toInt()
        val newHeight = (srcHeight * scale).toInt()
        val resized = Mat().also {
            Imgproc.resize(srcMat, it, Size(newWidth.toDouble(), newHeight.toDouble()))
            srcMat.release()
        }
        val padWidth = targetWidth - newWidth
        val padHeight = targetHeight - newHeight
        val (top, bottom) = padHeight / 2 to (padHeight - padHeight / 2)
        val (left, right) = padWidth / 2 to (padWidth - padWidth / 2)
        val letterboxed = Mat().also {
            Core.copyMakeBorder(
                resized, it,
                top, bottom,
                left, right,
                Core.BORDER_CONSTANT,
                padColor
            )
            resized.release()
        }
        val outputBitmap = Bitmap.createBitmap(
            letterboxed.cols(),
            letterboxed.rows(),
            srcBitmap.config
        ).apply {
            Utils.matToBitmap(letterboxed, this)
            letterboxed.release()
        }
        return Pair(outputBitmap, Pair(left, top))
    }

    /** Draws the detection circle and label on image */
    fun drawDetectionCircleWithLabel(
        mat: Mat,
        center: Point,
        box: BoundingBox,
        labelText: String
    ) {
        val boxWidth = box.x2 - box.x1
        val boxHeight = box.y2 - box.y1
        val baseRadius = (min(boxWidth, boxHeight) / 2).toInt()
        val radius = (baseRadius * 3.5f).toInt()
        Imgproc.circle(
            mat, center, radius,
            Settings.BoundingBox.boxColor,
            Settings.BoundingBox.boxThickness
        )
        // Label
        val fontScale = 2.0
        val thickness = 2
        val baseline = IntArray(1)
        val textSize = Imgproc.getTextSize(
            labelText, Imgproc.FONT_HERSHEY_SIMPLEX,
            fontScale, thickness,
            baseline
        )
        val textX = (center.x - textSize.width / 2).toInt().coerceAtLeast(0)
        val textY = (center.y - radius - 5).toInt().coerceAtLeast((textSize.height + baseline[0]).toInt())
        Imgproc.rectangle(
            mat,
            Point(textX.toDouble(), (textY + baseline[0]).toDouble()),
            Point((textX + textSize.width).toDouble(), (textY - textSize.height).toDouble()),
            Settings.BoundingBox.boxColor,
            Imgproc.FILLED
        )
        Imgproc.putText(
            mat, labelText,
            Point(textX.toDouble(), textY.toDouble()),
            Imgproc.FONT_HERSHEY_SIMPLEX,
            fontScale,
            Scalar(0.0, 0.0, 0.0),
            thickness
        )
    }
}