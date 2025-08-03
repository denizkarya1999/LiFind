package com.developer27.lifind.trilateration

import Trilateration
import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.os.Bundle
import android.os.Environment
import android.util.AttributeSet
import android.view.View
import androidx.appcompat.app.AppCompatActivity
import org.opencv.core.Point
import java.io.File

class MapActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Locate the log file in public Documents
        val docsDir = Environment
            .getExternalStoragePublicDirectory(Environment.DIRECTORY_DOCUMENTS)
        val logFile = File(docsDir, "LiFind_Log.txt")

        val ledCenters  = mutableListOf<Point>()
        val ledDistances = mutableListOf<Double>()

        if (logFile.exists()) {
            val lines = logFile.readLines()
            val groups = mutableListOf<MutableList<String>>()
            var current: MutableList<String>? = null

            // Split into blocks
            for (line in lines) {
                if (line.isBlank()) {
                    current?.let { groups.add(it); current = null }
                } else {
                    if (current == null) current = mutableListOf()
                    current!!.add(line)
                }
            }
            current?.let { groups.add(it) }

            // Parse the most recent block
            groups.lastOrNull()?.forEach { entry ->
                when {
                    entry.contains("LED") && entry.contains("Center:") -> {
                        // e.g. LED1 Center: x=123.4, y=567.8
                        Regex("LED\\d+ Center: x=([\\d\\.-]+), y=([\\d\\.-]+)")
                            .find(entry)
                            ?.destructured
                            ?.let { (px, py) ->
                                ledCenters.add(Point(px.toDouble(), py.toDouble()))
                            }
                    }
                    entry.contains("LED") && entry.contains("Distance:") -> {
                        // e.g. LED1 Distance: 345.6
                        Regex("LED\\d+ Distance: ([\\d\\.-]+)")
                            .find(entry)
                            ?.groupValues
                            ?.get(1)
                            ?.toDoubleOrNull()
                            ?.let { ledDistances.add(it) }
                    }
                }
            }
        }

        // Convert to world‐coord pairs
        val worldCoords = ledCenters.map { it.x to it.y }
        // Compute predicted position (only if we have exactly 3 LEDs + distances)
        val (predX, predY) = if (worldCoords.size == 3 && ledDistances.size == 3) {
            Trilateration.solve(worldCoords, ledDistances)
        } else {
            0.0 to 0.0
        }

        // Build view and pass parsed positions & distances
        val mapView = MapGridView(this).apply {
            setUserPixelPosition(predX, predY)
            setDetectedPixelData(ledCenters, ledDistances)
        }
        setContentView(mapView)
    }
}


class MapGridView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null
) : View(context, attrs) {

    private val paintGrid   = Paint().apply { color = Color.LTGRAY; strokeWidth = 2f; isAntiAlias = true }
    private val paintAxis   = Paint().apply { color = Color.DKGRAY; strokeWidth = 4f; isAntiAlias = true }
    private val paintCircle = Paint().apply { color = Color.MAGENTA; style = Paint.Style.FILL; isAntiAlias = true }
    private val paintUser   = Paint().apply { color = Color.BLUE; strokeWidth = 8f; isAntiAlias = true }

    private var userPoint: Point? = null
    private var detectedPts: List<Point> = emptyList()
    private var detectedDists: List<Double> = emptyList()

    /** Set last trilaterated position in pixel space */
    fun setUserPixelPosition(x: Double, y: Double) {
        userPoint = Point(x, y)
        invalidate()
    }

    /** Set raw LED centers *and* distances */
    fun setDetectedPixelData(pts: List<Point>, dists: List<Double>) {
        detectedPts  = pts
        detectedDists = dists
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        drawGrid(canvas)
        drawAxes(canvas)
        drawDetectedLeds(canvas)
        drawUser(canvas)
    }

    private fun drawGrid(canvas: Canvas) {
        val step = 100
        for (i in 0 until width step step) {
            canvas.drawLine(i.toFloat(), 0f, i.toFloat(), height.toFloat(), paintGrid)
        }
        for (j in 0 until height step step) {
            canvas.drawLine(0f, j.toFloat(), width.toFloat(), j.toFloat(), paintGrid)
        }
    }

    private fun drawAxes(canvas: Canvas) {
        val cx = width / 2f
        canvas.drawLine(cx, 0f, cx, height.toFloat(), paintAxis)
        val cy = height / 2f
        canvas.drawLine(0f, cy, width.toFloat(), cy, paintAxis)
    }

    private fun drawDetectedLeds(canvas: Canvas) {
        val textPaint = Paint().apply {
            color = Color.WHITE
            textSize = 48f
            isAntiAlias = true
            textAlign = Paint.Align.CENTER
        }
        val radius = 40f
        val labelOffset = radius + 16f

        detectedPts.forEachIndexed { i, pt ->
            val x = pt.x.toFloat()
            val y = pt.y.toFloat()
            canvas.drawCircle(x, y, radius, paintCircle)

            val distText = detectedDists.getOrNull(i)?.let { "%.2f".format(it) } ?: "?.??"
            val label = "LED${i+1} | $distText"

            canvas.drawText(label, x, y - labelOffset, textPaint)
        }
    }

    private fun drawUser(canvas: Canvas) {
        userPoint?.let { p ->
            val x = p.x.toFloat()
            val y = p.y.toFloat()
            val s = 30f
            canvas.drawLine(x - s, y - s, x + s, y + s, paintUser)
            canvas.drawLine(x - s, y + s, x + s, y - s, paintUser)
        }
    }
}