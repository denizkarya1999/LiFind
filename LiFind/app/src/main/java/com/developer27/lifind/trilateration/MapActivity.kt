package com.developer27.lifind.trilateration

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.os.Bundle
import android.os.Environment
import android.util.AttributeSet
import android.view.View
import androidx.appcompat.app.AppCompatActivity
import org.opencv.core.Point
import java.io.File

class MapActivity : AppCompatActivity() {
    // ----------------------------
    // Class-level (global) vars
    // ----------------------------
    private var led1: Point = Point(0.0, 0.0)
    private var led2: Point = Point(0.0, 0.0)
    private var led3: Point = Point(0.0, 0.0)
    private var distance: Double = 0.0

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        val docsDir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOCUMENTS)
        val logFile = File(docsDir, "LiFind_Log.txt")

        // Defaults
        var LED_1 = Point(0.0, 0.0)
        var LED_2 = Point(0.0, 0.0)
        var LED_3 = Point(0.0, 0.0)
        var Distance_1: Double = 0.0
        var Distance_2: Double = 0.0
        var Distance_3: Double = 0.0

        fun parseNum(token: String?): Double {
            return token?.trim()?.toDoubleOrNull() ?: 0.0
        }

        if (logFile.exists()) {
            logFile.useLines { seq ->
                val last = seq.lastOrNull { it.contains("LED_1") } ?: return@useLines

                // Parse LED coordinates
                val ledsRe = Regex(
                    """LED_1\s*-\s*\(x=([-\d.]+),\s*y=([-\d.]+)\)\s*,\s*LED_2\s*-\s*\(x=([-\d.]+),\s*y=([-\d.]+)\)\s*,\s*LED_3\s*-\s*\(x=([-\d.]+),\s*y=([-\d.]+)\)"""
                )
                ledsRe.find(last)?.destructured?.let { (x1, y1, x2, y2, x3, y3) ->
                    LED_1 = Point(x1.toDoubleOrNull() ?: 0.0, y1.toDoubleOrNull() ?: 0.0)
                    LED_2 = Point(x2.toDoubleOrNull() ?: 0.0, y2.toDoubleOrNull() ?: 0.0)
                    LED_3 = Point(x3.toDoubleOrNull() ?: 0.0, y3.toDoubleOrNull() ?: 0.0)
                }

                // Try NEW format with Distance_1, Distance_2, Distance_3
                val tripleRe = Regex(
                    """DISTANCE_1:\s*([-\w.]+)\s*,\s*DISTANCE_2:\s*([-\w.]+)\s*,\s*DISTANCE_3:\s*([-\w.]+)"""
                )
                val triple = tripleRe.find(last)?.destructured
                if (triple != null) {
                    val (t1, t2, t3) = triple
                    Distance_1 = parseNum(t1)
                    Distance_2 = parseNum(t2)
                    Distance_3 = parseNum(t3)
                } else {
                    // Fallback OLD format: single DISTANCE applied to all
                    val singleRe = Regex("""DISTANCE:\s*([-\d.]+)""")
                    singleRe.find(last)?.destructured?.let { (dist) ->
                        val v = parseNum(dist)
                        Distance_1 = v; Distance_2 = v; Distance_3 = v
                    }
                }
            }
        }

        // Pass explicitly labeled LED and Distance values
        val mapView = MapGridView(this).apply {
            setDetectedPixelData(
                listOf(LED_1, LED_2, LED_3),
                listOf(Distance_1, Distance_2, Distance_3)
            )
        }
        setContentView(mapView)
    }
}

class MapGridView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null
) : View(context, attrs) {

    private val paintGrid =
        Paint().apply { color = Color.LTGRAY; strokeWidth = 2f; isAntiAlias = true }
    private val paintAxis =
        Paint().apply { color = Color.DKGRAY; strokeWidth = 4f; isAntiAlias = true }
    private val paintCircle =
        Paint().apply { color = Color.MAGENTA; style = Paint.Style.FILL; isAntiAlias = true }
    private val paintUser =
        Paint().apply { color = Color.BLUE; strokeWidth = 8f; isAntiAlias = true }

    private var userPoint: Point? = null
    private var detectedPts: List<Point> = emptyList()
    private var detectedDists: List<Double> = emptyList()

    /** Set last trilaterated position in pixel space */
    fun setUserPixelPosition(x: Double, y: Double) {
        userPoint = Point(x, y)
        invalidate()
    }

    /** Set raw LED centers *and* distances (unused here) */
    fun setDetectedPixelData(pts: List<Point>, dists: List<Double>) {
        detectedPts = pts
        detectedDists = dists
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        drawGrid(canvas)
        drawAxes(canvas)
        drawDetectedLeds(canvas)  // LEDs stay fixed in this implementation
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
        val w = width.toFloat()
        val h = height.toFloat()

        // paints
        val paintRoom   = Paint().apply { color = Color.LTGRAY; style = Paint.Style.FILL }
        val paintBorder = Paint().apply { color = Color.DKGRAY; style = Paint.Style.STROKE; strokeWidth = 4f }
        val textPaint   = Paint().apply {
            color       = Color.BLACK
            textSize    = 48f
            isAntiAlias = true
            textAlign   = Paint.Align.CENTER
        }
        val paintLed    = Paint().apply { color = Color.MAGENTA; style = Paint.Style.FILL; isAntiAlias = true }

        // 1) Top‐left room “T3”
        val t3Rect = RectF(0f, 0f, w * 0.5f, h * 0.2f)
        canvas.drawRect(t3Rect, paintRoom)
        canvas.drawRect(t3Rect, paintBorder)
        canvas.drawText("T3",
            t3Rect.centerX(),
            t3Rect.centerY() + textPaint.textSize/2f,
            textPaint)

        // 2) Top‐right “STO”
        val stoRect = RectF(w * 0.8f, 0f, w, h * 0.2f)
        canvas.drawRect(stoRect, paintRoom)
        canvas.drawRect(stoRect, paintBorder)
        canvas.drawText("STO",
            stoRect.centerX(),
            stoRect.centerY() + textPaint.textSize/2f,
            textPaint)

        // 3) Bottom‐left two tables T1 & T2
        val bottomTop = h * 0.8f
        val t1Rect = RectF(0f,          bottomTop, w * 0.25f, h)
        val t2Rect = RectF(w * 0.25f,   bottomTop, w * 0.5f,  h)
        canvas.drawRect(t1Rect, paintRoom)
        canvas.drawRect(t1Rect, paintBorder)
        canvas.drawText("T1",
            t1Rect.centerX(),
            t1Rect.centerY() + textPaint.textSize/2f,
            textPaint)

        canvas.drawRect(t2Rect, paintRoom)
        canvas.drawRect(t2Rect, paintBorder)
        canvas.drawText("T2",
            t2Rect.centerX(),
            t2Rect.centerY() + textPaint.textSize/2f,
            textPaint)

        // 4) Draw the three LEDs as circles in a triangle (LED1 at center, LED2 bottom‐left, LED3 bottom‐right)
        val cx = w / 2f
        val cy = h / 2f
        val Horz = w * 0.25f
        val Vert = h * 0.25f
        val radius = 30f

        val ledPositions = listOf(
            Pair(cx,               cy),               // LED1
            Pair(cx - Horz, cy + Vert),     // LED2
            Pair(cx + Horz, cy + Vert)      // LED3
        )
        val ledLabels = listOf("LED1 (1010)","LED2 (1000)","LED3 (1001)")

        ledPositions.forEachIndexed { i, (x,y) ->
            canvas.drawCircle(x, y, radius, paintLed)
            canvas.drawText(ledLabels[i], x, y - radius - 12f, textPaint)
        }
    }

    private fun drawUser(canvas: Canvas) {
        userPoint?.let { p ->
            // Translate from pixel coords (where (0,0) is center)
            val cx = width  / 2f
            val cy = height / 2f
            val x = cx + p.x.toFloat()
            val y = cy - p.y.toFloat()  // invert Y if needed

            // Draw an “X” at the user position
            val s = 30f
            canvas.drawLine(x - s, y - s, x + s, y + s, paintUser)
            canvas.drawLine(x - s, y + s, x + s, y - s, paintUser)
        }
    }
}