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
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // 1) Locate the log file in public Documents
        val docsDir = Environment
            .getExternalStoragePublicDirectory(Environment.DIRECTORY_DOCUMENTS)
        val logFile = File(docsDir, "LiFind_Log.txt")

        // 2) Default to (0,0) if no entry found
        var userX = 0.0
        var userY = 0.0

        // 3) Read only the UserPosition line
        if (logFile.exists()) {
            logFile.useLines { lines ->
                lines.firstOrNull { it.startsWith("UserPosition:") }
                    ?.let { line ->
                        Regex("""UserPosition:\s*x=([\d\.\-]+),\s*y=([\d\.\-]+)""")
                            .find(line)
                            ?.destructured
                            ?.let { (xStr, yStr) ->
                                userX = xStr.toDoubleOrNull() ?: 0.0
                                userY = yStr.toDoubleOrNull() ?: 0.0
                            }
                    }
            }
        }

        // 4) Build your MapGridView with only the user point
        val mapView = MapGridView(this).apply {
            setUserPixelPosition(userX, userY)
            // LEDs are fixed, so we don't call setDetectedPixelData()
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