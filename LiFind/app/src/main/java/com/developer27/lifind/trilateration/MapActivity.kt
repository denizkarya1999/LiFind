package com.developer27.lifind.trilateration

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.os.Bundle
import android.util.AttributeSet
import android.view.View
import androidx.appcompat.app.AppCompatActivity

class MapActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Receive user position from intent. If missing or (0,0), do not mark.
        val userX = intent.getDoubleExtra("userX", 0.0)
        val userY = intent.getDoubleExtra("userY", 0.0)

        setContentView(
            MapGridView(this).apply {
                setUserPosition(userX, userY)
            }
        )
    }
}

class MapGridView @JvmOverloads constructor(
    context: Context, attrs: AttributeSet? = null
) : View(context, attrs) {

    // Map axis and display settings
    private val minX = -5
    private val maxX = 5
    private val minY = -5
    private val maxY = 5
    private val step = 1

    // LED layout: (x, y, color, label)
    private val ledList = listOf(
        Triple(0.0, 0.0, Pair(Color.RED, "LED1 (0,0)")),
        Triple(-2.0, -2.0, Pair(Color.BLUE, "LED2 (-2,-2)")),
        Triple(2.0, -2.0, Pair(Color.GREEN, "LED3 (2,-2)"))
    )

    private var userPoint: Pair<Double, Double>? = null

    // Padding/borders around drawable area
    private val sidePad = 50f
    private val labelPad = 15f

    // Coordinate mapping
    private fun worldToScreenX(x: Double): Float {
        val w = width - 2 * sidePad
        return ((x - minX) / (maxX - minX) * w + sidePad).toFloat()
    }
    private fun worldToScreenY(y: Double): Float {
        // Y up in world, down in screen!
        val h = height - 2 * sidePad
        return ((maxY - y) / (maxY - minY) * h + sidePad).toFloat()
    }

    /**
     * Sets and draws user position only if it's not (0,0).
     */
    fun setUserPosition(x: Double, y: Double) {
        userPoint = if (x != 0.0 || y != 0.0) Pair(x, y) else null
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        drawGrid(canvas)
        drawAxes(canvas)
        drawLEDs(canvas)
        drawUser(canvas)
    }

    private fun drawGrid(canvas: Canvas) {
        val paint = Paint().apply {
            color = Color.LTGRAY
            strokeWidth = 2f
        }
        for (i in minX..maxX) {
            val sx = worldToScreenX(i.toDouble())
            canvas.drawLine(sx, worldToScreenY(minY.toDouble()), sx, worldToScreenY(maxY.toDouble()), paint)
        }
        for (j in minY..maxY) {
            val sy = worldToScreenY(j.toDouble())
            canvas.drawLine(worldToScreenX(minX.toDouble()), sy, worldToScreenX(maxX.toDouble()), sy, paint)
        }
    }

    private fun drawAxes(canvas: Canvas) {
        val paint = Paint().apply {
            color = Color.DKGRAY
            strokeWidth = 5f
        }
        // X axis
        canvas.drawLine(
            worldToScreenX(minX.toDouble()), worldToScreenY(0.0),
            worldToScreenX(maxX.toDouble()), worldToScreenY(0.0), paint)
        // Y axis
        canvas.drawLine(
            worldToScreenX(0.0), worldToScreenY(minY.toDouble()),
            worldToScreenX(0.0), worldToScreenY(maxY.toDouble()), paint)
    }

    private fun drawLEDs(canvas: Canvas) {
        val circleRadius = 22f
        val textPaint = Paint().apply {
            color = Color.BLACK
            textSize = 34f
            isAntiAlias = true
        }
        ledList.forEach { (x, y, info) ->
            val (col, label) = info
            val paintCircle = Paint().apply {
                color = col
                isAntiAlias = true
            }
            val screenX = worldToScreenX(x)
            val screenY = worldToScreenY(y)
            canvas.drawCircle(screenX, screenY, circleRadius, paintCircle)
            // Label: right of marker
            canvas.drawText(label, screenX + labelPad, screenY - labelPad, textPaint)
        }
    }

    private fun drawUser(canvas: Canvas) {
        val user = userPoint ?: return
        val x = worldToScreenX(user.first)
        val y = worldToScreenY(user.second)

        // Draw a large "X" at user position
        val markSize = 30f
        val paint = Paint().apply {
            color = Color.BLACK
            strokeWidth = 8f
            isAntiAlias = true
        }
        canvas.drawLine(x - markSize, y - markSize, x + markSize, y + markSize, paint)
        canvas.drawLine(x - markSize, y + markSize, x + markSize, y - markSize, paint)

        // Show user coordinates as text
        val coordText = "(${String.format("%.2f", user.first)}, ${String.format("%.2f", user.second)})"
        val textPaint = Paint().apply {
            color = Color.BLACK
            textSize = 34f
            isAntiAlias = true
        }
        canvas.drawText(coordText, x + labelPad, y - labelPad, textPaint)
    }
}
