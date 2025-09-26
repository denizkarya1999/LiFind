package com.developer27.lifind.trilateration

import Trilateration
import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.PointF
import android.graphics.RectF
import android.os.Bundle
import android.os.Environment
import android.util.AttributeSet
import android.view.View
import androidx.appcompat.app.AppCompatActivity
import org.opencv.core.Point
import java.io.File
import java.util.Locale
import kotlin.math.max
import kotlin.math.min

class MapActivity : AppCompatActivity() {
    private var LED_1: Point = Point(0.0, 0.0)
    private var LED_2: Point = Point(0.0, 0.0)
    private var LED_3: Point = Point(0.0, 0.0)

    private var LED_1_Distance: Double = 0.0
    private var LED_2_Distance: Double = 0.0
    private var LED_3_Distance: Double = 0.0

    private var USER_POS: Pair<Double, Double> = 0.0 to 0.0

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // 1) Parse and populate LED_1..3 and LED_1_Distance..3 from the log
        readLatestLedAndDistancesFromLog()

        // 2) World LED anchors (same as Python layout)
        val ledCoords = listOf(
            0.0 to 2.0,   // LED_1
            -2.0 to -2.0, // LED_2
            2.0 to -2.0   // LED_3
        )
        val distances = listOf(
            LED_1_Distance,
            LED_2_Distance,
            LED_3_Distance
        )

        // 3) Trilaterate
        USER_POS = Trilateration.solve(ledCoords, distances)

        // 4) Show map; pass user position in *world* coords and footer info
        val mapView = MapGridView(this).apply {
            setUserPixelPosition(USER_POS.first, USER_POS.second) // world → compressed to 0..100 internally (with clamping at edges)
            setFooterInfo(USER_POS, LED_1_Distance, LED_2_Distance, LED_3_Distance)
        }
        setContentView(mapView)
    }

    private fun readLatestLedAndDistancesFromLog() {
        val docsDir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOCUMENTS)
        val logFile = File(docsDir, "LiFind_Log.txt")
        if (!logFile.exists()) return

        val re = Regex(
            """LED_(\d)\s*->\s*Coordinates:\s*\{x=([-\d]+),\s*y=([-\d]+)\}\s*-\s*Distance:\s*\{([^}]*)\}"""
        )
        fun parseDistance(token: String?): Double {
            if (token == null) return 0.0
            val m = Regex("""[-+]?\d+(?:\.\d+)?""").find(token)
            return m?.value?.toDoubleOrNull() ?: 0.0
        }

        logFile.useLines { seq ->
            seq.forEach { line ->
                val m = re.find(line) ?: return@forEach
                val idx = m.groupValues[1].toInt()
                val x = m.groupValues[2].toDoubleOrNull() ?: 0.0
                val y = m.groupValues[3].toDoubleOrNull() ?: 0.0
                val distVal = parseDistance(m.groupValues[4])

                when (idx) {
                    1 -> { LED_1 = Point(x, y); LED_1_Distance = distVal }
                    2 -> { LED_2 = Point(x, y); LED_2_Distance = distVal }
                    3 -> { LED_3 = Point(x, y); LED_3_Distance = distVal }
                }
            }
        }
    }
}

class MapGridView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null
) : View(context, attrs) {

    // ---- World setup: 30×30 (−15..+15) ----
    private val extentWorld = 30f
    private val halfExtent = extentWorld / 2f

    // ---- Compressed drawing space: 0..100 on both axes (without changing canvas size) ----
    private val coordMax = 100f
    private var userPoint100: PointF? = null // compressed [0..100] coords
    private var userRawWorld: PointF? = null // original world coords (for footer/off-map label)

    // Convert original world (-halfExtent..+halfExtent) -> [0..100]
    private fun worldTo100(v: Float): Float = ((v + halfExtent) / extentWorld) * coordMax

    // Convert [0..100] -> world (-halfExtent..+halfExtent) then to canvas
    private fun hundredToCanvas(px100: Float, py100: Float, cw: Float, ch: Float): PointF {
        val wx = (px100 / coordMax) * extentWorld - halfExtent
        val wy = (py100 / coordMax) * extentWorld - halfExtent
        return worldToCanvas(wx, wy, cw, ch)
    }

    // Clamp a [0..100] point to edges and report whether it was out-of-bounds
    private fun clampPoint100(p: PointF): Pair<PointF, Boolean> {
        val cx = min(max(p.x, 0f), coordMax)
        val cy = min(max(p.y, 0f), coordMax)
        val oob = (p.x < 0f || p.x > coordMax || p.y < 0f || p.y > coordMax)
        return PointF(cx, cy) to oob
    }

    // ---- Paints ----
    private val paintGrid = Paint().apply {
        color = Color.LTGRAY
        strokeWidth = 1.5f
        isAntiAlias = true
    }
    private val paintAxis = Paint().apply {
        color = Color.DKGRAY
        strokeWidth = 4f
        isAntiAlias = true
    }
    private val paintWorldBg = Paint().apply {
        color = Color.rgb(248, 246, 250) // light background inside world rect
        style = Paint.Style.FILL
        isAntiAlias = true
    }
    private val paintLed = Paint().apply {
        color = Color.MAGENTA
        style = Paint.Style.FILL
        isAntiAlias = true
    }
    private val paintUser = Paint().apply {
        color = Color.BLUE
        strokeWidth = 8f
        isAntiAlias = true
    }
    private val textPaint = Paint().apply {
        color = Color.MAGENTA
        textSize = 48f
        isAntiAlias = true
        textAlign = Paint.Align.CENTER
    }
    private val footerPaint = Paint().apply {
        color = Color.WHITE
        textSize = 40f
        isAntiAlias = true
        textAlign = Paint.Align.LEFT
    }
    private val badgePaint = Paint().apply {
        color = Color.RED
        textSize = 36f
        isAntiAlias = true
        textAlign = Paint.Align.LEFT
    }

    // ---- Data ----
    private var footerUser: Pair<Double, Double>? = null
    private var footerD1: Double? = null
    private var footerD2: Double? = null
    private var footerD3: Double? = null

    /** Accept world coords; store raw, and compressed (with possible clamping later on draw). */
    fun setUserPixelPosition(x: Double, y: Double) {
        userRawWorld = PointF(x.toFloat(), y.toFloat())
        val x100 = worldTo100(x.toFloat())
        val y100 = worldTo100(y.toFloat())
        userPoint100 = PointF(x100, y100)
        invalidate()
    }

    /** Footer info: user position + distances for LED1..3 (kept in world units). */
    fun setFooterInfo(userPos: Pair<Double, Double>, d1: Double, d2: Double, d3: Double) {
        footerUser = userPos
        footerD1 = d1
        footerD2 = d2
        footerD3 = d3
        invalidate()
    }

    // ---- Helpers ----
    private fun scale(viewW: Float, viewH: Float): Float =
        minOf(viewW / extentWorld, viewH / extentWorld)

    private fun worldRect(cw: Float, ch: Float): RectF {
        val s = scale(cw, ch)
        val cx = cw / 2f
        val cy = ch / 2f
        val halfW = halfExtent * s
        val halfH = halfExtent * s
        return RectF(cx - halfW, cy - halfH, cx + halfW, cy + halfH)
    }

    private fun worldToCanvas(px: Float, py: Float, cw: Float, ch: Float): PointF {
        val s = scale(cw, ch)
        val cx = cw / 2f
        val cy = ch / 2f
        return PointF(cx + px * s, cy - py * s) // y-up world → y-down canvas
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        val w = width.toFloat()
        val h = height.toFloat()
        val r = worldRect(w, h)

        // Outside world: black
        canvas.drawColor(Color.BLACK)

        // Inside world: light background
        canvas.drawRect(r, paintWorldBg)

        // Clip to 30×30 world, draw map content
        canvas.save()
        canvas.clipRect(r)
        drawGrid(canvas)
        drawAxes(canvas)
        drawAnchors(canvas)
        drawUser(canvas)
        canvas.restore()

        // Footer (drawn *outside* the clipped region, under the world rect)
        drawFooter(canvas, r)
    }

    private fun drawGrid(canvas: Canvas) {
        val w = width.toFloat()
        val h = height.toFloat()
        val s = scale(w, h)
        val cx = w / 2f
        val cy = h / 2f

        val stepWorld = 1f // 1-unit world grid
        var xw = -halfExtent
        while (xw <= halfExtent + 1e-3f) {
            val x = cx + xw * s
            canvas.drawLine(x, cy - halfExtent * s, x, cy + halfExtent * s, paintGrid)
            xw += stepWorld
        }
        var yw = -halfExtent
        while (yw <= halfExtent + 1e-3f) {
            val y = cy - yw * s
            canvas.drawLine(cx - halfExtent * s, y, cx + halfExtent * s, y, paintGrid)
            yw += stepWorld
        }
    }

    private fun drawAxes(canvas: Canvas) {
        val w = width.toFloat()
        val h = height.toFloat()
        val s = scale(w, h)
        val cx = w / 2f
        val cy = h / 2f
        canvas.drawLine(cx, cy - halfExtent * s, cx, cy + halfExtent * s, paintAxis)
        canvas.drawLine(cx - halfExtent * s, cy, cx + halfExtent * s, cy, paintAxis)
    }

    private fun drawAnchors(canvas: Canvas) {
        val w = width.toFloat()
        val h = height.toFloat()
        val radiusPx = 30f

        // Original world coordinates of LEDs
        val ledWorld = listOf(
            PointF( 0f,  2f), // LED_1
            PointF(-2f, -2f), // LED_2
            PointF( 2f, -2f)  // LED_3
        )
        val labels = listOf("A", "B", "C")

        ledWorld.forEachIndexed { i, pW ->
            // Compress to [0..100] then to canvas
            val x100 = worldTo100(pW.x)
            val y100 = worldTo100(pW.y)
            val (clamped, _) = clampPoint100(PointF(x100, y100))
            val pC = hundredToCanvas(clamped.x, clamped.y, w, h)
            canvas.drawCircle(pC.x, pC.y, radiusPx, paintLed)
            canvas.drawText(labels[i], pC.x, pC.y - radiusPx - 12f, textPaint)
        }
    }

    private fun drawUser(canvas: Canvas) {
        val w = width.toFloat()
        val h = height.toFloat()
        val s = 30f // size of the X marker
        val p100 = userPoint100 ?: return

        val (clamped, oob) = clampPoint100(p100)
        val pC = hundredToCanvas(clamped.x, clamped.y, w, h)

        // draw the "X" at the (possibly clamped) location
        canvas.drawLine(pC.x - s, pC.y - s, pC.x + s, pC.y + s, paintUser)
        canvas.drawLine(pC.x - s, pC.y + s, pC.x + s, pC.y - s, paintUser)

        if (oob) {
            // Add a small "off-map" badge with the raw world coords
            val rw = userRawWorld
            if (rw != null) {
                val offText = "off-map (x=${fmt(rw.x.toDouble())}, y=${fmt(rw.y.toDouble())})"
                // Shift badge slightly inward from the edge so it’s readable
                val dx = when {
                    clamped.x <= 0f -> 12f
                    clamped.x >= coordMax -> -badgePaint.measureText(offText) - 12f
                    else -> -badgePaint.measureText(offText) / 2f
                }
                val dy = when {
                    clamped.y <= 0f -> 36f
                    clamped.y >= coordMax -> -12f
                    else -> -12f
                }
                canvas.drawText(offText, pC.x + dx, pC.y + dy, badgePaint)
            }
        }
    }

    private fun drawFooter(canvas: Canvas, worldRect: RectF) {
        val pad = 24f
        val line = 44f

        // Left aligned, just below the world rect
        val startX = pad
        var y = worldRect.bottom + pad + footerPaint.textSize

        val (ux, uy) = footerUser ?: (0.0 to 0.0)
        val d1 = footerD1
        val d2 = footerD2
        val d3 = footerD3

        val lines = listOf(
            "User Position: {x=${fmt(ux)}, y=${fmt(uy)}}",
            "LED A (1000): {${fmtD(d1)}}",
            "LED B (1001): {${fmtD(d2)}}",
            "LED C (1010): {${fmtD(d3)}}"
        )

        for (t in lines) {
            canvas.drawText(t, startX, y, footerPaint)
            y += line
        }
    }

    private fun fmt(v: Double) = String.format(Locale.US, "%.4f", v)
    private fun fmtD(v: Double?) = if (v == null || v.isNaN()) "N/A" else String.format(Locale.US, "%.4f", v)
}
