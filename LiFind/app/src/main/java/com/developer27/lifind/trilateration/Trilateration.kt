
import kotlin.math.max
import kotlin.math.pow
import kotlin.math.sqrt

object Trilateration {
    // Legacy fixed LED positions for backward-compatibility
    private val LED_A = Pair(0.0, 0.0)
    private val LED_B = Pair(-2.0, -2.0)
    private val LED_C = Pair(2.0, -2.0)

    // ---- Config (no callsite change needed) ----
    @Volatile var interpretDistancesAsRadial3D: Boolean = false
    @Volatile var sensorHeight: Double = 0.0

    /** Optional: call once during setup if your 'distances' are raw 3D radii and camera is H above the LED plane. */
    fun setDistancesAreRadial3D(enabled: Boolean, height: Double) {
        interpretDistancesAsRadial3D = enabled
        sensorHeight = height
    }

    private fun toPlanarIfNeeded(distances: List<Double>): List<Double> {
        if (!interpretDistancesAsRadial3D || sensorHeight <= 0.0) return distances
        val h2 = sensorHeight * sensorHeight
        return distances.map { d -> sqrt(max(0.0, d * d - h2)) }
    }

    fun solve(
        ledCoords: List<Pair<Double, Double>>,
        distances: List<Double>
    ): Pair<Double, Double> {
        require(ledCoords.size == 3 && distances.size == 3) {
            "Trilateration.solve requires exactly 3 LED positions and 3 distances"
        }

        val (A, B, C) = ledCoords
        val (Ax, Ay) = A
        val (Bx, By) = B
        val (Cx, Cy) = C

        // Radial (3D) -> planar (XY) using fixed height H = 15
        val H = 15.0
        val h2 = H * H
        val (dA, dB, dC) = distances
        val DA = sqrt(max(0.0, dA.pow(2) - H.pow(2)))
        val DB = sqrt(max(0.0, dB.pow(2) - H.pow(2)))
        val DC = sqrt(max(0.0, dC.pow(2) - H.pow(2)))

        // From: (|p - A|^2 - DA^2) = (|p - B|^2 - DB^2) and (|p - A|^2 - DA^2) = (|p - C|^2 - DC^2)
        // Linear system: a1*x + b1*y = c1 ; a2*x + b2*y = c2
        val a1 = 2.0 * (Bx - Ax)
        val b1 = 2.0 * (By - Ay)
        val c1 = (DA * DA - DB * DB) - (Ax * Ax + Ay * Ay) + (Bx * Bx + By * By)

        val a2 = 2.0 * (Cx - Ax)
        val b2 = 2.0 * (Cy - Ay)
        val c2 = (DA * DA - DC * DC) - (Ax * Ax + Ay * Ay) + (Cx * Cx + Cy * Cy)

        val det = a1 * b2 - a2 * b1
        if (kotlin.math.abs(det) < 1e-12) return 0.0 to 0.0  // degenerate / collinear

        val x = (c1 * b2 - c2 * b1) / det
        val y = (a1 * c2 - a2 * c1) / det
        return x to y
    }

    /** Legacy API — unchanged */
    fun solve(DA: Double, DB: Double, DC: Double): Pair<Double, Double> =
        solve(listOf(LED_A, LED_B, LED_C), listOf(DA, DB, DC))
}
