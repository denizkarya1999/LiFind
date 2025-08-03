import kotlin.math.abs
import kotlin.math.pow

object Trilateration {
    // Legacy fixed LED positions for backward-compatibility
    private val LED_A = Pair(0.0, 0.0)
    private val LED_B = Pair(-2.0, -2.0)
    private val LED_C = Pair(2.0, -2.0)

    /**
     * New, dynamic API:
     * @param ledCoords  exactly three (x,y) world-coordinates of your LEDs
     * @param distances  exactly three distances from those LEDs
     * @return the (x,y) solution or (0,0) if none converged
     */
    fun solve(
        ledCoords: List<Pair<Double, Double>>,
        distances: List<Double>
    ): Pair<Double, Double> {
        require(ledCoords.size == 3 && distances.size == 3) {
            "Trilateration.solve requires exactly 3 LED positions and 3 distances"
        }
        val (A, B, C) = ledCoords
        val (DA, DB, DC) = distances
        fun equations(x: Double, y: Double): Pair<Double, Double> {
            val eqA = (x - A.first).pow(2) + (y - A.second).pow(2) - DA.pow(2)
            val eqB = (x - B.first).pow(2) + (y - B.second).pow(2) - DB.pow(2)
            val eqC = (x - C.first).pow(2) + (y - C.second).pow(2) - DC.pow(2)
            return Pair(eqA - eqB, eqA - eqC)
        }
        val guesses = listOf(
            A, B, C,
            Pair((A.first + B.first)/2, (A.second + B.second)/2),
            Pair((B.first + C.first)/2, (B.second + C.second)/2),
            Pair((C.first + A.first)/2, (C.second + A.second)/2)
        )
        val solutions = mutableListOf<Pair<Double, Double>>()
        for ((gx, gy) in guesses) {
            var x = gx; var y = gy; var converged = false
            for (i in 0 until 100) {
                val (f1, f2) = equations(x, y)
                val h = 1e-5
                val fx1 = (equations(x + h, y).first - f1) / h
                val fx2 = (equations(x, y + h).first - f1) / h
                val fy1 = (equations(x + h, y).second - f2) / h
                val fy2 = (equations(x, y + h).second - f2) / h
                val det = fx1 * fy2 - fx2 * fy1
                if (abs(det) < 1e-12) break
                val dx = (-f1*fy2 + f2*fx2) / det
                val dy = (-fx1*f2 + fy1*f1) / det
                x += dx; y += dy
                if (abs(dx) < 1e-6 && abs(dy) < 1e-6) { converged = true; break }
            }
            if (converged && solutions.none { (sx, sy) -> abs(sx - x) < 1e-3 && abs(sy - y) < 1e-3 }) {
                solutions.add(Pair(x, y))
            }
        }
        return solutions.firstOrNull() ?: Pair(0.0, 0.0)
    }

    /** Legacy API */
    fun solve(DA: Double, DB: Double, DC: Double): Pair<Double, Double> =
        solve(listOf(LED_A, LED_B, LED_C), listOf(DA, DB, DC))
}