package com.developer27.lifind.trilateration

import kotlin.math.abs
import kotlin.math.pow

object Trilateration {

    // Coordinates for the three LEDs (match your map convention!)
    private val LED_A = Pair(0.0, 0.0)
    private val LED_B = Pair(-2.0, -2.0)
    private val LED_C = Pair(2.0, -2.0)

    /**
     * Solve the trilateration system:
     *  - Returns a Pair(x, y) with the first solution found,
     *    or (0.0, 0.0) if unable to compute.
     *  - Finds the (x,y) that is approximately at distances DA, DB, DC
     *    from LEDs at (A, B, C).
     */
    fun solve(DA: Double, DB: Double, DC: Double): Pair<Double, Double> {

        // Function to evaluate the system of equations
        fun equations(x: Double, y: Double): Pair<Double, Double> {
            val eq1 = (x - LED_A.first).pow(2) + (y - LED_A.second).pow(2) - DA.pow(2)
            val eq2 = (x - LED_B.first).pow(2) + (y - LED_B.second).pow(2) - DB.pow(2)
            val eq3 = (x - LED_C.first).pow(2) + (y - LED_C.second).pow(2) - DC.pow(2)
            return Pair(eq1 - eq2, eq1 - eq3)
        }

        // Some reasonable initial guesses (map coverage!)
        val initialGuesses = listOf(
            Pair(0.0, 0.0), Pair(2.0, 0.0), Pair(-2.0, 0.0), Pair(0.0, -3.0),
            Pair(1.5, -2.0), Pair(-1.5, -2.0), Pair(3.0, -1.0), Pair(-3.0, -1.0)
        )

        // Basic multi-start Newton's method for two equations/two variables
        val solutions = mutableListOf<Pair<Double, Double>>()

        for (guess in initialGuesses) {
            var (x, y) = guess
            var converged = false
            for (iter in 0 until 100) {
                val (f1, f2) = equations(x, y)
                val eps = 1e-8
                val h = 1e-5

                // Numerical partial derivatives (Jacobian)
                val fx1 = (equations(x + h, y).first - f1) / h
                val fx2 = (equations(x, y + h).first - f1) / h
                val fy1 = (equations(x + h, y).second - f2) / h
                val fy2 = (equations(x, y + h).second - f2) / h

                // Jacobian determinant
                val det = fx1 * fy2 - fx2 * fy1
                if (abs(det) < 1e-12) break // singular Jacobian

                // Newton-Raphson step
                val dx = (-f1 * fy2 + f2 * fx2) / det
                val dy = (-fx1 * f2 + fy1 * f1) / det

                x += dx
                y += dy

                if (abs(dx) < 1e-6 && abs(dy) < 1e-6) {
                    converged = true
                    break
                }
            }
            if (converged) {
                // Don't add duplicates
                if (solutions.none { (sx, sy) -> abs(sx - x) < 1e-3 && abs(sy - y) < 1e-3 }) {
                    solutions.add(Pair(x, y))
                }
            }
        }

        // Return first found solution, or fallback (0,0).
        return if (solutions.isNotEmpty()) solutions.first() else Pair(0.0, 0.0)
    }
}