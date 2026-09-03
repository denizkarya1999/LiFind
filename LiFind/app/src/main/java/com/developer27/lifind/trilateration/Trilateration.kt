package com.developer27.lifind.trilateration

import kotlin.math.abs

object Trilateration {
    @Volatile var interpretDistancesAsRadial3D = false
    @Volatile var sensorHeight = LedLayout.sensorHeightCm

    fun setDistancesAreRadial3D(enabled: Boolean, height: Double) {
        require(height.isFinite() && height >= 0.0)
        interpretDistancesAsRadial3D = enabled
        sensorHeight = height
    }

    fun solve(
        ledCoords: List<Pair<Double, Double>>,
        distances: List<Double>,
        height: Double = if (interpretDistancesAsRadial3D) sensorHeight else 0.0
    ): Pair<Double, Double> {
        require(ledCoords.size == 3 && distances.size == 3) {
            "Exactly three LED positions and distances are required."
        }
        require(height.isFinite() && height >= 0.0)
        require(ledCoords.all { (x, y) -> x.isFinite() && y.isFinite() })
        require(distances.all { it.isFinite() && it >= height && it > 0.0 }) {
            "Distances must be positive and at least the sensor height."
        }

        val (a, b, c) = ledCoords
        val (ax, ay) = a
        val (bx, by) = b
        val (cx, cy) = c
        val squared = distances.map { it * it - height * height }
        val a1 = 2.0 * (bx - ax)
        val b1 = 2.0 * (by - ay)
        val c1 = squared[0] - squared[1] - ax * ax - ay * ay + bx * bx + by * by
        val a2 = 2.0 * (cx - ax)
        val b2 = 2.0 * (cy - ay)
        val c2 = squared[0] - squared[2] - ax * ax - ay * ay + cx * cx + cy * cy
        val determinant = a1 * b2 - a2 * b1
        require(abs(determinant) >= 1e-12) { "LED positions must not be collinear." }

        val x = (c1 * b2 - c2 * b1) / determinant
        val y = (a1 * c2 - a2 * c1) / determinant
        require(x.isFinite() && y.isFinite()) { "Distances are too large." }
        return x to y
    }

    fun solve(dA: Double, dB: Double, dC: Double): Pair<Double, Double> =
        solve(LedLayout.anchors, listOf(dA, dB, dC))
}
