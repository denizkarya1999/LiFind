package com.developer27.lifind.trilateration

import org.junit.Assert.*
import org.junit.After
import org.junit.Test
import kotlin.math.hypot
import kotlin.math.sqrt

class TrilaterationTest {
    private val anchors = listOf(0.0 to 4.0, 4.0 to 0.0, -4.0 to 0.0)
    private fun distances(x: Double, y: Double, height: Double = 0.0) = anchors.map { (ax, ay) ->
        val planar = hypot(x - ax, y - ay)
        sqrt(planar * planar + height * height)
    }
    private fun assertPosition(expected: Pair<Double, Double>, actual: Pair<Double, Double>) {
        assertEquals(expected.first, actual.first, 1e-8)
        assertEquals(expected.second, actual.second, 1e-8)
    }

    @After fun resetConfig() = Trilateration.setDistancesAreRadial3D(false, LedLayout.sensorHeightCm)

    @Test fun solvesShortPlanarDistancesWithoutClampingThemToSensorHeight() {
        assertPosition(1.25 to -2.5, Trilateration.solve(anchors, distances(1.25, -2.5)))
    }
    @Test fun honorsConfiguredRadialHeight() {
        Trilateration.setDistancesAreRadial3D(true, 10.0)
        assertPosition(-1.0 to 3.0, Trilateration.solve(anchors, distances(-1.0, 3.0, 10.0)))
    }
    @Test fun solvesRealWorldCentimetreCoordinates() {
        val target = 20.0 to -30.0
        val distances = LedLayout.anchors.map { (x, y) ->
            sqrt((x - target.first) * (x - target.first) + (y - target.second) * (y - target.second) + LedLayout.sensorHeightCm * LedLayout.sensorHeightCm)
        }
        assertPosition(target, Trilateration.solve(LedLayout.anchors, distances, LedLayout.sensorHeightCm))
    }
    @Test(expected = IllegalArgumentException::class)
    fun rejectsImpossibleRadialDistance() {
        Trilateration.solve(anchors, listOf(2.0, 3.0, 4.0), 10.0)
    }
    @Test(expected = IllegalArgumentException::class)
    fun rejectsMissingDistance() {
        Trilateration.solve(anchors, listOf(0.0, 3.0, 4.0))
    }
    @Test(expected = IllegalArgumentException::class)
    fun rejectsNonFiniteDistances() {
        Trilateration.solve(anchors, listOf(Double.NaN, 3.0, 4.0))
    }
    @Test(expected = IllegalArgumentException::class)
    fun rejectsCollinearAnchorsInsteadOfInventingOrigin() {
        Trilateration.solve(listOf(0.0 to 0.0, 1.0 to 0.0, 2.0 to 0.0), listOf(1.0, 2.0, 3.0))
    }
}
