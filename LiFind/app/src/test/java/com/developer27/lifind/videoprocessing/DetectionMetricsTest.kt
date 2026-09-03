package com.developer27.lifind.videoprocessing

import org.junit.Assert.*
import org.junit.Test

class DetectionMetricsTest {
    private fun box(cls: Int, x: Float = 0.5f) = DetectionResult(x, 0.5f, 0.2f, 0.2f, 0.9f, cls)

    @Test fun wrongClassAtCorrectLocationGoesOffDiagonal() {
        val metrics = DetectionMetrics(3, true)
        metrics.addImage(listOf(box(0)), listOf(box(2)), 0.5f)
        assertEquals(1, metrics.matrix[0][2])
        assertEquals(1, metrics.matched)
        assertEquals(0, metrics.missed)
        assertEquals(0, metrics.falsePositives)
    }
    @Test fun falsePositivesAreCountedWithoutNoneRow() {
        val metrics = DetectionMetrics(3, false)
        metrics.addImage(emptyList(), listOf(box(1)), 0.5f)
        assertEquals(1, metrics.falsePositives)
        assertEquals(3, metrics.matrix.size)
    }
    @Test fun eachPredictionMatchesAtMostOneTruth() {
        val metrics = DetectionMetrics(3, true)
        metrics.addImage(listOf(box(0), box(0)), listOf(box(0)), 0.5f)
        assertEquals(1, metrics.matched)
        assertEquals(1, metrics.missed)
        assertEquals(1, metrics.matrix[0][3])
    }
}
