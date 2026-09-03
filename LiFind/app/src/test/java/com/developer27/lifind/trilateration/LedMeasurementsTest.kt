package com.developer27.lifind.trilateration

import org.junit.Assert.*
import org.junit.Test

class LedMeasurementsTest {
    @Test fun preservesFractionalCentimetres() {
        val distances = listOf(236.22, 256.54, 304.8)
        assertEquals(distances, LedMeasurements.decode(LedMeasurements.encode(distances)))
    }
    @Test fun missingDetectionsRemainUnavailable() {
        assertEquals(listOf(241.3, null, null), LedMeasurements.decode(LedMeasurements.encode(listOf(241.3, null, null))))
        assertEquals(listOf(null, null, null), LedMeasurements.decode(""))
    }
    @Test fun invalidValuesCannotBecomeMapPositions() {
        assertEquals(listOf(null, null, null), LedMeasurements.decode(LedMeasurements.encode(listOf(0.0, -1.0, Double.NaN))))
    }
    @Test fun logCoordinatesMatchMapAndSolver() {
        val text = LedMeasurements.encode(listOf(240.0, 250.0, 260.0))
        assertTrue(text.contains("LED_2 -> Coordinates: {x=43.18, y=0.0}"))
        assertTrue(text.contains("LED_3 -> Coordinates: {x=-43.18, y=0.0}"))
    }
}
