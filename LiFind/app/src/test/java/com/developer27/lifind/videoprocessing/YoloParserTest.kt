package com.developer27.lifind.videoprocessing

import org.junit.Assert.*
import org.junit.Test

class YoloParserTest {
    @Test fun distanceRawHeadIncludesAll33Classes() {
        val raw = arrayOf(Array(37) { FloatArray(1) })
        raw[0][0][0] = 0.5f
        raw[0][1][0] = 0.5f
        raw[0][2][0] = 0.2f
        raw[0][3][0] = 0.2f
        raw[0][36][0] = 0.9f
        val detections = YOLODISTANCEHelper.parseTFLite(raw, 0.45f, true, false, YOLODISTANCEHelper.NUM_CLASSES)
        assertEquals(1, detections.size)
        assertEquals(32, detections.single().classId)
        assertEquals("304.8", YOLODISTANCEHelper.classNameForId(detections.single().classId))
    }
    @Test fun ledHeadAcceptsBothLayouts() {
        val hwc = arrayOf(arrayOf(floatArrayOf(0.5f, 0.5f, 0.2f, 0.2f, 0.01f, 0.9f, 0.01f)))
        val chw = arrayOf(Array(7) { c -> floatArrayOf(hwc[0][0][c]) })
        val a = YOLOLEDHelper.parseTFLite(hwc, 0.45f, false, false, 3)
        val b = YOLOLEDHelper.parseTFLite(chw, 0.45f, false, false, 3)
        assertEquals(a, b)
        assertEquals(1, a.single().classId)
    }
    @Test fun evaluationNmsDoesNotMutateLiveThresholds() {
        val raw = arrayOf(arrayOf(
            floatArrayOf(0.1f, 0.1f, 0.4f, 0.4f, 0.9f, 0f),
            floatArrayOf(0.15f, 0.15f, 0.45f, 0.45f, 0.8f, 0f)
        ))
        val liveThreshold = Settings.Inference.iouThreshold
        val suppressed = YOLODISTANCEHelper.parseTFLite(raw, 0.45f, true, false, 33, 0.25f)
        val retained = YOLODISTANCEHelper.parseTFLite(raw, 0.45f, true, false, 33, 0.95f)
        assertEquals(1, suppressed.size)
        assertEquals(2, retained.size)
        assertEquals(liveThreshold, Settings.Inference.iouThreshold)
    }
    @Test fun emptyHeadsReturnNoDetections() {
        assertTrue(YOLOLEDHelper.parseTFLite(emptyArray(), 0.45f, false, false, 3).isEmpty())
        assertTrue(YOLODISTANCEHelper.parseTFLite(arrayOf(emptyArray()), 0.45f, false, false, 33).isEmpty())
    }
}
