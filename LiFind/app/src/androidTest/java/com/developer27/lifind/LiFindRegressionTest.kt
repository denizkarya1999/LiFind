package com.developer27.lifind

import android.Manifest
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.Color
import android.view.View
import android.widget.Button
import androidx.lifecycle.Lifecycle
import androidx.test.core.app.ActivityScenario
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.developer27.lifind.trilateration.LedMeasurementStore
import com.developer27.lifind.trilateration.MapActivity
import com.developer27.lifind.videoprocessing.BitmapToFloatTensor
import com.developer27.lifind.videoprocessing.VideoProcessor
import org.junit.Assert.*
import org.junit.Test
import org.junit.runner.RunWith
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit

@RunWith(AndroidJUnit4::class)
class LiFindRegressionTest {
    private val instrumentation get() = InstrumentationRegistry.getInstrumentation()
    private val context get() = instrumentation.targetContext

    @Test fun normalizedInputUsesRgbValuesBetweenZeroAndOne() {
        val bitmap = Bitmap.createBitmap(1, 1, Bitmap.Config.ARGB_8888)
        bitmap.eraseColor(Color.rgb(255, 128, 0))
        val input = BitmapToFloatTensor.nhwc(bitmap, 1).asFloatBuffer()
        assertEquals(1f, input.get(), 0f)
        assertEquals(128f / 255f, input.get(), 0.000001f)
        assertEquals(0f, input.get(), 0f)
        bitmap.recycle()
    }

    @Test fun storagePreservesDecimalsWithoutStoragePermission() {
        val expected = listOf(236.22, 256.54, 304.8)
        val file = LedMeasurementStore.write(context, expected)
        assertTrue(file.exists())
        assertTrue(file.canonicalPath.startsWith(context.filesDir.canonicalPath))
        assertEquals(expected, LedMeasurementStore.read(context))
        LedMeasurementStore.write(context, listOf(null, null, null))
        ActivityScenario.launch<MapActivity>(Intent(context, MapActivity::class.java)).use { scenario ->
            scenario.onActivity { assertFalse(it.isFinishing) }
        }
    }

    @Test fun bothBundledModelsRunAndDoNotMutateEarlierFrames() {
        val processor = VideoProcessor(context)
        fun process(color: Int): Bitmap {
            val bitmap = Bitmap.createBitmap(640, 640, Bitmap.Config.ARGB_8888)
            bitmap.eraseColor(color)
            val latch = CountDownLatch(1)
            var output: Bitmap? = null
            processor.processFrame(bitmap) {
                output = it
                latch.countDown()
            }
            assertTrue("Inference timed out", latch.await(120, TimeUnit.SECONDS))
            return requireNotNull(output) { "Inference returned no frame" }
        }
        try {
            val first = process(Color.BLACK)
            val before = first.getPixel(320, 320)
            val second = process(Color.WHITE)
            assertNotSame(first, second)
            assertEquals(before, first.getPixel(320, 320))
            assertEquals(640, second.width)
            first.recycle()
            second.recycle()
        } finally {
            processor.close()
        }
    }

    @Test fun cameraSurvivesSwitchingAndPausingWithoutOpeningMap() {
        instrumentation.uiAutomation.grantRuntimePermission(context.packageName, Manifest.permission.CAMERA)
        ActivityScenario.launch(MainActivity::class.java).use { scenario ->
            repeat(3) {
                scenario.onActivity { activity ->
                    activity.findViewById<View>(R.id.switchCameraButton).performClick()
                    activity.findViewById<View>(R.id.clearButton).performClick()
                }
                scenario.moveToState(Lifecycle.State.CREATED)
                scenario.moveToState(Lifecycle.State.RESUMED)
            }
            scenario.onActivity { activity ->
                activity.findViewById<Button>(R.id.startProcessingButton).performClick()
            }
            scenario.moveToState(Lifecycle.State.CREATED)
            scenario.moveToState(Lifecycle.State.RESUMED)
            scenario.onActivity { activity ->
                assertFalse(activity.isFinishing)
                assertEquals("Start Tracking", activity.findViewById<Button>(R.id.startProcessingButton).text.toString())
                assertTrue(activity.findViewById<Button>(R.id.TakeSnapshotButton).isEnabled)
            }
        }
    }
}
