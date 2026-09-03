package com.developer27.lifind.camera

import android.Manifest
import android.annotation.SuppressLint
import android.content.Context
import android.content.SharedPreferences
import android.graphics.Rect
import android.graphics.SurfaceTexture
import android.hardware.camera2.CameraAccessException
import android.hardware.camera2.CameraCaptureSession
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraDevice
import android.hardware.camera2.CameraManager
import android.hardware.camera2.CameraMetadata
import android.hardware.camera2.CaptureRequest
import android.media.MediaRecorder
import android.os.Handler
import android.os.HandlerThread
import android.util.Log
import android.util.Size
import android.view.MotionEvent
import android.view.Surface
import android.widget.Toast
import com.developer27.lifind.MainActivity
import com.developer27.lifind.databinding.ActivityMainBinding

/**
 * CameraHelper is responsible for:
 *  - Opening & closing the camera
 *  - Switching front/back (NEW: persisted in SharedPreferences)
 *  - Creating a preview
 *  - Handling zoom & shutter speed
 *  - Starting a background thread for camera operations
 *
 *  This version forces a specific AWB mode & color correction to avoid color tint on Pixel 4a.
 */
class CameraHelper(
    private val activity: MainActivity,
    private val viewBinding: ActivityMainBinding,
    private val sharedPreferences: SharedPreferences
) {
    companion object {
        // Persist camera facing in prefs: "front" | "back"
        private const val PREF_CAMERA_FACING = "camera_facing"
        private const val FACING_FRONT = "front"
        private const val FACING_BACK = "back"
    }

    // The Android Camera2 API
    val cameraManager: CameraManager by lazy {
        activity.getSystemService(Context.CAMERA_SERVICE) as CameraManager
    }

    // Active camera device + capture session
    var cameraDevice: CameraDevice? = null
    var cameraCaptureSession: CameraCaptureSession? = null

    // Capture builder for preview (and record)
    var captureRequestBuilder: CaptureRequest.Builder? = null

    // Preview + video sizes
    var previewSize: Size? = null
    var videoSize: Size? = null

    // Sensor area for zoom
    var sensorArraySize: Rect? = null

    // Whether we are using the front camera (loaded from prefs)
    var isFrontCamera: Boolean = false
        private set

    // Thread for camera operations
    private var backgroundThread: HandlerThread? = null
    var backgroundHandler: Handler? = null
        private set

    // Zoom control
    private var zoomLevel = 1.0f
    private var maxZoom = 1.0f
    private val callbackHandler = Handler(activity.mainLooper)
    private var cameraGeneration = 0
    private var openingCamera = false
    private var previewSurface: Surface? = null

    init {
        loadFacingFromPrefs()
    }

    private fun loadFacingFromPrefs() {
        val facing = sharedPreferences.getString(PREF_CAMERA_FACING, FACING_BACK) ?: FACING_BACK
        isFrontCamera = (facing == FACING_FRONT)
    }

    private fun saveFacingToPrefs(front: Boolean) {
        sharedPreferences.edit()
            .putString(PREF_CAMERA_FACING, if (front) FACING_FRONT else FACING_BACK)
            .apply()
    }

    /**
     * Callback for camera device events
     */
    private fun stateCallback(generation: Int) = object : CameraDevice.StateCallback() {
        override fun onOpened(camera: CameraDevice) {
            if (generation != cameraGeneration) {
                camera.close()
                return
            }
            openingCamera = false
            cameraDevice = camera
            createCameraPreview()
        }

        override fun onDisconnected(camera: CameraDevice) {
            camera.close()
            if (generation == cameraGeneration) closeCamera()
        }

        override fun onError(camera: CameraDevice, error: Int) {
            camera.close()
            if (generation != cameraGeneration) return
            closeCamera()
            Toast.makeText(activity, "Camera unavailable. Tap Clear to retry.", Toast.LENGTH_LONG).show()
        }
    }

    // ------------------------------------------------------------------------
    // Background Thread Setup
    // ------------------------------------------------------------------------
    fun startBackgroundThread() {
        if (backgroundThread != null) return
        backgroundThread = HandlerThread("CameraBackground").also { it.start() }
        backgroundHandler = Handler(backgroundThread!!.looper)
    }

    fun stopBackgroundThread() {
        backgroundThread?.quitSafely()
        try {
            backgroundThread?.join()
            backgroundThread = null
            backgroundHandler = null
        } catch (e: InterruptedException) {
            e.printStackTrace()
        }
    }

    // ------------------------------------------------------------------------
    // Front/Back camera option (NEW)
    // ------------------------------------------------------------------------
    /**
     * Call this from UI (toggle/switch) to force front/back.
     * Persisted to SharedPreferences and restarts camera safely.
     */
    @SuppressLint("MissingPermission")
    fun setFrontCameraEnabled(enabled: Boolean) {
        if (isFrontCamera == enabled) return
        isFrontCamera = enabled
        saveFacingToPrefs(enabled)
        restartCamera()
    }

    /**
     * Call this from UI (button) to toggle front/back.
     */
    @SuppressLint("MissingPermission")
    fun toggleCameraFacing() {
        isFrontCamera = !isFrontCamera
        saveFacingToPrefs(isFrontCamera)
        restartCamera()
    }

    @SuppressLint("MissingPermission")
    private fun restartCamera() {
        // reset zoom (optional safety when switching cameras)
        zoomLevel = 1.0f

        // Close then reopen with new facing
        closeCamera()
        openCamera()
    }

    // ------------------------------------------------------------------------
    // Open/Close Camera
    // ------------------------------------------------------------------------
    @SuppressLint("MissingPermission")
    fun openCamera() {
        if (openingCamera || cameraDevice != null || !viewBinding.viewFinder.isAvailable) return
        try {
            // Decide which camera (front/back)
            val cameraId = getCameraId()
            val characteristics = cameraManager.getCameraCharacteristics(cameraId)

            // Grab the full sensor area for zoom
            sensorArraySize = characteristics.get(CameraCharacteristics.SENSOR_INFO_ACTIVE_ARRAY_SIZE)
            maxZoom = (characteristics.get(CameraCharacteristics.SCALER_AVAILABLE_MAX_DIGITAL_ZOOM) ?: 1f).coerceAtLeast(1f)
            zoomLevel = zoomLevel.coerceIn(1f, maxZoom)

            // Possible output sizes
            val map = characteristics.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP)
                ?: return

            // Choose your preview/video sizes
            previewSize = chooseOptimalSize(map.getOutputSizes(SurfaceTexture::class.java))
            videoSize = chooseOptimalSize(map.getOutputSizes(MediaRecorder::class.java))

            // Now open the selected camera
            openingCamera = true
            cameraManager.openCamera(cameraId, stateCallback(cameraGeneration), callbackHandler)
        } catch (e: Exception) {
            openingCamera = false
            Log.e("CameraHelper", "Could not open camera", e)
            Toast.makeText(activity, "Camera unavailable. Check camera permission and retry.", Toast.LENGTH_SHORT).show()
        }
    }

    fun closeCamera() {
        cameraGeneration++
        openingCamera = false
        cameraCaptureSession?.close()
        cameraCaptureSession = null
        cameraDevice?.close()
        cameraDevice = null
        captureRequestBuilder = null
        previewSurface?.release()
        previewSurface = null
    }

    // ------------------------------------------------------------------------
    // Create Preview
    // ------------------------------------------------------------------------
    fun createCameraPreview() {
        try {
            val texture = viewBinding.viewFinder.surfaceTexture ?: return
            // Match the texture view size to the chosen preview size
            previewSize?.let { texture.setDefaultBufferSize(it.width, it.height) }

            val camera = cameraDevice ?: return
            val generation = cameraGeneration
            val previewSurface = Surface(texture).also { this.previewSurface = it }
            // Build a preview request
            captureRequestBuilder = cameraDevice?.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW)
            // Add the preview surface as a target
            captureRequestBuilder?.addTarget(previewSurface)

            // Apply any manual or auto exposure logic
            applyRollingShutter()
            // Possibly set flash, lighting, zoom
            applyFlashIfEnabled()
            applyLightingMode()
            applyZoom()

            // ----------------------------------------------------------------
            // Force color correction to avoid greenish tint
            // 1) Auto White Balance (set to e.g. DAYLIGHT for consistent color)
            //    or CONTROL_AWB_MODE_AUTO for auto
            // 2) Color Correction Mode => HIGH_QUALITY for better color
            // ----------------------------------------------------------------
            captureRequestBuilder?.set(
                CaptureRequest.CONTROL_AWB_MODE,
                // For strictly "daylight" color:
                // CaptureRequest.CONTROL_AWB_MODE_DAYLIGHT
                // or if you prefer auto, do:
                CaptureRequest.CONTROL_AWB_MODE_AUTO
            )
            captureRequestBuilder?.set(
                CaptureRequest.COLOR_CORRECTION_MODE,
                CaptureRequest.COLOR_CORRECTION_MODE_HIGH_QUALITY
            )

            // Now create the capture session
            cameraDevice?.createCaptureSession(
                listOf(previewSurface),
                object : CameraCaptureSession.StateCallback() {
                    override fun onConfigured(session: CameraCaptureSession) {
                        if (generation != cameraGeneration || cameraDevice !== camera) {
                            session.close()
                            return
                        }
                        // Save the session
                        cameraCaptureSession = session
                        updatePreview() // Start the preview
                    }

                    override fun onConfigureFailed(session: CameraCaptureSession) {
                        session.close()
                        if (generation != cameraGeneration) return
                        Toast.makeText(
                            activity,
                            "Preview config failed.",
                            Toast.LENGTH_SHORT
                        ).show()
                    }
                },
                callbackHandler
            )
        } catch (e: CameraAccessException) {
            e.printStackTrace()
        }
    }

    /**
     * Update the camera preview with latest builder settings
     */
    fun updatePreview() {
        if (cameraDevice == null || captureRequestBuilder == null) return
        try {
            // Keep forcing color correction and AWB
            captureRequestBuilder?.set(
                CaptureRequest.CONTROL_AWB_MODE,
                CaptureRequest.CONTROL_AWB_MODE_AUTO
            )
            captureRequestBuilder?.set(
                CaptureRequest.COLOR_CORRECTION_MODE,
                CaptureRequest.COLOR_CORRECTION_MODE_HIGH_QUALITY
            )

            cameraCaptureSession?.setRepeatingRequest(
                captureRequestBuilder!!.build(),
                null,
                callbackHandler
            )
        } catch (e: CameraAccessException) {
            e.printStackTrace()
        }
    }

    // ------------------------------------------------------------------------
    // Camera Selection (Front/Back) - UPDATED: robust fallback
    // ------------------------------------------------------------------------
    fun getCameraId(): String {
        var backId: String? = null
        var frontId: String? = null

        for (id in cameraManager.cameraIdList) {
            val facing = cameraManager
                .getCameraCharacteristics(id)
                .get(CameraCharacteristics.LENS_FACING)

            if (facing == CameraCharacteristics.LENS_FACING_BACK && backId == null) {
                backId = id
            }
            if (facing == CameraCharacteristics.LENS_FACING_FRONT && frontId == null) {
                frontId = id
            }
        }

        return if (isFrontCamera) {
            frontId ?: backId ?: cameraManager.cameraIdList.firstOrNull() ?: error("No camera available")
        } else {
            backId ?: frontId ?: cameraManager.cameraIdList.firstOrNull() ?: error("No camera available")
        }
    }

    private fun chooseOptimalSize(choices: Array<Size>): Size {
        val targetWidth = 1280
        val targetHeight = 720

        // Try to find 1280x720 specifically
        val found720p = choices.find { it.width == targetWidth && it.height == targetHeight }
        if (found720p != null) {
            return found720p
        }
        // fallback to the smallest
        return choices.minByOrNull { it.width * it.height } ?: error("No supported preview sizes")
    }

    // ------------------------------------------------------------------------
    // Rolling shutter & exposure
    // ------------------------------------------------------------------------
    fun applyRollingShutter() {
        val cameraId = getCameraId()
        val characteristics = cameraManager.getCameraCharacteristics(cameraId)

        val capabilities = characteristics.get(CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES)
        val canManualExposure = capabilities?.contains(
            CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES_MANUAL_SENSOR
        ) == true

        val shutterFps = sharedPreferences.getString("shutter_speed", "60")?.toIntOrNull() ?: 60
        val shutterValueNs = if (shutterFps > 0) 1_000_000_000L / shutterFps else 0L

        // If no manual or user set 0, just do auto
        if (!canManualExposure || shutterValueNs <= 0 || !sharedPreferences.getBoolean("manual_iso_enabled", true)) {
            setAutoExposure()
            return
        }

        val exposureTimeRange = characteristics.get(CameraCharacteristics.SENSOR_INFO_EXPOSURE_TIME_RANGE)
        val isoRange = characteristics.get(CameraCharacteristics.SENSOR_INFO_SENSITIVITY_RANGE)

        if (exposureTimeRange == null || isoRange == null) {
            setAutoExposure()
            return
        }

        val safeExposureNs = shutterValueNs.coerceIn(exposureTimeRange.lower, exposureTimeRange.upper)

        // Read ISO prefs
        val manualIsoEnabled = sharedPreferences.getBoolean("manual_iso_enabled", true)
        val isoFromPrefs = sharedPreferences.getString("iso_value", "800")?.toIntOrNull() ?: 800
        val safeISO = isoFromPrefs.coerceIn(isoRange.lower, isoRange.upper)

        // Fully manual exposure; if manual ISO disabled, we still must set *some* ISO because AE is off.
        // Choose either the user ISO or a mid-range fallback.
        val isoToUse = if (manualIsoEnabled) safeISO else
            ((isoRange.lower + isoRange.upper) / 2).coerceIn(isoRange.lower, isoRange.upper)

        captureRequestBuilder?.set(CaptureRequest.CONTROL_MODE, CameraMetadata.CONTROL_MODE_AUTO)
        captureRequestBuilder?.set(CaptureRequest.CONTROL_AE_MODE, CameraMetadata.CONTROL_AE_MODE_OFF)
        captureRequestBuilder?.set(CaptureRequest.SENSOR_EXPOSURE_TIME, safeExposureNs)
        captureRequestBuilder?.set(CaptureRequest.SENSOR_SENSITIVITY, isoToUse)
    }

    private fun setAutoExposure() {
        captureRequestBuilder?.set(CaptureRequest.CONTROL_MODE, CameraMetadata.CONTROL_MODE_AUTO)
        captureRequestBuilder?.set(CaptureRequest.CONTROL_AE_MODE, CameraMetadata.CONTROL_AE_MODE_ON)
    }

    /**
     * If user changes shutter speed in settings, we re-apply
     */
    fun updateShutterSpeed() {
        if (cameraDevice == null || captureRequestBuilder == null) return
        applyRollingShutter()
        try {
            cameraCaptureSession?.setRepeatingRequest(
                captureRequestBuilder!!.build(),
                null,
                callbackHandler
            )
        } catch (e: CameraAccessException) {
            e.printStackTrace()
        }
    }

    // ------------------------------------------------------------------------
    // Flash & Lighting
    // ------------------------------------------------------------------------
    fun applyFlashIfEnabled() {
        val isFlashEnabled = sharedPreferences.getBoolean("enable_flash", false)
        captureRequestBuilder?.set(
            CaptureRequest.FLASH_MODE,
            if (isFlashEnabled) CaptureRequest.FLASH_MODE_TORCH
            else CaptureRequest.FLASH_MODE_OFF
        )
    }

    fun applyLightingMode() {
        // Only apply AE compensation if AE is ON
        val aeMode = captureRequestBuilder?.get(CaptureRequest.CONTROL_AE_MODE)
        if (aeMode == CameraMetadata.CONTROL_AE_MODE_ON) {
            val lightingMode = sharedPreferences.getString("lighting_mode", "normal")
            val cameraId = getCameraId()
            val compensationRange = cameraManager
                .getCameraCharacteristics(cameraId)
                .get(CameraCharacteristics.CONTROL_AE_COMPENSATION_RANGE)

            val exposureComp = when (lightingMode) {
                "low_light" -> compensationRange?.lower ?: 0
                "high_light" -> compensationRange?.upper ?: 0
                else -> 0
            }
            captureRequestBuilder?.set(
                CaptureRequest.CONTROL_AE_EXPOSURE_COMPENSATION,
                exposureComp
            )
        }
    }

    // ------------------------------------------------------------------------
    // Zoom
    // ------------------------------------------------------------------------
    fun setupZoomControls() {
        val zoomHandler = Handler(activity.mainLooper)
        var zoomInRunnable: Runnable? = null
        var zoomOutRunnable: Runnable? = null

        // Repetitive zoom in on long-press
        viewBinding.zoomInButton.setOnTouchListener { _, event ->
            when (event.action) {
                MotionEvent.ACTION_DOWN -> {
                    zoomInRunnable = object : Runnable {
                        override fun run() {
                            zoomIn()
                            zoomHandler.postDelayed(this, 50)
                        }
                    }
                    zoomHandler.post(zoomInRunnable!!)
                    true
                }
                MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> {
                    zoomInRunnable?.let { zoomHandler.removeCallbacks(it) }
                    true
                }
                else -> false
            }
        }

        // Repetitive zoom out on long-press
        viewBinding.zoomOutButton.setOnTouchListener { _, event ->
            when (event.action) {
                MotionEvent.ACTION_DOWN -> {
                    zoomOutRunnable = object : Runnable {
                        override fun run() {
                            zoomOut()
                            zoomHandler.postDelayed(this, 50)
                        }
                    }
                    zoomHandler.post(zoomOutRunnable!!)
                    true
                }
                MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> {
                    zoomOutRunnable?.let { zoomHandler.removeCallbacks(it) }
                    true
                }
                else -> false
            }
        }
    }

    private fun zoomIn() {
        if (zoomLevel < maxZoom) {
            zoomLevel = (zoomLevel + 0.1f).coerceAtMost(maxZoom)
            applyZoom()
        }
    }

    private fun zoomOut() {
        if (zoomLevel > 1.0f) {
            zoomLevel = (zoomLevel - 0.1f).coerceAtLeast(1f)
            applyZoom()
        }
    }

    /**
     * Applies digital zoom by setting the SCALER_CROP_REGION
     */
    fun applyZoom() {
        if (sensorArraySize == null || captureRequestBuilder == null) return
        val ratio = 1 / zoomLevel
        val croppedWidth = sensorArraySize!!.width() * ratio
        val croppedHeight = sensorArraySize!!.height() * ratio

        val left = sensorArraySize!!.left + ((sensorArraySize!!.width() - croppedWidth) / 2).toInt()
        val top = sensorArraySize!!.top + ((sensorArraySize!!.height() - croppedHeight) / 2).toInt()
        val right = (left + croppedWidth).toInt()
        val bottom = (top + croppedHeight).toInt()

        val zoomRect = Rect(left, top, right, bottom)
        captureRequestBuilder?.set(CaptureRequest.SCALER_CROP_REGION, zoomRect)

        try {
            cameraCaptureSession?.setRepeatingRequest(
                captureRequestBuilder!!.build(),
                null,
                callbackHandler
            )
        } catch (e: CameraAccessException) {
            e.printStackTrace()
        }
    }
}