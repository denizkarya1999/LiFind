package com.developer27.lifind

import android.Manifest
import android.annotation.SuppressLint
import android.content.Intent
import android.content.SharedPreferences
import android.content.pm.ActivityInfo
import android.content.pm.PackageManager
import android.graphics.BitmapFactory
import android.graphics.SurfaceTexture
import android.hardware.camera2.CameraManager
import android.net.Uri
import android.os.Bundle
import androidx.preference.PreferenceManager
import android.text.InputType
import android.util.Log
import android.util.SparseIntArray
import android.view.Surface
import android.view.TextureView
import android.view.View
import android.view.ViewGroup
import android.view.WindowManager
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.Toast
import androidx.activity.result.ActivityResultLauncher
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.core.splashscreen.SplashScreen.Companion.installSplashScreen
import androidx.lifecycle.lifecycleScope
import com.developer27.lifind.camera.CameraHelper
import com.developer27.lifind.databinding.ActivityMainBinding
import com.developer27.lifind.trilateration.MapActivity
import com.developer27.lifind.trilateration.LedLayout
import com.developer27.lifind.trilateration.LedMeasurementStore
import com.developer27.lifind.videoprocessing.SingleModelEvaluator
import com.developer27.lifind.videoprocessing.VideoProcessor
import kotlinx.coroutines.launch
import java.io.File

class MainActivity : AppCompatActivity() {
    private lateinit var viewBinding: ActivityMainBinding
    private lateinit var sharedPreferences: SharedPreferences
    private lateinit var cameraManager: CameraManager
    private lateinit var cameraHelper: CameraHelper
    private var videoProcessor: VideoProcessor? = null
    private var isRecording = false
    private var isProcessing = false
    private var isProcessingFrame = false
    private var activityResumed = false
    private var permissionRequested = false
    private var frameGeneration = 0
    private var pendingMap = false
    private val preferenceListener = SharedPreferences.OnSharedPreferenceChangeListener { _, key ->
        if (key in setOf("shutter_speed", "iso_value", "manual_iso_enabled")) {
            cameraHelper.updateShutterSpeed()
        }
    }

    private val REQUIRED_PERMISSIONS = arrayOf(
        Manifest.permission.CAMERA
    )

    private lateinit var requestPermissionLauncher: ActivityResultLauncher<Array<String>>
    private lateinit var pickMediaLauncher: ActivityResultLauncher<String>

    // ----------------------------
    // ZIP Evaluation
    // ----------------------------
    private enum class EvalModel { LED, DISTANCE }
    private var evalModel: EvalModel? = null

    private lateinit var pickTrainZip: ActivityResultLauncher<String>
    private lateinit var pickValZip: ActivityResultLauncher<String>
    private lateinit var pickTestZip: ActivityResultLauncher<String>

    private var trainZipUri: Uri? = null
    private var valZipUri: Uri? = null
    private var testZipUri: Uri? = null

    companion object {
        private const val SETTINGS_REQUEST_CODE = 1
        private val ORIENTATIONS = SparseIntArray().apply {
            append(Surface.ROTATION_0, 90)
            append(Surface.ROTATION_90, 0)
            append(Surface.ROTATION_180, 270)
            append(Surface.ROTATION_270, 180)
        }
    }

    private val textureListener = object : TextureView.SurfaceTextureListener {
        @SuppressLint("MissingPermission")
        override fun onSurfaceTextureAvailable(surface: SurfaceTexture, width: Int, height: Int) {
            if (activityResumed && allPermissionsGranted()) cameraHelper.openCamera()
        }
        override fun onSurfaceTextureSizeChanged(surface: SurfaceTexture, width: Int, height: Int) {}
        override fun onSurfaceTextureDestroyed(surface: SurfaceTexture): Boolean {
            cameraHelper.closeCamera()
            return true
        }
        override fun onSurfaceTextureUpdated(surface: SurfaceTexture) {
            if (isProcessing) processFrameWithVideoProcessor()
        }
    }

    @SuppressLint("MissingPermission")
    override fun onCreate(savedInstanceState: Bundle?) {
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        requestedOrientation = ActivityInfo.SCREEN_ORIENTATION_PORTRAIT
        installSplashScreen()
        super.onCreate(savedInstanceState)
        viewBinding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(viewBinding.root)

        sharedPreferences = PreferenceManager.getDefaultSharedPreferences(this)
        cameraManager = getSystemService(CAMERA_SERVICE) as CameraManager
        cameraHelper = CameraHelper(this, viewBinding, sharedPreferences)
        videoProcessor = VideoProcessor(this)

        viewBinding.processedFrameView.visibility = View.GONE

        viewBinding.titleContainer.setOnClickListener {
            val url = "https://www.zhangxiao.me/"
            val intent = Intent(Intent.ACTION_VIEW, Uri.parse(url))
            startActivity(intent)
        }

        requestPermissionLauncher =
            registerForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) { permissions ->
                if (permissions[Manifest.permission.CAMERA] == true) {
                    if (activityResumed && viewBinding.viewFinder.isAvailable) cameraHelper.openCamera()
                } else {
                    Toast.makeText(this, "Camera permission is required for tracking.", Toast.LENGTH_SHORT).show()
                }
            }
        viewBinding.viewFinder.surfaceTextureListener = textureListener

        // Switch camera button
        viewBinding.switchCameraButton.setOnClickListener {
            resetTracking()
            if (allPermissionsGranted()) {
                cameraHelper.toggleCameraFacing()
            } else {
                requestPermissionLauncher.launch(REQUIRED_PERMISSIONS)
            }
        }

        // ------------------------------------------------------------
        // ZIP evaluation pickers - user selects train/val/test zips
        // ------------------------------------------------------------
        pickTrainZip = registerForActivityResult(ActivityResultContracts.GetContent()) { uri ->
            trainZipUri = uri
            if (uri != null) {
                Toast.makeText(this, "Selected TRAIN zip", Toast.LENGTH_SHORT).show()
                pickValZip.launch("application/zip")
            } else {
                Toast.makeText(this, "Train zip not selected.", Toast.LENGTH_SHORT).show()
            }
        }

        pickValZip = registerForActivityResult(ActivityResultContracts.GetContent()) { uri ->
            valZipUri = uri
            if (uri != null) {
                Toast.makeText(this, "Selected VAL zip", Toast.LENGTH_SHORT).show()
                pickTestZip.launch("application/zip")
            } else {
                Toast.makeText(this, "Val zip not selected.", Toast.LENGTH_SHORT).show()
            }
        }

        pickTestZip = registerForActivityResult(ActivityResultContracts.GetContent()) { uri ->
            testZipUri = uri
            if (uri == null) {
                Toast.makeText(this, "Test zip not selected.", Toast.LENGTH_SHORT).show()
                return@registerForActivityResult
            }
            Toast.makeText(this, "Selected TEST zip", Toast.LENGTH_SHORT).show()

            val m = evalModel ?: return@registerForActivityResult
            val t = trainZipUri ?: return@registerForActivityResult
            val v = valZipUri ?: return@registerForActivityResult
            val te = testZipUri ?: return@registerForActivityResult

            runZipEvaluation(m, t, v, te)
        }

        // Eval ZIP button
        viewBinding.evaluateZipButton.setOnClickListener {
            showModelPickerAndStartZipSelection()
        }

        viewBinding.startProcessingButton.setOnClickListener {
            if (isRecording) {
                stopProcessingAndRecording()
            } else {
                startProcessingAndRecording()
            }
        }

        viewBinding.TakeSnapshotButton.setOnClickListener {
            takeSnapshotAndProcessOnce()
        }

        viewBinding.hardCodedDistancesButton.setOnClickListener {
            showHardCodedDistancesDialog()
        }

        viewBinding.clearButton.setOnClickListener {
            resetTracking()
            restartCamera()
        }

        viewBinding.aboutButton.setOnClickListener {
            startActivity(Intent(this, AboutXameraActivity::class.java))
        }
        viewBinding.settingsButton.setOnClickListener {
            startActivity(Intent(this, SettingsActivity::class.java))
        }

        cameraHelper.setupZoomControls()
        sharedPreferences.registerOnSharedPreferenceChangeListener(preferenceListener)

        viewBinding.viewMapButton.setOnClickListener {
            openMapActivity()
        }

        pickMediaLauncher = registerForActivityResult(ActivityResultContracts.GetContent()) { uri: Uri? ->
            uri?.let { handlePickedMedia(it) }
        }

        // Optional: keep or remove
        viewBinding.settingsButton.setOnLongClickListener {
            showModelPickerAndStartZipSelection()
            true
        }
    }

    // ----------------------------
    // ZIP Evaluation helpers
    // ----------------------------
    private fun showModelPickerAndStartZipSelection() {
        if (!viewBinding.evaluateZipButton.isEnabled) return
        resetTracking()
        val options = arrayOf("LED Detection Model", "Distance Estimation Model")
        var selected = 0 // default = LED

        AlertDialog.Builder(this)
            .setTitle("Evaluate which model?")
            .setSingleChoiceItems(options, selected) { _, which ->
                selected = which
            }
            .setPositiveButton("Next") { _, _ ->
                evalModel = if (selected == 0) EvalModel.LED else EvalModel.DISTANCE

                trainZipUri = null
                valZipUri = null
                testZipUri = null

                pickTrainZip.launch("application/zip")
            }
            .setNegativeButton("Cancel", null)
            .show()
    }

    private fun runZipEvaluation(model: EvalModel, train: Uri, v: Uri, test: Uri) {
        val modelType =
            if (model == EvalModel.LED) SingleModelEvaluator.ModelType.LED
            else SingleModelEvaluator.ModelType.DISTANCE

        // Disable button during evaluation (no progress UI, no title updates)
        viewBinding.evaluateZipButton.isEnabled = false

        lifecycleScope.launch {
            Toast.makeText(this@MainActivity, "Running ${model.name} evaluation…", Toast.LENGTH_SHORT).show()

            val results = SingleModelEvaluator.evaluateTrainValTest(
                context = this@MainActivity,
                model = modelType,
                trainZip = train,
                valZip = v,
                testZip = test,
                includeNoneClass = true,
                onProgress = { _, _, _ ->
                    // Intentionally no UI updates (no title, no progress bar)
                }
            )

            if (results == null) {
                Toast.makeText(this@MainActivity, "Evaluation failed.", Toast.LENGTH_LONG).show()
                viewBinding.evaluateZipButton.isEnabled = true
                return@launch
            }

            val files = results.mapNotNull { res ->
                SingleModelEvaluator.writeCsv(this@MainActivity, modelType, res)
            }

            Toast.makeText(
                this@MainActivity,
                if (files.size == results.size) "Saved:\n" + files.joinToString("\n") { it.absolutePath }
                else "Could not save all evaluation reports.",
                Toast.LENGTH_LONG
            ).show()

            viewBinding.evaluateZipButton.isEnabled = true
        }
    }

    // Handle picked image/video
    private fun handlePickedMedia(uri: Uri) {
        val mimeType = contentResolver.getType(uri)
        if (mimeType?.startsWith("image") == true) {
            val bitmap = runCatching {
                contentResolver.openInputStream(uri).use { BitmapFactory.decodeStream(it) }
            }.getOrNull()
            if (bitmap == null) {
                Toast.makeText(this, "Could not read this image.", Toast.LENGTH_SHORT).show()
                return
            }
            videoProcessor?.processFrame(bitmap) { processedFrames ->
                processedFrames?.let { outputBitmap ->
                    viewBinding.processedFrameView.setImageBitmap(outputBitmap)
                    viewBinding.processedFrameView.visibility = View.VISIBLE
                }
            }
        } else if (mimeType?.startsWith("video") == true) {
            Toast.makeText(this, "Video processing not implemented.", Toast.LENGTH_SHORT).show()
        } else {
            Toast.makeText(this, "Unsupported file type!", Toast.LENGTH_SHORT).show()
        }
    }

    private fun openMapActivity() {
        val intent = Intent(this, MapActivity::class.java)
        startActivity(intent)
    }

    private fun startProcessingAndRecording() {
        if (!allPermissionsGranted()) {
            requestPermissionLauncher.launch(REQUIRED_PERMISSIONS)
            return
        }
        if (isProcessingFrame || pendingMap) return
        videoProcessor?.resetMeasurements()
        cameraHelper.openCamera()
        isRecording = true
        isProcessing = true
        viewBinding.TakeSnapshotButton.isEnabled = false
        viewBinding.startProcessingButton.text = "Stop Tracking"
        viewBinding.startProcessingButton.backgroundTintList = ContextCompat.getColorStateList(this, R.color.red)
        viewBinding.processedFrameView.visibility = View.VISIBLE
        viewBinding.processedFrameView.setImageBitmap(null)
    }

    private fun stopProcessingAndRecording() {
        isRecording = false
        isProcessing = false
        pendingMap = true
        viewBinding.startProcessingButton.isEnabled = false
        if (!isProcessingFrame) finishTracking()
    }

    private fun finishTracking() {
        if (!pendingMap) return
        pendingMap = false
        val saved = videoProcessor?.writeLedDistLogToFile() != null
        resetTracking()
        if (saved) openMapActivity()
        else Toast.makeText(this, "Could not save distances.", Toast.LENGTH_LONG).show()
    }

    private fun resetTracking() {
        frameGeneration++
        pendingMap = false
        isRecording = false
        isProcessing = false
        isProcessingFrame = false
        videoProcessor?.resetMeasurements()
        viewBinding.startProcessingButton.isEnabled = true
        viewBinding.TakeSnapshotButton.isEnabled = true
        viewBinding.startProcessingButton.text = "Start Tracking"
        viewBinding.startProcessingButton.backgroundTintList = ContextCompat.getColorStateList(this, R.color.blue)
        viewBinding.processedFrameView.visibility = View.GONE
        viewBinding.processedFrameView.setImageDrawable(null)
    }

    private fun processFrameWithVideoProcessor() {
        val vp = videoProcessor ?: return
        if (isProcessingFrame || !activityResumed) return
        val bitmap = viewBinding.viewFinder.bitmap ?: return
        val generation = frameGeneration
        isProcessingFrame = true
        vp.processFrame(bitmap) { processedFrames ->
            if (generation != frameGeneration || !activityResumed) return@processFrame
            processedFrames?.let { outputBitmap ->
                if (isProcessing) viewBinding.processedFrameView.setImageBitmap(outputBitmap)
            }
            isProcessingFrame = false
            if (pendingMap) finishTracking()
        }
    }

    private fun restartCamera() {
        try {
            cameraHelper.closeCamera()
            cameraHelper.stopBackgroundThread()
        } catch (_: Throwable) { /* ignore */ }

        cameraHelper.startBackgroundThread()

        if (allPermissionsGranted()) {
            if (viewBinding.viewFinder.isAvailable) {
                cameraHelper.openCamera()
            } else {
                viewBinding.viewFinder.surfaceTextureListener = textureListener
            }
        } else {
            requestPermissionLauncher.launch(REQUIRED_PERMISSIONS)
        }
    }

    private fun takeSnapshotAndProcessOnce() {
        val vp = videoProcessor ?: return
        if (isProcessing || isProcessingFrame || pendingMap) return
        if (!allPermissionsGranted()) {
            requestPermissionLauncher.launch(REQUIRED_PERMISSIONS)
            return
        }
        val snapshot = if (cameraHelper.cameraDevice != null) viewBinding.viewFinder.bitmap else null
        if (snapshot == null) {
            restartCamera()
            Toast.makeText(this, "Waiting for the camera. Tap Snapshot again.", Toast.LENGTH_SHORT).show()
            return
        }
        vp.resetMeasurements()
        val generation = frameGeneration
        isProcessingFrame = true
        viewBinding.TakeSnapshotButton.isEnabled = false
        viewBinding.startProcessingButton.isEnabled = false
        Toast.makeText(this, "Capturing snapshot…", Toast.LENGTH_SHORT).show()
        vp.processFrame(snapshot) { processed ->
            if (generation != frameGeneration || !activityResumed) return@processFrame
            isProcessingFrame = false
            viewBinding.TakeSnapshotButton.isEnabled = true
            viewBinding.startProcessingButton.isEnabled = true
            if (processed == null) {
                Toast.makeText(this, "Could not process this frame. Please retry.", Toast.LENGTH_LONG).show()
                return@processFrame
            }
            viewBinding.processedFrameView.setImageBitmap(processed)
            viewBinding.processedFrameView.visibility = View.VISIBLE
            if (vp.writeLedDistLogToFile() == null) {
                Toast.makeText(this, "Could not save distances.", Toast.LENGTH_LONG).show()
            }
            cameraHelper.closeCamera()
        }
    }

    private fun showHardCodedDistancesDialog() {
        val container = ScrollView(this).apply {
            val pad = (16 * resources.displayMetrics.density).toInt()
            setPadding(pad, pad, pad, pad)
            layoutParams = ViewGroup.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT
            )
        }

        val form = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            layoutParams = ViewGroup.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT
            )
        }

        val led1Et = EditText(this).apply {
            hint = "LED A Distance (cm)"
            inputType = InputType.TYPE_CLASS_NUMBER or InputType.TYPE_NUMBER_FLAG_DECIMAL
            setText("")
        }
        val led2Et = EditText(this).apply {
            hint = "LED B Distance (cm)"
            inputType = InputType.TYPE_CLASS_NUMBER or InputType.TYPE_NUMBER_FLAG_DECIMAL
            setText("")
        }
        val led3Et = EditText(this).apply {
            hint = "LED C Distance (cm)"
            inputType = InputType.TYPE_CLASS_NUMBER or InputType.TYPE_NUMBER_FLAG_DECIMAL
            setText("")
        }

        form.addView(led1Et)
        form.addView(led2Et)
        form.addView(led3Et)
        container.addView(form)

        val dialog = AlertDialog.Builder(this)
            .setTitle("Enter LED distances (cm)")
            .setView(container)
            .setNegativeButton("Cancel", null)
            .setPositiveButton("Save", null)
            .create()
        dialog.setOnShowListener {
            dialog.getButton(AlertDialog.BUTTON_POSITIVE).setOnClickListener {
                val d1 = led1Et.text.toString().trim().toDoubleOrNull()
                val d2 = led2Et.text.toString().trim().toDoubleOrNull()
                val d3 = led3Et.text.toString().trim().toDoubleOrNull()

                if (listOf(d1, d2, d3).any { it == null || !it.isFinite() || it < LedLayout.sensorHeightCm }) {
                    Toast.makeText(this, "Enter three distances of at least ${LedLayout.sensorHeightCm} cm (sensor height).", Toast.LENGTH_LONG).show()
                    return@setOnClickListener
                }
                if (writeLedDistLogToFileFromInputs(d1!!, d2!!, d3!!) != null) {
                    dialog.dismiss()
                    openMapActivity()
                } else {
                    Toast.makeText(this, "Could not save distances.", Toast.LENGTH_LONG).show()
                }
            }
        }
        dialog.show()
    }

    private fun writeLedDistLogToFileFromInputs(d1Cm: Double, d2Cm: Double, d3Cm: Double): File? = try {
        LedMeasurementStore.write(this, listOf(d1Cm, d2Cm, d3Cm))
    } catch (error: Exception) {
        Log.e("LedDistLogger", "Failed to write log", error)
        null
    }

    override fun onResume() {
        super.onResume()
        activityResumed = true
        cameraHelper.startBackgroundThread()
        if (allPermissionsGranted()) {
            if (viewBinding.viewFinder.isAvailable) cameraHelper.openCamera()
        } else if (!permissionRequested) {
            permissionRequested = true
            requestPermissionLauncher.launch(REQUIRED_PERMISSIONS)
        }
    }

    override fun onPause() {
        activityResumed = false
        if (isRecording) videoProcessor?.writeLedDistLogToFile()
        resetTracking()
        cameraHelper.closeCamera()
        cameraHelper.stopBackgroundThread()
        super.onPause()
    }

    override fun onDestroy() {
        sharedPreferences.unregisterOnSharedPreferenceChangeListener(preferenceListener)
        videoProcessor?.close()
        videoProcessor = null
        super.onDestroy()
    }

    private fun allPermissionsGranted(): Boolean = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(this, it) == PackageManager.PERMISSION_GRANTED
    }
}