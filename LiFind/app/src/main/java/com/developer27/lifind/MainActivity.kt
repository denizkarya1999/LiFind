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
import android.os.Environment
import android.preference.PreferenceManager
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
import com.developer27.lifind.camera.CameraHelper
import com.developer27.lifind.databinding.ActivityMainBinding
import com.developer27.lifind.trilateration.MapActivity
import com.developer27.lifind.videoprocessing.VideoProcessor
import java.io.BufferedWriter
import java.io.File
import java.io.FileWriter
import java.io.PrintWriter

class MainActivity : AppCompatActivity() {
    private lateinit var viewBinding: ActivityMainBinding
    private lateinit var sharedPreferences: SharedPreferences
    private lateinit var cameraManager: CameraManager
    private lateinit var cameraHelper: CameraHelper
    private var videoProcessor: VideoProcessor? = null
    private var isRecording = false
    private var isProcessing = false
    private var isProcessingFrame = false

    private val REQUIRED_PERMISSIONS = arrayOf(
        Manifest.permission.CAMERA,
        Manifest.permission.RECORD_AUDIO
    )

    private lateinit var requestPermissionLauncher: ActivityResultLauncher<Array<String>>
    private lateinit var pickMediaLauncher: ActivityResultLauncher<String>

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
            if (allPermissionsGranted()) {
                cameraHelper.openCamera()
            } else {
                requestPermissionLauncher.launch(REQUIRED_PERMISSIONS)
            }
        }
        override fun onSurfaceTextureSizeChanged(surface: SurfaceTexture, width: Int, height: Int) {}
        override fun onSurfaceTextureDestroyed(surface: SurfaceTexture): Boolean = false
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
                val camGranted = permissions[Manifest.permission.CAMERA] ?: false
                val micGranted = permissions[Manifest.permission.RECORD_AUDIO] ?: false
                if (camGranted && micGranted) {
                    if (viewBinding.viewFinder.isAvailable) {
                        cameraHelper.openCamera()
                    } else {
                        viewBinding.viewFinder.surfaceTextureListener = textureListener
                    }
                } else {
                    Toast.makeText(this, "Camera & Audio permissions are required.", Toast.LENGTH_SHORT).show()
                }
            }

        if (allPermissionsGranted()) {
            if (viewBinding.viewFinder.isAvailable) {
                cameraHelper.openCamera()
            } else {
                viewBinding.viewFinder.surfaceTextureListener = textureListener
            }
        } else {
            requestPermissionLauncher.launch(REQUIRED_PERMISSIONS)
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
            // 1) Clean overlay
            viewBinding.processedFrameView.setImageDrawable(null)
            viewBinding.processedFrameView.visibility = View.GONE
            viewBinding.processedFrameView.invalidate()

            // 2) Restart camera preview
            restartCamera()
        }

        viewBinding.aboutButton.setOnClickListener {
            startActivity(Intent(this, AboutXameraActivity::class.java))
        }
        viewBinding.settingsButton.setOnClickListener {
            startActivity(Intent(this, SettingsActivity::class.java))
        }

        cameraHelper.setupZoomControls()
        sharedPreferences.registerOnSharedPreferenceChangeListener { _, key ->
            if (key == "shutter_speed") {
                cameraHelper.updateShutterSpeed()
            }
        }

        viewBinding.viewMapButton.setOnClickListener {
            openMapActivity()
        }

        pickMediaLauncher = registerForActivityResult(ActivityResultContracts.GetContent()) { uri: Uri? ->
            uri?.let { handlePickedMedia(it) }
        }
    }

    // Handle picked image/video
    private fun handlePickedMedia(uri: Uri) {
        val mimeType = contentResolver.getType(uri)
        if (mimeType?.startsWith("image") == true) {
            val inputStream = contentResolver.openInputStream(uri)
            val bitmap = BitmapFactory.decodeStream(inputStream)
            inputStream?.close()

            videoProcessor?.processFrame(bitmap) { processedFrames ->
                processedFrames?.let { (outputBitmap, _) ->
                    viewBinding.processedFrameView.setImageBitmap(outputBitmap)
                    viewBinding.processedFrameView.visibility = View.VISIBLE
                }
            }
        } else if (mimeType?.startsWith("video") == true) {
            Toast.makeText(this,
                "Video processing not implemented.",
                Toast.LENGTH_SHORT).show()
        } else {
            Toast.makeText(this,
                "Unsupported file type!",
                Toast.LENGTH_SHORT).show()
        }
    }

    private fun openMapActivity() {
        val intent = Intent(this, MapActivity::class.java)
        startActivity(intent)
    }

    private fun startProcessingAndRecording() {
        isRecording = true
        isProcessing = true
        viewBinding.startProcessingButton.text = "Stop Tracking"
        viewBinding.startProcessingButton.backgroundTintList =
            ContextCompat.getColorStateList(this, R.color.red)
        viewBinding.processedFrameView.visibility = View.VISIBLE
    }

    private fun stopProcessingAndRecording() {
        isRecording = false
        isProcessing = false
        viewBinding.startProcessingButton.text = "Start Tracking"
        viewBinding.startProcessingButton.backgroundTintList =
            ContextCompat.getColorStateList(this, R.color.blue)
        viewBinding.processedFrameView.visibility = View.GONE
        viewBinding.processedFrameView.setImageBitmap(null)
        videoProcessor?.writeLedDistLogToFile()
    }

    private fun processFrameWithVideoProcessor() {
        val vp = videoProcessor ?: return
        if (isProcessingFrame) return
        val bitmap = viewBinding.viewFinder.bitmap ?: return
        isProcessingFrame = true
        vp.processFrame(bitmap) { processedFrames ->
            runOnUiThread {
                processedFrames?.let { (outputBitmap, _) ->
                    if (isProcessing) {
                        viewBinding.processedFrameView.setImageBitmap(outputBitmap)
                    }
                }
                isProcessingFrame = false
            }
        }
    }

    private fun restartCamera() {
        try {
            // Stop and fully reset the camera pipeline
            cameraHelper.closeCamera()
            cameraHelper.stopBackgroundThread()
        } catch (_: Throwable) { /* ignore */ }

        // Start threads again
        cameraHelper.startBackgroundThread()

        // Re-open camera or set listener if the TextureView isn't ready yet
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

    @SuppressLint("MissingPermission")
    private fun takeSnapshotAndProcessOnce() {
        val vp = videoProcessor ?: return
        val btn = viewBinding.TakeSnapshotButton

        // Always start with a clean processed view
        viewBinding.processedFrameView.setImageDrawable(null)
        viewBinding.processedFrameView.visibility = View.GONE
        viewBinding.processedFrameView.invalidate()

        // UI debounce
        btn.isEnabled = false
        Toast.makeText(this, "Capturing snapshot…", Toast.LENGTH_SHORT).show()

        // get a single frame from the preview
        val snapshot = viewBinding.viewFinder.bitmap
        if (snapshot == null) {
            btn.isEnabled = true
            Toast.makeText(this, "No frame available yet.", Toast.LENGTH_SHORT).show()
            return
        }

        // run your current single-frame pipeline
        vp.processFrame(snapshot) { processed ->
            processed?.let { (annotated, _) ->
                // briefly show it
                viewBinding.processedFrameView.setImageBitmap(annotated)
                viewBinding.processedFrameView.visibility = View.VISIBLE
            }

            // use your EXISTING log-writer (unchanged)
            vp.writeLedDistLogToFile()

            cameraHelper.closeCamera()
            btn.isEnabled = true
        }
    }

    private fun showHardCodedDistancesDialog() {
        // Container with simple vertical form
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
            hint = "LED 1 distance (cm)"
            inputType = InputType.TYPE_CLASS_NUMBER or InputType.TYPE_NUMBER_FLAG_DECIMAL
            setText("") // optionally prefill e.g., "9"
        }
        val led2Et = EditText(this).apply {
            hint = "LED 2 distance (cm)"
            inputType = InputType.TYPE_CLASS_NUMBER or InputType.TYPE_NUMBER_FLAG_DECIMAL
            setText("") // optionally prefill e.g., "12"
        }
        val led3Et = EditText(this).apply {
            hint = "LED 3 distance (cm)"
            inputType = InputType.TYPE_CLASS_NUMBER or InputType.TYPE_NUMBER_FLAG_DECIMAL
            setText("") // optionally prefill e.g., "15"
        }

        form.addView(led1Et)
        form.addView(led2Et)
        form.addView(led3Et)
        container.addView(form)

        AlertDialog.Builder(this)
            .setTitle("Enter LED distances (cm)")
            .setView(container)
            .setNegativeButton("Cancel", null)
            .setPositiveButton("Save") { _, _ ->
                val d1 = led1Et.text.toString().trim().toDoubleOrNull()
                val d2 = led2Et.text.toString().trim().toDoubleOrNull()
                val d3 = led3Et.text.toString().trim().toDoubleOrNull()

                if (d1 == null || d2 == null || d3 == null) {
                    Toast.makeText(this, "Please enter valid numbers for all three distances.", Toast.LENGTH_SHORT).show()
                    return@setPositiveButton
                }

                writeLedDistLogToFileFromInputs(d1, d2, d3)

                openMapActivity()
            }
            .show()
    }

    private fun writeLedDistLogToFileFromInputs(d1Cm: Double, d2Cm: Double, d3Cm: Double): File? {
        fun fmtDistanceBraced(v: Double?): String =
            if (v == null) "{N/A}" else "{${v.toInt()} CM}"

        val name = "LiFind_Log.txt"
        val docsDir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOCUMENTS)
        if (!docsDir.exists()) docsDir.mkdirs()

        val outFile = File(docsDir, name)
        return try {
            PrintWriter(BufferedWriter(FileWriter(outFile))).use { out ->
                out.println("LED_1 -> Coordinates: {x=0, y=2} - Distance: ${fmtDistanceBraced(d1Cm)}")
                out.println("LED_2 -> Coordinates: {x=2, y=-2} - Distance: ${fmtDistanceBraced(d2Cm)}")
                out.println("LED_3 -> Coordinates: {x=-2, y=-2} - Distance: ${fmtDistanceBraced(d3Cm)}")
            }
            outFile
        } catch (t: Throwable) {
            Log.e("LedDistLogger", "Failed to write log", t)
            null
        }
    }

    override fun onResume() {
        super.onResume()
        cameraHelper.startBackgroundThread()
        if (viewBinding.viewFinder.isAvailable) {
            if (allPermissionsGranted()) {
                cameraHelper.openCamera()
            } else {
                requestPermissionLauncher.launch(REQUIRED_PERMISSIONS)
            }
        } else {
            viewBinding.viewFinder.surfaceTextureListener = textureListener
        }
    }

    override fun onPause() {
        if (isRecording) stopProcessingAndRecording()
        cameraHelper.closeCamera()
        cameraHelper.stopBackgroundThread()
        super.onPause()
    }

    private fun allPermissionsGranted(): Boolean = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(this, it) == PackageManager.PERMISSION_GRANTED
    }
}