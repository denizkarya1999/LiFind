package com.developer27.lifind

import android.Manifest
import android.annotation.SuppressLint
import android.content.Intent
import android.content.SharedPreferences
import android.content.pm.ActivityInfo
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.SurfaceTexture
import android.hardware.camera2.CameraManager
import android.net.Uri
import android.os.Bundle
import android.preference.PreferenceManager
import android.util.Log
import android.util.SparseIntArray
import android.view.Surface
import android.view.TextureView
import android.view.View
import android.view.WindowManager
import android.widget.Toast
import androidx.activity.result.ActivityResultLauncher
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.core.splashscreen.SplashScreen.Companion.installSplashScreen

import com.developer27.lifind.camera.CameraHelper
import com.developer27.lifind.camera.RecorderHelper
import com.developer27.lifind.databinding.ActivityMainBinding
import com.developer27.lifind.videoprocessing.VideoProcessor
import com.developer27.lifind.trilateration.Trilateration

import java.io.File
import java.io.FileOutputStream
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.nnapi.NnApiDelegate

class MainActivity : AppCompatActivity() {
    private lateinit var viewBinding: ActivityMainBinding
    private lateinit var sharedPreferences: SharedPreferences
    private lateinit var cameraManager: CameraManager
    private lateinit var cameraHelper: CameraHelper
    private var videoProcessor: VideoProcessor? = null

    private lateinit var recorderHelper: RecorderHelper
    private var isRecordingVideo = false
    private var isRecording = false
    private var isProcessing = false
    private var isProcessingFrame = false

    private val REQUIRED_PERMISSIONS = arrayOf(
        Manifest.permission.CAMERA,
        Manifest.permission.RECORD_AUDIO
    )

    private lateinit var requestPermissionLauncher: ActivityResultLauncher<Array<String>>
    private lateinit var pickMediaLauncher: ActivityResultLauncher<String>

    /** Stores last detected trilateration result for manual map display */
    private var lastUserPosition: Pair<Double, Double>? = null

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
        recorderHelper = RecorderHelper(this, cameraHelper, sharedPreferences, viewBinding)

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
        viewBinding.startRecordingButton.setOnClickListener {
            if (isRecordingVideo) {
                recorderHelper.stopRecordingVideo()
                isRecordingVideo = false
                viewBinding.startRecordingButton.text = "Capture Video"
                viewBinding.startRecordingButton.backgroundTintList =
                    ContextCompat.getColorStateList(this, R.color.green)
            } else {
                recorderHelper.startRecordingVideo()
                isRecordingVideo = true
                viewBinding.startRecordingButton.text = "Stop Video"
                viewBinding.startRecordingButton.backgroundTintList =
                    ContextCompat.getColorStateList(this, R.color.red)
            }
        }
        viewBinding.switchCameraButton.setOnClickListener { switchCamera() }
        viewBinding.aboutButton.setOnClickListener {
            startActivity(Intent(this, AboutXameraActivity::class.java))
        }
        viewBinding.settingsButton.setOnClickListener {
            startActivity(Intent(this, SettingsActivity::class.java))
        }

        loadTFLiteModelOnStartupThreaded("best-fp16.tflite")
        loadTFLiteModelOnStartupThreaded("Distance_YOLOv8_float32.tflite")

        cameraHelper.setupZoomControls()
        sharedPreferences.registerOnSharedPreferenceChangeListener { _, key ->
            if (key == "shutter_speed") {
                cameraHelper.updateShutterSpeed()
            }
        }

        // --- UPLOAD BUTTON LOGIC ---
        pickMediaLauncher = registerForActivityResult(ActivityResultContracts.GetContent()) { uri: Uri? ->
            uri?.let { handlePickedMedia(it) }
        }
        viewBinding.uploadButton.setOnClickListener {
            pickMediaLauncher.launch("*/*")
        }

        // --- VIEW MAP BUTTON LOGIC ---
        viewBinding.viewMapButton.setOnClickListener {
            Log.d("MainActivity", "View Map clicked") // For debugging -- see logcat!
            val intent = Intent(this, MapActivity::class.java)
            val userX = lastUserPosition?.first ?: 0.0
            val userY = lastUserPosition?.second ?: 0.0
            intent.putExtra("userX", userX)
            intent.putExtra("userY", userY)
            startActivity(intent)
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

                    // Extract your detections (classId, distance) from videoProcessor
                    val detections: List<Pair<Int, Double>> = videoProcessor?.getLastLedDistances() ?: emptyList()
                    val DA = detections.getOrNull(0)?.second ?: 0.0
                    val DB = detections.getOrNull(1)?.second ?: 0.0
                    val DC = detections.getOrNull(2)?.second ?: 0.0

                    if (DA > 0 && DB > 0 && DC > 0) {
                        val (x, y) = Trilateration.solve(DA, DB, DC)
                        lastUserPosition = Pair(x, y) // Save for View Map button!
                        val intent = Intent(this, MapActivity::class.java)
                        intent.putExtra("userX", x)
                        intent.putExtra("userY", y)
                        intent.putExtra("DA", DA)
                        intent.putExtra("DB", DB)
                        intent.putExtra("DC", DC)
                        startActivity(intent)
                    } else {
                        Toast.makeText(this, "Could not detect all three LEDs!", Toast.LENGTH_LONG).show()
                    }
                }
            }
        } else if (mimeType?.startsWith("video") == true) {
            Toast.makeText(this, "Video processing not implemented.", Toast.LENGTH_SHORT).show()
        } else {
            Toast.makeText(this, "Unsupported file type!", Toast.LENGTH_SHORT).show()
        }
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
        Toast.makeText(this, "Tracking stopped", Toast.LENGTH_LONG).show()
    }

    private fun processFrameWithVideoProcessor() {
        if (isProcessingFrame) return
        val bitmap = viewBinding.viewFinder.bitmap ?: return
        isProcessingFrame = true
        videoProcessor?.processFrame(bitmap) { processedFrames ->
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

    private fun loadTFLiteModelOnStartupThreaded(modelName: String) {
        Thread {
            val bestLoadedPath = copyAssetModelBlocking(modelName)
            runOnUiThread {
                if (bestLoadedPath.isNotEmpty()) {
                    try {
                        val options = Interpreter.Options().apply {
                            setNumThreads(Runtime.getRuntime().availableProcessors())
                        }
                        var delegateAdded = false
                        try {
                            val nnApiDelegate = NnApiDelegate()
                            options.addDelegate(nnApiDelegate)
                            delegateAdded = true
                            Log.d("MainActivity", "NNAPI delegate added successfully.")
                        } catch (e: Exception) {
                            Log.d("MainActivity", "NNAPI delegate unavailable, falling back to GPU delegate.", e)
                        }
                        if (!delegateAdded) {
                            try {
                                val gpuDelegate = GpuDelegate()
                                options.addDelegate(gpuDelegate)
                                Log.d("MainActivity", "GPU delegate added successfully.")
                            } catch (e: Exception) {
                                Log.d("MainActivity", "GPU delegate unavailable, will use CPU only.", e)
                            }
                        }
                        when (modelName) {
                            "best-fp16.tflite" -> {
                                videoProcessor?.setYoloInterpreter(Interpreter(loadMappedFile(bestLoadedPath), options))
                            }
                            "Distance_YOLOv8_float32.tflite" -> {
                                videoProcessor?.setDistanceInterpreter(Interpreter(loadMappedFile(bestLoadedPath), options))
                            }
                            else -> Log.d("MainActivity", "No model processing method defined for $modelName")
                        }
                    } catch (e: Exception) {
                        Toast.makeText(this, "Error loading TFLite model: ${e.message}", Toast.LENGTH_LONG).show()
                        Log.d("MainActivity", "TFLite Interpreter error", e)
                    }
                } else {
                    Toast.makeText(this, "Failed to copy or load $modelName", Toast.LENGTH_SHORT).show()
                }
            }
        }.start()
    }

    private fun loadMappedFile(modelPath: String): MappedByteBuffer {
        val file = File(modelPath)
        val fileInputStream = file.inputStream()
        val fileChannel = fileInputStream.channel
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, 0, file.length())
    }

    private fun copyAssetModelBlocking(assetName: String): String {
        return try {
            val outFile = File(filesDir, assetName)
            if (outFile.exists() && outFile.length() > 0) {
                return outFile.absolutePath
            }
            assets.open(assetName).use { input ->
                FileOutputStream(outFile).use { output ->
                    val buffer = ByteArray(4 * 1024)
                    var bytesRead: Int
                    while (input.read(buffer).also { bytesRead = it } != -1) {
                        output.write(buffer, 0, bytesRead)
                    }
                    output.flush()
                }
            }
            outFile.absolutePath
        } catch (e: Exception) {
            Log.e("MainActivity", "Error copying asset $assetName: ${e.message}")
            ""
        }
    }

    private var isFrontCamera = false
    private fun switchCamera() {
        if (isRecording) stopProcessingAndRecording()
        isFrontCamera = !isFrontCamera
        cameraHelper.isFrontCamera = isFrontCamera
        cameraHelper.closeCamera()
        cameraHelper.openCamera()
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
        if (isRecordingVideo) {
            recorderHelper.stopRecordingVideo()
            isRecordingVideo = false
            viewBinding.startRecordingButton.text = "Capture Video"
            viewBinding.startRecordingButton.backgroundTintList =
                ContextCompat.getColorStateList(this, R.color.green)
        }
        cameraHelper.closeCamera()
        cameraHelper.stopBackgroundThread()
        super.onPause()
    }

    private fun allPermissionsGranted(): Boolean = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(this, it) == PackageManager.PERMISSION_GRANTED
    }
}
