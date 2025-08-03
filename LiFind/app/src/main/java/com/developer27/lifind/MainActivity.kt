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
import android.os.Handler
import android.os.Looper
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
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.core.splashscreen.SplashScreen.Companion.installSplashScreen
import com.developer27.lifind.camera.CameraHelper
import com.developer27.lifind.databinding.ActivityMainBinding
import com.developer27.lifind.trilateration.MapActivity
import com.developer27.lifind.videoprocessing.VideoProcessor
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.nnapi.NnApiDelegate
import java.io.File
import java.io.FileOutputStream
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

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

        // --- LIVE VIEW “View Map” button logic ---
        viewBinding.viewMapButton.setOnClickListener {
            Log.d("MainActivity", "View Map clicked")

            // 1) Get trilaterated user position
            val userX = lastUserPosition?.first ?: 0.0
            val userY = lastUserPosition?.second ?: 0.0

            // 2) Get raw LED centers from VideoProcessor
            val centers = videoProcessor?.getLastLedCenters() ?: emptyList()
            val ledIds = IntArray(centers.size)   { centers[it].first }
            val ledXs  = FloatArray(centers.size) { centers[it].second.x.toFloat() }
            val ledYs  = FloatArray(centers.size) { centers[it].second.y.toFloat() }

            // 2a) Grab camera/preview dimensions
            val camW = viewBinding.viewFinder.width
            val camH = viewBinding.viewFinder.height

            // 3) Pack into Intent
            Intent(this, MapActivity::class.java).also { intent ->
                intent.putExtra("userX",    userX)
                intent.putExtra("userY",    userY)
                intent.putExtra("ledIds",   ledIds)
                intent.putExtra("ledXs",    ledXs)
                intent.putExtra("ledYs",    ledYs)
                intent.putExtra("camWidth",  camW)
                intent.putExtra("camHeight", camH)
                startActivity(intent)
            }
        }

        // --- IMAGE PICKER logic (same pattern) ---
        viewBinding.uploadButton.setOnClickListener {
            pickMediaLauncher.launch("*/*")
        }

        // --- UPLOAD BUTTON LOGIC ---
        pickMediaLauncher = registerForActivityResult(ActivityResultContracts.GetContent()) { uri: Uri? ->
            uri?.let { handlePickedMedia(it) }
        }
    }

    // Handle picked image/video
    private fun handlePickedMedia(uri: Uri) {
        val mimeType = contentResolver.getType(uri)
        if (mimeType?.startsWith("image") == true) {
            // 1) Load bitmap
            val inputStream = contentResolver.openInputStream(uri)
            val bitmap = BitmapFactory.decodeStream(inputStream)
            inputStream?.close()

            // 2) Process frame
            videoProcessor?.processFrame(bitmap) { processedFrames ->
                processedFrames?.let { (outputBitmap, _) ->
                    viewBinding.processedFrameView.setImageBitmap(outputBitmap)
                    viewBinding.processedFrameView.visibility = View.VISIBLE

                    // 3) Get your three distances
                    val detections: List<Pair<Int, Double>> =
                        videoProcessor?.getLastLedDistances() ?: emptyList()
                    val DA = detections.getOrNull(0)?.second ?: 0.0
                    val DB = detections.getOrNull(1)?.second ?: 0.0
                    val DC = detections.getOrNull(2)?.second ?: 0.0

                    if (DA > 0 && DB > 0 && DC > 0) {
                        // 4) Trilaterate user X/Y
                        val (x, y) = Trilateration.solve(DA, DB, DC)
                        lastUserPosition = Pair(x, y)

                        // 5) **NEW**: grab raw LED centers
                        val centers = videoProcessor?.getLastLedCenters() ?: emptyList()
                        val ledIds = IntArray(centers.size)   { centers[it].first }
                        val ledXs  = FloatArray(centers.size) { centers[it].second.x.toFloat() }
                        val ledYs  = FloatArray(centers.size) { centers[it].second.y.toFloat() }

                        // 6) Build Intent with everything
                        Intent(this, MapActivity::class.java).also { intent ->
                            intent.putExtra("userX", x)
                            intent.putExtra("userY", y)
                            intent.putExtra("DA", DA)
                            intent.putExtra("DB", DB)
                            intent.putExtra("DC", DC)
                            // pass the LED-center arrays
                            intent.putExtra("ledIds", ledIds)
                            intent.putExtra("ledXs",  ledXs)
                            intent.putExtra("ledYs",  ledYs)
                            startActivity(intent)
                        }
                    } else {
                        Toast.makeText(this,
                            "Could not detect all three LEDs!",
                            Toast.LENGTH_LONG).show()
                    }
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

    private fun startProcessingAndRecording() {
        isRecording = true
        isProcessing = true
        viewBinding.startProcessingButton.text = "Stop Tracking"
        viewBinding.startProcessingButton.backgroundTintList =
            ContextCompat.getColorStateList(this, R.color.red)
        viewBinding.processedFrameView.visibility = View.VISIBLE

        // After 3 seconds, ask the user whether to launch MapActivity
        Handler(Looper.getMainLooper()).postDelayed({
            // 1) stop processing immediately
            stopProcessingAndRecording()

            AlertDialog.Builder(this)
                .setTitle("Open Map?")
                .setMessage("Do you want to view the map now?")
                .setPositiveButton("Yes") { _, _ ->
                    // Gather data and launch map
                    val userX = lastUserPosition?.first ?: 0.0
                    val userY = lastUserPosition?.second ?: 0.0
                    val centers = videoProcessor?.getLastLedCenters() ?: emptyList()
                    val ledIds = IntArray(centers.size)   { centers[it].first }
                    val ledXs  = FloatArray(centers.size) { centers[it].second.x.toFloat() }
                    val ledYs  = FloatArray(centers.size) { centers[it].second.y.toFloat() }
                    val camW = viewBinding.viewFinder.width
                    val camH = viewBinding.viewFinder.height

                    Intent(this, MapActivity::class.java).also { intent ->
                        intent.putExtra("userX",    userX)
                        intent.putExtra("userY",    userY)
                        intent.putExtra("ledIds",   ledIds)
                        intent.putExtra("ledXs",    ledXs)
                        intent.putExtra("ledYs",    ledYs)
                        intent.putExtra("camWidth",  camW)
                        intent.putExtra("camHeight", camH)
                        startActivity(intent)
                    }
                }
                .setNegativeButton("No", null)
                .show()
        }, 3000L)
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
