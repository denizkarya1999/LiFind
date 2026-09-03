package com.developer27.lifind.videoprocessing

import android.app.AlertDialog
import android.app.Dialog
import android.content.ContentResolver
import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Environment
import android.os.Handler
import android.os.Looper
import android.util.Log
import android.view.Gravity
import android.view.ViewGroup
import android.widget.LinearLayout
import android.widget.ProgressBar
import android.widget.TextView
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.BufferedInputStream
import java.io.BufferedWriter
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.io.FileWriter
import java.io.PrintWriter
import java.util.concurrent.atomic.AtomicBoolean
import java.util.zip.ZipEntry
import java.util.zip.ZipInputStream

/** Evaluation thresholds are independent of live tracking settings. */
object InferenceConfig {
    @Volatile var confidenceThreshold: Float = 0.001f
    @Volatile var iouThreshold: Float = 0.25f

    @Volatile var classAgnosticNms: Boolean = false
    @Volatile var multiLabelPerBox: Boolean = true

    /**
     * IoU threshold used to MATCH a predicted box to a GT box for confusion-matrix counting.
     * (This is separate from NMS IoU.)
     */
    @Volatile var matchIouThreshold: Float = 0.50f
}

object SingleModelEvaluator {

    private const val TAG = "SingleModelEvaluator"

    enum class ModelType { LED, DISTANCE }

    data class SplitResult(
        val splitName: String,               // train | val | test
        val classCount: Int,                 // K
        val includeNone: Boolean,
        val matrix: Array<IntArray>,         // [gt][pred] with optional NONE row/col
        val totalGtBoxes: Int,               // total GT objects processed
        val skippedNoLabel: Int,             // images skipped due to missing label file
        val skippedDecodeFail: Int,           // images skipped due to decode failure
        val matched: Int,                    // # matched GT boxes
        val missed: Int,                     // # GT boxes with no matching prediction
        val falsePositives: Int              // # predictions not matched to any GT
    )

    // -------------------- Progress Dialog --------------------

    private class ProgressDialogController(private val context: Context) {
        private val mainHandler = Handler(Looper.getMainLooper())
        private var dialog: Dialog? = null
        private var progressBar: ProgressBar? = null
        private var titleView: TextView? = null
        private var msgView: TextView? = null
        private val isShown = AtomicBoolean(false)

        fun show(initialTitle: String, initialMsg: String) {
            mainHandler.post {
                if (isShown.getAndSet(true)) return@post

                val root = LinearLayout(context).apply {
                    orientation = LinearLayout.VERTICAL
                    setPadding(48, 32, 48, 16)
                    layoutParams = ViewGroup.LayoutParams(
                        ViewGroup.LayoutParams.MATCH_PARENT,
                        ViewGroup.LayoutParams.WRAP_CONTENT
                    )
                }

                titleView = TextView(context).apply {
                    textSize = 18f
                    text = initialTitle
                    setPadding(0, 0, 0, 18)
                }
                msgView = TextView(context).apply {
                    textSize = 14f
                    text = initialMsg
                    setPadding(0, 0, 0, 18)
                }

                progressBar = ProgressBar(context, null, android.R.attr.progressBarStyleHorizontal).apply {
                    isIndeterminate = false
                    max = 100
                    progress = 0
                    layoutParams = LinearLayout.LayoutParams(
                        ViewGroup.LayoutParams.MATCH_PARENT,
                        ViewGroup.LayoutParams.WRAP_CONTENT
                    ).apply { gravity = Gravity.CENTER_HORIZONTAL }
                }

                root.addView(titleView)
                root.addView(msgView)
                root.addView(progressBar)

                dialog = AlertDialog.Builder(context)
                    .setView(root)
                    .setCancelable(false)
                    .create()

                try {
                    dialog?.show()
                } catch (t: Throwable) {
                    Log.w(TAG, "Failed to show progress dialog", t)
                    isShown.set(false)
                }
            }
        }

        fun update(title: String? = null, msg: String? = null, progressPercent: Int? = null) {
            mainHandler.post {
                if (!isShown.get()) return@post
                title?.let { titleView?.text = it }
                msg?.let { msgView?.text = it }
                progressPercent?.let { progressBar?.progress = it.coerceIn(0, 100) }
            }
        }

        fun dismiss() {
            mainHandler.post {
                try { dialog?.dismiss() } catch (_: Throwable) { }
                dialog = null
                progressBar = null
                titleView = null
                msgView = null
                isShown.set(false)
            }
        }
    }

    // -------------------- Public API --------------------

    suspend fun evaluateTrainValTest(
        context: Context,
        model: ModelType,
        trainZip: Uri,
        valZip: Uri,
        testZip: Uri,
        includeNoneClass: Boolean = true,
        showProgressDialog: Boolean = true,
        onProgress: (split: String, done: Int, total: Int) -> Unit = { _, _, _ -> }
    ): List<SplitResult>? = withContext(Dispatchers.IO) {

        val progressDialog = if (showProgressDialog) ProgressDialogController(context) else null

        try {
            progressDialog?.show(
                initialTitle = "Evaluating ${model.name} Detection Model",
                initialMsg = "Preparing..."
            )


            val k = when (model) {
                ModelType.LED -> YOLOLEDHelper.labels.size
                ModelType.DISTANCE -> YOLODISTANCEHelper.NUM_CLASSES
            }

            fun updateOverall(splitIndex: Int, splitName: String, done: Int, total: Int) {
                val splitFrac = if (total > 0) done.toFloat() / total.toFloat() else 0f
                val overall = ((splitIndex + splitFrac) / 3f * 100f).toInt().coerceIn(0, 100)
                progressDialog?.update(
                    title = "Evaluating ${model.name} model",
                    msg = "Split: $splitName ($done / $total)",
                    progressPercent = overall
                )
                onProgress(splitName, done, total)
            }

            val results = buildList {
                add(evalOneSplit(context, model, k, "Train", trainZip, includeNoneClass) { s, d, t -> updateOverall(0, s, d, t) })
                add(evalOneSplit(context, model, k, "Val",   valZip,   includeNoneClass) { s, d, t -> updateOverall(1, s, d, t) })
                add(evalOneSplit(context, model, k, "Test",  testZip,  includeNoneClass) { s, d, t -> updateOverall(2, s, d, t) })
            }

            progressDialog?.update(msg = "Done.", progressPercent = 100)
            results
        } catch (t: CancellationException) {
            throw t
        } catch (t: Exception) {
            Log.e(TAG, "evaluateTrainValTest failed", t)
            null
        } finally {
            progressDialog?.dismiss()
        }
    }

    /**
     * Save into:
     *   Documents/LiFind/Zip Inference Results/
     *
     * ⚠️ Android 10+ may block direct writes to public Documents.
     * This method tries Documents first; if blocked it falls back to:
     *   Android/data/<package>/files/LiFind/Zip Inference Results/
     */
    fun writeCsv(context: Context, model: ModelType, res: SplitResult): File? {
        val modelTag = if (model == ModelType.LED) "led" else "distance"
        val fileName = "confusion_${modelTag}_${res.splitName}.csv"

        val outDir = getLiFindZipResultsDirOrFallback(context)
        if (!outDir.exists()) outDir.mkdirs()

        val outFile = File(outDir, fileName)

        fun labelFor(idx: Int): String {
            if (res.includeNone && idx == res.classCount) return "NONE"
            return when (model) {
                ModelType.LED -> YOLOLEDHelper.classNameForId(idx)
                ModelType.DISTANCE -> YOLODISTANCEHelper.classNameForId(idx)
            }
        }

        fun csvCell(s: String): String {
            val needsQuotes = s.contains(",") || s.contains("\"") || s.contains("\n")
            return if (!needsQuotes) s else "\"" + s.replace("\"", "\"\"") + "\""
        }

        return try {
            PrintWriter(BufferedWriter(FileWriter(outFile))).use { out ->
                val size = if (res.includeNone) res.classCount + 1 else res.classCount

                out.println("gt\\pred," + (0 until size).joinToString(",") { csvCell(labelFor(it)) })

                for (gt in 0 until size) {
                    out.println(csvCell(labelFor(gt)) + "," + res.matrix[gt].joinToString(","))
                }

                out.println()
                out.println("confidence_threshold,${InferenceConfig.confidenceThreshold}")
                out.println("nms_iou_threshold,${InferenceConfig.iouThreshold}")
                out.println("match_iou_threshold,${InferenceConfig.matchIouThreshold}")
                out.println("class_agnostic_nms,${InferenceConfig.classAgnosticNms}")
                out.println("multi_label_per_box,${InferenceConfig.multiLabelPerBox}")
                out.println("total_gt_boxes,${res.totalGtBoxes}")
                out.println("matched,${res.matched}")
                out.println("missed,${res.missed}")
                out.println("false_positives,${res.falsePositives}")
                out.println("skipped_no_label,${res.skippedNoLabel}")
                out.println("skipped_decode_fail,${res.skippedDecodeFail}")
            }
            outFile
        } catch (t: Throwable) {
            Log.e(TAG, "writeCsv failed", t)
            null
        }
    }

    // -------------------- internal eval --------------------

    private data class Box(
        val classId: Int,
        val xc: Float,
        val yc: Float,
        val w: Float,
        val h: Float
    )

    private suspend fun evalOneSplit(
        context: Context,
        model: ModelType,
        classCount: Int,
        splitName: String,
        zipUri: Uri,
        includeNone: Boolean,
        onProgress: (split: String, done: Int, total: Int) -> Unit
    ): SplitResult = withContext(Dispatchers.IO) {


        val workDir = File(context.cacheDir, "lifind_eval_${model.name.lowercase()}_$splitName").apply {
            deleteRecursively()
            mkdirs()
        }

        try {
            val zipFile = File(workDir, "$splitName.zip")
            copyUriToFile(context.contentResolver, zipUri, zipFile)

            val unzipDir = File(workDir, "unzipped").apply { mkdirs() }
            unzip(zipFile, unzipDir)

            val images = unzipDir.walkTopDown()
                .filter { it.isFile && it.extension.lowercase() in setOf("jpg", "jpeg", "png") }
                .toList()

            Log.d(TAG, "Split=$splitName images_found=${images.size} root=${unzipDir.absolutePath}")

            val metrics = DetectionMetrics(classCount, includeNone)
            var skippedNoLabelImgs = 0
            var skippedDecodeFailImgs = 0
            var totalGtBoxes = 0

            require(images.isNotEmpty()) { "No supported images in $splitName ZIP." }
            for ((idx, imgFile) in images.withIndex()) {
                currentCoroutineContext().ensureActive()
                val bmp = decodeBitmap(imgFile)
                if (bmp == null) {
                    skippedDecodeFailImgs++
                    Log.w(TAG, "Decode failed: ${imgFile.absolutePath}")
                    onProgress(splitName, idx + 1, images.size)
                    continue
                }

                // Read ALL gt boxes (multiple lines) for this image
                val gtBoxes = readGtBoxesForImage(unzipDir, imgFile)
                if (gtBoxes == null) {
                    skippedNoLabelImgs++
                    bmp.recycle()
                    Log.w(TAG, "No label for image: ${imgFile.absolutePath}")
                    onProgress(splitName, idx + 1, images.size)
                    continue
                }

                totalGtBoxes += gtBoxes.size

                // Run inference -> multiple predicted boxes
                val predDetections = try {
                    when (model) {
                        ModelType.LED -> inferLedDetections(context, bmp)
                        ModelType.DISTANCE -> inferDistanceDetections(context, bmp)
                    }
                } finally {
                    bmp.recycle()
                }

                metrics.addImage(
                    gtBoxes.map { DetectionResult(it.xc, it.yc, it.w, it.h, 1f, it.classId) },
                    predDetections,
                    InferenceConfig.matchIouThreshold
                )

                onProgress(splitName, idx + 1, images.size)
            }

            SplitResult(
                splitName = splitName,
                classCount = classCount,
                includeNone = includeNone,
                matrix = metrics.matrix,
                totalGtBoxes = totalGtBoxes,
                skippedNoLabel = skippedNoLabelImgs,
                skippedDecodeFail = skippedDecodeFailImgs,
                matched = metrics.matched,
                missed = metrics.missed,
                falsePositives = metrics.falsePositives
            )
        } finally {
            workDir.deleteRecursively()
        }
    }

    // -------------------- inference (use helpers) --------------------

    private fun inferLedDetections(context: Context, src: Bitmap): List<DetectionResult> {
        val interpreter = YOLOLEDHelper.ensureInterpreter(context)

        val inputBmp = YOLOLEDHelper.autoOrientAndResize(src)
        val input = BitmapToFloatTensor.nhwc(inputBmp, YOLOLEDHelper.INPUT_SIZE)

        val outTensor = interpreter.getOutputTensor(0)
        val s = outTensor.shape()
        val out = Array(s[0]) { Array(s[1]) { FloatArray(s[2]) } }

        synchronized(interpreter) {
            interpreter.run(input, out)
        }
        if (inputBmp !== src) inputBmp.recycle()

        return YOLOLEDHelper.parseTFLite(
            raw = out,
            confidenceThreshold = InferenceConfig.confidenceThreshold,
            classAgnosticNms = InferenceConfig.classAgnosticNms,
            multiLabelPerBox = InferenceConfig.multiLabelPerBox,
            expectedClasses = YOLOLEDHelper.labels.size,
            iouThreshold = InferenceConfig.iouThreshold
        )
    }

    private fun inferDistanceDetections(context: Context, src: Bitmap): List<DetectionResult> {
        val interpreter = YOLODISTANCEHelper.ensureInterpreter(context)

        val inputBmp = YOLODISTANCEHelper.autoOrientAndResize(src)
        val input = BitmapToFloatTensor.nhwc(inputBmp, YOLODISTANCEHelper.INPUT_SIZE)

        val outTensor = interpreter.getOutputTensor(0)
        val s = outTensor.shape()
        val out = Array(s[0]) { Array(s[1]) { FloatArray(s[2]) } }

        synchronized(interpreter) {
            interpreter.run(input, out)
        }
        if (inputBmp !== src) inputBmp.recycle()

        return YOLODISTANCEHelper.parseTFLite(
            raw = out,
            confidenceThreshold = InferenceConfig.confidenceThreshold,
            classAgnosticNms = InferenceConfig.classAgnosticNms,
            multiLabelPerBox = InferenceConfig.multiLabelPerBox,
            expectedClasses = YOLODISTANCEHelper.NUM_CLASSES,
            iouThreshold = InferenceConfig.iouThreshold
        )
    }

    // -------------------- GT parsing (multiple lines) --------------------

    /**
     * Reads YOLO label file for an image (multiple lines => multiple objects).
     * Returns null if label file not found.
     * Returns empty list if file exists but has no valid lines.
     *
     * Each line: class xc yc w h (normalized 0..1)
     */
    private fun readGtBoxesForImage(root: File, imageFile: File): List<Box>? {
        val labelFile = findLabelFileForImage(root, imageFile) ?: return null

        val out = ArrayList<Box>(8)
        labelFile.bufferedReader().useLines { seq ->
            seq.forEach { line ->
                val t = line.trim()
                if (t.isEmpty()) return@forEach
                val parts = t.split(Regex("\\s+"))
                if (parts.size < 5) return@forEach

                val cls = parts[0].toIntOrNull() ?: return@forEach
                val xc = parts[1].toFloatOrNull() ?: return@forEach
                val yc = parts[2].toFloatOrNull() ?: return@forEach
                val w  = parts[3].toFloatOrNull() ?: return@forEach
                val h  = parts[4].toFloatOrNull() ?: return@forEach

                out.add(
                    Box(
                        classId = cls,
                        xc = xc.coerceIn(0f, 1f),
                        yc = yc.coerceIn(0f, 1f),
                        w = w.coerceIn(0f, 1f),
                        h = h.coerceIn(0f, 1f)
                    )
                )
            }
        }
        return out
    }

    /**
     * Finds matching label file for an image.
     * Supports common YOLO layouts:
     *  - .../images/xxx.jpg and .../labels/xxx.txt
     *  - .../train/images/xxx.jpg and .../train/labels/xxx.txt
     *  - root/labels/**/xxx.txt (fallback)
     */
    private fun findLabelFileForImage(root: File, imageFile: File): File? {
        val baseName = imageFile.nameWithoutExtension
        val imgPathUnix = imageFile.absolutePath.replace("\\", "/")

        // Case 1: replace /images/ with /labels/ (case-insensitive folder names not guaranteed by string replace,
        // but this catches most lower-case datasets)
        val candidate1 = if (imgPathUnix.contains("/images/")) {
            val guess = imgPathUnix
                .replace("/images/", "/labels/")
                .replaceAfterLast("/", "$baseName.txt")
            File(guess)
        } else null

        // Case 2: sibling labels folder next to an Images/images folder (case-insensitive scan upwards)
        var candidate2: File? = null
        var p: File? = imageFile.parentFile
        repeat(7) {
            val cur = p ?: return@repeat
            if (cur.name.equals("images", ignoreCase = true)) {
                val parent = cur.parentFile
                if (parent != null) {
                    val d1 = File(parent, "labels")
                    val d2 = File(parent, "Labels")
                    val f1 = File(d1, "$baseName.txt")
                    val f2 = File(d2, "$baseName.txt")
                    if (f1.exists()) candidate2 = f1
                    if (candidate2 == null && f2.exists()) candidate2 = f2
                }
            }
            p = cur.parentFile
        }

        // Case 3: root/labels/**/xxx.txt fallback
        val labelsDir = File(root, "labels")
        val candidate3 = if (labelsDir.exists()) {
            labelsDir.walkTopDown().firstOrNull { it.isFile && it.name == "$baseName.txt" }
        } else null

        return listOfNotNull(candidate1, candidate2, candidate3).firstOrNull { it.exists() }
    }

    // -------------------- IoU (xywh normalized) --------------------

    // -------------------- IO helpers --------------------

    private fun decodeBitmap(file: File): Bitmap? =
        try { BitmapFactory.decodeFile(file.absolutePath) } catch (_: Throwable) { null }

    private fun copyUriToFile(resolver: ContentResolver, uri: Uri, out: File) {
        resolver.openInputStream(uri).use { input ->
            requireNotNull(input) { "Cannot open input stream for uri=$uri" }
            FileOutputStream(out).use { output -> input.copyTo(output) }
        }
    }

    internal fun unzip(zipFile: File, destDir: File) {
        ZipInputStream(BufferedInputStream(FileInputStream(zipFile))).use { zis ->
            var entry: ZipEntry? = zis.nextEntry
            while (entry != null) {
                val safeName = entry.name.replace("\\", "/")
                val outFile = File(destDir, safeName)

                // ZipSlip protection
                val canonicalDest = destDir.canonicalPath
                val canonicalOut = outFile.canonicalPath
                require(canonicalOut.startsWith(canonicalDest + File.separator)) { "Blocked ZipSlip: ${entry!!.name}" }

                if (entry.isDirectory) {
                    outFile.mkdirs()
                } else {
                    outFile.parentFile?.mkdirs()
                    FileOutputStream(outFile).use { fos -> zis.copyTo(fos) }
                }

                zis.closeEntry()
                entry = zis.nextEntry
            }
        }
    }

    // -------------------- output dir --------------------

    private fun getLiFindZipResultsDirOrFallback(context: Context): File {
        val docsDir = try {
            Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOCUMENTS)
        } catch (_: Throwable) {
            null
        }

        if (docsDir != null) {
            val target = File(docsDir, "LiFind_Zip_Inference_Results")
            if (isDirWritable(target)) return target
        }

        val base = context.getExternalFilesDir(null) ?: context.filesDir
        return File(base, "LiFind_Zip_Inference_Results")
    }

    private fun isDirWritable(dir: File): Boolean {
        return try {
            if (!dir.exists()) dir.mkdirs()
            val probe = File(dir, ".probe_${System.currentTimeMillis()}.tmp")
            probe.writeText("ok")
            probe.delete()
            true
        } catch (_: Throwable) {
            false
        }
    }

}
