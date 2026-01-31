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
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.tensorflow.lite.Interpreter
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

/**
 * Global inference thresholds (set once from anywhere).
 *
 * IMPORTANT:
 * - YOLOLEDHelper.applyNms() reads Settings.Inference.iouThreshold
 * - YOLODISTANCEHelper.applyNms() reads Settings.Inference.iouThreshold
 *
 * This file includes a best-effort sync so InferenceConfig also updates
 * Settings.Inference.* if those are mutable vars. If Settings uses val getters,
 * the sync will be ignored safely.
 */
object InferenceConfig {
    @Volatile var confidenceThreshold: Float = 0.001f
    @Volatile var iouThreshold: Float = 0.25f

    @Volatile var classAgnosticNms: Boolean = false
    @Volatile var multiLabelPerBox: Boolean = false
}

object SingleModelEvaluator {

    private const val TAG = "SingleModelEvaluator"

    enum class ModelType { LED, Distance }

    data class SplitResult(
        val splitName: String,               // train | val | test
        val classCount: Int,                 // K
        val includeNone: Boolean,
        val matrix: Array<IntArray>,         // [gt][pred]
        val totalUsed: Int,
        val skippedNoLabel: Int,
        val noDetection: Int,
        val skippedDecodeFail: Int
    )

    // -------------------- Progress Dialog --------------------

    /**
     * Simple non-cancelable progress dialog that can be updated from any thread.
     * Uses AlertDialog + horizontal ProgressBar + message TextView.
     */
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

                progressBar = ProgressBar(
                    context,
                    null,
                    android.R.attr.progressBarStyleHorizontal
                ).apply {
                    isIndeterminate = false
                    max = 100
                    progress = 0
                    layoutParams = LinearLayout.LayoutParams(
                        ViewGroup.LayoutParams.MATCH_PARENT,
                        ViewGroup.LayoutParams.WRAP_CONTENT
                    ).apply {
                        gravity = Gravity.CENTER_HORIZONTAL
                    }
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
                progressPercent?.let { p ->
                    val clamped = p.coerceIn(0, 100)
                    progressBar?.progress = clamped
                }
            }
        }

        fun dismiss() {
            mainHandler.post {
                try {
                    dialog?.dismiss()
                } catch (_: Throwable) {
                    // ignore
                } finally {
                    dialog = null
                    progressBar = null
                    titleView = null
                    msgView = null
                    isShown.set(false)
                }
            }
        }
    }

    /**
     * Optional: Show a progress bar message box while processing.
     *
     * - If showProgressDialog=true, a non-cancelable dialog is shown.
     * - Progress is updated using:
     *     split progress (0..100) within each split
     *     and overall progress across train/val/test (0..100)
     */
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
            // Show dialog early (main thread)
            progressDialog?.show(
                initialTitle = "Evaluating ${model.name} Detection Model",
                initialMsg = "Preparing..."
            )

            // Keep helper NMS IoU consistent with InferenceConfig
            syncSettingsFromInferenceConfig()

            val (interp, k) = when (model) {
                ModelType.LED -> YOLOLEDHelper.ensureInterpreter(context) to 3
                ModelType.Distance -> YOLODISTANCEHelper.ensureInterpreter(context) to YOLODISTANCEHelper.NUM_CLASSES
            }

            // Helper to update overall progress (3 splits total)
            fun updateOverall(splitIndex: Int, splitName: String, done: Int, total: Int) {
                // splitIndex: 0=train, 1=val, 2=test
                val splitFrac = if (total > 0) done.toFloat() / total.toFloat() else 0f
                val overall = ((splitIndex + splitFrac) / 3f * 100f).toInt().coerceIn(0, 100)
                val splitPct = (splitFrac * 100f).toInt().coerceIn(0, 100)

                progressDialog?.update(
                    title = "Evaluating ${model.name} model",
                    msg = "Split: $splitName ($done / $total)",
                    progressPercent = overall
                )

                // keep your external callback intact
                onProgress(splitName, done, total)

                // (Optional) log split percentage sometimes
                if (done == 1 || done == total || done % 25 == 0) {
                    Log.d(TAG, "Progress $splitName: $splitPct% (overall $overall%)")
                }
            }

            val results = buildList {
                add(
                    evalOneSplit(
                        context = context,
                        model = model,
                        interpreter = interp,
                        classCount = k,
                        splitName = "train",
                        zipUri = trainZip,
                        includeNone = includeNoneClass
                    ) { split, done, total -> updateOverall(0, split, done, total) }
                )
                add(
                    evalOneSplit(
                        context = context,
                        model = model,
                        interpreter = interp,
                        classCount = k,
                        splitName = "val",
                        zipUri = valZip,
                        includeNone = includeNoneClass
                    ) { split, done, total -> updateOverall(1, split, done, total) }
                )
                add(
                    evalOneSplit(
                        context = context,
                        model = model,
                        interpreter = interp,
                        classCount = k,
                        splitName = "test",
                        zipUri = testZip,
                        includeNone = includeNoneClass
                    ) { split, done, total -> updateOverall(2, split, done, total) }
                )
            }

            progressDialog?.update(msg = "Done.", progressPercent = 100)
            results
        } catch (t: Throwable) {
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
                ModelType.Distance -> YOLODISTANCEHelper.classNameForId(idx)
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
                out.println("iou_threshold,${InferenceConfig.iouThreshold}")
                out.println("class_agnostic_nms,${InferenceConfig.classAgnosticNms}")
                out.println("multi_label_per_box,${InferenceConfig.multiLabelPerBox}")
                out.println("total_used,${res.totalUsed}")
                out.println("skipped_no_label,${res.skippedNoLabel}")
                out.println("skipped_decode_fail,${res.skippedDecodeFail}")
                out.println("no_detection,${res.noDetection}")
            }
            outFile
        } catch (t: Throwable) {
            Log.e(TAG, "writeCsv failed", t)
            null
        }
    }

    // -------------------- paths --------------------

    private fun getLiFindZipResultsDirOrFallback(context: Context): File {
        val docsDir = try {
            Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOCUMENTS)
        } catch (_: Throwable) {
            null
        }

        // EXACT requested path: Documents/LiFind/Zip Inference Results
        if (docsDir != null) {
            val target = File(docsDir, "LiFind_Zip_Inference_Results")
            if (isDirWritable(target)) return target
        }

        // Fallback: always writable
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

    // -------------------- internal --------------------

    private suspend fun evalOneSplit(
        context: Context,
        model: ModelType,
        interpreter: Interpreter,
        classCount: Int,
        splitName: String,
        zipUri: Uri,
        includeNone: Boolean,
        onProgress: (split: String, done: Int, total: Int) -> Unit
    ): SplitResult = withContext(Dispatchers.IO) {

        // Keep helper thresholds in sync
        syncSettingsFromInferenceConfig()

        val workDir = File(context.cacheDir, "lifind_eval_${model.name.lowercase()}_$splitName").apply {
            deleteRecursively()
            mkdirs()
        }

        val zipFile = File(workDir, "$splitName.zip")
        copyUriToFile(context.contentResolver, zipUri, zipFile)

        val unzipDir = File(workDir, "unzipped").apply { mkdirs() }
        unzip(zipFile, unzipDir)

        val images = unzipDir.walkTopDown()
            .filter { it.isFile && it.extension.lowercase() in setOf("jpg", "jpeg", "png") }
            .toList()

        Log.d(TAG, "Split=$splitName images_found=${images.size} root=${unzipDir.absolutePath}")

        val size = if (includeNone) classCount + 1 else classCount
        val matrix = Array(size) { IntArray(size) }

        var used = 0
        var skippedNoLabel = 0
        var skippedDecodeFail = 0
        var noDet = 0

        for ((idx, imgFile) in images.withIndex()) {

            val bmp = decodeBitmap(imgFile)
            if (bmp == null) {
                skippedDecodeFail++
                Log.w(TAG, "Decode failed: ${imgFile.absolutePath}")
                onProgress(splitName, idx + 1, images.size)
                continue
            }

            // Robust GT label lookup (supports many YOLO zip layouts)
            val gt = readGtClassForImage(unzipDir, imgFile)
            if (gt == null) {
                skippedNoLabel++
                Log.w(TAG, "No label for image: ${imgFile.absolutePath}")
                onProgress(splitName, idx + 1, images.size)
                continue
            }
            val gtIdx = gt.coerceIn(0, classCount - 1)

            val pred = when (model) {
                ModelType.LED -> inferLed(interpreter, bmp)
                ModelType.Distance -> inferDistance(interpreter, bmp)
            }

            val predIdx = if (pred == null) {
                noDet++
                if (includeNone) {
                    classCount // NONE index
                } else {
                    onProgress(splitName, idx + 1, images.size)
                    continue
                }
            } else {
                pred.coerceIn(0, classCount - 1)
            }

            matrix[gtIdx][predIdx]++
            used++

            onProgress(splitName, idx + 1, images.size)
        }

        SplitResult(
            splitName = splitName,
            classCount = classCount,
            includeNone = includeNone,
            matrix = matrix,
            totalUsed = used,
            skippedNoLabel = skippedNoLabel,
            noDetection = noDet,
            skippedDecodeFail = skippedDecodeFail
        )
    }

    private fun inferLed(interpreter: Interpreter, src: Bitmap): Int? {
        val inputBmp = YOLOLEDHelper.autoOrientAndResize(src)
        val input = BitmapToFloatTensor.nhwc(inputBmp)

        val outTensor = interpreter.getOutputTensor(0)
        val s = outTensor.shape()
        val out = Array(s[0]) { Array(s[1]) { FloatArray(s[2]) } }

        interpreter.run(input, out)

        val dets = YOLOLEDHelper.parseTFLite(
            raw = out,
            confidenceThreshold = InferenceConfig.confidenceThreshold,
            classAgnosticNms = InferenceConfig.classAgnosticNms,
            multiLabelPerBox = InferenceConfig.multiLabelPerBox,
            expectedClasses = 3
        )
        return dets.maxByOrNull { it.confidence }?.classId
    }

    private fun inferDistance(interpreter: Interpreter, src: Bitmap): Int? {
        val inputBmp = YOLODISTANCEHelper.autoOrientAndResize(src)
        val input = BitmapToFloatTensor.nhwc(inputBmp)

        val outTensor = interpreter.getOutputTensor(0)
        val s = outTensor.shape()
        val out = Array(s[0]) { Array(s[1]) { FloatArray(s[2]) } }

        interpreter.run(input, out)

        val dets = YOLODISTANCEHelper.parseTFLite(
            raw = out,
            confidenceThreshold = InferenceConfig.confidenceThreshold,
            classAgnosticNms = InferenceConfig.classAgnosticNms,
            multiLabelPerBox = InferenceConfig.multiLabelPerBox,
            expectedClasses = YOLODISTANCEHelper.NUM_CLASSES
        )
        return YOLODISTANCEHelper.bestOne(dets)?.classId
    }

    /**
     * Robust GT label lookup for common YOLO zip layouts.
     *
     * Supports:
     * - .../images/xxx.jpg   with sibling .../labels/xxx.txt (case-insensitive folder names)
     * - train/val/test splits (labels folder near the images folder)
     * - alternative naming: Labels, annotations, Annotations
     * - a global root "labels/**/xxx.txt" as a fallback
     */
    private fun readGtClassForImage(root: File, imageFile: File): Int? {
        val baseName = imageFile.nameWithoutExtension

        val labelDirs = LinkedHashSet<File>()
        val labelFolderNames = listOf("labels", "annotations")

        var p: File? = imageFile.parentFile
        repeat(7) {
            val cur = p ?: return@repeat

            for (name in labelFolderNames) {
                val d = File(cur, name)
                if (d.exists() && d.isDirectory) labelDirs.add(d)

                val d2 = File(cur, name.replaceFirstChar { it.uppercaseChar() })
                if (d2.exists() && d2.isDirectory) labelDirs.add(d2)
            }

            if (cur.name.equals("images", ignoreCase = true)) {
                val parent = cur.parentFile
                if (parent != null) {
                    for (name in labelFolderNames) {
                        val d = File(parent, name)
                        if (d.exists() && d.isDirectory) labelDirs.add(d)

                        val d2 = File(parent, name.replaceFirstChar { it.uppercaseChar() })
                        if (d2.exists() && d2.isDirectory) labelDirs.add(d2)
                    }
                }
            }

            p = cur.parentFile
        }

        try {
            root.walkTopDown()
                .maxDepth(8)
                .filter { it.isDirectory && (it.name.equals("labels", true) || it.name.equals("annotations", true)) }
                .forEach { labelDirs.add(it) }
        } catch (_: Throwable) {
            // ignore
        }

        val labelFile = labelDirs
            .asSequence()
            .map { dir -> File(dir, "$baseName.txt") }
            .firstOrNull { it.exists() }
            ?: run {
                val same = File(imageFile.parentFile, "$baseName.txt")
                if (same.exists()) same else null
            }
            ?: return null

        val firstLine = labelFile.bufferedReader().useLines { seq ->
            seq.firstOrNull { it.isNotBlank() }
        } ?: return null

        val clsStr = firstLine.trim().split(Regex("\\s+")).firstOrNull() ?: return null
        return clsStr.toIntOrNull()
    }

    private fun decodeBitmap(file: File): Bitmap? =
        try { BitmapFactory.decodeFile(file.absolutePath) } catch (_: Throwable) { null }

    private fun copyUriToFile(resolver: ContentResolver, uri: Uri, out: File) {
        resolver.openInputStream(uri).use { input ->
            requireNotNull(input) { "Cannot open input stream for uri=$uri" }
            FileOutputStream(out).use { output -> input.copyTo(output) }
        }
    }

    private fun unzip(zipFile: File, destDir: File) {
        ZipInputStream(BufferedInputStream(FileInputStream(zipFile))).use { zis ->
            var entry: ZipEntry? = zis.nextEntry
            while (entry != null) {
                val safeName = entry.name.replace("\\", "/")
                val outFile = File(destDir, safeName)

                val canonicalDest = destDir.canonicalPath
                val canonicalOut = outFile.canonicalPath
                require(canonicalOut.startsWith(canonicalDest)) { "Blocked ZipSlip: ${entry!!.name}" }

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

    /**
     * Best-effort: keep helper NMS IoU consistent with InferenceConfig by syncing into Settings.Inference.*
     * without taking a compile-time dependency on Settings having mutable vars.
     */
    private fun syncSettingsFromInferenceConfig() {
        try {
            val inferenceCls = Class.forName("com.developer27.lifind.videoprocessing.Settings\$Inference")
            val instance = inferenceCls.getField("INSTANCE").get(null)

            fun callSetter(name: String, arg: Any) {
                try {
                    val m = inferenceCls.methods.firstOrNull { it.name == name } ?: return
                    m.invoke(instance, arg)
                } catch (_: Throwable) {
                    // ignore
                }
            }

            callSetter("setConfidenceThreshold", InferenceConfig.confidenceThreshold)
            callSetter("setIouThreshold", InferenceConfig.iouThreshold)
            callSetter("setClassAgnosticNms", InferenceConfig.classAgnosticNms)
            callSetter("setMultiLabelPerBox", InferenceConfig.multiLabelPerBox)
        } catch (_: Throwable) {
            // ignore
        }
    }
}