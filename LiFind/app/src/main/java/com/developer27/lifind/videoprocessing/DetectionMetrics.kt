package com.developer27.lifind.videoprocessing

/** Accumulates spatial matches, including misclassifications, in a detection confusion matrix. */
internal class DetectionMetrics(private val classCount: Int, includeNone: Boolean) {
    val matrix = Array(classCount + if (includeNone) 1 else 0) {
        IntArray(classCount + if (includeNone) 1 else 0)
    }
    private val noneIndex = if (includeNone) classCount else null
    var matched = 0; private set
    var missed = 0; private set
    var falsePositives = 0; private set

    fun addImage(groundTruth: List<DetectionResult>, predictions: List<DetectionResult>, matchIou: Float) {
        val used = BooleanArray(predictions.size)
        for (truth in groundTruth) {
            require(truth.classId in 0 until classCount) { "Invalid ground-truth class ${truth.classId}" }
            var best = -1
            var bestIou = 0f
            for (index in predictions.indices) {
                if (used[index]) continue
                val overlap = iou(truth, predictions[index])
                if (overlap >= matchIou && overlap > bestIou) {
                    best = index
                    bestIou = overlap
                }
            }
            if (best >= 0) {
                val prediction = predictions[best]
                require(prediction.classId in 0 until classCount)
                used[best] = true
                matrix[truth.classId][prediction.classId]++
                matched++
            } else {
                missed++
                noneIndex?.let { matrix[truth.classId][it]++ }
            }
        }
        predictions.forEachIndexed { index, prediction ->
            require(prediction.classId in 0 until classCount)
            if (!used[index]) {
                falsePositives++
                noneIndex?.let { matrix[it][prediction.classId]++ }
            }
        }
    }

    private fun iou(a: DetectionResult, b: DetectionResult): Float {
        val width = (minOf(a.xCenter + a.width / 2, b.xCenter + b.width / 2) -
            maxOf(a.xCenter - a.width / 2, b.xCenter - b.width / 2)).coerceAtLeast(0f)
        val height = (minOf(a.yCenter + a.height / 2, b.yCenter + b.height / 2) -
            maxOf(a.yCenter - a.height / 2, b.yCenter - b.height / 2)).coerceAtLeast(0f)
        val intersection = width * height
        val union = a.width * a.height + b.width * b.height - intersection
        return if (union > 0f) intersection / union else 0f
    }
}
