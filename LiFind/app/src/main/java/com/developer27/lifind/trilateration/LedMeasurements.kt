package com.developer27.lifind.trilateration

/** All anchor coordinates and measured distances are in centimetres. */
object LedLayout {
    val anchors = listOf(0.0 to 43.18, 43.18 to 0.0, -43.18 to 0.0)
    const val sensorHeightCm = 228.6
}

object LedMeasurements {
    fun encode(distances: List<Double?>): String {
        require(distances.size == 3)
        return LedLayout.anchors.mapIndexed { index, (x, y) ->
            val distance = distances[index]?.takeIf { it.isFinite() && it > 0.0 }
            "LED_${index + 1} -> Coordinates: {x=$x, y=$y} - Distance: {${distance?.let { "$it CM" } ?: "N/A"}}"
        }.joinToString("\n", postfix = "\n")
    }

    fun decode(text: String): List<Double?> {
        val distances = MutableList<Double?>(3) { null }
        val linePattern = Regex("""LED_([1-3])\s*->.*Distance:\s*\{([^}]*)\}""")
        text.lineSequence().forEach { line ->
            val match = linePattern.find(line) ?: return@forEach
            distances[match.groupValues[1].toInt() - 1] = match.groupValues[2]
                .removeSuffix("CM").trim().toDoubleOrNull()
                ?.takeIf { it.isFinite() && it > 0.0 }
        }
        return distances
    }
}
