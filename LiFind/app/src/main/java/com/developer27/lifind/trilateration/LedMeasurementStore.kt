package com.developer27.lifind.trilateration

import android.content.Context
import android.util.AtomicFile
import java.io.File

/** Private app storage works on every supported Android version without permissions. */
object LedMeasurementStore {
    private fun file(context: Context) = File(context.filesDir, "LiFind_Log.txt")

    @Synchronized
    fun write(context: Context, distances: List<Double?>): File {
        val target = file(context)
        val atomicFile = AtomicFile(target)
        val stream = atomicFile.startWrite()
        try {
            stream.write(LedMeasurements.encode(distances).toByteArray(Charsets.UTF_8))
            atomicFile.finishWrite(stream)
        } catch (error: Exception) {
            atomicFile.failWrite(stream)
            throw error
        }
        return target
    }

    @Synchronized
    fun read(context: Context): List<Double?> = try {
        LedMeasurements.decode(AtomicFile(file(context)).readFully().toString(Charsets.UTF_8))
    } catch (_: java.io.IOException) {
        listOf(null, null, null)
    }
}
