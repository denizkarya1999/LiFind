package com.developer27.lifind.videoprocessing

import android.graphics.Bitmap
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Preprocess + tensor conversion aligned with your helpers:
 *  - force ARGB_8888 (same as YOLOLEDHelper/YOLODISTANCEHelper autoOrientAndResize)
 *  - resize to inputSize x inputSize
 *  - NHWC float32 [1,H,W,3] normalized to 0..1
 *
 * Notes:
 * - This is intentionally minimal and consistent with the helpers.
 * - Returns a direct ByteBuffer suitable for Interpreter.run().
 */
object BitmapToFloatTensor {

    fun nhwc(src: Bitmap, inputSize: Int): ByteBuffer {
        val bmp = toArgbAndResize(src, inputSize)
        val w = bmp.width
        val h = bmp.height

        val buf = ByteBuffer
            .allocateDirect(4 * h * w * 3)
            .order(ByteOrder.nativeOrder())

        val pixels = IntArray(w * h)
        bmp.getPixels(pixels, 0, w, 0, 0, w, h)

        for (p in pixels) {
            buf.putFloat(((p ushr 16) and 0xFF) / 255f) // R
            buf.putFloat(((p ushr 8) and 0xFF) / 255f)  // G
            buf.putFloat((p and 0xFF) / 255f)           // B
        }

        buf.rewind()
        return buf
    }

    private fun toArgbAndResize(src: Bitmap, inputSize: Int): Bitmap {
        val base = if (src.config != Bitmap.Config.ARGB_8888) {
            src.copy(Bitmap.Config.ARGB_8888, false)
        } else src

        return if (base.width == inputSize && base.height == inputSize) {
            base
        } else {
            Bitmap.createScaledBitmap(base, inputSize, inputSize, true)
        }
    }
}