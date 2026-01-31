package com.developer27.lifind.videoprocessing

import android.graphics.Bitmap
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Converts a Bitmap into a float32 NHWC tensor for TFLite:
 *   shape: [1, H, W, 3]
 *   range: 0..1
 *
 * Notes:
 * - Assumes Bitmap is already ARGB_8888 (helpers usually ensure this).
 * - Uses native byte order for direct buffers.
 */
object BitmapToFloatTensor {

    fun nhwc(bmp: Bitmap): ByteBuffer {
        val w = bmp.width
        val h = bmp.height

        // 1 * H * W * 3 float32
        val buf = ByteBuffer
            .allocateDirect(4 * h * w * 3)
            .order(ByteOrder.nativeOrder())

        val pixels = IntArray(w * h)
        bmp.getPixels(pixels, 0, w, 0, 0, w, h)

        for (p in pixels) {
            val r = ((p shr 16) and 0xFF) / 255f
            val g = ((p shr 8) and 0xFF) / 255f
            val b = (p and 0xFF) / 255f
            buf.putFloat(r)
            buf.putFloat(g)
            buf.putFloat(b)
        }

        buf.rewind()
        return buf
    }
}