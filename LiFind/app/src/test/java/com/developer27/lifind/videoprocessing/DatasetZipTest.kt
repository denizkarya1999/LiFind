package com.developer27.lifind.videoprocessing

import org.junit.Assert.*
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.io.File
import java.util.zip.ZipEntry
import java.util.zip.ZipOutputStream

class DatasetZipTest {
    @get:Rule val temporary = TemporaryFolder()

    @Test fun extractsRegularDatasetEntry() {
        val zip = temporary.newFile("data.zip")
        ZipOutputStream(zip.outputStream()).use {
            it.putNextEntry(ZipEntry("labels/frame.txt"))
            it.write("0 0.5 0.5 0.1 0.1".toByteArray())
            it.closeEntry()
        }
        val destination = temporary.newFolder("unzipped")
        SingleModelEvaluator.unzip(zip, destination)
        assertTrue(File(destination, "labels/frame.txt").exists())
    }
    @Test fun rejectsTraversalIntoDirectoryWithSamePrefix() {
        val zip = temporary.newFile("data.zip")
        ZipOutputStream(zip.outputStream()).use {
            it.putNextEntry(ZipEntry("../unzipped-sibling/escaped.txt"))
            it.write("unsafe".toByteArray())
            it.closeEntry()
        }
        val destination = temporary.newFolder("unzipped")
        assertThrows(IllegalArgumentException::class.java) {
            SingleModelEvaluator.unzip(zip, destination)
        }
        assertFalse(File(temporary.root, "unzipped-sibling/escaped.txt").exists())
    }
}
