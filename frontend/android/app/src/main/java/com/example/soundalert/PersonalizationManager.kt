package com.example.soundalert

import android.content.Context
import java.io.File
import java.io.FileOutputStream

class PersonalizationManager(private val context: Context) {

    private val templatesDir = File(context.filesDir, "templates")

    init {
        if (!templatesDir.exists()) templatesDir.mkdirs()
    }

    /**
     * Saves a recorded spectrogram as a template for a custom sound.
     */
    fun saveTemplate(label: String, spectrogram: FloatArray) {
        val file = File(templatesDir, "$label.bin")
        val buffer = java.nio.ByteBuffer.allocate(spectrogram.size * 4)
        for (f in spectrogram) buffer.putFloat(f)
        
        FileOutputStream(file).use { it.write(buffer.array()) }
    }

    /**
     * Loads all saved templates.
     */
    fun loadTemplates(): List<Pair<String, FloatArray>> {
        val templates = mutableListOf<Pair<String, FloatArray>>()
        templatesDir.listFiles()?.forEach { file ->
            val label = file.nameWithoutExtension
            val bytes = file.readBytes()
            val floatArray = FloatArray(bytes.size / 4)
            val buffer = java.nio.ByteBuffer.wrap(bytes)
            for (i in floatArray.indices) floatArray[i] = buffer.getFloat()
            templates.add(label to floatArray)
        }
        return templates
    }
}
