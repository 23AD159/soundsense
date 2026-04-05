package com.example.soundalert

import android.content.Context
import android.content.res.AssetFileDescriptor
import org.tensorflow.lite.Interpreter
import java.io.BufferedReader
import java.io.FileInputStream
import java.io.InputStreamReader
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel

class SoundClassifier(context: Context) {

    private var interpreter: Interpreter? = null
    private val modelFile = "sound_classifier.tflite"
    private val labelFile = "labels.txt"
    private val labels = mutableListOf<String>()

    init {
        try {
            interpreter = Interpreter(loadModelFile(context))
            loadLabels(context)
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }

    private fun loadModelFile(context: Context): ByteBuffer {
        val fileDescriptor: AssetFileDescriptor = context.assets.openFd(modelFile)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel: FileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    private fun loadLabels(context: Context) {
        val reader = BufferedReader(InputStreamReader(context.assets.open(labelFile)))
        var line: String?
        while (reader.readLine().also { line = it } != null) {
            labels.add(line!!)
        }
        reader.close()
    }

    data class Recognition(
        val label: String, 
        val confidence: Float,
        var direction: LocalizationEngine.Direction = LocalizationEngine.Direction.UNKNOWN,
        var priority: Int = 0 
    )

    fun classify(spectrogramData: FloatArray): Recognition {
        // Input: [1, 64, 94, 1] for 3s audio
        // FloatArray size is 64 * 94 = 6016
        val inputBuffer = ByteBuffer.allocateDirect(spectrogramData.size * 4)
        inputBuffer.order(ByteOrder.nativeOrder())
        for (value in spectrogramData) {
            inputBuffer.putFloat(value)
        }
        inputBuffer.rewind()

        val output = Array(1) { FloatArray(labels.size) }
        interpreter?.run(inputBuffer, output)
        
        // Find max
        val probabilities = output[0]
        var maxIdx = -1
        var maxProb = 0f
        for (i in probabilities.indices) {
            if (probabilities[i] > maxProb) {
                maxProb = probabilities[i]
                maxIdx = i
            }
        }

        return if (maxIdx != -1) {
            Recognition(labels[maxIdx], maxProb)
        } else {
            Recognition("Unknown", 0f)
        }
    }
    
    fun close() {
        interpreter?.close()
    }
}
