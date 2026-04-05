package com.example.soundalert

import android.content.Context
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel

class SiameseClassifier(context: Context) {

    private var interpreter: Interpreter? = null
    private val modelFile = "siamese_model.tflite"

    init {
        try {
            val fileDescriptor = context.assets.openFd(modelFile)
            val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
            val fileChannel = inputStream.channel
            val startOffset = fileDescriptor.startOffset
            val declaredLength = fileDescriptor.declaredLength
            val buffer = fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
            interpreter = Interpreter(buffer)
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }

    /**
     * Returns a similarity distance (lower is more similar).
     */
    fun calculateDistance(inputA: FloatArray, inputB: FloatArray): Float {
        // Siamese model expects two inputs [1, 64, 94, 1]
        val bufferA = toByteBuffer(inputA)
        val bufferB = toByteBuffer(inputB)
        
        val inputs = arrayOf<Any>(bufferA, bufferB)
        val outputs = mutableMapOf<Int, Any>()
        val outputDist = Array(1) { FloatArray(1) }
        outputs[0] = outputDist
        
        interpreter?.runForMultipleInputsOutputs(inputs, outputs)
        
        return outputDist[0][0]
    }

    private fun toByteBuffer(data: FloatArray): ByteBuffer {
        val buffer = ByteBuffer.allocateDirect(data.size * 4)
        buffer.order(ByteOrder.nativeOrder())
        for (v in data) buffer.putFloat(v)
        buffer.rewind()
        return buffer
    }

    fun close() {
        interpreter?.close()
    }
}
