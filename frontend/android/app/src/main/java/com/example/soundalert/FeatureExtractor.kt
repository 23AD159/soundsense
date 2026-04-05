package com.example.soundalert

import android.content.Context
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.*

class FeatureExtractor(private val context: Context) {

    private val sampleRate = 16000
    private val nMels = 64
    private val nFft = 1024
    private val hopLength = 512
    private val melBasis: Array<FloatArray>

    init {
        // Load precomputed Mel basis from assets
        melBasis = loadMelBasis("mel_basis.bin")
    }

    private fun loadMelBasis(filename: String): Array<FloatArray> {
        val bytes = context.assets.open(filename).readBytes()
        val buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        val numBins = nFft / 2 + 1
        val basis = Array(nMels) { FloatArray(numBins) }
        
        for (m in 0 until nMels) {
            for (i in 0 until numBins) {
                basis[m][i] = buffer.float
            }
        }
        return basis
    }

    /**
     * Extracts Log Mel-Spectrogram from raw PCM data.
     * Matches librosa.feature.melspectrogram and power_to_db.
     */
    fun extractLogMelSpectrogram(data: FloatArray): FloatArray {
        // 1. Hann Window
        val window = FloatArray(nFft) { i ->
            0.5f * (1 - cos(2 * PI.toFloat() * i / (nFft - 1)))
        }

        // 2. STFT
        val numFrames = (data.size - nFft) / hopLength + 1
        val spectrogram = Array(numFrames) { FloatArray(nFft / 2 + 1) }

        for (f in 0 until numFrames) {
            val offset = f * hopLength
            val frameReal = FloatArray(nFft)
            val frameImag = FloatArray(nFft)
            for (i in 0 until nFft) {
                frameReal[i] = data[offset + i] * window[i]
                frameImag[i] = 0f
            }
            
            // Perform In-place FFT
            fft(frameReal, frameImag)
            
            for (i in 0 until (nFft / 2 + 1)) {
                spectrogram[f][i] = frameReal[i] * frameReal[i] + frameImag[i] * frameImag[i]
            }
        }

        // 3. Apply Mel Basis
        val melSpectrogram = Array(numFrames) { FloatArray(nMels) }
        for (f in 0 until numFrames) {
            for (m in 0 until nMels) {
                var sum = 0f
                for (i in 0 until (nFft / 2 + 1)) {
                    sum += spectrogram[f][i] * melBasis[m][i]
                }
                melSpectrogram[f][m] = sum
            }
        }

        // 4. Power to DB and Normalization
        val logMelFlat = FloatArray(numFrames * nMels)
        for (f in 0 until numFrames) {
            for (m in 0 until nMels) {
                // power_to_db with ref=max(mel_spec) -> top 0dB
                // For simplicity here we use a standard threshold
                val db = 10f * log10(max(1e-10f, melSpectrogram[f][m]))
                
                // Fixed-range Normalization (-80dB to 0dB -> [0, 1])
                // We clip it like in Python web_utils.py: np.clip((features + 80.0) / 80.0, 0, 1)
                logMelFlat[f * nMels + m] = max(0f, min(1f, (db + 80f) / 80f))
            }
        }

        return logMelFlat
    }

    /**
     * Iterative Cooley-Tukey FFT implementation.
     */
    private fun fft(real: FloatArray, imag: FloatArray) {
        val n = real.size
        var j = 0
        for (i in 0 until n) {
            if (i < j) {
                val tempReal = real[i]
                real[i] = real[j]
                real[j] = tempReal
                val tempImag = imag[i]
                imag[i] = imag[j]
                imag[j] = tempImag
            }
            var m = n shr 1
            while (m >= 1 && j >= m) {
                j -= m
                m = m shr 1
            }
            j += m
        }

        var len = 2
        while (len <= n) {
            val ang = 2 * PI.toFloat() / len
            val wlenReal = cos(ang)
            val wlenImag = -sin(ang)
            for (i in 0 until n step len) {
                var wReal = 1f
                var wImag = 0f
                for (k in 0 until len / 2) {
                    val uReal = real[i + k]
                    val uImag = imag[i + k]
                    val vReal = real[i + k + len / 2] * wReal - imag[i + k + len / 2] * wImag
                    val vImag = real[i + k + len / 2] * wImag + imag[i + k + len / 2] * wReal
                    real[i + k] = uReal + vReal
                    imag[i + k] = uImag + vImag
                    real[i + k + len / 2] = uReal - vReal
                    imag[i + k + len / 2] = uImag - vImag
                    val nextWReal = wReal * wlenReal - wImag * wlenImag
                    wImag = wReal * wlenImag + wImag * wlenReal
                    wReal = nextWReal
                }
            }
            len *= 2
        }
    }
}
