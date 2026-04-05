package com.example.soundalert

import android.annotation.SuppressLint
import android.content.Context
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import java.nio.ByteBuffer
import java.nio.ByteOrder

class AudioProcessor(
    private val context: Context,
    private val onSoundDetected: (SoundClassifier.Recognition) -> Unit
) {

    private val sampleRate = 16000
    private val channelConfig = AudioFormat.CHANNEL_IN_STEREO
    private val audioFormat = AudioFormat.ENCODING_PCM_16BIT
    private val bufferSize = AudioRecord.getMinBufferSize(sampleRate, channelConfig, audioFormat)
    
    private var audioRecord: AudioRecord? = null
    private var isRecording = false
    private var recordingThread: Thread? = null

    private val featureExtractor = FeatureExtractor(context)
    private val soundClassifier = SoundClassifier(context)
    private val localizationEngine = LocalizationEngine()
    private val personalizationManager = PersonalizationManager(context)
    private val siameseClassifier = SiameseClassifier(context)

    @SuppressLint("MissingPermission")
    fun startProcessing() {
        if (isRecording) return

        try {
            audioRecord = AudioRecord(
                MediaRecorder.AudioSource.MIC,
                sampleRate,
                channelConfig,
                audioFormat,
                bufferSize
            )

            audioRecord?.startRecording()
            isRecording = true

            recordingThread = Thread {
                processAudioLoop()
            }
            recordingThread?.start()

        } catch (e: Exception) {
            e.printStackTrace()
        }
    }

    private fun processAudioLoop() {
        // Model expects 3 seconds of audio
        // For stereo, we read 2 channels * 48000 samples = 96,000 shorts
        val stereoAudioBuffer = ShortArray(96000)
        var shortsRead = 0
        
        while (isRecording) {
            val chunk = ShortArray(3200) // 100ms stereo chunk (1600 samples * 2 channels)
            val readResult = audioRecord?.read(chunk, 0, chunk.size) ?: 0
            
            if (readResult > 0) {
                if (shortsRead + readResult <= stereoAudioBuffer.size) {
                    System.arraycopy(chunk, 0, stereoAudioBuffer, shortsRead, readResult)
                    shortsRead += readResult
                }
                
                if (shortsRead >= stereoAudioBuffer.size) {
                    // 1. Separate Channels for Localization
                    val leftChannel = FloatArray(48000)
                    val rightChannel = FloatArray(48000)
                    
                    for (i in 0 until 48000) {
                        leftChannel[i] = stereoAudioBuffer[i * 2] / 32768.0f
                        rightChannel[i] = stereoAudioBuffer[i * 2 + 1] / 32768.0f
                    }
                    
                    // 2. Localize Direction
                    val direction = localizationEngine.localize(leftChannel, rightChannel)
                    
                    // 3. Extract Spectrogram (using mono mix for classification)
                    val monoMix = FloatArray(48000) { i -> (leftChannel[i] + rightChannel[i]) / 2.0f }
                    val spectrogram = featureExtractor.extractLogMelSpectrogram(monoMix)
                    
                    // 4. Classify using Base Model
                    var recognition = soundClassifier.classify(spectrogram)
                    
                    // 5. Personalization check: If base classifier is weak, check custom templates
                    if (recognition.confidence < 0.6f) {
                        val templates = personalizationManager.loadTemplates()
                        var bestCustomLabel = ""
                        var minDistance = 100f // Lower is better in Siamese

                        for ((customLabel, templateSpec) in templates) {
                            val dist = siameseClassifier.calculateDistance(spectrogram, templateSpec)
                            if (dist < minDistance && dist < 0.5f) { // Threshold for match
                                minDistance = dist
                                bestCustomLabel = customLabel
                            }
                        }

                        if (bestCustomLabel.isNotEmpty()) {
                            // Map distance to pseudo-confidence (0.5 threshold)
                            val customConfidence = 1.0f - (minDistance / 0.5f) * 0.4f
                            recognition = SoundClassifier.Recognition("User: $bestCustomLabel", customConfidence)
                        }
                    }

                    // 6. Update recognition with direction info
                    recognition.direction = direction
                    
                    if (recognition.confidence > 0.15f) {
                        onSoundDetected(recognition)
                    }
                    
                    shortsRead = 0 
                }
            }
        }
    }

    fun stopProcessing() {
        isRecording = false
        try {
            audioRecord?.stop()
            audioRecord?.release()
        } catch (e: Exception) {
            e.printStackTrace()
        }
        audioRecord = null
        soundClassifier.close()
        siameseClassifier.close()
    }
}
