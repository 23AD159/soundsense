package com.example.soundalert

import android.content.Context
import android.os.Build
import android.os.VibrationEffect
import android.os.Vibrator
import android.os.VibratorManager

class AlertManager(private val context: Context) {

    private val vibrator: Vibrator = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
        val vibratorManager = context.getSystemService(Context.VIBRATOR_MANAGER_SERVICE) as VibratorManager
        vibratorManager.defaultVibrator
    } else {
        context.getSystemService(Context.VIBRATOR_SERVICE) as Vibrator
    }

    // Priority mapping for Indian Sound Classes
    private val priorityMap = mapOf(
        "fire_alarm" to 3,
        "lpg_leak" to 3,
        "baby_crying" to 2,
        "indian_doorbell_electric" to 2,
        "traditional_doorbell" to 2,
        "glass_breaking" to 2,
        "knocking" to 2,
        "pressure_cooker" to 1,
        "auto_rickshaw_horn" to 1,
        "motorcycle_horn" to 1,
        "phone_ringing" to 1,
        "mixer_grinder" to 1,
        "cooker_whistle" to 1,
        "temple_bells" to 1,
        "name_calling" to 1,
        "clapping" to 1,
        "footsteps" to 1,
        "washing_machine" to 1,
        "tv_sounds" to 0
    )

    /**
     * Triggers the alert based on sound recognition results.
     */
    fun triggerAlert(recognition: SoundClassifier.Recognition) {
        val priority = priorityMap[recognition.label] ?: 1
        recognition.priority = priority

        if (priority == 0) return // Ignore neutral sounds

        showNotification(recognition)
        triggerVibration(priority)
        
        // Directional visual alert would be triggered here in UI
    }

    private fun showNotification(recognition: SoundClassifier.Recognition) {
        // Implementation for system notification with ISL video link
        // This would typically use NotificationCompat.Builder
    }

    private fun triggerVibration(priority: Int) {
        val pattern = when (priority) {
            3 -> longArrayOf(0, 100, 100, 100, 100, 100, 300, 300, 100, 300, 100, 300, 100) // SOS
            2 -> longArrayOf(0, 200, 100, 200) // Double Pulse
            else -> longArrayOf(0, 500) // Single Pulse
        }

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            vibrator.vibrate(VibrationEffect.createWaveform(pattern, -1))
        } else {
            vibrator.vibrate(pattern, -1)
        }
    }
}
