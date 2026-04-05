package com.example.soundalert

import android.content.Context
import android.os.Build
import android.os.VibrationEffect
import android.os.Vibrator
import android.os.VibratorManager

class VibrationHelper(context: Context) {

    private val vibrator: Vibrator = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
        val vibratorManager = context.getSystemService(Context.VIBRATOR_MANAGER_SERVICE) as VibratorManager
        vibratorManager.defaultVibrator
    } else {
        @Suppress("DEPRECATION")
        context.getSystemService(Context.VIBRATOR_SERVICE) as Vibrator
    }

    fun triggerVibration(soundClass: String) {
        if (!vibrator.hasVibrator()) return

        // Cancel previous vibration if any
        vibrator.cancel()

        val pattern = getPatternForSound(soundClass)
        
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val effect = VibrationEffect.createWaveform(pattern, -1)
            vibrator.vibrate(effect)
        } else {
            @Suppress("DEPRECATION")
            vibrator.vibrate(pattern, -1)
        }
    }

    fun stopVibration() {
        vibrator.cancel()
    }

    private fun getPatternForSound(soundClass: String): LongArray {
        return when (soundClass.lowercase()) {
            "fire_alarm", "siren", "smoke_detector" -> 
                longArrayOf(0, 100, 50, 100, 50, 100, 50, 400) // SOS / Rapid Danger
            
            "door_knock", "doorbell" -> 
                longArrayOf(0, 300, 200, 300) // Double Pulse
            
            "crying_baby" -> 
                longArrayOf(0, 500, 200, 500, 200, 500) // Slow rhythmic pulse
            
            "glass_breaking", "fireworks" -> 
                longArrayOf(0, 50, 50, 50, 50, 50, 200) // Sudden sharp bursts
            
            "car_horn", "train", "airplane", "helicopter" -> 
                longArrayOf(0, 800) // Continuous long warning
            
            "dog", "clock_tick", "radio" -> 
                longArrayOf(0, 150) // Short casual notification
                
            else -> longArrayOf(0, 100) // Default minor blip
        }
    }
}
