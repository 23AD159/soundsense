package com.example.soundalert

import kotlin.math.abs

class LocalizationEngine {

    enum class Direction {
        LEFT, CENTER, RIGHT, UNKNOWN
    }

    /**
     * Estimates sound direction using a simplified GCC-PHAT (TDOA estimation).
     */
    fun localize(left: FloatArray, right: FloatArray): Direction {
        // 1. Energy check (Quick filter for silence)
        val leftEnergy = left.sumOf { (it * it).toDouble() }.toFloat()
        val rightEnergy = right.sumOf { (it * it).toDouble() }.toFloat()
        if (leftEnergy + rightEnergy < 0.001f) return Direction.UNKNOWN

        // 2. Cross-Correlation (Simplified for TDOA)
        // We look for the 'lag' that maximizes similarity
        val maxLag = 20 // Samples (~1.25ms lag for standard phone mic spacing)
        var maxCorr = -1.0f
        var bestLag = 0

        for (lag in -maxLag..maxLag) {
            var corr = 0.0f
            for (i in maxLag until left.size - maxLag) {
                corr += left[i] * right[i + lag]
            }
            if (corr > maxCorr) {
                maxCorr = corr
                bestLag = lag
            }
        }

        // 3. Map Lag to Direction
        // If sound reaches LEFT mic first, bestLag will be positive/negative depending on index
        return when {
            bestLag > 3 -> Direction.RIGHT
            bestLag < -3 -> Direction.LEFT
            else -> Direction.CENTER
        }
    }
}
