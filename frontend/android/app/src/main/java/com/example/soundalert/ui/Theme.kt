package com.example.soundalert.ui

import androidx.compose.material.MaterialTheme
import androidx.compose.material.darkColors
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color

val DarkGray = Color(0xFF121212)
val CardGray = Color(0xFF1E1E1E)
val EmergencyRed = Color(0xFFFF3B30)
val ImportantYellow = Color(0xFFFF9500)
val NormalGreen = Color(0xFF34C759)

private val DarkColorPalette = darkColors(
    primary = Color(0xFF4F8EF7),
    primaryVariant = Color(0xFF3700B3),
    secondary = Color(0xFF03DAC5),
    background = DarkGray,
    surface = CardGray,
    onPrimary = Color.White,
    onSecondary = Color.Black,
    onBackground = Color.White,
    onSurface = Color.White,
    error = EmergencyRed
)

@Composable
fun DeafAlertTheme(content: @Composable () -> Unit) {
    MaterialTheme(
        colors = DarkColorPalette,
        typography = androidx.compose.material.Typography(),
        shapes = androidx.compose.material.Shapes(),
        content = content
    )
}
