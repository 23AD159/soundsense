package com.example.soundalert.ui.screens

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.example.soundalert.ui.*

@Composable
fun DashboardScreen(
    onNavigateToSettings: () -> Unit,
    onNavigateToCustomSound: () -> Unit
) {
    // These states would normally be hoisted to a ViewModel monitoring the TFLite inference
    val detectedLabel by remember { mutableStateOf("WATER POURING") }
    val confidence by remember { mutableStateOf(92) }
    val priority by remember { mutableStateOf("NORMAL") } // NORMAL, IMPORTANT, EMERGENCY

    var bgColor = DarkGray
    var borderColor = NormalGreen

    when(priority) {
        "EMERGENCY" -> {
            bgColor = EmergencyRed.copy(alpha = 0.1f)
            borderColor = EmergencyRed
        }
        "IMPORTANT" -> {
            bgColor = ImportantYellow.copy(alpha = 0.1f)
            borderColor = ImportantYellow
        }
        "NORMAL" -> {
            bgColor = NormalGreen.copy(alpha = 0.05f)
            borderColor = NormalGreen
        }
    }

    Scaffold(
        backgroundColor = DarkGray,
        topBar = {
            TopAppBar(
                title = { Text("Sound Dashboard", color = Color.White) },
                backgroundColor = CardGray,
                elevation = 0.dp
            )
        }
    ) { padding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(padding)
                .padding(16.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            
            // Fake animated waveform area
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .height(150.dp)
                    .background(CardGray, RoundedCornerShape(16.dp)),
                contentAlignment = Alignment.Center
            ) {
                Text("Waveform Canvas (Listening...)", color = Color.Gray)
            }

            Spacer(modifier = Modifier.height(32.dp))

            // Main Detection Card
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .border(2.dp, borderColor, RoundedCornerShape(16.dp))
                    .background(bgColor, RoundedCornerShape(16.dp))
                    .padding(24.dp),
                contentAlignment = Alignment.Center
            ) {
                Column(horizontalAlignment = Alignment.CenterHorizontally) {
                    Text(
                        text = "🔊 CURRENT EVENT",
                        color = Color.Gray,
                        fontSize = 12.sp,
                        fontWeight = FontWeight.Bold
                    )
                    Spacer(modifier = Modifier.height(12.dp))
                    Text(
                        text = detectedLabel,
                        color = Color.White,
                        fontSize = 32.sp,
                        fontWeight = FontWeight.ExtraBold
                    )
                    Spacer(modifier = Modifier.height(8.dp))
                    Text(
                        text = "Confidence: $confidence%",
                        color = borderColor,
                        fontSize = 16.sp
                    )
                }
            }

            Spacer(modifier = Modifier.weight(1f))

            // Navigation Buttons (Temporary for testing scaffolding)
            Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceEvenly) {
                OutlinedButton(
                    onClick = onNavigateToCustomSound,
                    colors = ButtonDefaults.outlinedButtonColors(backgroundColor = CardGray)
                ) {
                    Text("Add Custom Sound", color = Color.White)
                }
                OutlinedButton(
                    onClick = onNavigateToSettings,
                    colors = ButtonDefaults.outlinedButtonColors(backgroundColor = CardGray)
                ) {
                    Text("Settings", color = Color.White)
                }
            }
        }
    }
}

@androidx.compose.ui.tooling.preview.Preview(showBackground = true, backgroundColor = 0xFF121212)
@Composable
fun DashboardScreenPreview() {
    DashboardScreen(
        onNavigateToSettings = {},
        onNavigateToCustomSound = {}
    )
}
