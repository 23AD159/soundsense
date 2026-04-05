package com.example.soundalert.ui.navigation

import androidx.compose.runtime.Composable
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import com.example.soundalert.ui.screens.LoginScreen
import com.example.soundalert.ui.screens.DashboardScreen

@Composable
fun AppNavigation() {
    val navController = rememberNavController()

    NavHost(navController = navController, startDestination = "login") {
        
        composable("login") {
            // Screen 9: Mock Login Authentication
            LoginScreen(
                onLoginSuccess = {
                    // Navigate to Dashboard and clear backstack
                    navController.navigate("dashboard") {
                        popUpTo("login") { inclusive = true }
                    }
                }
            )
        }

        composable("dashboard") {
            // Screen 1: Real-time Sound Alert Dashboard
            DashboardScreen(
                onNavigateToSettings = { navController.navigate("settings") },
                onNavigateToCustomSound = { navController.navigate("custom_sound") }
            )
        }
        
        // Additional Screens would be mapped here
        // composable("custom_sound") { ... }
        // composable("settings") { ... }
    }
}
