package com.tracking.client

import android.Manifest
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import com.tracking.client.ui.MainScreen
import com.tracking.client.ui.MainViewModel
import com.tracking.client.ui.SettingsScreen
import com.tracking.client.ui.theme.TrackingTheme

class MainActivity : ComponentActivity() {

    private val permissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { /* permissions handled; camera/mic checks happen at use site */ }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        permissionLauncher.launch(
            arrayOf(
                Manifest.permission.CAMERA,
                Manifest.permission.RECORD_AUDIO,
                Manifest.permission.CALL_PHONE,
                Manifest.permission.READ_CONTACTS,
                Manifest.permission.READ_CALENDAR,
                Manifest.permission.WRITE_CALENDAR,
            )
        )

        setContent {
            TrackingTheme {
                val navController = rememberNavController()
                val mainViewModel: MainViewModel = viewModel()

                NavHost(navController = navController, startDestination = "main") {
                    composable("main") {
                        MainScreen(
                            viewModel = mainViewModel,
                            onOpenSettings = { navController.navigate("settings") },
                        )
                    }
                    composable("settings") {
                        SettingsScreen(
                            mainViewModel = mainViewModel,
                            onConnect = { host, port, frameIntervalMs, scanIntervalMs, recentBufferMs, vadThreshold, startThreshold, apiKey, ocrUrl, locationId ->
                                mainViewModel.connect(host, port, frameIntervalMs, scanIntervalMs, recentBufferMs, vadThreshold, startThreshold, apiKey, ocrUrl, locationId)
                            },
                            onBack = { navController.popBackStack() }
                        )
                    }
                }
            }
        }
    }
}
