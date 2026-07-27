package com.tracking.client

import android.Manifest
import android.os.Build
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
import com.tracking.client.ui.EdgeTestScreen
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

        val permissions = mutableListOf(
            Manifest.permission.CAMERA,
            Manifest.permission.RECORD_AUDIO,
            Manifest.permission.CALL_PHONE,
            Manifest.permission.READ_CONTACTS,
            Manifest.permission.READ_CALENDAR,
            Manifest.permission.WRITE_CALENDAR,
            // Telephony/SMS — CallBackgroundReceiver/SmsBackgroundReceiver +
            // answer_phone_call/send_sms/check_unread_sms tools. Sideloaded/
            // personal-use only, not Play Store distributed (see CLAUDE.md) —
            // the Play Store default-handler restriction on READ_SMS/
            // RECEIVE_SMS doesn't apply here.
            Manifest.permission.ANSWER_PHONE_CALLS,
            Manifest.permission.READ_PHONE_STATE,
            Manifest.permission.READ_CALL_LOG,
            Manifest.permission.RECEIVE_SMS,
            Manifest.permission.READ_SMS,
            Manifest.permission.SEND_SMS,
        )
        // LiveAssistantService's ongoing notification (API 33+ requires
        // this to actually display it; the service still runs without it).
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            permissions.add(Manifest.permission.POST_NOTIFICATIONS)
        }
        permissionLauncher.launch(permissions.toTypedArray())

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
                            onConnect = { host, port, frameIntervalMs, scanIntervalMs, recentBufferMs, avoidanceIntervalMs, beaconElevationDeg, beaconRadiusM, vadThreshold, startThreshold, apiKey, ocrApiKey, locationId, blurSharpnessThreshold, saveDebugOcrFrames, youtubeApiKey, cueVolume, geminiVoiceVolume, otherSoundVolume, useRemoteEdgeDevice, edgeDeviceHost ->
                                mainViewModel.connect(host, port, frameIntervalMs, scanIntervalMs, recentBufferMs, avoidanceIntervalMs, beaconElevationDeg, beaconRadiusM, vadThreshold, startThreshold, apiKey, ocrApiKey, locationId, blurSharpnessThreshold, saveDebugOcrFrames, youtubeApiKey, cueVolume, geminiVoiceVolume, otherSoundVolume, useRemoteEdgeDevice, edgeDeviceHost)
                            },
                            onBack = { navController.popBackStack() },
                            onOpenEdgeTest = { navController.navigate("edge_test") },
                        )
                    }
                    composable("edge_test") {
                        EdgeTestScreen(onBack = { navController.popBackStack() })
                    }
                }
            }
        }
    }
}
