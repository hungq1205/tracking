package com.tracking.client.ui

import android.graphics.BitmapFactory
import androidx.camera.view.PreviewView
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import kotlinx.coroutines.delay

@Composable
fun MainScreen(
    viewModel: MainViewModel,
    onOpenSettings: () -> Unit,
) {
    val isRemoteEdgeActive by viewModel.isRemoteEdgeActive.collectAsState()
    val edgeFrame by viewModel.edgeFrame.collectAsState()
    val ocrFrame by viewModel.ocrFrame.collectAsState()

    // Frame capture runs continuously against LiveAssistantService's own
    // lifecycle regardless (see CameraManager.bind()) — this only plugs/
    // unplugs the on-screen preview surface while this screen is actually
    // visible, per CLAUDE.md's "Client-Orchestrated Live Session" note on
    // the Preview/ImageAnalysis binding split.
    DisposableEffect(Unit) {
        onDispose { viewModel.detachCameraPreview() }
    }

    Box(modifier = Modifier.fillMaxSize()) {
        // Camera preview — the phone's own CameraX feed normally, or
        // whatever a remote edge device's camera last sent when "Use Remote
        // Edge Device" is active (edgeFrame is only ever non-null in that
        // mode — see LiveAssistantService.edgeFrame). Keeping the
        // PreviewView mounted underneath (rather than swapping composables)
        // means attachCameraPreview()'s Preview use case stays bound
        // regardless, matching CameraManager's "bind once, attach/detach
        // just the surface" design — only the remote-edge Image is drawn
        // over it.
        AndroidView(
            factory = { ctx ->
                PreviewView(ctx).also { previewView ->
                    viewModel.attachCameraPreview(previewView)
                }
            },
            modifier = Modifier.fillMaxSize()
        )
        if (isRemoteEdgeActive) {
            val bitmap = remember(edgeFrame) {
                edgeFrame?.let { bytes -> BitmapFactory.decodeByteArray(bytes, 0, bytes.size) }
            }
            if (bitmap != null) {
                // ContentScale.Fit (not Crop) — the edge device's frame_out
                // is now sent at the camera SENSOR's own native aspect
                // ratio (e.g. 4608x2592, 16:9-ish for the IMX708 — see
                // client/pi_edge/main.py's camera_capture_thread), which
                // essentially never matches the phone's own screen aspect.
                // Crop would silently cut off part of the frame to fill the
                // screen; Fit letterboxes instead, so the WHOLE frame is
                // always visible, matching the "downscale, never crop"
                // intent this feature was built around from the start.
                Image(
                    bitmap = bitmap.asImageBitmap(),
                    contentDescription = "Remote edge device camera",
                    contentScale = ContentScale.Fit,
                    modifier = Modifier.fillMaxSize(),
                )
            }
        }

        // OCR scan preview — whatever frame was just sent to OCR.space
        // (scan_current_view()/live-reading), REPLACING the normal preview
        // full-screen for a few seconds after each scan (same "cover the
        // live preview" precedent as the remote-edge-device Image above),
        // so a sighted companion looking at the screen can see exactly
        // what got scanned at full resolution — full sensor res/quality,
        // see CameraManager.captureFullResFrame(), not the downscaled
        // 640px/quality-50 JPEG the live preview otherwise implies. Then
        // auto-reverts back to the live camera feed once the scan is stale.
        var showOcrPreview by remember { mutableStateOf(false) }
        LaunchedEffect(ocrFrame) {
            if (ocrFrame != null) {
                showOcrPreview = true
                delay(4000)
                showOcrPreview = false
            }
        }
        if (showOcrPreview && ocrFrame != null) {
            val ocrBitmap = remember(ocrFrame) {
                ocrFrame?.let { bytes -> BitmapFactory.decodeByteArray(bytes, 0, bytes.size) }
            }
            if (ocrBitmap != null) {
                Image(
                    bitmap = ocrBitmap.asImageBitmap(),
                    contentDescription = "Last OCR scan (full resolution)",
                    contentScale = ContentScale.Fit,
                    modifier = Modifier.fillMaxSize(),
                )
            }
        }

        // Settings button — top-right
        Row(
            modifier = Modifier
                .align(Alignment.TopEnd)
                .padding(end = 4.dp, top = 4.dp)
        ) {
            IconButton(onClick = onOpenSettings) {
                Icon(Icons.Default.Settings, contentDescription = "Settings", tint = Color.White)
            }
        }
    }
}
