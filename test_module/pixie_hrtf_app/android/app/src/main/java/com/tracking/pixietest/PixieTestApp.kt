package com.tracking.pixietest

import android.app.Application
import android.util.Log
import org.opencv.android.OpenCVLoader

/**
 * Loads OpenCV's native library once, at process start — RotationTracker's
 * `matcher` field calls BFMatcher.create() eagerly (not lazily), so without
 * this it crashes with UnsatisfiedLinkError the instant MainActivity is
 * constructed (BFMatcher.create is a native call, and nothing else in this
 * app touches OpenCV before RotationTracker does). Direct copy of the main
 * client app's TrackingApp.kt, which exists for the exact same reason.
 */
class PixieTestApp : Application() {
    override fun onCreate() {
        super.onCreate()
        if (!OpenCVLoader.initDebug()) {
            Log.e(TAG, "OpenCV initialization failed")
        } else {
            Log.i(TAG, "OpenCV initialized successfully")
        }
    }

    companion object {
        private const val TAG = "PixieTestApp"
    }
}
