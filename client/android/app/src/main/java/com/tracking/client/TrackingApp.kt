package com.tracking.client

import android.app.Application
import android.util.Log
import org.opencv.android.OpenCVLoader

class TrackingApp : Application() {
    override fun onCreate() {
        super.onCreate()
        if (!OpenCVLoader.initDebug()) {
            Log.e(TAG, "OpenCV initialization failed")
        } else {
            Log.i(TAG, "OpenCV initialized successfully")
        }
    }

    companion object {
        private const val TAG = "TrackingApp"
    }
}

