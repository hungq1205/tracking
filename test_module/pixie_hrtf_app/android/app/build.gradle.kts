plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.tracking.pixietest"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.tracking.pixietest"
        minSdk = 26
        targetSdk = 35
        versionCode = 1
        versionName = "1.0"
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions {
        jvmTarget = "17"
    }
}

// Same dependency versions as client/android/app/build.gradle.kts, for the
// same reasons documented there — this app duplicates RotationTracker.kt/
// HrtfConvolver.kt/HrtfBeaconPlayer.kt from that project (a standalone test
// harness, not a shared module — see this app's README) and needs the same
// OpenCV/CameraX builds those files were written against.
val cameraxVersion = "1.3.4"
val opencvVersion = "4.11.0"

dependencies {
    implementation("androidx.core:core-ktx:1.13.1")
    implementation("androidx.appcompat:appcompat:1.7.0")
    implementation("com.google.android.material:material:1.12.0")

    implementation("androidx.camera:camera-core:$cameraxVersion")
    implementation("androidx.camera:camera-camera2:$cameraxVersion")
    implementation("androidx.camera:camera-lifecycle:$cameraxVersion")
    implementation("androidx.camera:camera-view:$cameraxVersion")

    // RotationTracker.kt's Essential-matrix RANSAC drift detector.
    implementation("org.opencv:opencv:$opencvVersion")

    // Raw WebSocket client to pixie_hrtf_server.py — same reason
    // GeminiLiveClient.kt in the main app uses OkHttp directly. org.json.*
    // (used for the wire messages) needs no separate dependency — it's
    // part of the Android platform SDK, same as GeminiLiveClient.kt's use.
    implementation("com.squareup.okhttp3:okhttp:4.12.0")

    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.8.1")
}
