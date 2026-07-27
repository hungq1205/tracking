plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.tracking.edgemock"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.tracking.edgemock"
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

dependencies {
    implementation("androidx.core:core-ktx:1.13.1")
    implementation("androidx.appcompat:appcompat:1.7.0")
    implementation("com.google.android.material:material:1.12.0")

    // Pure-Java ZeroMQ (no NDK/native libzmq needed) — binds the same
    // PUSH/PULL ports a real Pi Zero 2 edge device would, so the main
    // tracking app's Edge ZMQ Test screen can connect to this app exactly
    // as it would to real hardware. See MainActivity.kt.
    implementation("org.zeromq:jeromq:0.6.0")
}
