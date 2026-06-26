#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ANDROID_HOME="$HOME/Android/Sdk"

echo "=== Step 1: Install JDK 17 ==="
if ! java -version 2>&1 | grep -q "version \"17"; then
    sudo apt-get install -y openjdk-17-jdk
fi
export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
export PATH="$JAVA_HOME/bin:$PATH"
java -version

echo "=== Step 2: Download Android cmdline-tools ==="
if [ ! -d "$ANDROID_HOME/cmdline-tools/latest/bin" ]; then
    mkdir -p "$ANDROID_HOME/cmdline-tools"
    cd /tmp
    TOOLS_ZIP="commandlinetools-linux-11076708_latest.zip"
    if [ ! -f "$TOOLS_ZIP" ]; then
        wget -q "https://dl.google.com/android/repository/$TOOLS_ZIP"
    fi
    unzip -q "$TOOLS_ZIP" -d /tmp/cmdtools_extract
    mv /tmp/cmdtools_extract/cmdline-tools "$ANDROID_HOME/cmdline-tools/latest"
fi

export PATH="$ANDROID_HOME/cmdline-tools/latest/bin:$ANDROID_HOME/platform-tools:$PATH"

echo "=== Step 3: Accept licenses & install SDK packages ==="
yes | sdkmanager --licenses > /dev/null 2>&1 || true
sdkmanager "platform-tools" "platforms;android-35" "build-tools;35.0.0"

echo "=== Step 4: Write local.properties ==="
cd "$SCRIPT_DIR"
echo "sdk.dir=$ANDROID_HOME" > local.properties

echo "=== Step 5: Build APK ==="
export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
./gradlew assembleDebug

APK="$SCRIPT_DIR/app/build/outputs/apk/debug/app-debug.apk"
echo ""
echo "=== BUILD COMPLETE ==="
echo "APK: $APK"
echo ""
echo "To install on a connected Android device:"
echo "  adb install -r $APK"
echo ""
echo "To install and launch:"
echo "  adb install -r $APK && adb shell am start -n com.tracking.client/.MainActivity"
