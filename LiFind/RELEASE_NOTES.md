# LiFind 1.0.1

- Normalize live RGB input to 0–1, matching dataset inference, and parse all 33 distance classes.
- Recognize distance output layouts with fewer detections than channels.
- Serialize shared model inference and frame buffers; preserve already displayed frames.
- Guard camera opening and preview callbacks across switching, pause, resume, and Clear.
- Stop tracking on pause without unexpectedly opening the map. Tracking requires only camera permission.
- Save the latest measurement atomically in app storage, preserving fractional centimetres. Missing detections remain unavailable.
- Use the same physical LED coordinates in the log, position solver, and map. Reject invalid distances and degenerate geometry.
- Honor configured sensor height; the app's radial-distance setup remains 228.6 cm above the LED plane.
- Persist validated ISO values and apply camera settings consistently.
- Keep dataset thresholds separate from tracking; count misclassifications and false positives correctly.
- Reject ZIP entries outside the extraction directory and clean up evaluation cache files.
- Package only the two model exports used by the app.
- Replace OpenCV 4.10 and LiteRT 1.1 with 16 KB-compatible native runtimes (OpenCV 4.14.0 and LiteRT 1.4.1); remove unused PyTorch, AR, GPU, and CameraX dependencies.

## Build and verify

Requirements: JDK 17 or 21, Android SDK 34, and the SDK/NDK dependencies requested by Gradle. Set `sdk.dir` in the ignored `local.properties` file or configure `ANDROID_HOME`.

From the `LiFind` Android project directory:

```sh
bash gradlew :app:testDebugUnitTest :app:lintDebug :app:assembleRelease
```

The installable ARMv7/ARM64 APK is `app/build/outputs/apk/release/app-release.apk`. The release build retains the repository's existing debug-keystore signing configuration; distributing updates requires the same signing key. Version: `1.0.1` (code `2`).

For x86_64 emulator tests only:

```sh
bash gradlew :app:assembleDebug :app:assembleDebugAndroidTest -PlifindTestAbi=x86_64
adb -s EMULATOR_SERIAL install -r app/build/outputs/apk/debug/app-debug.apk
adb -s EMULATOR_SERIAL install -r app/build/outputs/apk/androidTest/debug/app-debug-androidTest.apk
adb -s EMULATOR_SERIAL shell am instrument -w com.developer27.lifind.test/androidx.test.runner.AndroidJUnitRunner
```

Use an emulator with sufficient memory for Android and both models (6 GB was used for verification). Do not pass `lifindTestAbi` when building the ARM release.

Measurements used by the map are stored in the app's private `files/LiFind_Log.txt`. CSV evaluation reports retain the existing public Documents / app-specific storage fallback.
