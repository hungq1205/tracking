pluginManagement {
    repositories {
        google()
        mavenCentral()
        gradlePluginPortal()
    }
}
dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        google()
        mavenCentral()
        // NewPipeExtractor (com.github.TeamNewPipe:NewPipeExtractor) is only
        // published on JitPack, not Maven Central — resolves a YouTube video
        // id to a direct playable stream URL so YouTube audio can flow
        // through PlaybackService's own ExoPlayer (already tapped for the
        // remote edge device), replacing the WebView-based IFrame player.
        maven { url = uri("https://jitpack.io") }
    }
}

rootProject.name = "TrackingClient"
include(":app")
