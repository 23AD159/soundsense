package com.example.soundalert

import android.net.Uri
import android.os.Bundle
import android.widget.VideoView
import androidx.appcompat.app.AppCompatActivity
import com.example.soundalert.databinding.ActivityVideoBinding

class VideoActivity : AppCompatActivity() {

    private lateinit var binding: ActivityVideoBinding

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityVideoBinding.inflate(layoutInflater)
        setContentView(binding.root)

        val soundLabel = intent.getStringExtra("SOUND_LABEL") ?: "unknown"
        binding.tvVideoTitle.text = "ISL Signal: $soundLabel"

        setupVideoPlayback(soundLabel)

        binding.btnCloseVideo.setOnClickListener {
            finish()
        }
    }

    private fun setupVideoPlayback(label: String) {
        val videoView: VideoView = binding.videoView
        
        // Define potential video name (e.g., "isl_pressure_cooker")
        val videoName = "isl_" + label.lowercase()
        val videoResId = resources.getIdentifier(videoName, "raw", packageName)

        if (videoResId != 0) {
            try {
                val path = "android.resource://" + packageName + "/" + videoResId
                videoView.setVideoURI(Uri.parse(path))
                videoView.setOnPreparedListener { mp ->
                    mp.isLooping = true
                    videoView.start()
                }
            } catch (e: Exception) {
                e.printStackTrace()
            }
        } else {
            // Handle missing video - maybe show a placeholder image or toast
            binding.tvVideoTitle.text = "ISL Signal: $label (Video missing)"
        }
    }
}
