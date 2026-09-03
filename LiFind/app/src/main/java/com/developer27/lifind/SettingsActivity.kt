package com.developer27.lifind

import android.content.Intent
import android.os.Bundle
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.preference.EditTextPreference
import androidx.preference.PreferenceFragmentCompat
import androidx.preference.SwitchPreference

class SettingsActivity : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.settings_activity)

        supportFragmentManager
            .beginTransaction()
            .replace(R.id.settings_container, SettingsFragment())
            .commit()
    }

    class SettingsFragment : PreferenceFragmentCompat() {
        override fun onCreatePreferences(savedInstanceState: Bundle?, rootKey: String?) {
            setPreferencesFromResource(R.xml.root_preferences, rootKey)
            val manualIsoPref = findPreference<SwitchPreference>("manual_iso_enabled")

            val isoPref = findPreference<EditTextPreference>("iso_value")
            isoPref?.setOnBindEditTextListener { edit ->
                // numeric only from XML; optionally add min/max hints
                edit.hint = "100–6400"
                edit.inputType = android.text.InputType.TYPE_CLASS_NUMBER
            }

            isoPref?.setOnPreferenceChangeListener { _, newValue ->
                val entered = (newValue as? String)?.trim().orEmpty()
                val iso = entered.toIntOrNull()
                val clamped = when {
                    iso == null -> null
                    iso < 50 -> 50               // soft min (safer for low light)
                    iso > 25600 -> 25600         // soft max; actual max will be clamped by camera
                    else -> iso
                }
                if (clamped == null) {
                    Toast.makeText(context, "Please enter a valid ISO number.", Toast.LENGTH_SHORT).show()
                    false
                } else {
                    // Store the (soft) clamped value
                    isoPref.text = clamped.toString()
                    Toast.makeText(context, "ISO set to $clamped", Toast.LENGTH_SHORT).show()
                    false // The normalized value was already persisted above.
                }
            }
        }
    }

    override fun onBackPressed() {
        super.onBackPressed()
        // Save settings and go back to MainActivity with result
        setResult(RESULT_OK, Intent())
        finish()
    }
}
