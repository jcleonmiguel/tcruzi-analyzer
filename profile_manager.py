"""Saved parameter profiles for the analyzer (JSON in the user data folder)."""

import os
import json
from pathlib import Path
from tkinter import simpledialog, messagebox


def _get_user_data_dir() -> Path:
    """User data folder, writable even when the program is installed under
    Program Files: %APPDATA%\\TCruzi Analyzer on Windows,
    ~/.tcruzi_analyzer elsewhere.
    """
    if os.name == "nt":
        base = os.environ.get("APPDATA", str(Path.home()))
        d = Path(base) / "TCruzi Analyzer"
    else:
        d = Path.home() / ".tcruzi_analyzer"
    d.mkdir(parents=True, exist_ok=True)
    return d


class ProfileManager:
    """Load, save, rename and delete parameter profiles."""

    def __init__(self, config_file="tcruzi_profiles.json"):
        # profiles live in the user data folder, which stays writable even
        # when the .exe sits in Program Files
        self.config_file = _get_user_data_dir() / config_file
        self.profiles = {}
        self.last_used = "Recommended"
        self.load_profiles()
    
    def load_profiles(self):
        """Load profiles from the JSON file."""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.profiles = data.get('profiles', {})
                    self.last_used = data.get('last_used', 'Recommended')
            except Exception as e:
                print(f"[Profiles] could not load: {e}")
                self._create_default_profile()
        else:
            self._create_default_profile()
    
    def _create_default_profile(self):
        """Create the built-in profiles; Recommended is the default."""
        self.profiles = {
            "Recommended": {
                "protected": True,
                "blue_ch": 2,
                "red_ch": 0,
                "amas_color": "red",
                "amas_channel_indices": [0],
                "amas_ch_red": True,
                "amas_ch_green": False,
                "amas_ch_blue": False,
                "amas_ch_custom": False,
                "amas_ch_custom_idx": 3,
                "amas_detection_mode": "classic",
                "amas_cellpose_diameter": None,
                "diameter": "auto",
                "flow_threshold": 0.4,
                "cellprob_threshold": -1.0,
                "amas_min_size": 2,
                "amas_max_size": 150,
                "threshold_method": "otsu",
                "dilation_radius": 120,
                "min_amastigotes": 1,
                "avg_amastigote_size": 0,
                "use_auto_calibration": True,
                "hardware_mode": "auto"
            },
            "Default": {
                "protected": True,
                "blue_ch": 2,
                "red_ch": 0,
                "diameter": "auto",
                "flow_threshold": 0.4,
                "cellprob_threshold": -1.0,
                "amas_min_size": 2,
                "amas_max_size": 150,
                "threshold_method": "otsu",
                "dilation_radius": 100,
                "min_amastigotes": 1,
                "avg_amastigote_size": 35,
                "use_auto_calibration": True,
                "hardware_mode": "auto"
            }
        }
        self.last_used = "Recommended"
        self.save_profiles()
    
    def save_profiles(self):
        """Write profiles to the JSON file."""
        try:
            data = {
                "last_used": self.last_used,
                "profiles": self.profiles
            }
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[Profiles] could not save: {e}")
    
    def get_profile(self, name):
        """Return a profile by name (Default if it does not exist)."""
        return self.profiles.get(name, self.profiles.get("Default"))
    
    def save_profile(self, name, config):
        """Save or update a profile."""
        if not name or name.strip() == "":
            messagebox.showerror("Error", "The name cannot be empty")
            return False
        
        if name in self.profiles and name != self.last_used:
            respuesta = messagebox.askyesno(
                "Confirm",
                f"Profile '{name}' already exists.\nOverwrite it?"
            )
            if not respuesta:
                return False
        
        protected = self.profiles.get(name, {}).get("protected", False)
        config["protected"] = protected
        
        self.profiles[name] = config
        self.save_profiles()
        return True
    
    def delete_profile(self, name):
        """Delete a profile."""
        if name not in self.profiles:
            messagebox.showerror("Error", f"Profile '{name}' does not exist")
            return False
        
        if self.profiles[name].get("protected", False):
            messagebox.showwarning(
                "Protected",
                f"Profile '{name}' is protected and cannot be deleted"
            )
            return False
        
        respuesta = messagebox.askyesno(
            "Confirm",
            f"Delete profile '{name}'?\nThis cannot be undone."
        )
        
        if respuesta:
            del self.profiles[name]
            if self.last_used == name:
                self.last_used = "Default"
            self.save_profiles()
            return True
        
        return False
    
    def rename_profile(self, old_name, new_name):
        """Rename a profile."""
        if old_name not in self.profiles:
            messagebox.showerror("Error", f"Profile '{old_name}' does not exist")
            return False
        
        if self.profiles[old_name].get("protected", False):
            messagebox.showwarning(
                "Protected",
                f"Profile '{old_name}' is protected and cannot be renamed"
            )
            return False
        
        if not new_name or new_name.strip() == "":
            messagebox.showerror("Error", "The new name cannot be empty")
            return False
        
        if new_name in self.profiles:
            messagebox.showerror("Error", f"A profile called '{new_name}' already exists")
            return False
        
        self.profiles[new_name] = self.profiles.pop(old_name)
        
        if self.last_used == old_name:
            self.last_used = new_name
        
        self.save_profiles()
        return True
    
    def get_profile_names(self):
        """Return the profile names."""
        return list(self.profiles.keys())
    
    def set_last_used(self, name):
        """Remember the last profile used."""
        if name in self.profiles:
            self.last_used = name
            self.save_profiles()
    
    def is_protected(self, name):
        """True if the profile is protected."""
        return self.profiles.get(name, {}).get("protected", False)
