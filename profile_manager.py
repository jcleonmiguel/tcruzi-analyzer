"""
PROFILE MANAGER - Sistema de Gestión de Perfiles
================================================
Gestiona configuraciones guardadas del T.cruzi Analyzer
"""

import os
import json
from pathlib import Path
from tkinter import simpledialog, messagebox


def _get_user_data_dir() -> Path:
    """
    Carpeta de datos del usuario donde SÍ se puede escribir aunque el programa
    esté instalado en 'Archivos de programa'.
    Windows: %APPDATA%\\TCruzi Analyzer
    Otros:   ~/.tcruzi_analyzer
    """
    if os.name == "nt":
        base = os.environ.get("APPDATA", str(Path.home()))
        d = Path(base) / "TCruzi Analyzer"
    else:
        d = Path.home() / ".tcruzi_analyzer"
    d.mkdir(parents=True, exist_ok=True)
    return d


class ProfileManager:
    """Gestor de perfiles de configuración"""

    def __init__(self, config_file="tcruzi_profiles.json"):
        # Los perfiles viven en la carpeta de datos del usuario (escribible
        # aunque el .exe esté en 'Archivos de programa').
        self.config_file = _get_user_data_dir() / config_file
        self.profiles = {}
        self.last_used = "Recomendado"
        self.load_profiles()
    
    def load_profiles(self):
        """Carga perfiles desde archivo JSON"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.profiles = data.get('profiles', {})
                    self.last_used = data.get('last_used', 'Recomendado')
            except Exception as e:
                print(f"[Perfiles] Error al cargar: {e}")
                self._create_default_profile()
        else:
            self._create_default_profile()
    
    def _create_default_profile(self):
        """Crea los perfiles de fábrica. Recomendado es el perfil por defecto."""
        self.profiles = {
            "Recomendado": {
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
        self.last_used = "Recomendado"
        self.save_profiles()
    
    def save_profiles(self):
        """Guarda perfiles a archivo JSON"""
        try:
            data = {
                "last_used": self.last_used,
                "profiles": self.profiles
            }
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[Perfiles] Error al guardar: {e}")
    
    def get_profile(self, name):
        """Obtiene un perfil por nombre"""
        return self.profiles.get(name, self.profiles.get("Default"))
    
    def save_profile(self, name, config):
        """Guarda o actualiza un perfil"""
        if not name or name.strip() == "":
            messagebox.showerror("Error", "El nombre no puede estar vacío")
            return False
        
        if name in self.profiles and name != self.last_used:
            respuesta = messagebox.askyesno(
                "Confirmar",
                f"El perfil '{name}' ya existe.\n¿Deseas sobreescribirlo?"
            )
            if not respuesta:
                return False
        
        protected = self.profiles.get(name, {}).get("protected", False)
        config["protected"] = protected
        
        self.profiles[name] = config
        self.save_profiles()
        return True
    
    def delete_profile(self, name):
        """Elimina un perfil"""
        if name not in self.profiles:
            messagebox.showerror("Error", f"El perfil '{name}' no existe")
            return False
        
        if self.profiles[name].get("protected", False):
            messagebox.showwarning(
                "Protegido",
                f"El perfil '{name}' está protegido y no se puede eliminar"
            )
            return False
        
        respuesta = messagebox.askyesno(
            "Confirmar",
            f"¿Seguro que deseas eliminar el perfil '{name}'?\nEsta acción no se puede deshacer."
        )
        
        if respuesta:
            del self.profiles[name]
            if self.last_used == name:
                self.last_used = "Default"
            self.save_profiles()
            return True
        
        return False
    
    def rename_profile(self, old_name, new_name):
        """Renombra un perfil"""
        if old_name not in self.profiles:
            messagebox.showerror("Error", f"El perfil '{old_name}' no existe")
            return False
        
        if self.profiles[old_name].get("protected", False):
            messagebox.showwarning(
                "Protegido",
                f"El perfil '{old_name}' está protegido y no se puede renombrar"
            )
            return False
        
        if not new_name or new_name.strip() == "":
            messagebox.showerror("Error", "El nuevo nombre no puede estar vacío")
            return False
        
        if new_name in self.profiles:
            messagebox.showerror("Error", f"Ya existe un perfil llamado '{new_name}'")
            return False
        
        self.profiles[new_name] = self.profiles.pop(old_name)
        
        if self.last_used == old_name:
            self.last_used = new_name
        
        self.save_profiles()
        return True
    
    def get_profile_names(self):
        """Retorna lista de nombres de perfiles"""
        return list(self.profiles.keys())
    
    def set_last_used(self, name):
        """Establece el último perfil usado"""
        if name in self.profiles:
            self.last_used = name
            self.save_profiles()
    
    def is_protected(self, name):
        """Verifica si un perfil está protegido"""
        return self.profiles.get(name, {}).get("protected", False)
