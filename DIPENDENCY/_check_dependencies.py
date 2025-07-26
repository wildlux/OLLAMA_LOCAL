
import os
import subprocess
import sys
import shutil
import platform
from datetime import datetime, timedelta

# --- CONFIGURAZIONE ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VENV_DIR = os.path.join(SCRIPT_DIR, "venv_local_assistant")
LOG_FILE = os.path.join(SCRIPT_DIR, "_dependency_check_log.txt")
CACHE_DURATION_HOURS = 24 # Durata del cache in ore (es. 24 ore = 1 giorno)

# --- FUNZIONI DI UTILITY ---

def log_message(message, indent=0):
    """Scrive un messaggio sul log file e sulla console."""
    prefix = " " * indent
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_message = f"[{timestamp}] {prefix}{message}"
    print(full_message)
    with open(LOG_FILE, 'a') as f:
        f.write(full_message + "\n")

def is_cache_valid(flag_name):
    """
    Controlla se un flag specifico nel log file è presente e recente.
    """
    if not os.path.exists(LOG_FILE):
        return False
    
    with open(LOG_FILE, 'r') as f:
        lines = f.readlines()
        
    for line in reversed(lines): # Controlla le linee più recenti per prime
        if flag_name in line and "✅ Controllato" in line:
            try:
                # Estrai il timestamp dalla riga di log
                timestamp_str = line.split(']')[0].strip('[ ')
                log_time = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
                if datetime.now() - log_time < timedelta(hours=CACHE_DURATION_HOURS):
                    log_message(f"Cache valido per '{flag_name}'. Saltato il controllo.", indent=4)
                    return True
            except (ValueError, IndexError):
                # Errore nel parsing del timestamp, tratta come cache non valido
                pass
    return False

def check_system_command(cmd, apt_pkg=None, install_cmd=None):
    """Controlla e installa comandi di sistema, loggando i risultati."""
    log_message(f"Verifica disponibilità: {cmd}...", indent=4)
    if shutil.which(cmd):
        log_message(f"  ✅ '{cmd}' trovato.", indent=4)
        return True
    else:
        log_message(f"  ⛔ '{cmd}' non trovato.", indent=4)
        if platform.system() == "Linux":
            if install_cmd:
                log_message(f"  Tentativo di installazione: {install_cmd}...", indent=4)
                try:
                    subprocess.check_call(install_cmd.split(), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    if shutil.which(cmd):
                        log_message(f"  ✅ '{cmd}' installato con successo.", indent=4)
                        return True
                    else:
                        log_message(f"  ❌ Installazione di '{cmd}' fallita o non trovato dopo l'installazione.", indent=4)
                        log_message(f"     Per favore, prova a eseguire manualmente: {install_cmd}", indent=4)
                except subprocess.CalledProcessError:
                    log_message(f"  ❌ Errore durante l'esecuzione di '{install_cmd}'.", indent=4)
                    log_message(f"     Per favor, prova a eseguire manualmente: {install_cmd}", indent=4)
            elif apt_pkg:
                log_message(f"  Per installarlo, esegui: sudo apt update && sudo apt install {apt_pkg}", indent=4)
        return False

def install_pip_package(package):
    """Installa un pacchetto Python nell'ambiente virtuale, loggando i risultati."""
    log_message(f"Installazione pacchetto Python '{package}'...", indent=4)
    try:
        subprocess.check_call([os.path.join(VENV_DIR, "bin", "python"), "-m", "pip", "install", package], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        log_message(f"  ✅ '{package}' installato.", indent=4)
        return True
    except subprocess.CalledProcessError:
        log_message(f"  ❌ Installazione di '{package}' fallita. Potrebbe essere un problema di rete o di installazione del pacchetto.", indent=4)
        return False

def main():
    """Funzione principale per il controllo delle dipendenze."""
    log_message("--- Avvio controllo dipendenze ---")

    # 1. Controlla e installa le dipendenze di sistema
    system_deps_flag_name = "SYSTEM_DEPS_CHECKED"
    if is_cache_valid(system_deps_flag_name):
        log_message(f"✅ Dipendenze di sistema già controllate di recente.", indent=2)
        system_deps_ok = True
    else:
        log_message(f"Verifica delle dipendenze di sistema (cache scaduto o non presente)...", indent=2)
        system_deps_ok = True
        # Controlla tutti i comandi necessari
        if not check_system_command("python3", apt_pkg="python3"): system_deps_ok = False
        if not check_system_command("pip", apt_pkg="python3-pip"): system_deps_ok = False
        if not check_system_command("sox", apt_pkg="sox libsox-fmt-all"): system_deps_ok = False
        if not check_system_command("curl", apt_pkg="curl"): system_deps_ok = False
        if not check_system_command("jq", apt_pkg="jq"): system_deps_ok = False
        if not check_system_command("espeak", apt_pkg="espeak"): system_deps_ok = False
        if not check_system_command("dos2unix", install_cmd="sudo apt-get install -y dos2unix"): system_deps_ok = False
        if not check_system_command("ollama"):
            log_message("    Ollama binary not found in PATH. Download it from ollama.com and/or ensure it's in your PATH.", indent=4)
            system_deps_ok = False

        if system_deps_ok:
            log_message(f"✅ Controllato: {system_deps_flag_name}", indent=2)
        else:
            log_message(f"❌ Errore: Alcune dipendenze di sistema non sono state trovate o installate.", indent=2)
            sys.exit(1)
    
    # 2. Crea o riutilizza l'ambiente virtuale (solo creazione qui, attivazione in setup_and_run)
    log_message(f"Creazione/Verifica ambiente virtuale in '{VENV_DIR}'...", indent=2)
    if not os.path.exists(VENV_DIR):
        try:
            subprocess.check_call([sys.executable, "-m", "venv", VENV_DIR])
            log_message(f"  ✅ Ambiente virtuale creato.", indent=2)
        except Exception as e:
            log_message(f"  ❌ Errore durante la creazione dell'ambiente virtuale: {e}", indent=2)
            sys.exit(1)
    else:
        log_message(f"  ✅ Ambiente virtuale già esistente.", indent=2)

    # 3. Controlla e installa le dipendenze Python nell'ambiente virtuale.
    python_deps_flag_name = "PYTHON_DEPS_CHECKED"
    if is_cache_valid(python_deps_flag_name):
        log_message(f"✅ Dipendenze Python già controllate di recente.", indent=2)
        python_deps_ok = True
    else:
        log_message(f"Verifica e installazione delle dipendenze Python (cache scaduto o non presente)...", indent=2)
        python_deps_ok = True
        if not install_pip_package("vosk"): python_deps_ok = False
        if not install_pip_package("speechrecognition"): python_deps_ok = False
        if not install_pip_package("pynput"): python_deps_ok = False

        if python_deps_ok:
            log_message(f"✅ Controllato: {python_deps_flag_name}", indent=2)
        else:
            log_message(f"❌ Errore: Alcune dipendenze Python non sono state installate correttamente nell'ambiente virtuale.", indent=2)
            sys.exit(1)

    log_message("--- Controllo dipendenze completato ---")
    
if __name__ == "__main__":
    main()
