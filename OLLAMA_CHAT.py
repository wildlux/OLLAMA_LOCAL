# Importazione della funzione chat da ollama
from ollama import chat
import os
import datetime

# Funzione per fare una domanda al modello LLaVA
def AI(query="Di che colore è il cielo?"):
    # Chiamiamo il modello specificando il messaggio dell'utente
    response = chat(
        model='llava:7b',
        messages=[
            {
                'role': 'user',
                'content': query,  # uso corretto della variabile, non 'QUERY' come stringa
            },
        ]
    )
    
    # Estraiamo il contenuto della risposta dal messaggio
    risposta = response.message.content
    return risposta  # restituiamo la risposta per poterla usare altrove

# Funzione principale
def main():
    language = 'it'  # codice lingua per espeak e pico2wave
    name = 'Paolo'
    
    # Otteniamo l'orario corrente per mostrarlo a schermo
    ora = datetime.datetime.now()
    print("Ora attuale:", ora)

    # Dizionario dei comandi vocali disponibili
    VOCI = {
        "espeak":     f'echo "Ciao {name}, questa è la voce espeak." | espeak -v {language}',
        "espeak-f":   f'echo "Ciao {name}, questa è una voce femminile." | espeak -v {language}+f3',
        "festival":   'echo "Ciao questa è Festival" | festival --tts',
        "pico2wave":  f'pico2wave -l=it-IT -w=output.wav "Ciao {name}, questa è una voce pico." && aplay output.wav',
        "RHVoice":    'echo "Ciao questa è RHVoice." | RHVoice-test -p anna'
    }

    # Frase iniziale per testare l'audio
    os.system(f'echo "Ciao {name}, sto funzionando. È una bella giornata!" | espeak -v {language}')

    # Usiamo l'AI per generare una risposta a una domanda
    domanda = "Perché il cielo è blu?"
    risposta_ai = AI(domanda)
    print("Risposta AI:", risposta_ai)

    # Leggiamo la risposta usando espeak
    comando_voce = f'echo "{risposta_ai}" | espeak -v {language}+f3'
    os.system(comando_voce)

    # Possiamo anche testare le varie voci disponibili (decommenta per provarle)
    # for nome, comando in VOCI.items():
    #     print(f"Provo voce: {nome}")
    #     os.system(comando)

# Avviamo lo script
if __name__ == "__main__":
    main()
