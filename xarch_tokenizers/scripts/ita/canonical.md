# Model Performance Summary

## 📊 Accuracy Summary

| Rank | Model | Accuracy | Correct | Total | Avg Confidence |
|------|-------|----------|---------|-------|----------------|
| 1 | supertoken_models-llama_google-gemma-2-2b | 95.2% | 40 | 42 | 0.506 |
| 2 | supertoken_models-llama_meta-llama-Llama-3.2-1B | 95.2% | 40 | 42 | 0.492 |
| 3 | supertoken_models-llama_common-pile-comma-v0.1 | 92.9% | 39 | 42 | 0.454 |
| 4 | supertoken_models-llama_microsoft-Phi-3-mini-4k-instruct | 90.5% | 38 | 42 | 0.455 |
| 5 | supertoken_models-llama_gpt2 | 90.5% | 38 | 42 | 0.439 |
| 6 | supertoken_models-llama_bigscience-bloom | 90.5% | 38 | 42 | 0.420 |
| 7 | supertoken_models-llama_facebook-xglm-564M | 90.5% | 38 | 42 | 0.444 |


## 📋 Detailed Question Results

| Q# | Question | Correct Answer | supertoken_models-llama_google-gemma-2-2b | supertoken_models-llama_common-pile-comma-v0.1 | supertoken_models-llama_meta-llama-Llama-3.2-1B | supertoken_models-llama_microsoft-Phi-3-mini-4k-instruct | supertoken_models-llama_gpt2 | supertoken_models-llama_bigscience-bloom | supertoken_models-llama_facebook-xglm-564M |
|---|---------|---------------|------------------------------------------|-----------------------------------------------|------------------------------------------------|---------------------------------------------------------|-----------------------------|-----------------------------------------|-------------------------------------------|
| 1 | Il Dr Rossi è un dottore. L'occupazione del Dr.... | Dottore | ❌ Insegnante (0.29) | ❌ Insegnante (0.33) | ✅ Dottore (0.33) | ❌ Insegnante (0.32) | ✅ Dottore (0.36) | ✅ Dottore (0.33) | ✅ Dottore (0.33) |
| 2 | Il colore del cielo è | azzurro | ✅ azzurro (0.35) | ✅ azzurro (0.29) | ✅ azzurro (0.27) | ✅ azzurro (0.26) | ✅ azzurro (0.27) | ✅ azzurro (0.29) | ✅ azzurro (0.30) |
| 3 | Il prezzo di questa casa è 300,000 euro. Il cos... | 300,000 euro | ✅ 300,000 euro (0.37) | ✅ 300,000 euro (0.36) | ✅ 300,000 euro (0.38) | ✅ 300,000 euro (0.36) | ✅ 300,000 euro (0.37) | ✅ 300,000 euro (0.37) | ✅ 300,000 euro (0.43) |
| 4 | La data di oggi è 15/8/2025. Oggi è | 15/8/2025 | ✅ 15/8/2025 (0.35) | ✅ 15/8/2025 (0.33) | ✅ 15/8/2025 (0.34) | ✅ 15/8/2025 (0.31) | ✅ 15/8/2025 (0.31) | ✅ 15/8/2025 (0.33) | ✅ 15/8/2025 (0.33) |
| 5 | Il numero di continenti sulla Terra è | 7 | ✅ 7 (0.80) | ❌ 5 (0.39) | ✅ 7 (0.60) | ✅ 7 (0.37) | ✅ 7 (0.47) | ✅ 7 (0.41) | ❌ 6 (0.35) |
| 6 | La capitale dell'Italia è | Roma | ✅ Roma (0.36) | ✅ Roma (0.33) | ✅ Roma (0.33) | ✅ Roma (0.33) | ✅ Roma (0.35) | ✅ Roma (0.33) | ✅ Roma (0.39) |
| 7 | Il numero di giorni in una settimana è | 7 | ✅ 7 (0.94) | ✅ 7 (0.84) | ✅ 7 (0.94) | ✅ 7 (0.75) | ✅ 7 (0.59) | ✅ 7 (0.64) | ✅ 7 (0.68) |
| 8 | Il numero di ore in un giorno è | 24 | ✅ 24 (0.83) | ✅ 24 (0.82) | ✅ 24 (0.85) | ✅ 24 (0.70) | ✅ 24 (0.42) | ✅ 24 (0.37) | ✅ 24 (0.45) |
| 9 | Il numero di zampe che ha una mucca è | 4 | ✅ 4 (0.59) | ✅ 4 (0.46) | ✅ 4 (0.60) | ✅ 4 (0.33) | ❌ 5 (0.30) | ❌ 8 (0.32) | ❌ 3 (0.27) |
| 10 | Il numero di minuti in 2 ore è | 120 | ✅ 120 (0.47) | ✅ 120 (0.40) | ✅ 120 (0.38) | ✅ 120 (0.37) | ❌ 100 (0.40) | ❌ 100 (0.37) | ✅ 120 (0.41) |
| 11 | Il numero di mesi in un anno è | 12 | ✅ 12 (0.67) | ✅ 12 (0.58) | ✅ 12 (0.87) | ✅ 12 (0.75) | ✅ 12 (0.45) | ✅ 12 (0.47) | ✅ 12 (0.53) |
| 12 | I secondi che compongono un minuto sono | 60 | ✅ 60 (0.46) | ✅ 60 (0.62) | ✅ 60 (0.69) | ✅ 60 (0.67) | ❌ 100 (0.49) | ❌ 100 (0.39) | ✅ 60 (0.40) |
| 13 | Il numero di lati che ha un esagono è | 6 | ✅ 6 (0.67) | ✅ 6 (0.57) | ✅ 6 (0.95) | ❌ 5 (0.45) | ✅ 6 (0.66) | ✅ 6 (0.38) | ✅ 6 (0.63) |
| 14 | Il numero di lati che ha un triangolo è | 3 | ✅ 3 (0.93) | ✅ 3 (0.59) | ✅ 3 (0.73) | ✅ 3 (0.62) | ✅ 3 (0.71) | ✅ 3 (0.42) | ✅ 3 (0.56) |
| 15 | Quando qualcuno dice "Io lavoro a Lamborghini",... | un'azienda | ✅ un'azienda (0.28) | ✅ un'azienda (0.30) | ✅ un'azienda (0.32) | ✅ un'azienda (0.30) | ✅ un'azienda (0.30) | ✅ un'azienda (0.32) | ✅ un'azienda (0.32) |
| 16 | Nella frase "Io lavoro in Bialetti", Bialetti r... | un'azienda | ✅ un'azienda (0.29) | ✅ un'azienda (0.30) | ✅ un'azienda (0.29) | ✅ un'azienda (0.33) | ✅ un'azienda (0.33) | ✅ un'azienda (0.27) | ✅ un'azienda (0.32) |
| 17 | In "Logitech ha appena rilasciato un aggiorname... | un'azienda | ✅ un'azienda (0.39) | ✅ un'azienda (0.39) | ✅ un'azienda (0.38) | ✅ un'azienda (0.38) | ✅ un'azienda (0.37) | ✅ un'azienda (0.43) | ✅ un'azienda (0.38) |
| 18 | In "Il gatto è seduto sul divano", il soggetto è | il gatto | ✅ il gatto (0.59) | ✅ il gatto (0.56) | ✅ il gatto (0.58) | ✅ il gatto (0.57) | ✅ il gatto (0.63) | ✅ il gatto (0.67) | ✅ il gatto (0.62) |
| 19 | Per vivere, gli essere umani devono respirare | ossigeno | ✅ ossigeno (0.63) | ✅ ossigeno (0.52) | ✅ ossigeno (0.56) | ✅ ossigeno (0.58) | ✅ ossigeno (0.56) | ✅ ossigeno (0.56) | ✅ ossigeno (0.55) |
| 20 | Il 10% di 100 è | 10 | ✅ 10 (0.86) | ✅ 10 (0.76) | ✅ 10 (0.97) | ✅ 10 (0.77) | ✅ 10 (0.54) | ✅ 10 (0.63) | ✅ 10 (0.83) |
| 21 | Il 25% di 80 è | 20 | ✅ 20 (0.46) | ✅ 20 (0.38) | ✅ 20 (0.46) | ❌ 25 (0.43) | ✅ 20 (0.36) | ❌ 30 (0.32) | ❌ 25 (0.35) |
| 22 | Il 50% di 60 è | 30 | ✅ 30 (0.56) | ✅ 30 (0.42) | ❌ 40 (0.36) | ✅ 30 (0.39) | ✅ 30 (0.43) | ✅ 30 (0.43) | ✅ 30 (0.47) |
| 23 | La capitale del Ciad è | N'Djamena | ✅ N'Djamena (0.64) | ✅ N'Djamena (0.63) | ✅ N'Djamena (0.54) | ✅ N'Djamena (0.66) | ✅ N'Djamena (0.63) | ✅ N'Djamena (0.44) | ✅ N'Djamena (0.56) |
| 24 | La capitale della Francia è | Parigi | ✅ Parigi (0.56) | ✅ Parigi (0.48) | ✅ Parigi (0.52) | ✅ Parigi (0.53) | ✅ Parigi (0.56) | ✅ Parigi (0.61) | ✅ Parigi (0.54) |
| 25 | La capitale del Giappone è | Tokyo | ✅ Tokyo (0.45) | ✅ Tokyo (0.37) | ✅ Tokyo (0.33) | ✅ Tokyo (0.40) | ✅ Tokyo (0.43) | ✅ Tokyo (0.37) | ✅ Tokyo (0.47) |
| 26 | La capitale della Turchia è | Ankara | ✅ Ankara (0.51) | ✅ Ankara (0.49) | ✅ Ankara (0.42) | ✅ Ankara (0.44) | ✅ Ankara (0.46) | ✅ Ankara (0.40) | ✅ Ankara (0.47) |
| 27 | La formula chimica dell'acqua è | H2O | ✅ H2O (0.84) | ✅ H2O (0.80) | ✅ H2O (0.74) | ✅ H2O (0.80) | ✅ H2O (0.77) | ✅ H2O (0.77) | ✅ H2O (0.83) |
| 28 | L'intento in "A che ora chiude il negozio?" è | ottenere un'informazione | ❌ prenotare un appuntamento (0.28) | ✅ ottenere un'informazione (0.29) | ✅ ottenere un'informazione (0.30) | ✅ ottenere un'informazione (0.29) | ✅ ottenere un'informazione (0.30) | ✅ ottenere un'informazione (0.30) | ✅ ottenere un'informazione (0.29) |
| 29 | Il mammifero più grande del mondo è | la balenottera azzurra | ✅ la balenottera azzurra (0.37) | ✅ la balenottera azzurra (0.33) | ✅ la balenottera azzurra (0.35) | ✅ la balenottera azzurra (0.33) | ✅ la balenottera azzurra (0.32) | ✅ la balenottera azzurra (0.33) | ✅ la balenottera azzurra (0.34) |
| 30 | Nel Sistema Internazionale, la temperatura si m... | Kelvin | ✅ Kelvin (0.37) | ✅ Kelvin (0.35) | ✅ Kelvin (0.32) | ✅ Kelvin (0.32) | ✅ Kelvin (0.30) | ✅ Kelvin (0.35) | ✅ Kelvin (0.40) |
| 31 | Il paese la cui agenzia spaziale è la NASA è | Stati Uniti | ✅ Stati Uniti (0.36) | ✅ Stati Uniti (0.39) | ✅ Stati Uniti (0.46) | ✅ Stati Uniti (0.42) | ✅ Stati Uniti (0.45) | ✅ Stati Uniti (0.41) | ✅ Stati Uniti (0.45) |
| 32 | La lingua parlata in Brasile è | il portoghese | ✅ il portoghese (0.31) | ✅ il portoghese (0.34) | ✅ il portoghese (0.34) | ✅ il portoghese (0.34) | ✅ il portoghese (0.36) | ✅ il portoghese (0.35) | ✅ il portoghese (0.35) |
| 33 | Il metallo con simbolo chimico 'Fe' è | Ferro | ✅ Ferro (0.41) | ✅ Ferro (0.40) | ❌ Piombo (0.41) | ❌ Piombo (0.41) | ❌ Piombo (0.47) | ✅ Ferro (0.40) | ❌ Piombo (0.37) |
| 34 | L'elemento chimico che ha come simbolo 'Fe' è | Ferro | ✅ Ferro (0.53) | ✅ Ferro (0.53) | ✅ Ferro (0.40) | ✅ Ferro (0.46) | ✅ Ferro (0.41) | ✅ Ferro (0.54) | ✅ Ferro (0.43) |
| 35 | L'organo che pompa il sangue nel corpo umano è il | cuore | ✅ cuore (0.58) | ✅ cuore (0.46) | ✅ cuore (0.57) | ✅ cuore (0.49) | ✅ cuore (0.46) | ✅ cuore (0.50) | ✅ cuore (0.54) |
| 36 | Il pianeta che ha la distanza minore dal Sole è | Mercurio | ✅ Mercurio (0.37) | ✅ Mercurio (0.36) | ✅ Mercurio (0.33) | ✅ Mercurio (0.33) | ✅ Mercurio (0.37) | ✅ Mercurio (0.35) | ✅ Mercurio (0.33) |
| 37 | Il pianeta con il diametro più grande del Siste... | Giove | ✅ Giove (0.37) | ❌ Marte (0.29) | ✅ Giove (0.34) | ✅ Giove (0.36) | ✅ Giove (0.34) | ✅ Giove (0.36) | ✅ Giove (0.33) |
| 38 | Il meccanismo con cui le piante creano nutrimen... | Fotosintesi | ✅ Fotosintesi (0.38) | ✅ Fotosintesi (0.34) | ✅ Fotosintesi (0.32) | ✅ Fotosintesi (0.31) | ✅ Fotosintesi (0.32) | ✅ Fotosintesi (0.32) | ✅ Fotosintesi (0.31) |
| 39 | L'autore che ha scritto la commedia "Romeo e Gi... | William Shakespeare | ✅ William Shakespeare (0.41) | ✅ William Shakespeare (0.40) | ✅ William Shakespeare (0.38) | ✅ William Shakespeare (0.39) | ✅ William Shakespeare (0.41) | ✅ William Shakespeare (0.40) | ✅ William Shakespeare (0.42) |
| 40 | Le api sono famose per produrre | miele | ✅ miele (0.61) | ✅ miele (0.57) | ✅ miele (0.57) | ✅ miele (0.56) | ✅ miele (0.55) | ✅ miele (0.55) | ✅ miele (0.50) |
| 41 | Quello di cui hanno bisogno le piante dall'aria... | anidride carbonica | ✅ anidride carbonica (0.45) | ✅ anidride carbonica (0.43) | ✅ anidride carbonica (0.52) | ✅ anidride carbonica (0.54) | ✅ anidride carbonica (0.49) | ✅ anidride carbonica (0.48) | ✅ anidride carbonica (0.50) |
| 42 | In "Puoi prenotare il volo per Parigi?", la per... | fare una prenotazione | ✅ fare una prenotazione (0.32) | ✅ fare una prenotazione (0.31) | ✅ fare una prenotazione (0.33) | ✅ fare una prenotazione (0.36) | ✅ fare una prenotazione (0.33) | ✅ fare una prenotazione (0.34) | ✅ fare una prenotazione (0.32) |


### Legend
- ✅ = Correct answer
- ❌ = Incorrect answer
- ⚠️ = Error occurred
- Numbers in parentheses = Confidence score
