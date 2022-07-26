# Entwicklung einer automatisierten Prüfung der Strafbarkeit von Hate Speech
## Klassifizierung des §130 StGB, Volksverhetzung, in Tweets

Repository zur Bachelorarbeit [*Entwicklung einer automatisierten Prüfung der Strafbarkeit von Hate Speech*](https://github.com/cmbirle/legal-hate-speech/docs/HateSpeech-Bachelorarbeit-Birle.pdf) von Celia Birle, Universität Potsdam, verfasst im Rahmen der Tätigkeit bei Fraunhofer FOKUS

### Struktur

#### Notebooks
1. [*src/create_refcorpus.ipynb*](https://github.com/cmbirle/legal-hate-speech/src/create_refcorpus.ipynb): Vereinheitlichung und Zusammenfügen der Korpora
2. [*src/refcorp_annotation.ipynb*](https://github.com/cmbirle/legal-hate-speech/src/refcorp_annotation.ipynb): Hinzufügen und Prüfung der neuen Annotation der Volksverhetzung, § 130 StGB
3. [*src/refcorp_visual.ipynb*](https://github.com/cmbirle/legal-hate-speech/src/refcorp_visual.ipynb): Darstellung der Verteilung der der Annotationslabels im Datensatz
4. [*src/classify.ipynb*](https://github.com/cmbirle/legal-hate-speech/src/classify.ipynb): Klassifikationspipelines für Lineare Regression und BERT, Entscheidungsbaum

#### Korpora
Die vier den Referenzdatensatz konstituierenden Korpora unterliegen unterschiedlichen Lizenzen; sie sind nicht alle frei verfügbar und können nicht hier hochgeladen werden. Der Ordner dient der Verständlichkeit des Arbeitsprozesses.

#### Modelle
Alle mit Logistischer Regression und auf BERT trainierten Modelle. Für Details s. [*src/classify.ipynb*](https://github.com/cmbirle/legal-hate-speech/src/classify.ipynb).



<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.