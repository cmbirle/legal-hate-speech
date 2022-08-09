# Entwicklung einer automatisierten Prüfung der Strafbarkeit von Hate Speech
## Klassifizierung des §130 StGB, Volksverhetzung, in Tweets

Repository zur Bachelorarbeit [*Entwicklung einer automatisierten Prüfung der Strafbarkeit von Hate Speech*](https://github.com/cmbirle/legal-hate-speech/blob/main/docs/Bachelorarbeit-HateSpeech-Birle-2022.pdf) von Celia Birle, Universität Potsdam, verfasst im Rahmen der Tätigkeit als wissenschaftlich Hilfskraft bei Fraunhofer FOKUS


Diese Bachelorarbeit untersucht, inwiefern es möglich ist, die Prüfung von Hate Speech auf ihre potentielle Strafbarkeit zu automatisieren. Dazu wurde beispielhaft für die Volksverhetzung (§ 130 StGB) ein Annotationsschema entwickelt. Ein Referenzdatensatz wurde aus bestehenden deutschsprachigen Datensätzen zusammengestellt und ein Ausschnitt dieser Daten anhand des neuen Annotationsschemas annotiert. Diese Daten dienten als Grundlage für das Training zweier Klassifikationsverfahren: Der logistischen Regression und dem Transfer Learning mit BERT. Anstatt Aussagen direkt als volksverhetzend zu klassifizieren, wurden separat zwei zentrale Tatbestandsmerkmale klassifiziert, die zusammen ein starkes Indiz dafür sind: Gruppe und Tathandlung. Durch eine Kombination dieser Teilklassifikation in einem Entscheidungsbaum konnte die Vorhersage 'volksverhetzend - Ja/Nein' getroffen werden.

Die durch verschiedene Merkmale definierten Gruppen können erfolgreich klassifiziert werden; es wird mit BERT ein MCC-Wert zwischen 0.58 und 0.83 erreicht. Durch einen Mangel an Beispielen für die Tathandlungen können diese automatisiert jedoch mit einem MCC-Wert von 0.38 nur ungenügend erkannt werden. Die Kombination der Klassifikatoren der beiden Merkmale zur Klassifikation der Volksverhetzung als Ganzes verbessert die Ergebnisse der direkten Klassifikation nur geringfügig. Eine automatisierte Prüfung von Hate Speech auf ihre Strafbarkeit unter dem Tatbestand der Volksverhetzung ist grundsätzlich möglich, für bessere Ergebnisse wären jedoch noch mehr annotierte Daten notwendig.

In diesem Repository finden sich die Notebooks zur Erstellung des Referenzdatensatzes und zur Klassifikation sowie die daraus resultierenden Modelle für die einzelnen Merkmale und zuletzt die Bachelorarbeit selbst.

### Struktur

#### Notebooks
1. [*src/create_refcorpus.ipynb*](https://github.com/cmbirle/legal-hate-speech/blob/main/src/create_refcorpus.ipynb): Vereinheitlichung und Zusammenfügen der Korpora
2. [*src/refcorp_annotation.ipynb*](https://github.com/cmbirle/legal-hate-speech/blob/main/src/refcorp_annotation.ipynb): Hinzufügen und Prüfung der neuen Annotation der Volksverhetzung, § 130 StGB
3. [*src/refcorp_visual.ipynb*](https://github.com/cmbirle/legal-hate-speech/blob/main/src/refcorp_visual.ipynb): Darstellung der Verteilung der der Annotationslabels im Datensatz
4. [*src/classify.ipynb*](https://github.com/cmbirle/legal-hate-speech/blob/main/src/classify.ipynb): Klassifikationspipelines für die logistische Regression, das Transfer Learning mit BERT und den Entscheidungsbaum

#### Korpora
Die vier den Referenzdatensatz konstituierenden Korpora unterliegen unterschiedlichen Lizenzen; sie sind nicht alle frei verfügbar und können nicht hier hochgeladen werden. Der Ordner dient der Verständlichkeit des Arbeitsprozesses.

#### Modelle
Die mit logistischer Regression und auf BERT trainierten Modelle für das Gruppen- und Handlungsmerkmal. Für Details s. [*src/classify.ipynb*](https://github.com/cmbirle/legal-hate-speech/blob/main/src/classify.ipynb).



<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.