# GEM Learnings - Erkenntnisse aus der Validierungsarbeit

Diese Dokumentation fasst alle Erkenntnisse zusammen, die während der Arbeit mit gym-electric-motor (GEM) zur Validierung gegen MATLAB/Simulink gewonnen wurden.

---

## Ausgangssituation und Ziel

Das übergeordnete Ziel war es, die Python/GEM-Simulation gegen das bestehende MATLAB/Simulink PMSM-Modell zu validieren. Bevor überhaupt bewertet werden konnte, ob beide Simulationen gleiche oder zumindest ähnliche Ergebnisse liefern, mussten die Eingangs- und Sollwerte vereinheitlicht werden, um eine saubere Baseline zu schaffen.

Die anfänglichen Vergleiche zeigten massive Diskrepanzen: unterschiedliche Drehzahlen, invertierte Vorzeichen bei den Strömen, und völlig verschiedene Zeitverläufe. Diese Probleme mussten systematisch identifiziert und behoben werden.

---

## Problem 1: Step-Zeitpunkt Inkonsistenz

**Beobachtung:** Beim ersten Vergleich waren die Zeitverläufe komplett unterschiedlich. MATLAB zeigte bis t=0.1s nahezu keine Aktivität, während Python sofort reagierte.

**Ursache:** Das MATLAB-Modell verwendet Step-Blöcke, die den Sollwert erst bei t=0.1s aktivieren. Die ursprüngliche Python-Simulation hatte keine entsprechende Verzögerung und wendete den Sollwert von Anfang an an.

**Lösung:** Die Python-Simulation wurde angepasst, sodass ein konfigurierbarer step_time Parameter verfügbar ist. Für den Vergleich wird dieser auf 0 gesetzt, da auch die MATLAB-Simulation später so angepasst wurde, dass die Sollwerte sofort aktiv sind.

**Learning:** Beim Vergleich zweier Simulationen muss der zeitliche Ablauf der Sollwert-Aktivierung exakt übereinstimmen. Scheinbar triviale Unterschiede wie ein verzögerter Step können den gesamten Vergleich invalidieren.

---

## Problem 2: Vorzeichen-Inversion bei i_q

**Beobachtung:** Nach der Korrektur des Step-Zeitpunkts zeigte sich, dass das Vorzeichen von i_q invertiert war. MATLAB lieferte positive Werte, Python negative.

**Vermutete Ursache:** Unterschiedliche Konventionen bei der dq-Transformation oder der Definition des Referenzsystems. GEM könnte eine andere Transformationsrichtung verwenden als das MATLAB-Modell.

**Status:** Dieses Problem wurde im Laufe der Arbeiten durch die korrekte Konfiguration und das Debugging der Normalisierung gelöst. Der GEM Standard Controller liefert nun korrekte Vorzeichen.

**Learning:** Simulationsbibliotheken können unterschiedliche Konventionen für Koordinatensysteme und Transformationen verwenden. Diese müssen explizit geprüft und dokumentiert werden.

---

## Problem 3: Drehzahl nicht direkt steuerbar

**Beobachtung:** GEM startete bei etwa 716 rpm, während MATLAB bei 1000 rpm (später 500, 1500, 2500 rpm für die Tests) lief. Die Drehzahl ließ sich nicht über einen Reference Generator setzen.

**Ursache:** Das Current-Control-Environment Cont-CC-PMSM-v0 akzeptiert nur Strom-Referenzen. Die Drehzahl wird intern durch die Physik des Motors und die Last bestimmt, nicht durch einen externen Sollwert. Ein Versuch, die Drehzahl über einen ConstReferenceGenerator zu setzen, schlug fehl.

**Lösung:** Verwendung der ConstantSpeedLoad als Last-Modell. Diese erzwingt eine konstante mechanische Drehzahl und ermöglicht so den Vergleich bei definierten Drehzahlen.

**Learning:** In GEM ist die Drehzahl beim Current-Control-Environment ein Zustandsparameter, der die Back-EMF beeinflusst, aber nicht direkt als Eingabe gesetzt werden kann. Für feste Drehzahlen muss die Last entsprechend konfiguriert werden.

---

## Problem 4: Unterschied zwischen GEM-Controller und MATLAB-Controller

**Beobachtung:** Auch bei gleicher Drehzahl wichen die Ergebnisse ab. Der GEM-Controller verhielt sich anders als der MATLAB-Regler.

**Ursache:** Der Standard-GEM-Controller (gem_controllers) ist als universeller Regler konzipiert, der automatisch zu einem gegebenen Motor-Environment passt. Die PI-Gains werden intern über Polplatzierung berechnet, wobei ein Tuning-Parameter die Bandbreite bestimmt. Im Gegensatz dazu verwendet das MATLAB-Modell explizit definierte PI-Gains nach der Technischen Optimaleinstellung mit den Formeln Kp = L/(2*Ts) und Ki = R/(2*Ts).

**Maßnahme:** Ein eigener PI-Controller wurde implementiert, der exakt die MATLAB-Parameter verwendet. Dieser Controller hat dieselben Gain-Formeln, dieselbe Back-EMF-Entkopplung und sollte dieselbe Anti-Windup-Logik wie das MATLAB-Modell haben.

**Ergebnis:** Der eigene Controller funktioniert konzeptionell, erreicht aber die Sollwerte in GEM nicht korrekt. Der GEM Standard Controller hingegen erreicht perfekte Übereinstimmung bei den Strömen.

**Learning:** Der GEM-Controller fungiert als Black-Box, deren internes Verhalten nicht direkt an ein bestehendes MATLAB-Modell angepasst werden kann. Für exakte Replikation muss entweder der Controller transparent sein oder man akzeptiert den GEM-Controller und validiert nur die Physik.

---

## Problem 5: Parameter-Übergabe an GEM wird ignoriert

**Beobachtung:** Trotz Übergabe von motor_parameter und limit_values an gem.make() verwendete GEM die Default-Werte. Die Drehzahl war systematisch um 25% niedriger als erwartet.

**Ursache:** Die GEM-Funktion gem.make() reicht Top-Level-Kwargs nicht korrekt an die internen Komponenten weiter. Die Parameter wurden schlicht ignoriert und die Bibliotheks-Defaults verwendet.

**Lösung:** Motor- und Load-Objekte müssen explizit mit den Klassen PermanentMagnetSynchronousMotor und ConstantSpeedLoad erstellt werden. Diese Objekte werden dann an gem.make() übergeben.

**Learning:** GEM's API-Design hat Schwächen bei der Parameter-Übergabe. Komponenten sollten immer direkt instantiiert werden, um sicherzustellen, dass die Parameter ankommen.

---

## Problem 6: Falsche Denormalisierung

**Beobachtung:** Selbst nach korrekter Parameter-Übergabe war die angezeigte Drehzahl falsch (375 rpm statt 500 rpm, also 75% des Sollwerts).

**Ursache:** Für die Denormalisierung wurden die eigenen limit_values verwendet, aber GEM normalisiert intern mit den Physical System Limits. Da diese unterschiedlich waren, kam bei der Denormalisierung ein falscher Wert heraus.

**Lösung:** Die tatsächlichen Limits müssen vom Physical System ausgelesen werden. Diese Limits werden dann für alle Denormalisierungen verwendet.

**Learning:** GEM's Normalisierung ist nicht transparent. Die verwendeten Limits müssen explizit vom Physical System abgefragt werden, nicht aus den übergebenen Parametern angenommen werden.

---

## Problem 7: State-Tuple Handling

**Beobachtung:** Nach jedem Simulationsschritt driftete die Regelung ab. Ein isoliertes Debug-Skript mit identischem Code funktionierte, das Hauptskript nicht.

**Ursache:** Der Rückgabewert von env.step() ist ein 5-Tuple, wobei das erste Element (observation) selbst ein Tuple sein kann. Der ursprüngliche Code behandelte das inkonsistent, wodurch im nächsten Schritt ein Tuple statt eines numpy-Arrays an den Controller übergeben wurde.

**Lösung:** Explizites und robustes Auspacken mit isinstance-Checks an allen Stellen, wo der Zustand verarbeitet wird.

**Learning:** GEM's API-Rückgaben sind nicht konsistent typisiert. An jeder Stelle, wo States verarbeitet werden, muss auf verschachtelte Tuples geprüft werden.

---

## Problem 8: Constraints und Simulationsabbruch

**Beobachtung:** Die Simulation terminierte unerwartet mit done=True, was zu Resets führte und die Regelung destabilisierte.

**Ursache:** Das Cont-CC-PMSM-v0 Environment hat standardmäßig SquaredConstraint aktiviert, das die Simulation beendet wenn die Strom-Limits überschritten werden.

**Lösung:** Constraints beim Environment-Erstellen deaktivieren durch Übergabe von constraints=().

**Learning:** GEM-Environments haben standardmäßig Sicherheits-Constraints aktiv. Für Validierungszwecke sollten diese deaktiviert werden, um unkontrollierte Resets zu vermeiden.

---

## Problem 9: Controller-Reset bei Termination

**Beobachtung:** Wenn done=True auftrat und das Environment resetet wurde, verlor der Controller seinen Zustand und die Regelung begann von vorn.

**Ursache:** Der ursprüngliche Code rief bei done=True sowohl env.reset() als auch controller.reset() auf. Der Controller-Reset setzte den Integrator auf Null zurück.

**Lösung:** Bei Termination nur das Environment resetten, den Controller-Zustand aber behalten.

**Learning:** Environment-Reset und Controller-Reset müssen getrennt behandelt werden. Der Integrator-Zustand sollte bei kurzfristigen Environment-Resets erhalten bleiben.

---

## Problem 10: MATLAB Simulink-Verdrahtung

**Beobachtung (von Dennis):** Die MATLAB-Simulation reagierte nicht auf Änderungen der Variablen im Skript. Die Step-Blöcke schienen feste Werte zu nutzen.

**Ursache:** Es gab einen Naming-Konflikt in Simulink: Die Variable und der Block hatten den gleichen Namen, was Simulink nicht korrekt verarbeitete.

**Lösung (Dennis):** Umbenennung des Blocks, sodass die Variable korrekt als Final-Value verwendet werden kann.

**Learning:** Bei MATLAB/Simulink-Modellen muss geprüft werden, ob Variablen aus dem Skript tatsächlich in der Simulation ankommen. Naming-Konflikte können zu subtilen Bugs führen.

---

## Architektonische Erkenntnisse

**GEM Environment-Architektur:** Das Current-Control-Environment ist für reine Stromregelung ohne übergeordnete Drehzahlregelung konzipiert. Die Drehzahl ergibt sich aus der Motordynamik und der Last. Dies entspricht dem MATLAB-Modell, das ebenfalls Direct Current Control ohne Kaskadenregelung implementiert.

**Drehzahlabhängigkeit der Stromregelung:** Bei niedriger Drehzahl ist die induzierte Gegenspannung klein. Bei hoher Drehzahl dominiert die Gegenspannung und der Regler muss deutlich höhere Spannungen ausgeben. Diese Abhängigkeit muss in Trainingsdaten abgebildet werden.

**Normalisierung als Designprinzip:** GEM ist für Reinforcement Learning konzipiert und normalisiert deshalb alle Größen. Für konventionelle Regelungstechnik muss diese Normalisierung explizit berücksichtigt werden.

---

## Validierungsstrategie

Das finale Testsetup verwendet einheitliche Parameter für alle drei Simulationsvarianten:

**Sollwerte:** id,ref = 0 A, iq,ref = 2 A, sofort aktiv

**Drehzahlen:** 500 rpm, 1500 rpm, 2500 rpm zur Abdeckung des drehzahlabhängigen Verhaltens

**Metriken:** MAE und RMSE für Ströme und Spannungen, getrennt nach Transient- und Steady-State-Phase

**Ergebnis:** Der GEM Standard Controller erreicht MAE = 0.0000 A für alle Drehzahlen, was eine exakte Übereinstimmung der Stromregelung mit MATLAB bestätigt.

---

## Zusammenfassung der wichtigsten Learnings

1. **Parameter explizit übergeben:** Motor- und Load-Objekte direkt instantiieren, nicht als kwargs an gem.make()

2. **Limits vom Physical System lesen:** Nicht die eigenen limit_values für Denormalisierung verwenden

3. **Tuple-Handling robust implementieren:** step() und reset() können verschachtelte Tuples zurückgeben

4. **Constraints deaktivieren:** Für Validierung constraints=() setzen

5. **Controller-Reset vermeiden:** Bei Environment-Reset den Controller-Zustand behalten

6. **Drehzahl über Last steuern:** ConstantSpeedLoad für feste Drehzahlen verwenden

7. **GEM-Controller als Black-Box akzeptieren:** Der Standard-Controller funktioniert besser als eine eigene Implementierung

8. **MATLAB-Verdrahtung prüfen:** Variablen müssen tatsächlich in der Simulation ankommen

---

*Letzte Aktualisierung: Dezember 2025*


