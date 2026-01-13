# Edge Impulse Regression: 2 Outputs konfigurieren

## Problem
Edge Impulse zeigt nur **"1 (Scalar value)"** statt **2 Outputs** (`u_d`, `u_q`).

## Lösung

### Schritt 1: Regression Block öffnen
1. Klicken Sie auf den **Regression** Block (lila/blauer Block)
2. Es sollte sich ein Einstellungsfenster öffnen

### Schritt 2: Output Features konfigurieren
Im Regression-Block sollten Sie sehen:
- **Input features:** "Flatten" (von vorherigem Block)
- **Output features:** "1 (Scalar value)" ← Hier ist das Problem!

**Optionen zum Fixen:**

**Option A: Output-Spalten direkt auswählen**
- Suchen Sie nach **"Output columns"** oder **"Label columns"**
- Sollte eine Liste mit Checkboxen zeigen: `timestamp`, `i_d`, `i_q`, `n`, `u_d`, `u_q`
- **Checken Sie:** `u_d` und `u_q`
- **NICHT checken:** `timestamp`, `i_d`, `i_q`, `n`

**Option B: Anzahl der Outputs ändern**
- Suchen Sie nach **"Number of outputs"** oder **"Output count"**
- Ändern Sie von **1** auf **2**
- Dann müssen Sie die Spalten zuweisen:
  - Output 1 → `u_d`
  - Output 2 → `u_q`

**Option C: CSV Transformations verwenden**
1. Gehen Sie zu **Data acquisition** Tab
2. Suchen Sie nach **"CSV Transformations"** oder **"Transform"** Button
3. Wählen Sie Ihre hochgeladene CSV-Datei
4. Im Transformation-Wizard:
   - **Features:** `timestamp`, `i_d`, `i_q`, `n`
   - **Labels:** `u_d`, `u_q`
5. Transformation anwenden
6. Zurück zu **Impulse design** → sollte jetzt 2 Outputs zeigen

### Schritt 3: Verifizieren
Nach der Konfiguration sollte der **Regression** Block zeigen:
- **Output features:** **"2"** (statt "1")
- Oder: `u_d` und `u_q` explizit aufgelistet

### Schritt 4: Save Impulse
- Klicken Sie auf **"Save Impulse"** (grüner Button rechts)
- Die Konfiguration wird gespeichert

## Falls nichts funktioniert

**Alternative: CSV-Struktur ändern**

Edge Impulse erwartet möglicherweise ein bestimmtes Format. Versuchen Sie:

1. **Separate Label-Spalte:** Fügen Sie eine kombinierte Label-Spalte hinzu
2. **Zwei separate Regression-Modelle:** Trainieren Sie zwei separate Modelle (eines für u_d, eines für u_q)
3. **Edge Impulse CLI verwenden:** Mehr Kontrolle über die Konfiguration

## Wichtig
- **Flatten Block:** Nur `i_d`, `i_q`, `n` als Input (✓)
- **Regression Block:** `u_d`, `u_q` als Output (✓)
- **Timestamp:** Kann als Feature verwendet werden oder ignoriert werden
