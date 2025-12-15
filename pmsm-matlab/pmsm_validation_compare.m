%% PMSM Validierungs-Simulation für Python-Vergleich
% ============================================================
% Dieses Skript erzeugt eine Referenzsimulation mit festen
% Parametern, die mit der Python/GEM-Simulation verglichen
% werden kann.
%
% WICHTIG: Die Drehzahl wird auf 716 RPM gesetzt, da sich
%          die GEM-Simulation ohne externe Drehzahlregelung
%          bei diesem Wert einpendelt.
%
% Output: export/validation/validation_sim.csv
% ============================================================

clear; close all; clc;

fprintf('=======================================================\n');
fprintf('PMSM Validierungs-Simulation für Python-Vergleich\n');
fprintf('=======================================================\n\n');

%% ==================== Maschinenparameter ====================
% Diese Parameter MÜSSEN mit simulate_pmsm.py übereinstimmen!

I_nenn = 4.2;                        % Nennstrom [A]
I_max = 10.8;                        % Maximalstrom [A]
V_DC = 48;                           % DC-Spannung [V]
max_modulation_index = 1/sqrt(3);    % Raumzeigermodulation
V_max = V_DC * max_modulation_index; % Maximale Ausgangsspannung
n_nenn = 3000;                       % Nenndrehzahl [RPM]
L_d = 0.00113;                       % d-Achsen Induktivität [H]
L_q = 0.00142;                       % q-Achsen Induktivität [H]
R_s = 0.543;                         % Statorwiderstand [Ohm]
polepairs = 3;                       % Polpaarzahl
Psi_PM = 0.0169;                     % Permanentmagnet-Fluss [Wb]
Ts = 1/10000;                        % Abtastzeit [s] = 100µs

fprintf('Motorparameter:\n');
fprintf('  R_s = %.3f Ohm\n', R_s);
fprintf('  L_d = %.5f H, L_q = %.5f H\n', L_d, L_q);
fprintf('  Psi_PM = %.4f Wb\n', Psi_PM);
fprintf('  Polpaare = %d\n', polepairs);
fprintf('  Ts = %.0f µs\n', Ts * 1e6);
fprintf('\n');

%% ==================== Validierungsparameter ====================
% Diese Werte müssen mit simulate_pmsm.py übereinstimmen!

% Sollströme (Step bei t=0.1s wie in Python)
id_ref = 0.0;    % [A] - d-Achsen Sollstrom (typisch 0 für MTPA)
iq_ref = 2.0;    % [A] - q-Achsen Sollstrom (Drehmoment)

% KRITISCH: Drehzahl auf 716 RPM setzen!
% Dies ist der Gleichgewichtspunkt der GEM-Simulation
n_ref = 716;     % [RPM] - Angepasst an GEM-Ergebnis

fprintf('Validierungsparameter:\n');
fprintf('  id_ref = %.1f A\n', id_ref);
fprintf('  iq_ref = %.1f A\n', iq_ref);
fprintf('  n_ref  = %.0f RPM (GEM Gleichgewichtspunkt)\n', n_ref);
fprintf('\n');

%% ==================== Modell öffnen und simulieren ====================
model = "foc_pmsm";

fprintf('Öffne Simulink-Modell: %s\n', model);
open_system(model);

% Simulationseinstellungen
in = Simulink.SimulationInput(model);
in = setVariable(in, 'setpoint_step_id', id_ref);
in = setVariable(in, 'setpoint_step_iq', iq_ref);
in = setVariable(in, 'setpoint_step_n',  n_ref);

fprintf('Starte Simulation...\n');
tic;
out = sim(in);
sim_time = toc;
fprintf('Simulation abgeschlossen in %.2f Sekunden.\n\n', sim_time);

%% ==================== Daten extrahieren ====================
signals_from_simulink = {'i_d', 'i_q', 'n', 'u_d', 'u_q'};

L = out.logsout;
if isempty(L)
    error('Kein logsout gefunden! Prüfe Simulink-Modell.');
end

try
    avail = getElementNames(L);
catch
    avail = arrayfun(@(k) L{k}.Name, 1:L.numElements, 'UniformOutput', false);
end

fprintf('Verfügbare Signale: %s\n', strjoin(avail, ', '));

% Zeitbasis finden
base_ts = [];
for k = 1:numel(signals_from_simulink)
    if any(strcmp(signals_from_simulink{k}, avail))
        base_ts = getElement(L, signals_from_simulink{k}).Values;
        break
    end
end

if isempty(base_ts)
    error('Keine Zeitbasis gefunden.');
end

% Gleichmäßige Zeitachse
t = (base_ts.Time(1):Ts:base_ts.Time(end)).';
T = table(t, 'VariableNames', {'time'});

fprintf('Zeitbereich: %.4f s bis %.4f s (%d Punkte)\n', t(1), t(end), length(t));

% Signale hinzufügen
for k = 1:numel(signals_from_simulink)
    nm = signals_from_simulink{k};
    if ~any(strcmp(nm, avail))
        warning('Signal "%s" nicht gefunden.', nm);
        continue
    end
    ts = getElement(L, nm).Values;
    if ~isa(ts, 'timeseries')
        warning('Signal "%s" ist keine timeseries.', nm);
        continue
    end
    tsr = resample(ts, t);
    vn = matlab.lang.makeValidName(nm);
    dat = tsr.Data;
    if size(dat, 2) == 1
        T.(vn) = dat;
    else
        for c = 1:size(dat, 2)
            T.([vn '_' num2str(c)]) = dat(:, c);
        end
    end
end

% Referenzwerte hinzufügen
n_rows = height(T);
T.i_d_ref = repmat(id_ref, n_rows, 1);
T.i_q_ref = repmat(iq_ref, n_rows, 1);
T.n_ref   = repmat(n_ref, n_rows, 1);

fprintf('Exportierte Spalten: %s\n', strjoin(T.Properties.VariableNames, ', '));

%% ==================== Export ====================
out_dir = fullfile(pwd, 'export', 'validation');
if ~exist(out_dir, 'dir')
    mkdir(out_dir);
end

filename = fullfile(out_dir, 'validation_sim.csv');
writetable(T, filename);
fprintf('\n✓ Validierungsdaten exportiert: %s\n', filename);

%% ==================== Vorschau-Plots ====================
fprintf('\nErstelle Vorschau-Plots...\n');

figure('Name', 'MATLAB Validierungs-Simulation', 'Position', [100, 100, 1200, 800]);

% Strom id
subplot(3, 2, 1);
plot(T.time * 1000, T.i_d, 'b', 'LineWidth', 1.5);
hold on;
yline(id_ref, 'r--', 'LineWidth', 1);
xlabel('Zeit [ms]');
ylabel('i_d [A]');
title('d-Achsen Strom');
legend('i_d', 'Sollwert');
grid on;

% Strom iq
subplot(3, 2, 2);
plot(T.time * 1000, T.i_q, 'b', 'LineWidth', 1.5);
hold on;
yline(iq_ref, 'r--', 'LineWidth', 1);
xlabel('Zeit [ms]');
ylabel('i_q [A]');
title('q-Achsen Strom');
legend('i_q', 'Sollwert');
grid on;

% Drehzahl
subplot(3, 2, 3);
plot(T.time * 1000, T.n, 'b', 'LineWidth', 1.5);
hold on;
yline(n_ref, 'r--', 'LineWidth', 1);
xlabel('Zeit [ms]');
ylabel('n [RPM]');
title('Drehzahl');
legend('n', 'Sollwert');
grid on;

% Spannung ud
subplot(3, 2, 4);
plot(T.time * 1000, T.u_d, 'b', 'LineWidth', 1.5);
xlabel('Zeit [ms]');
ylabel('u_d [V]');
title('d-Achsen Spannung');
grid on;

% Spannung uq
subplot(3, 2, 5);
plot(T.time * 1000, T.u_q, 'b', 'LineWidth', 1.5);
xlabel('Zeit [ms]');
ylabel('u_q [V]');
title('q-Achsen Spannung');
grid on;

% Phasenportrait
subplot(3, 2, 6);
plot(T.i_d, T.i_q, 'b', 'LineWidth', 1.5);
hold on;
plot(T.i_d(1), T.i_q(1), 'go', 'MarkerSize', 10, 'MarkerFaceColor', 'g');
plot(T.i_d(end), T.i_q(end), 'rs', 'MarkerSize', 10, 'MarkerFaceColor', 'r');
xlabel('i_d [A]');
ylabel('i_q [A]');
title('Phasenportrait i_d vs i_q');
legend('Trajektorie', 'Start', 'Ende');
grid on;
axis equal;

sgtitle(sprintf('MATLAB Validierung: n_{ref}=%d RPM, i_{d,ref}=%.1f A, i_{q,ref}=%.1f A', ...
    n_ref, id_ref, iq_ref), 'FontWeight', 'bold');

% Plot speichern
plot_filename = fullfile(out_dir, 'validation_preview.png');
saveas(gcf, plot_filename);
fprintf('✓ Vorschau-Plot gespeichert: %s\n', plot_filename);

%% ==================== Zusammenfassung ====================
fprintf('\n=======================================================\n');
fprintf('ZUSAMMENFASSUNG\n');
fprintf('=======================================================\n');
fprintf('Drehzahl:     %d RPM (GEM-Gleichgewichtspunkt)\n', n_ref);
fprintf('Sollströme:   id=%.1f A, iq=%.1f A\n', id_ref, iq_ref);
fprintf('Datenpunkte:  %d\n', n_rows);
fprintf('Export:       %s\n', filename);
fprintf('\nNÄCHSTE SCHRITTE:\n');
fprintf('1. Python-Simulation mit gleichen Parametern ausführen:\n');
fprintf('   cd pmsm-pem && python simulate_pmsm.py\n');
fprintf('2. Vergleichsskript ausführen:\n');
fprintf('   python compare_simulations.py\n');
fprintf('=======================================================\n');

