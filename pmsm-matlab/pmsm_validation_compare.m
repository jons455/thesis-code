%% PMSM Validierungs-Simulation für Python-Vergleich (Testsetup mit Drehzahl-Sweep)
% ============================================================
% Dieses Skript erzeugt eine Referenzsimulation mit festen
% Parametern, die mit der Python/GEM-Simulation verglichen
% werden kann.
%
% Testsetup:
% - Fixe Sollwerte: id_ref=0, iq_ref=2 (Step bei t=0.1s)
% - Fixe Drehzahlen: n_ref ∈ {500, 1500, 2500} RPM
% - Pro Drehzahl wird eine Referenz-CSV exportiert
%
% Output: export/validation/validation_sim_nXXXX.csv
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
iq_ref = 3.5;    % [A] - q-Achsen Sollstrom (Drehmoment) - Run 003: erhöht

% Test-Drehzahlen (RPM) für Vergleich MATLAB vs. Python/GEM
% Run 003: Andere Drehzahlen für Validierung
n_ref_list = [750, 1250, 2000];

fprintf('Validierungsparameter:\n');
fprintf('  id_ref = %.1f A\n', id_ref);
fprintf('  iq_ref = %.1f A\n', iq_ref);
fprintf('  n_ref  = [%s] RPM\n', strjoin(string(n_ref_list), ', '));
fprintf('\n');

%% ==================== Modell öffnen und simulieren ====================
model = "foc_pmsm";

fprintf('Öffne Simulink-Modell: %s\n', model);
open_system(model);

% Export-Verzeichnis
out_dir = fullfile(pwd, 'export', 'validation');
if ~exist(out_dir, 'dir')
    mkdir(out_dir);
end

signals_from_simulink = {'i_d', 'i_q', 'n', 'u_d', 'u_q'};

for idx_case = 1:numel(n_ref_list)
    n_ref = n_ref_list(idx_case);
    fprintf('-------------------------------------------------------\n');
    fprintf('Case %d/%d: n_ref = %d RPM\n', idx_case, numel(n_ref_list), n_ref);
    fprintf('-------------------------------------------------------\n');

    % Simulationseinstellungen
    in = Simulink.SimulationInput(model);
    in = setVariable(in, 'setpoint_step_id', id_ref);
    in = setVariable(in, 'setpoint_step_iq', iq_ref);
    in = setVariable(in, 'setpoint_step_n',  n_ref);

    fprintf('Starte Simulation...\n');
    tic;
    out = sim(in);
    sim_time = toc;
    fprintf('Simulation abgeschlossen in %.2f Sekunden.\n', sim_time);

    % ==================== Daten extrahieren ====================
    L = out.logsout;
    if isempty(L)
        error('Kein logsout gefunden! Prüfe Simulink-Modell.');
    end

    try
        avail = getElementNames(L);
    catch
        avail = arrayfun(@(k) L{k}.Name, 1:L.numElements, 'UniformOutput', false);
    end

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

    % ==================== Export ====================
    filename = fullfile(out_dir, sprintf('validation_sim_n%04d.csv', n_ref));
    writetable(T, filename);
    fprintf('✓ Validierungsdaten exportiert: %s\n', filename);

    % Optional: Vorschauplot pro Case
    figure('Name', sprintf('MATLAB Validierung n=%d RPM', n_ref), 'Position', [100, 100, 1200, 800]);
    subplot(3, 2, 1); plot(T.time*1000, T.i_d, 'b', 'LineWidth', 1.2); hold on; yline(id_ref, 'r--'); grid on; title('i_d'); xlabel('ms'); ylabel('A');
    subplot(3, 2, 2); plot(T.time*1000, T.i_q, 'b', 'LineWidth', 1.2); hold on; yline(iq_ref, 'r--'); grid on; title('i_q'); xlabel('ms'); ylabel('A');
    subplot(3, 2, 3); plot(T.time*1000, T.n,   'b', 'LineWidth', 1.2); hold on; yline(n_ref, 'r--'); grid on; title('n');   xlabel('ms'); ylabel('RPM');
    subplot(3, 2, 4); plot(T.time*1000, T.u_d, 'b', 'LineWidth', 1.2); grid on; title('u_d'); xlabel('ms'); ylabel('V');
    subplot(3, 2, 5); plot(T.time*1000, T.u_q, 'b', 'LineWidth', 1.2); grid on; title('u_q'); xlabel('ms'); ylabel('V');
    subplot(3, 2, 6); plot(T.i_d, T.i_q, 'b', 'LineWidth', 1.2); grid on; axis equal; title('i_d vs i_q'); xlabel('A'); ylabel('A');
    sgtitle(sprintf('MATLAB Validierung: n_{ref}=%d RPM, i_{d,ref}=%.1f A, i_{q,ref}=%.1f A', n_ref, id_ref, iq_ref), 'FontWeight', 'bold');

    plot_filename = fullfile(out_dir, sprintf('validation_preview_n%04d.png', n_ref));
    saveas(gcf, plot_filename);
    fprintf('✓ Vorschau-Plot gespeichert: %s\n\n', plot_filename);
end

fprintf('=======================================================\n');
fprintf('ZUSAMMENFASSUNG\n');
fprintf('=======================================================\n');
fprintf('Drehzahlen:   [%s] RPM\n', strjoin(string(n_ref_list), ', '));
fprintf('Sollströme:   id=%.1f A, iq=%.1f A (Step bei t=0.1s)\n', id_ref, iq_ref);
fprintf('Export:       %s\n', out_dir);
fprintf('\nNÄCHSTE SCHRITTE (pro Drehzahl):\n');
fprintf('1. Python-Simulationen ausführen (mit gleicher Drehzahl):\n');
fprintf('   cd pmsm-pem && python simulate_pmsm.py --n-rpm <n>\n');
fprintf('   cd pmsm-pem && python simulate_pmsm_matlab_match.py --n-rpm <n>\n');
fprintf('2. Vergleich fahren (MATLAB CSV auswählen / Pfad im Vergleichsskript setzen)\n');
fprintf('=======================================================\n');

