%% PMSM Arbeitspunkt-Variation - Validierung für Python-Vergleich
% ============================================================
% Dieses Skript erzeugt Referenzsimulationen mit verschiedenen
% id/iq-Arbeitspunkten bei fester Drehzahl (1000 rpm).
%
% Testmatrix:
% | Testfall | id [A] | iq [A] | |I| [A] | Beschreibung             |
% |----------|--------|--------|--------|--------------------------|
% | 1        | 0      | 2      | 2.0    | Baseline niedrige Last   |
% | 2        | 0      | 5      | 5.0    | Mittlere Last            |
% | 3        | 0      | 8      | 8.0    | Hohe Last                |
% | 4        | -3     | 2      | 3.6    | Moderate Feldschwächung  |
% | 5        | -3     | 5      | 5.8    | Feldschw. + mittlere Last|
% | 6        | -5     | 5      | 7.1    | Stärkere Feldschw. + Last|
%
% Output: export/validation/validation_op_nXXXX_idXXX_iqXXX.csv
% ============================================================

clear; close all; clc;

fprintf('=======================================================\n');
fprintf('PMSM Arbeitspunkt-Variation für Python-Vergleich\n');
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
fprintf('  I_max = %.1f A\n', I_max);
fprintf('\n');

%% ==================== Testparameter ====================
% Run 003: Andere Drehzahl und andere Arbeitspunkte
n_ref = 1500;  % RPM (vorher 1000)

% Arbeitspunkt-Matrix: [id_ref, iq_ref, Beschreibung]
% Run 003: Andere Kombinationen für breitere Validierung
operating_points = {
    0.0, 1.0, 'very_low_load';        % Sehr geringe Last
    0.0, 3.5, 'mid_low_load';         % Niedrige-mittlere Last  
    0.0, 6.0, 'mid_high_load';        % Mittlere-hohe Last
    -2.0, 3.0, 'fw_light';            % Leichte Feldschwächung
    -4.0, 4.0, 'fw_balanced';         % Balancierte Feldschwächung
    -6.0, 3.0, 'fw_strong_low_torque';% Starke FS, niedriges Drehmoment
};

n_tests = size(operating_points, 1);

fprintf('Testparameter:\n');
fprintf('  Drehzahl: %d rpm (konstant)\n', n_ref);
fprintf('  Anzahl Arbeitspunkte: %d\n', n_tests);
fprintf('\n');

% Validiere Arbeitspunkte
fprintf('Arbeitspunkte:\n');
for idx = 1:n_tests
    id_val = operating_points{idx, 1};
    iq_val = operating_points{idx, 2};
    name = operating_points{idx, 3};
    i_total = sqrt(id_val^2 + iq_val^2);
    
    if i_total > I_max
        status = '✗ ÜBERSCHRITTEN';
    else
        status = '✓';
    end
    
    fprintf('  %s %s: id=%+.1f A, iq=%.1f A, |I|=%.2f A\n', ...
        status, name, id_val, iq_val, i_total);
end
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

% Ergebnistabelle für Zusammenfassung
summary_results = cell(n_tests, 6);  % id_ref, iq_ref, name, id_ss, iq_ss, error

for idx_case = 1:n_tests
    id_ref = operating_points{idx_case, 1};
    iq_ref = operating_points{idx_case, 2};
    op_name = operating_points{idx_case, 3};
    
    fprintf('-------------------------------------------------------\n');
    fprintf('Case %d/%d: id_ref = %+.1f A, iq_ref = %.1f A (%s)\n', ...
        idx_case, n_tests, id_ref, iq_ref, op_name);
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

    % Steady-State Werte berechnen (ab t >= 0.1s)
    ss_mask = T.time >= 0.1;
    id_steady = mean(T.i_d(ss_mask));
    iq_steady = mean(T.i_q(ss_mask));
    id_error = id_steady - id_ref;
    iq_error = iq_steady - iq_ref;
    
    fprintf('Steady-State: id=%.4f A (Δ=%+.4f), iq=%.4f A (Δ=%+.4f)\n', ...
        id_steady, id_error, iq_steady, iq_error);

    % Ergebnis speichern
    summary_results{idx_case, 1} = id_ref;
    summary_results{idx_case, 2} = iq_ref;
    summary_results{idx_case, 3} = op_name;
    summary_results{idx_case, 4} = id_steady;
    summary_results{idx_case, 5} = iq_steady;
    summary_results{idx_case, 6} = sqrt(id_error^2 + iq_error^2);

    % ==================== Export ====================
    % Dateiname mit Vorzeichen-Formatierung
    if id_ref >= 0
        id_str = sprintf('+%02d', round(id_ref));
    else
        id_str = sprintf('%03d', round(id_ref));
    end
    if iq_ref >= 0
        iq_str = sprintf('+%02d', round(iq_ref));
    else
        iq_str = sprintf('%03d', round(iq_ref));
    end
    
    filename = fullfile(out_dir, sprintf('validation_op_n%04d_id%s_iq%s.csv', ...
        n_ref, id_str, iq_str));
    writetable(T, filename);
    fprintf('✓ Daten exportiert: %s\n\n', filename);
end

%% ==================== Zusammenfassung ====================
fprintf('=======================================================\n');
fprintf('ZUSAMMENFASSUNG\n');
fprintf('=======================================================\n');
fprintf('\nSteady-State Ergebnisse bei %d rpm:\n', n_ref);
fprintf('%-25s %8s %8s %10s %10s %10s\n', ...
    'Arbeitspunkt', 'id_ref', 'iq_ref', 'id_ss', 'iq_ss', '|Fehler|');
fprintf('%s\n', repmat('-', 1, 75));

for idx = 1:n_tests
    fprintf('%-25s %+8.1f %+8.1f %+10.4f %+10.4f %10.4f\n', ...
        summary_results{idx, 3}, ...  % name
        summary_results{idx, 1}, ...  % id_ref
        summary_results{idx, 2}, ...  % iq_ref
        summary_results{idx, 4}, ...  % id_ss
        summary_results{idx, 5}, ...  % iq_ss
        summary_results{idx, 6});     % error
end

fprintf('\n');
fprintf('Export-Verzeichnis: %s\n', out_dir);
fprintf('\n');
fprintf('NÄCHSTE SCHRITTE:\n');
fprintf('1. Python-Simulationen ausführen:\n');
fprintf('   cd pmsm-pem && .\\venv\\Scripts\\activate\n');
fprintf('   python simulation/run_operating_point_tests.py\n');
fprintf('2. Vergleich:\n');
fprintf('   python validation/compare_operating_points.py\n');
fprintf('=======================================================\n');

%% ==================== Visualisierung ====================
figure('Name', 'Arbeitspunkt-Übersicht', 'Position', [100, 100, 1400, 900]);

% Subplot 1: id/iq Arbeitspunkte
subplot(2, 3, 1);
hold on;
for idx = 1:n_tests
    id_ref = summary_results{idx, 1};
    iq_ref = summary_results{idx, 2};
    err = summary_results{idx, 6};
    
    % Farbe basierend auf Fehler
    if err < 0.01
        c = [0, 0.7, 0];  % Grün
    elseif err < 0.1
        c = [1, 0.7, 0];  % Orange
    else
        c = [1, 0, 0];    % Rot
    end
    
    scatter(id_ref, iq_ref, 150, c, 'filled', 'MarkerEdgeColor', 'k');
    text(id_ref + 0.3, iq_ref, sprintf('%.3f', err), 'FontSize', 8);
end
xlabel('id_{ref} [A]');
ylabel('iq_{ref} [A]');
title('Arbeitspunkte mit Tracking-Fehler');
grid on;
xlim([-7, 1]);
ylim([0, 10]);

% Stromlimit-Kreis
theta = linspace(0, 2*pi, 100);
plot(I_max * cos(theta), I_max * sin(theta), 'r--', 'LineWidth', 1);
legend('Testpunkte', 'I_{max}', 'Location', 'southwest');

% Subplot 2-6: Zeitverläufe (für jeden Arbeitspunkt)
for idx = 1:min(5, n_tests)
    subplot(2, 3, idx + 1);
    
    id_ref = operating_points{idx, 1};
    iq_ref = operating_points{idx, 2};
    
    % Datei laden
    if id_ref >= 0
        id_str = sprintf('+%02d', round(id_ref));
    else
        id_str = sprintf('%03d', round(id_ref));
    end
    if iq_ref >= 0
        iq_str = sprintf('+%02d', round(iq_ref));
    else
        iq_str = sprintf('%03d', round(iq_ref));
    end
    
    filename = fullfile(out_dir, sprintf('validation_op_n%04d_id%s_iq%s.csv', ...
        n_ref, id_str, iq_str));
    
    if exist(filename, 'file')
        T_plot = readtable(filename);
        
        yyaxis left;
        plot(T_plot.time * 1000, T_plot.i_d, 'b-', 'LineWidth', 1);
        hold on;
        yline(id_ref, 'b--', 'LineWidth', 0.5);
        ylabel('i_d [A]');
        
        yyaxis right;
        plot(T_plot.time * 1000, T_plot.i_q, 'r-', 'LineWidth', 1);
        yline(iq_ref, 'r--', 'LineWidth', 0.5);
        ylabel('i_q [A]');
        
        xlabel('Zeit [ms]');
        title(sprintf('id=%+.0f, iq=%.0f', id_ref, iq_ref));
        grid on;
    end
end

sgtitle(sprintf('MATLAB Arbeitspunkt-Variation @ %d rpm', n_ref), 'FontWeight', 'bold');

% Plot speichern
plot_filename = fullfile(out_dir, sprintf('operating_points_overview_n%04d.png', n_ref));
saveas(gcf, plot_filename);
fprintf('✓ Übersichts-Plot gespeichert: %s\n', plot_filename);

