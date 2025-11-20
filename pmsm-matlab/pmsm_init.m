clear; close all;

% ==================== Maschinenparameter ====================
I_nenn = 4.2;
I_max = 10.8;
V_DC = 48;
max_modulation_index = 1/sqrt(3);
V_max = V_DC * max_modulation_index;  % [V] verfügbare Spannung
n_nenn = 3000;
L_d = 0.00113;
L_q = 0.00142;
R_s = 0.543;
polepairs = 3;
Psi_PM = 0.0169;
Ts = 1/10000;  % Abtastzeit

% ==================== Modell öffnen ====================
model = "foc_pmsm";
open_system(model);

% ==================== Simulationen ====================
number_of_simulations = 2;
ergebnisse = cell(1, number_of_simulations);  % 1×N, nicht N×N!

for i = 1:number_of_simulations
    % Zufällige Setpoints
    id_ref = rand * I_nenn;  
    iq_ref = rand * I_nenn;
    n_ref  = rand * n_nenn;

    % Simulation Input erzeugen
    in = Simulink.SimulationInput(model);
    in = setVariable(in, 'setpoint_step_id', id_ref);
    in = setVariable(in, 'setpoint_step_iq', iq_ref);
    in = setVariable(in, 'setpoint_step_n',  n_ref);

    % Simulation starten
    out = sim(in);
    ergebnisse{i} = out;
end

% ==================== Export der Daten ====================
features = {'i_d','i_q','n'};   % einfache Namen, keine Wildcards nötig
targets  = {'u_d','u_q'};
allnames = [features, targets];

out_dir = fullfile(pwd, 'export', 'train');
if ~exist(out_dir, 'dir'), mkdir(out_dir); end

for s = 1:numel(ergebnisse)
    L = ergebnisse{s}.logsout;  % Dataset

    if isempty(L)
        warning('Simulation %d: Kein logsout gefunden.', s);
        continue;
    end

    % Liste vorhandener Signale
    try
        avail = getElementNames(L);
    catch
        avail = arrayfun(@(k) L{k}.Name, 1:L.numElements, 'UniformOutput', false);
    end

    % Zeitbasis finden
    base_ts = [];
    for k = 1:numel(allnames)
        if any(strcmp(allnames{k}, avail))
            base_ts = getElement(L, allnames{k}).Values;
            break
        end
    end
    if isempty(base_ts)
        if L.numElements > 0 && isa(L{1}.Values, 'timeseries')
            base_ts = L{1}.Values;
        else
            warning('Run %d: Keine timeseries gefunden.', s);
            continue;
        end
    end

    % gleichmäßiges Zeitraster
    t = (base_ts.Time(1):Ts:base_ts.Time(end)).';
    T = table(t, 'VariableNames', {'time'});

    % Signale hinzufügen
    for k = 1:numel(allnames)
        nm = allnames{k};
        if ~any(strcmp(nm, avail)), continue; end
        ts  = getElement(L, nm).Values;
        if ~isa(ts, 'timeseries'), continue; end
        tsr = resample(ts, t);
        vn  = matlab.lang.makeValidName(nm);
        dat = tsr.Data;
        if size(dat, 2) == 1
            T.(vn) = dat;
        else
            for c = 1:size(dat, 2)
                T.([vn '_' num2str(c)]) = dat(:, c);
            end
        end
    end

    % CSV schreiben
    filename = fullfile(out_dir, sprintf('sim_%04d.csv', s));
    writetable(T, filename);
    fprintf('→ Simulation %d exportiert: %s\n', s, filename);
end

disp('Export abgeschlossen.');
