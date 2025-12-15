clear; close all;

% ==================== Maschinenparameter ====================
I_nenn = 4.2;
I_max = 10.8;
V_DC = 48;
max_modulation_index = 1/sqrt(3);
V_max = V_DC * max_modulation_index;
n_nenn = 3000;
L_d = 0.00113;
L_q = 0.00142;
R_s = 0.543;
polepairs = 3;
Psi_PM = 0.0169;
Ts = 1/10000;

% ==================== Modell öffnen ====================
model = "foc_pmsm";
open_system(model);

% ==================== Simulationen ====================
number_of_simulations = 1;  
ergebnisse = cell(1, number_of_simulations);
setpoints = zeros(number_of_simulations, 3);

for i = 1:number_of_simulations
%    id_ref = rand * I_nenn;  
%    iq_ref = rand * I_nenn;
%    n_ref  = rand * n_nenn;
% Feste Test-Werte:
id_ref = 0;      % [A] - typisch 0 für MTPA
iq_ref = 2.0;    % [A] - Drehmoment-Strom
n_ref  = 716;   % [RPM] - mittlere Drehzahl
    setpoints(i, :) = [id_ref, iq_ref, n_ref];

    in = Simulink.SimulationInput(model);
    in = setVariable(in, 'setpoint_step_id', id_ref);
    in = setVariable(in, 'setpoint_step_iq', iq_ref);
    in = setVariable(in, 'setpoint_step_n',  n_ref);

    out = sim(in);
    ergebnisse{i} = out;
end

% ==================== Export der Daten ====================
signals_from_simulink = {'i_d','i_q','n','u_d','u_q'};  % ← Nur echte Simulink-Signale

out_dir = fullfile(pwd, 'export', 'train');
if ~exist(out_dir, 'dir'), mkdir(out_dir); end

for s = 1:numel(ergebnisse)
    id_ref_val = setpoints(s, 1);
    iq_ref_val = setpoints(s, 2);

    L = ergebnisse{s}.logsout;
    if isempty(L)
        warning('Simulation %d: Kein logsout gefunden.', s);
        continue;
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
        warning('Run %d: Keine timeseries gefunden.', s);
        continue;
    end

    t = (base_ts.Time(1):Ts:base_ts.Time(end)).';
    T = table(t, 'VariableNames', {'time'});

    % Simulink-Signale hinzufügen
    for k = 1:numel(signals_from_simulink)
        nm = signals_from_simulink{k};
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

    % Referenzwerte manuell hinzufügen
    n_rows = height(T);
    T.i_d_ref = repmat(id_ref_val, n_rows, 1);
    T.i_q_ref = repmat(iq_ref_val, n_rows, 1);

    filename = fullfile(out_dir, sprintf('sim_%04d.csv', s));
    writetable(T, filename);
    fprintf('→ Simulation %d exportiert: %s\n', s, filename);
end

disp('Export abgeschlossen.');