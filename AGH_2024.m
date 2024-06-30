filename = 'C:\Users\Paweł\Desktop\data.xlsx';
opts = detectImportOptions(filename);
opts.VariableNamingRule = 'preserve';
data = readtable(filename, opts);

disp('Dostępne kolumny:');
disp(data.Properties.VariableNames);

data.Properties.VariableNames = {'Czas', 'P_in', 'P_out', 'VFlow', 'T_in', 'T_out', 'T_after_Heating', 'Q_Heating', 'Q_generated'};

if ismember('Czas', data.Properties.VariableNames)
    data.Czas = [];
end

varNames = data.Properties.VariableNames;
for i = 1:numel(varNames)
    if iscell(data.(varNames{i}))
        data.(varNames{i}) = str2double(strrep(data.(varNames{i}), ',', '.'));
    end
end

data = rmmissing(data);

for i = 1:numel(varNames)
    if ~ismember(varNames{i}, {'Q_Heating', 'Q_generated'})
        outliers = isoutlier(data.(varNames{i}));
        data(outliers, :) = [];
    end
end

scaler_X = struct();
scaler_y = struct();
input_names = {'P_in', 'P_out', 'VFlow', 'T_in', 'T_out'};
output_names = {'Q_Heating', 'Q_generated'};

for i = 1:numel(input_names)
    scaler_X.(input_names{i}).min = min(data.(input_names{i}));
    scaler_X.(input_names{i}).max = max(data.(input_names{i}));
    data.(input_names{i}) = (data.(input_names{i}) - scaler_X.(input_names{i}).min) / (scaler_X.(input_names{i}).max - scaler_X.(input_names{i}).min);
end

for i = 1:numel(output_names)
    scaler_y.(output_names{i}).min = min(data.(output_names{i}));
    scaler_y.(output_names{i}).max = max(data.(output_names{i}));
    data.(output_names{i}) = (data.(output_names{i}) - scaler_y.(output_names{i}).min) / (scaler_y.(output_names{i}).max - scaler_y.(output_names{i}).min);
end

X = data{:, input_names};
y = data{:, output_names};

[X_train, X_test, y_train, y_test] = train_test_split(X, y, 0.8);

hiddenLayerSizes = {[10], [20], [10, 10], [20, 20]};
learningRates = [0.01, 0.05, 0.1];
epochs = [1000, 2000, 3000];

best_rmse = inf;
best_net = [];

for i = 1:length(hiddenLayerSizes)
    for j = 1:length(learningRates)
        for k = 1:length(epochs)
            net = fitnet(hiddenLayerSizes{i});
            net.trainParam.lr = learningRates(j);
            net.trainParam.epochs = epochs(k);
            [net, tr] = train(net, X_train', y_train');
            y_pred = net(X_test');
            rmse = sqrt(mean((y_test' - y_pred).^2));
            if rmse < best_rmse
                best_rmse = rmse;
                best_net = net;
            end
            disp(['Hidden Layers: ', num2str(hiddenLayerSizes{i}), ', Learning Rate: ', num2str(learningRates(j)), ', Epochs: ', num2str(epochs(k)), ', RMSE: ', num2str(rmse)]);
        end
    end
end

disp(['Best RMSE: ', num2str(best_rmse)]);

P_in = input("Podaj wartosc P_in: ");
P_out = input("Podaj wartosc P_out: ");
VFlow = input("Podaj wartosc VFlow: ");
T_in = input("Podaj wartosc T_in: ");
T_out = input("Podaj wartosc T_out: ");

predicted_values = predict_energy(best_net, scaler_X, scaler_y, P_in, P_out, VFlow, T_in, T_out);
disp(['Przewidywane wartosci Q_Heating i Q_generated: ', num2str(predicted_values)]);

while true
    end_program = input("Czy zakonczyc dzialanie programu? (yes/no): ", 's');
    if strcmpi(end_program, 'yes')
        disp("Program zakonczony.");
        break;
    elseif strcmpi(end_program, 'no')
        P_in = input("Podaj wartosc P_in: ");
        P_out = input("Podaj wartosc P_out: ");
        VFlow = input("Podaj wartosc VFlow: ");
        T_in = input("Podaj wartosc T_in: ");
        T_out = input("Podaj wartosc T_out: ");
        predicted_values = predict_energy(best_net, scaler_X, scaler_y, P_in, P_out, VFlow, T_in, T_out);
        disp(['Przewidywane wartosci Q_Heating i Q_generated: ', num2str(predicted_values)]);
    else
        disp("Niepoprawna odpowiedz. Wpisz 'yes' lub 'no'.");
    end
end

function predicted_values = predict_energy(net, scaler_X, scaler_y, P_in, P_out, VFlow, T_in, T_out)
    user_input = [P_in, P_out, VFlow, T_in, T_out];
    input_names = {'P_in', 'P_out', 'VFlow', 'T_in', 'T_out'};
    for i = 1:length(user_input)
        user_input(i) = (user_input(i) - scaler_X.(input_names{i}).min) / (scaler_X.(input_names{i}).max - scaler_X.(input_names{i}).min);
    end
    
    prediction_scaled = net(user_input');
    prediction = zeros(1, 2);
    prediction(1) = prediction_scaled(1) * (scaler_y.Q_Heating.max - scaler_y.Q_Heating.min) + scaler_y.Q_Heating.min;
    prediction(2) = prediction_scaled(2) * (scaler_y.Q_generated.max - scaler_y.Q_generated.min) + scaler_y.Q_generated.min;
    predicted_values = prediction;
end

function [X_train, X_test, y_train, y_test] = train_test_split(X, y, train_size)
    num_train = round(train_size * size(X, 1));
    idx = randperm(size(X, 1));
    X_train = X(idx(1:num_train), :);
    X_test = X(idx(num_train+1:end), :);
    y_train = y(idx(1:num_train), :);
    y_test = y(idx(num_train+1:end), :);
end
