%% ============================================================
%                 SMART-GRID SHIELD
%             TinyML Decision Tree Training
%
% NOTE:
% Dataset is synthetic/generated data.
% Accuracy is NOT field-validation accuracy.
%% ============================================================

clc;
clear;
close all;

fprintf('\n');
fprintf('====================================================\n');
fprintf('             SMART-GRID SHIELD - TinyML\n');
fprintf('====================================================\n\n');


%% ============================================================
% 1. LOAD DATASET
%% ============================================================

filename = 'Smart_Grid_Shield_All_Fault_Analysis_Data.xlsx';

if ~isfile(filename)
    error(['Dataset not found: ', filename, ...
        newline, ...
        'Place the Excel file in the MATLAB Current Folder.']);
end

T = readtable(filename);

fprintf('Dataset loaded: %d rows\n', height(T));


%% ============================================================
% 2. DISPLAY DATASET COLUMNS
%% ============================================================

fprintf('\nAvailable dataset columns:\n');

disp(T.Properties.VariableNames');


%% ============================================================
% 3. REMOVE SENSOR ERROR ROWS
%% ============================================================

if ismember('Status', T.Properties.VariableNames)

    T = T(~strcmp(string(T.Status), "SENSOR_ERROR"), :);

    fprintf('Rows after sensor-error removal: %d\n', height(T));

else

    warning('Status column not found.');

end


%% ============================================================
% 4. DEFINE INPUT FEATURES
%% ============================================================

features = { ...
    'PV_Temp_C', ...
    'Ambient_Temp_C', ...
    'DeltaT_C', ...
    'LDR_ADC', ...
    'PV_Voltage_V'};


%% ============================================================
% 5. CHECK REQUIRED COLUMNS
%% ============================================================

for i = 1:length(features)

    if ~ismember(features{i}, T.Properties.VariableNames)

        error(['Required feature column missing: ', features{i}]);

    end

end


if ~ismember('Condition', T.Properties.VariableNames)

    error('Condition column is missing from the dataset.');

end


%% ============================================================
% 6. CREATE FEATURE MATRIX AND TARGET
%% ============================================================

X = T{:, features};

Y = categorical(T.Condition);


%% ============================================================
% 7. REMOVE INVALID DATA
%% ============================================================

valid = all(isfinite(X), 2) & ~isundefined(Y);

X = X(valid, :);
Y = Y(valid);

fprintf('Valid ML samples: %d\n', size(X,1));


%% ============================================================
% 8. DISPLAY CLASS DISTRIBUTION
%% ============================================================

fprintf('\n====================================================\n');
fprintf('              CLASS DISTRIBUTION\n');
fprintf('====================================================\n');

classes = categories(Y);

for i = 1:length(classes)

    count = sum(Y == classes{i});

    fprintf('%-25s : %d\n', classes{i}, count);

end


%% ============================================================
% 9. TRAIN / TEST SPLIT
%% ============================================================

rng(1);

cv = cvpartition(Y, 'HoldOut', 0.20);

Xtr = X(training(cv), :);
Ytr = Y(training(cv));

Xte = X(test(cv), :);
Yte = Y(test(cv));


fprintf('\n====================================================\n');
fprintf('              DATA SPLIT\n');
fprintf('====================================================\n');

fprintf('Training samples : %d\n', size(Xtr,1));
fprintf('Testing samples  : %d\n', size(Xte,1));


%% ============================================================
% 10. TRAIN TINYML DECISION TREE
%% ============================================================

tree = fitctree( ...
    Xtr, ...
    Ytr, ...
    'MaxNumSplits', 12, ...
    'MinLeafSize', 5);


%% ============================================================
% 11. PREDICTIONS
%% ============================================================

pTr = predict(tree, Xtr);

pTe = predict(tree, Xte);


%% ============================================================
% 12. CALCULATE ACCURACY
%% ============================================================

trainingAccuracy = mean(pTr == Ytr) * 100;

testingAccuracy = mean(pTe == Yte) * 100;


fprintf('\n====================================================\n');
fprintf('                 MODEL RESULTS\n');
fprintf('====================================================\n');

fprintf('Training Accuracy = %.2f %%\n', trainingAccuracy);

fprintf('Testing Accuracy  = %.2f %%\n', testingAccuracy);


%% ============================================================
% 13. FIGURE 1 - CONFUSION MATRIX
%% ============================================================

figure(1);
clf;

cm = confusionchart(Yte, pTe);

cm.Title = 'TinyML Decision Tree Confusion Matrix';

cm.RowSummary = 'row-normalized';

cm.ColumnSummary = 'column-normalized';


%% ============================================================
% 14. FIGURE 2 - DECISION TREE
%% ============================================================

figure(2);
clf;

% Graphical tree display
view(tree, 'Mode', 'graph');


%% ============================================================
% 15. FEATURE IMPORTANCE
%% ============================================================

importance = predictorImportance(tree);


fprintf('\n====================================================\n');
fprintf('              FEATURE IMPORTANCE\n');
fprintf('====================================================\n');

importanceTable = table( ...
    features', ...
    importance', ...
    'VariableNames', ...
    {'Feature', 'Importance'});

disp(importanceTable);


%% ============================================================
% 16. FIGURE 3 - FEATURE IMPORTANCE
%% ============================================================

figure(3);
clf;

bar(importance);

set(gca, ...
    'XTick', 1:length(features), ...
    'XTickLabel', features);

xtickangle(45);

ylabel('Importance');

title('TinyML Decision Tree Feature Importance');

grid on;


%% ============================================================
% 17. TREE INFORMATION
%% ============================================================

fprintf('\n====================================================\n');
fprintf('               TREE INFORMATION\n');
fprintf('====================================================\n');

fprintf('Number of Tree Nodes : %d\n', tree.NumNodes);

% Find leaf nodes
children = tree.Children;

leafNodes = all(children == 0, 2);

fprintf('Number of Leaves     : %d\n', sum(leafNodes));


%% ============================================================
% 18. DISPLAY TREE RULES IN COMMAND WINDOW
%% ============================================================

fprintf('\n====================================================\n');
fprintf('              DECISION TREE RULES\n');
fprintf('====================================================\n\n');

view(tree);


%% ============================================================
% 19. LIVE TEST SAMPLE
%
% Format:
% [PV Temperature, Ambient Temperature,
%  Delta T, LDR, PV Voltage]
%% ============================================================

liveSample = [33, 31.5, 1.5, 450, 5.8];

livePrediction = predict(tree, liveSample);


fprintf('\n====================================================\n');
fprintf('                LIVE TEST SAMPLE\n');
fprintf('====================================================\n');

fprintf('PV Temperature : %.2f C\n', liveSample(1));

fprintf('Ambient Temp   : %.2f C\n', liveSample(2));

fprintf('Delta T        : %.2f C\n', liveSample(3));

fprintf('LDR            : %.0f\n', liveSample(4));

fprintf('PV Voltage     : %.2f V\n', liveSample(5));

fprintf('\nTinyML Prediction : %s\n', string(livePrediction));


%% ============================================================
% 20. SAVE TRAINED MODEL
%% ============================================================

save( ...
    'SmartGridShield_TinyML_DecisionTree.mat', ...
    'tree', ...
    'features');


fprintf('\n====================================================\n');
fprintf('                 MODEL SAVED\n');
fprintf('====================================================\n');

fprintf('SmartGridShield_TinyML_DecisionTree.mat\n');


%% ============================================================
% 21. FINAL MESSAGE
%% ============================================================

fprintf('\n====================================================\n');
fprintf('             TRAINING COMPLETE\n');
fprintf('====================================================\n');

fprintf('\nFigures created:\n');
fprintf('Figure 1 -> Confusion Matrix\n');
fprintf('Figure 2 -> TinyML Decision Tree\n');
fprintf('Figure 3 -> Feature Importance\n');

fprintf('\nIMPORTANT:\n');
fprintf('Accuracy is based on the generated/synthetic dataset.\n');
fprintf('Real-world field validation is still required.\n');

fprintf('\n====================================================\n');