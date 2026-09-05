%% ============================================================
%                  SMART-GRID SHIELD
%              TinyML SVM TRAINING
%
%  Multiclass Linear SVM + ECOC
%
%  Dataset:
%  Smart_Grid_Shield_Realistic_Overlapping_85_90_Dataset-3.xlsx
%
%  NOTE:
%  Dataset contains derived/synthetic samples based on
%  field-level observations.
%
%  Accuracy is NOT field-validation accuracy.
%% ============================================================

clc;
clear;
close all;

fprintf('\n');
fprintf('====================================================\n');
fprintf('          SMART-GRID SHIELD - TinyML SVM\n');
fprintf('====================================================\n\n');


%% ============================================================
% 1. LOAD DATASET
%% ============================================================

filename = 'Smart_Grid_Shield_Realistic_Overlapping_85_90_Dataset-3.xlsx';

if ~isfile(filename)

    error(['Dataset not found: ', filename, ...
        newline, ...
        'Place the Excel file in the MATLAB Current Folder.']);

end

T = readtable(filename);

fprintf('Dataset loaded successfully.\n');
fprintf('Total rows: %d\n', height(T));


%% ============================================================
% 2. DISPLAY DATASET COLUMNS
%% ============================================================

fprintf('\n====================================================\n');
fprintf('              DATASET COLUMNS\n');
fprintf('====================================================\n');

disp(T.Properties.VariableNames');


%% ============================================================
% 3. REMOVE SENSOR ERROR ROWS IF STATUS EXISTS
%% ============================================================

if ismember('Status', T.Properties.VariableNames)

    beforeRows = height(T);

    T = T(~strcmp(string(T.Status), "SENSOR_ERROR"), :);

    afterRows = height(T);

    fprintf('\nSensor-error rows removed: %d\n', ...
        beforeRows - afterRows);

else

    fprintf('\nStatus column not found.\n');
    fprintf('No sensor-error filtering applied.\n');

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

fprintf('\n====================================================\n');
fprintf('          CHECKING REQUIRED COLUMNS\n');
fprintf('====================================================\n');

for i = 1:length(features)

    if ~ismember(features{i}, T.Properties.VariableNames)

        error(['Required feature column missing: ', ...
            features{i}]);

    end

    fprintf('Found: %s\n', features{i});

end


if ~ismember('Condition', T.Properties.VariableNames)

    error('Condition column is missing from the dataset.');

end

fprintf('Found: Condition\n');


%% ============================================================
% 6. CREATE FEATURE MATRIX AND TARGET
%% ============================================================

X = T{:, features};

Y = categorical(T.Condition);


fprintf('\nFeature matrix size: %d x %d\n', ...
    size(X,1), size(X,2));

fprintf('Target classes: %d\n', numel(categories(Y)));


%% ============================================================
% 7. REMOVE INVALID DATA
%% ============================================================

validRows = ...
    all(isfinite(X), 2) & ...
    ~isundefined(Y);

X = X(validRows, :);
Y = Y(validRows);

fprintf('\nValid ML samples: %d\n', size(X,1));


%% ============================================================
% 8. DISPLAY CLASS DISTRIBUTION
%% ============================================================

fprintf('\n====================================================\n');
fprintf('              CLASS DISTRIBUTION\n');
fprintf('====================================================\n');

classes = categories(Y);

for i = 1:length(classes)

    count = sum(Y == classes{i});

    fprintf('%-25s : %d samples\n', ...
        classes{i}, count);

end


%% ============================================================
% 9. TRAIN / TEST SPLIT
%% ============================================================

rng(1);

cvHoldout = cvpartition(Y, 'HoldOut', 0.20);

trainingRows = training(cvHoldout);
testingRows  = test(cvHoldout);

Xtr = X(trainingRows, :);
Ytr = Y(trainingRows);

Xte = X(testingRows, :);
Yte = Y(testingRows);


fprintf('\n====================================================\n');
fprintf('                DATA SPLIT\n');
fprintf('====================================================\n');

fprintf('Total samples    : %d\n', size(X,1));

fprintf('Training samples  : %d\n', size(Xtr,1));

fprintf('Testing samples   : %d\n', size(Xte,1));

fprintf('Training ratio    : 80%%\n');

fprintf('Testing ratio     : 20%%\n');


%% ============================================================
% 10. STANDARDIZE FEATURES
%
% IMPORTANT:
%
% Mean and standard deviation are calculated ONLY from
% training data.
%
% This prevents data leakage.
%% ============================================================

mu = mean(Xtr, 1);

sigma = std(Xtr, 0, 1);

% Prevent division by zero

sigma(sigma == 0) = 1;


XtrStd = ...
    (Xtr - mu) ./ sigma;

XteStd = ...
    (Xte - mu) ./ sigma;


fprintf('\nFeature standardization completed.\n');


%% ============================================================
% 11. TRAIN MULTICLASS LINEAR SVM
%% ============================================================

fprintf('\n====================================================\n');
fprintf('              TRAINING LINEAR SVM\n');
fprintf('====================================================\n');

% Linear SVM learner

template = templateSVM( ...
    'KernelFunction', 'linear', ...
    'Standardize', false);


% ECOC converts multiple binary SVM classifiers
% into a multiclass classifier.

svmModel = fitcecoc( ...
    XtrStd, ...
    Ytr, ...
    'Learners', template);


fprintf('Linear SVM training completed.\n');


%% ============================================================
% 12. TRAINING PREDICTIONS
%% ============================================================

pTr = predict( ...
    svmModel, ...
    XtrStd);


%% ============================================================
% 13. TESTING PREDICTIONS
%% ============================================================

pTe = predict( ...
    svmModel, ...
    XteStd);


%% ============================================================
% 14. CALCULATE ACCURACY
%% ============================================================

trainingAccuracy = ...
    mean(pTr == Ytr) * 100;

testingAccuracy = ...
    mean(pTe == Yte) * 100;


fprintf('\n====================================================\n');
fprintf('                 SVM ACCURACY\n');
fprintf('====================================================\n');

fprintf('Training Accuracy = %.2f %%\n', ...
    trainingAccuracy);

fprintf('Testing Accuracy  = %.2f %%\n', ...
    testingAccuracy);


%% ============================================================
% 15. CALCULATE PRECISION / RECALL / F1
%% ============================================================

classes = categories(Yte);

nClasses = numel(classes);

precision = zeros(nClasses,1);

recall = zeros(nClasses,1);

f1 = zeros(nClasses,1);

support = zeros(nClasses,1);


for k = 1:nClasses

    currentClass = classes{k};

    TP = sum( ...
        (Yte == currentClass) & ...
        (pTe == currentClass));

    FP = sum( ...
        (Yte ~= currentClass) & ...
        (pTe == currentClass));

    FN = sum( ...
        (Yte == currentClass) & ...
        (pTe ~= currentClass));


    % Number of actual samples

    support(k) = sum(Yte == currentClass);


    % Precision

    if (TP + FP) == 0

        precision(k) = 0;

    else

        precision(k) = ...
            TP / (TP + FP);

    end


    % Recall

    if (TP + FN) == 0

        recall(k) = 0;

    else

        recall(k) = ...
            TP / (TP + FN);

    end


    % F1

    if (precision(k) + recall(k)) == 0

        f1(k) = 0;

    else

        f1(k) = ...
            2 * precision(k) * recall(k) / ...
            (precision(k) + recall(k));

    end

end


% Convert to percentage

precisionMacro = mean(precision) * 100;

recallMacro = mean(recall) * 100;

f1Macro = mean(f1) * 100;


%% ============================================================
% 16. DISPLAY OVERALL PERFORMANCE
%% ============================================================

fprintf('\n====================================================\n');
fprintf('             OVERALL SVM PERFORMANCE\n');
fprintf('====================================================\n');

fprintf('Accuracy       : %.2f %%\n', ...
    testingAccuracy);

fprintf('Macro Precision: %.2f %%\n', ...
    precisionMacro);

fprintf('Macro Recall   : %.2f %%\n', ...
    recallMacro);

fprintf('Macro F1 Score : %.2f %%\n', ...
    f1Macro);


%% ============================================================
% 17. DISPLAY PER-CLASS PERFORMANCE
%% ============================================================

fprintf('\n====================================================\n');
fprintf('             PER-CLASS PERFORMANCE\n');
fprintf('====================================================\n');

fprintf('\n');

fprintf('%-25s %-12s %-12s %-12s %-10s\n', ...
    'Class', ...
    'Precision', ...
    'Recall', ...
    'F1', ...
    'Samples');

fprintf('----------------------------------------------------\n');

for k = 1:nClasses

    fprintf('%-25s %8.2f %%   %8.2f %%   %8.2f %%   %5d\n', ...
        classes{k}, ...
        precision(k)*100, ...
        recall(k)*100, ...
        f1(k)*100, ...
        support(k));

end


%% ============================================================
% 18. FIGURE 1 - CONFUSION MATRIX
%% ============================================================

figure(1);

clf;

cm = confusionchart( ...
    Yte, ...
    pTe);

cm.Title = ...
    'Smart-Grid Shield - Linear SVM Confusion Matrix';

cm.RowSummary = ...
    'row-normalized';

cm.ColumnSummary = ...
    'column-normalized';


%% ============================================================
% 19. FIGURE 2 - PERFORMANCE COMPARISON
%
% Vertical bars:
% Accuracy
% Precision
% Recall
% F1 Score
%
% Percentage values displayed inside bars.
%% ============================================================

figure(2);

clf;


performanceData = [ ...
    testingAccuracy, ...
    precisionMacro, ...
    recallMacro, ...
    f1Macro];


b = bar( ...
    performanceData, ...
    'FaceColor', 'flat');


ylim([0 100]);

ylabel('Performance (%)');

title( ...
    'Smart-Grid Shield - SVM Performance');


xticklabels({ ...
    'Accuracy', ...
    'Precision', ...
    'Recall', ...
    'F1 Score'});


grid on;


%% ============================================================
% 20. DISPLAY VALUES INSIDE BARS
%% ============================================================

for j = 1:length(performanceData)

    value = performanceData(j);

    xPosition = b.XEndPoints(j);

    text( ...
        xPosition, ...
        value/2, ...
        sprintf('%.1f%%', value), ...
        ...
        'HorizontalAlignment', ...
        'center', ...
        ...
        'VerticalAlignment', ...
        'middle', ...
        ...
        'FontWeight', ...
        'bold', ...
        ...
        'FontSize', ...
        10, ...
        ...
        'Color', ...
        'white', ...
        ...
        'Rotation', ...
        90);

end


%% ============================================================
% 21. 5-FOLD CROSS-VALIDATION
%% ============================================================

fprintf('\n====================================================\n');
fprintf('            5-FOLD CROSS-VALIDATION\n');
fprintf('====================================================\n');

rng(1);

cv5 = cvpartition(Y, 'KFold', 5);

cvAccuracy = zeros(5,1);

for fold = 1:5

    trainIdx = training(cv5, fold);

    testIdx = test(cv5, fold);


    XtrainCV = X(trainIdx, :);

    YtrainCV = Y(trainIdx);


    XtestCV = X(testIdx, :);

    YtestCV = Y(testIdx);


    % Standardization using ONLY fold training data

    muCV = mean(XtrainCV, 1);

    sigmaCV = std(XtrainCV, 0, 1);

    sigmaCV(sigmaCV == 0) = 1;


    XtrainCVStd = ...
        (XtrainCV - muCV) ./ sigmaCV;

    XtestCVStd = ...
        (XtestCV - muCV) ./ sigmaCV;


    % Train SVM

    svmCV = fitcecoc( ...
        XtrainCVStd, ...
        YtrainCV, ...
        'Learners', template);


    % Predict

    predictionCV = predict( ...
        svmCV, ...
        XtestCVStd);


    % Accuracy

    cvAccuracy(fold) = ...
        mean(predictionCV == YtestCV) * 100;


    fprintf( ...
        'Fold %d Accuracy = %.2f %%\n', ...
        fold, ...
        cvAccuracy(fold));

end


meanCVAccuracy = mean(cvAccuracy);

stdCVAccuracy = std(cvAccuracy);


fprintf('\n');

fprintf( ...
    'Mean 5-Fold CV Accuracy = %.2f %%\n', ...
    meanCVAccuracy);

fprintf( ...
    'CV Standard Deviation   = %.2f %%\n', ...
    stdCVAccuracy);


%% ============================================================
% 22. LIVE TEST SAMPLE
%
% Format:
%
% [PV Temperature,
%  Ambient Temperature,
%  Delta T,
%  LDR,
%  PV Voltage]
%% ============================================================

liveSample = [ ...
    33, ...
    31.5, ...
    1.5, ...
    450, ...
    5.8];


%% ============================================================
% 23. STANDARDIZE LIVE SAMPLE
%
% Use the SAME training mean and standard deviation.
%% ============================================================

liveSampleStd = ...
    (liveSample - mu) ./ sigma;


%% ============================================================
% 24. SVM LIVE PREDICTION
%% ============================================================

livePrediction = predict( ...
    svmModel, ...
    liveSampleStd);


fprintf('\n====================================================\n');
fprintf('                LIVE TEST SAMPLE\n');
fprintf('====================================================\n');

fprintf( ...
    'PV Temperature : %.2f C\n', ...
    liveSample(1));

fprintf( ...
    'Ambient Temp   : %.2f C\n', ...
    liveSample(2));

fprintf( ...
    'Delta T        : %.2f C\n', ...
    liveSample(3));

fprintf( ...
    'LDR            : %.0f\n', ...
    liveSample(4));

fprintf( ...
    'PV Voltage     : %.2f V\n', ...
    liveSample(5));


fprintf('\n');

fprintf( ...
    'SVM Prediction : %s\n', ...
    string(livePrediction));


%% ============================================================
% 25. CREATE RESULTS TABLE
%% ============================================================

resultsTable = table( ...
    classes, ...
    precision*100, ...
    recall*100, ...
    f1*100, ...
    support, ...
    'VariableNames', ...
    { ...
    'Condition', ...
    'Precision_Percent', ...
    'Recall_Percent', ...
    'F1_Percent', ...
    'Support'});


%% ============================================================
% 26. CREATE OVERALL RESULTS TABLE
%% ============================================================

overallResults = table( ...
    trainingAccuracy, ...
    testingAccuracy, ...
    precisionMacro, ...
    recallMacro, ...
    f1Macro, ...
    meanCVAccuracy, ...
    stdCVAccuracy, ...
    'VariableNames', ...
    { ...
    'TrainingAccuracy', ...
    'TestingAccuracy', ...
    'MacroPrecision', ...
    'MacroRecall', ...
    'MacroF1', ...
    'Mean5FoldCVAccuracy', ...
    'CVStandardDeviation'});


%% ============================================================
% 27. SAVE RESULTS TO EXCEL
%% ============================================================

resultsFilename = ...
    'SmartGridShield_SVM_Results.xlsx';


writetable( ...
    overallResults, ...
    resultsFilename, ...
    'Sheet', ...
    'Overall_Results');


writetable( ...
    resultsTable, ...
    resultsFilename, ...
    'Sheet', ...
    'Per_Class_Results');


cvTable = table( ...
    (1:5)', ...
    cvAccuracy, ...
    'VariableNames', ...
    {'Fold', 'Accuracy_Percent'});


writetable( ...
    cvTable, ...
    resultsFilename, ...
    'Sheet', ...
    'Cross_Validation');


fprintf('\nResults saved to:\n');

fprintf('%s\n', resultsFilename);


%% ============================================================
% 28. SAVE TRAINED MODEL
%
% Save:
% svmModel
% features
% mu
% sigma
%
% mu and sigma are required to standardize future
% Arduino/live measurements.
%% ============================================================

modelFilename = ...
    'SmartGridShield_TinyML_SVM.mat';


save( ...
    modelFilename, ...
    'svmModel', ...
    'features', ...
    'mu', ...
    'sigma');


fprintf('\nModel saved to:\n');

fprintf('%s\n', modelFilename);


%% ============================================================
% 29. FINAL SUMMARY
%% ============================================================

fprintf('\n');
fprintf('====================================================\n');
fprintf('                FINAL SVM SUMMARY\n');
fprintf('====================================================\n');

fprintf('\n');

fprintf( ...
    'Training Accuracy : %.2f %%\n', ...
    trainingAccuracy);

fprintf( ...
    'Testing Accuracy  : %.2f %%\n', ...
    testingAccuracy);

fprintf( ...
    'Precision         : %.2f %%\n', ...
    precisionMacro);

fprintf( ...
    'Recall            : %.2f %%\n', ...
    recallMacro);

fprintf( ...
    'F1 Score          : %.2f %%\n', ...
    f1Macro);

fprintf( ...
    '5-Fold CV         : %.2f +/- %.2f %%\n', ...
    meanCVAccuracy, ...
    stdCVAccuracy);


fprintf('\n');

fprintf('Features used:\n');

for i = 1:length(features)

    fprintf('  %d. %s\n', ...
        i, ...
        features{i});

end


fprintf('\n');

fprintf('IMPORTANT:\n');

fprintf(['The reported performance is based on the supplied ', ...
    'dataset.\n']);

fprintf(['The dataset contains derived/synthetic samples ', ...
    'based on field-level observations.\n']);

fprintf(['Independent real-world PV field validation is ', ...
    'required before deployment claims.\n']);


fprintf('\n');

fprintf('====================================================\n');
fprintf('             SVM TRAINING COMPLETE\n');
fprintf('====================================================\n');